from __future__ import annotations
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
from q38_stream import CommittedEventPump
from q38_control_plane import ControlPlaneError, ExecutorTransport, Q38ControlPlane
from test_control_plane import FakeTransport


class StreamPumpTests(unittest.TestCase):
    def test_order_and_source_close_belong_to_the_producer(self):
        closed = []
        def source():
            try:
                yield from range(31)
            finally:
                closed.append(threading.get_ident())
        stop = threading.Event()
        with CommittedEventPump(source(), stop, lambda e: e.set(), 2) as pump:
            self.assertEqual(list(pump), list(range(31)))
        self.assertEqual(closed, [pump.thread.ident])
        self.assertFalse(pump.thread.is_alive())

    def test_backpressure_is_bounded_and_disconnect_drains(self):
        ready = threading.Event()
        produced, closed = [], []
        def source():
            try:
                for i in range(100):
                    produced.append(i)
                    if i == 1:
                        ready.set()
                    yield i
            finally:
                closed.append(threading.get_ident())
        with CommittedEventPump(source(), threading.Event(), lambda e: e.set(), 1) as pump:
            self.assertTrue(ready.wait(2))
            self.assertEqual(produced, [0, 1])
        self.assertEqual(produced, [0, 1])
        self.assertEqual(closed, [pump.thread.ident])
        self.assertFalse(pump.thread.is_alive())

    def test_error_follows_committed_prefix_and_releases_thread(self):
        def source():
            yield 17
            raise ValueError('failed writer')
        with CommittedEventPump(source(), threading.Event(), lambda e: e.set()) as pump:
            iterator = iter(pump)
            self.assertEqual(next(iterator), 17)
            with self.assertRaisesRegex(ValueError, 'failed writer'):
                next(iterator)
        self.assertFalse(pump.thread.is_alive())

    def test_cancel_between_rpcs_releases_lease_and_next_request_recovers(self):
        transport = FakeTransport()
        control = Q38ControlPlane(transport, context_limit=200, vocabulary=1000)
        control.create_session('s1')
        stop = threading.Event()
        source = control.complete_full_history('s1', [1, 2], 80, mode='mtp', mtp_width=4, stop_event=stop)
        with CommittedEventPump(source, stop, control.stop_stream, 1) as pump:
            self.assertIsNotNone(next(iter(pump)))
            result = control.cancel()
            self.assertEqual(result['status'], 'cancel_requested')
            self.assertTrue(stop.is_set())
            with self.assertRaises(ControlPlaneError) as caught:
                list(pump)
            self.assertEqual(caught.exception.code, 'stream_cancelled')
        self.assertFalse(pump.thread.is_alive())
        self.assertEqual(control._session.canonical_tokens, transport.tokens)
        self.assertEqual(sum(len(e.token_ids) for e, _ in control.complete_full_history('s1', [1, 2], 8)), 8)

    def test_disconnect_cannot_cancel_a_later_request_owner(self):
        transport = ExecutorTransport('/unused', 7)
        old, new = threading.Event(), threading.Event()
        transport._active_owner, transport._active_request_id = new, 42
        with patch.object(transport, '_cancel_request', return_value=None) as cancel:
            transport.cancel_owned(old)
            cancel.assert_called_once_with(None)
        self.assertTrue(old.is_set())
        self.assertFalse(new.is_set())
        with patch.object(transport, '_cancel_request', return_value=None) as cancel:
            transport.cancel_owned(new)
            cancel.assert_called_once_with(42)

    def test_stopped_owner_never_admits_another_writer(self):
        transport = ExecutorTransport('/unused', 7)
        stop = threading.Event()
        stop.set()
        transport._connection_scope.stop = stop
        with self.assertRaises(ControlPlaneError) as caught:
            from q38_rpc_client import OPCODES
            transport.call(OPCODES['decode'])
        self.assertEqual(caught.exception.code, 'stream_cancelled')
        self.assertIsNone(transport.active_request_id)


if __name__ == '__main__':
    unittest.main()
