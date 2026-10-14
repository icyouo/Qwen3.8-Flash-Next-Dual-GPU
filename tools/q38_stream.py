"""Bounded, request-owned handoff of committed events to an SSE consumer."""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar('T')
_END = object()


class CommittedEventPump(Generic[T]):
    def __init__(self, source: Iterator[T], stop: threading.Event,
                 cancel: Callable[[threading.Event], None], capacity: int = 2):
        if not 1 <= capacity <= 16:
            raise ValueError('stream queue capacity must be in 1..16')
        self.source = source
        self.stop = stop
        self.cancel = cancel
        self.queue: queue.Queue = queue.Queue(capacity)
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._produce, name='q38-committed-events')

    def _put(self, value: object) -> bool:
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=.025)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self) -> None:
        try:
            while not self.stop.is_set():
                try:
                    event = next(self.source)
                except StopIteration:
                    break
                if not self._put(event):
                    break
        except BaseException as error:
            self.error = error
        finally:
            # A generator holding the writer lease must be closed on this
            # producer thread, including consumer disconnect/backpressure.
            try:
                close = getattr(self.source, 'close', None)
                if close is not None:
                    close()
            except BaseException as error:
                if self.error is None:
                    self.error = error
            finally:
                if not self.stop.is_set():
                    self._put(_END)
                self.done.set()

    def __enter__(self) -> 'CommittedEventPump[T]':
        self.thread.start()
        return self

    def __iter__(self) -> Iterator[T]:
        while True:
            try:
                value = self.queue.get(timeout=.025)
            except queue.Empty:
                if not self.done.is_set():
                    continue
                if self.error is not None:
                    raise self.error
                if self.stop.is_set():
                    # Use the control-plane error type only on the cancelled
                    # path; the pump has no model/tokenizer dependency.
                    from q38_control_plane import ControlPlaneError
                    raise ControlPlaneError(499, 'stream_cancelled', 'stream was cancelled')
                return
            if self.stop.is_set():
                from q38_control_plane import ControlPlaneError
                raise ControlPlaneError(499, 'stream_cancelled', 'stream was cancelled')
            if value is _END:
                if self.error is not None:
                    raise self.error
                return
            yield value

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        try:
            if not self.done.is_set():
                self.cancel(self.stop)
        finally:
            self.thread.join()
