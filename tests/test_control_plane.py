from __future__ import annotations

import json
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from q38_control_plane import (  # noqa: E402
    ControlPlaneError,
    HuggingFaceCodec,
    Q38ControlPlane,
    SessionRecord,
    StreamingTextDecoder,
    block_hashes,
)
from q38_rpc_client import OPCODES, Response  # noqa: E402


class FakeTransport:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.target = 0
        self.epoch = 0
        self._active: int | None = None
        self.next_token = 100
        self.calls: list[int] = []

    @property
    def active_request_id(self) -> int | None:
        return self._active

    def response(self, opcode: int, tokens: tuple[int, ...] = (), message: str = "") -> Response:
        return Response(
            request_id=1,
            session_hash=7,
            status=0,
            opcode=opcode,
            tokens=tokens,
            message=message,
            frontiers=(
                len(self.tokens),
                self.target,
                self.target,
                self.target,
                self.target,
                self.epoch,
            ),
        )

    def call(
        self,
        opcode: int,
        *,
        tokens=None,
        argument0: int = 0,
        argument1: int = 0,
        timeout_ms: int = 0,
    ) -> Response:
        del timeout_ms
        self.calls.append(opcode)
        if opcode == OPCODES["append"]:
            values = list(tokens or ())
            self.tokens.extend(values)
            self.target = len(self.tokens)
            self.epoch += 1
            return self.response(opcode)
        if opcode == OPCODES["seed"]:
            value = self.next_token
            self.next_token += 1
            self.tokens.append(value)
            self.epoch += 1
            return self.response(opcode, (value,))
        if opcode in (OPCODES["decode"], OPCODES["spec"]):
            count = 1 if opcode == OPCODES["decode"] else argument1 + 1
            values = tuple(range(self.next_token, self.next_token + count))
            self.next_token += count
            self.target += count
            self.tokens.extend(values)
            self.epoch += 1
            return self.response(opcode, values)
        if opcode == OPCODES["state"]:
            return self.response(opcode)
        if opcode == OPCODES['replace_pending_terminal']:
            expected, replacement = tokens
            assert len(self.tokens) == argument0 and self.epoch == argument1
            assert self.target == len(self.tokens) - 1 and self.tokens[-1] == expected
            self.tokens[-1] = replacement
            self.epoch += 1
            return self.response(opcode)
        if opcode == OPCODES["reset"]:
            self.tokens.clear()
            self.target = 0
            self.epoch = 0
            return self.response(opcode, message="session state reset")
        if opcode == OPCODES["stats"]:
            return self.response(opcode, message=json.dumps({"schema": "q38.metrics.v1"}))
        raise AssertionError(opcode)

    def cancel_active(self) -> Response | None:
        if self._active is None:
            return None
        return self.response(OPCODES["cancel"])


class ControlPlaneTest(unittest.TestCase):
    def make(self, limit: int = 32) -> tuple[FakeTransport, Q38ControlPlane, str]:
        transport = FakeTransport()
        control = Q38ControlPlane(
            transport, context_limit=limit, vocabulary=1000, default_timeout_ms=1000
        )
        session = control.create_session("s1")
        return transport, control, str(session["id"])

    def test_block_hashes_are_stable_and_bounded(self) -> None:
        values = list(range(2050))
        self.assertEqual(block_hashes(values), block_hashes(tuple(values)))
        self.assertEqual(len(block_hashes(values)), 3)

    def test_incremental_hashes_cross_blocks_and_reset_to_longer_prefix(self):
        record = SessionRecord("hash", 262144, "q38")
        values = list(range(4099))
        for end in (1, 1023, 1024, 1025, 2049, 4099):
            record.canonical_tokens.extend(values[len(record.canonical_tokens):end])
            record.refresh_hashes()
            self.assertEqual(record.hashes, block_hashes(record.canonical_tokens))
        record.canonical_tokens.clear()
        record.refresh_hashes(rebuild=True)
        record.canonical_tokens.extend([17] * 5000)
        record.refresh_hashes()
        self.assertEqual(record.hashes, block_hashes(record.canonical_tokens))

    def test_interposed_writer_divergence_never_publishes(self):
        transport, control, session = self.make()
        control.append_tokens(session, [1, 2])
        stream = control.generate(session, 8, mode="mtp", mtp_width=4)
        self.assertEqual(next(stream).token_ids, (100,))
        transport.call(OPCODES["append"], tokens=[999])
        with self.assertRaises(ControlPlaneError) as caught:
            next(stream)
        self.assertEqual(caught.exception.code, "frontier_divergence")

    def test_append_and_plain_generation_publish_only_commits(self) -> None:
        transport, control, session = self.make()
        state = control.append_tokens(session, [1, 2, 3])
        self.assertEqual(state["committed_tokens"], 3)
        events = list(control.generate(session, 3))
        self.assertEqual([token for event in events for token in event.token_ids], [100, 101, 102])
        self.assertEqual(events[-1].finish_reason, "length")
        self.assertEqual(events[-1].committed_tokens, 6)
        self.assertEqual(transport.calls.count(OPCODES["seed"]), 1)
        self.assertEqual(transport.calls.count(OPCODES["decode"]), 2)

    def test_mtp_never_overshoots_exact_budget(self) -> None:
        _transport, control, session = self.make()
        control.append_tokens(session, [1])
        events = list(control.generate(session, 6, mode="mtp", mtp_width=4))
        self.assertEqual(sum(len(event.token_ids) for event in events), 6)
        self.assertEqual(events[-1].committed_tokens, 7)

    def test_mtp_capability_is_fail_closed(self) -> None:
        transport = FakeTransport()
        control = Q38ControlPlane(
            transport,
            context_limit=32,
            vocabulary=1000,
            default_timeout_ms=1000,
            mtp_enabled=False,
            mtp_max_draft=1,
        )
        session = str(control.create_session("plain-only")["id"])
        control.append_tokens(session, [1])
        with self.assertRaises(ControlPlaneError) as caught:
            list(control.generate(session, 2, mode="mtp", mtp_width=1))
        self.assertEqual(caught.exception.code, "mtp_disabled")
        self.assertNotIn(OPCODES["spec"], transport.calls)

        enabled = Q38ControlPlane(
            FakeTransport(),
            context_limit=32,
            vocabulary=1000,
            default_timeout_ms=1000,
            mtp_enabled=True,
            mtp_max_draft=1,
        )
        enabled_session = str(enabled.create_session("mtp-one")["id"])
        enabled.append_tokens(enabled_session, [1])
        with self.assertRaises(ControlPlaneError) as width:
            list(enabled.generate(enabled_session, 3, mode="mtp", mtp_width=2))
        self.assertEqual(width.exception.code, "invalid_generation_mode")

        width_four = Q38ControlPlane(
            FakeTransport(),
            context_limit=32,
            vocabulary=1000,
            default_timeout_ms=1000,
            mtp_enabled=True,
            mtp_max_draft=4,
        )
        width_four_session = str(width_four.create_session("mtp-four")["id"])
        width_four.append_tokens(width_four_session, [1])
        events = list(
            width_four.generate(
                width_four_session, 6, mode="mtp", mtp_width=4
            )
        )
        self.assertEqual(sum(len(event.token_ids) for event in events), 6)
        with self.assertRaises(ControlPlaneError) as too_wide:
            list(
                width_four.generate(
                    width_four_session, 2, mode="mtp", mtp_width=5
                )
            )
        self.assertEqual(too_wide.exception.code, "invalid_generation_mode")

    def test_full_history_rebuilds_non_extension_and_keeps_fast_path(self) -> None:
        transport, control, session = self.make()
        control.append_tokens(session, [1, 2, 3])
        state, timing = control.append_full_history(session, [1, 2, 3, 4, 5])
        self.assertEqual(state["committed_tokens"], 5)
        self.assertEqual(timing["prefix_tokens"], 3)
        self.assertFalse(timing["cold_rebuild"])
        self.assertEqual(timing["cache_status"], "incremental_hit")
        state, timing = control.append_full_history(session, [1, 9, 3, 4, 5])
        self.assertEqual(control.session(session)["committed_tokens"], 5)
        self.assertEqual(state["committed_tokens"], 5)
        self.assertEqual(timing["prefix_tokens"], 0)
        self.assertEqual(timing["suffix_tokens"], 5)
        self.assertTrue(timing["cold_rebuild"])
        self.assertEqual(timing["cache_status"], "cold_rebuild")
        self.assertEqual(transport.calls.count(OPCODES["reset"]), 1)

        _state, timing = control.append_full_history(
            session, [1, 9, 3, 4, 5, 6]
        )
        self.assertFalse(timing["cold_rebuild"])
        self.assertEqual(timing["cache_status"], "incremental_hit")
        self.assertEqual(timing["prefix_tokens"], 5)

    def test_replacing_explicit_session_evicts_resident_state(self) -> None:
        transport, control, session = self.make()
        control.append_tokens(session, [1, 2, 3])
        replacement = control.replace_session("s2")
        self.assertEqual(replacement["id"], "s2")
        self.assertEqual(replacement["committed_tokens"], 0)
        self.assertTrue(replacement["cold_rebuild"])
        state, timing = control.append_full_history("s2", [7, 8])
        self.assertEqual(state["committed_tokens"], 2)
        self.assertTrue(timing["cold_rebuild"])
        self.assertEqual(transport.tokens, [7, 8])

    def test_full_history_continues_after_generated_pending_token(self) -> None:
        transport, control, session = self.make()
        control.append_tokens(session, [1, 2, 3])
        generated = list(control.generate(session, 2))
        history = [1, 2, 3] + [
            token for event in generated for token in event.token_ids
        ]
        state, timing = control.append_full_history(session, history + [4, 5, 6])
        self.assertEqual(state["committed_tokens"], 8)
        self.assertEqual(timing["prefix_tokens"], 5)
        self.assertEqual(timing["suffix_tokens"], 3)
        self.assertEqual(transport.target, 8)

    def test_budget_is_hard_and_non_mutating(self) -> None:
        _transport, control, session = self.make(limit=4)
        control.append_tokens(session, [1, 2, 3])
        with self.assertRaises(ControlPlaneError) as caught:
            list(control.generate(session, 2))
        self.assertEqual(caught.exception.code, "context_length_exceeded")
        self.assertEqual(control.session(session)["committed_tokens"], 3)

    def test_metrics_are_versioned_json(self) -> None:
        _transport, control, session = self.make()
        _state, timing = control.append_full_history(session, [1, 2])
        self.assertEqual(timing["cache_status"], "cold_start")
        metrics = control.metrics()
        self.assertEqual(metrics["schema"], "q38.metrics.v1")
        self.assertEqual(metrics["session_cache"]["cold_starts"], 1)
        self.assertEqual(metrics["session_cache"]["resident_slots"], 1)

    def test_only_one_live_session(self) -> None:
        _transport, control, _session = self.make()
        with self.assertRaises(ControlPlaneError) as caught:
            control.create_session("s2")
        self.assertEqual(caught.exception.code, "session_exists")

    def test_no_active_cancel_fails_without_state_change(self) -> None:
        _transport, control, session = self.make()
        with self.assertRaises(ControlPlaneError) as caught:
            control.cancel()
        self.assertEqual(caught.exception.code, "no_active_writer")
        self.assertEqual(control.session(session)["committed_tokens"], 0)


class TerminalPrefixTests(unittest.TestCase):
    def make(self, enabled=True):
        transport = FakeTransport()
        control = Q38ControlPlane(transport, context_limit=100, vocabulary=1000,
                                 stop_token_ids=[100], reuse_terminal_prefix=enabled)
        control.create_session('chat')
        list(control.complete_full_history('chat', [1, 2], 4))
        self.assertEqual(control._session.canonical_tokens, [1, 2, 100])
        return transport, control

    def test_unprocessed_terminal_reuses_only_the_evaluated_prefix(self):
        transport, control = self.make()
        count = transport.calls.count(OPCODES['reset'])
        _state, timing = control.append_full_history('chat', [1, 2, 3, 4])
        self.assertEqual(transport.calls.count(OPCODES['reset']), count)
        self.assertEqual(transport.calls.count(OPCODES['replace_pending_terminal']), 1)
        self.assertEqual(timing['prefix_tokens'], 2)
        self.assertEqual(timing['suffix_tokens'], 2)
        self.assertTrue(timing['terminal_prefix_reused'])
        self.assertEqual(control._session.canonical_tokens, [1, 2, 3, 4])
        self.assertEqual(control._session.hashes, block_hashes([1, 2, 3, 4]))
        self.assertEqual(control.metrics()['session_cache']['terminal_prefix_hits'], 1)

    def test_disabled_divergent_or_evaluated_terminal_keeps_cold_rebuild(self):
        for mode in ('disabled', 'divergent', 'evaluated', 'nonterminal'):
            transport, control = self.make(mode != 'disabled')
            if mode == 'evaluated':
                transport.target = len(transport.tokens)
            elif mode == 'nonterminal':
                transport.tokens[-1] = control._session.canonical_tokens[-1] = 101
            values = [1, 2, 3, 4] if mode != 'divergent' else [8, 2, 3, 4]
            before = transport.calls.count(OPCODES['reset'])
            _state, timing = control.append_full_history('chat', values)
            self.assertTrue(timing['cold_rebuild'])
            self.assertFalse(timing['terminal_prefix_reused'])
            self.assertEqual(transport.calls.count(OPCODES['reset']), before + 1)
            self.assertNotIn(OPCODES['replace_pending_terminal'], transport.calls)

    def test_declined_operation_falls_back_only_with_unchanged_frontiers(self):
        transport, control = self.make()
        original = transport.call
        def call(opcode, **kwargs):
            if opcode == OPCODES['replace_pending_terminal']:
                state = transport.response(opcode)
                return Response(**{**state.__dict__, 'status': 1})
            return original(opcode, **kwargs)
        transport.call = call
        _, timing = control.append_full_history('chat', [1, 2, 3, 4])
        self.assertTrue(timing['cold_rebuild'])
        self.assertFalse(timing['terminal_prefix_reused'])
        self.assertEqual(transport.tokens, [1, 2, 3, 4])

    def test_cancel_after_replacement_preserves_committed_bookkeeping(self):
        transport, control = self.make()
        stop = threading.Event()
        original = transport.call
        def call(opcode, **kwargs):
            if opcode == OPCODES['append'] and stop.is_set():
                raise ControlPlaneError(499, 'stream_cancelled', 'stream was cancelled')
            response = original(opcode, **kwargs)
            if opcode == OPCODES['replace_pending_terminal']:
                stop.set()
            return response
        transport.call = call
        with self.assertRaises(ControlPlaneError):
            list(control.complete_full_history('chat', [1, 2, 3, 4], 2, stop_event=stop))
        self.assertEqual(control._session.canonical_tokens, [1, 2, 3])
        self.assertEqual(transport.tokens, [1, 2, 3])
        stop.clear()
        _, timing = control.append_full_history('chat', [1, 2, 3, 4])
        self.assertFalse(timing['cold_rebuild'])
        self.assertEqual(transport.tokens, [1, 2, 3, 4])


class CodecTerminationTests(unittest.TestCase):
    def test_terminal_tokens_hide_suffix_without_losing_parser_tags(self):
        pieces = {1: "<think>", 2: "reason", 3: "</think>",
                  4: '<tool_call>{"name":"x"}</tool_call>',
                  5: "<|im_end|>", 6: "\n", 7: "<|endoftext|>"}
        def decode(tokens, **options):
            self.assertFalse(options["skip_special_tokens"])
            return "".join(pieces[t] for t in tokens)
        tokenizer = SimpleNamespace(eos_token_id=7,
            get_vocab=lambda: {v: k for k, v in pieces.items()}, decode=decode)
        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: tokenizer))
        with patch.dict(sys.modules, {"transformers": module}):
            codec = HuggingFaceCodec("unused")
        prefix = [1, 2, 3, 4]
        expected = "".join(pieces[t] for t in prefix)
        for end in (5, 7):
            sequence = prefix + [end, 6, 7]
            self.assertEqual(codec.decode(sequence), expected)
            chunks, previous = [], ""
            for length in range(1, len(sequence) + 1):
                current = codec.decode(sequence[:length])
                self.assertTrue(current.startswith(previous))
                chunks.append(current[len(previous):])
                previous = current
            self.assertEqual("".join(chunks), expected)
        self.assertEqual(codec.stop_token_ids, (7,))


class StreamingCodecTests(unittest.TestCase):
    def test_split_unicode_and_truncated_bytes_match_final_decode(self):
        class Codec:
            output_end_token_ids = {999}
            def decode(self, tokens):
                return bytes(tokens).decode('utf-8', errors='replace')
        for text in ('中文🙂 café\n', 'x\ufffdy', '日本語 👩🏽\u200d💻'):
            values = list(text.encode())
            for width in (1, 2, 5):
                decoder = StreamingTextDecoder(Codec())
                chunks = [decoder.push(values[i:i + width]) for i in range(0, len(values), width)]
                chunks.append(decoder.push([999, 65], final=True))
                self.assertEqual(''.join(chunks), text)
            decoder = StreamingTextDecoder(Codec())
            values = list('中文🙂'.encode())[:-1]
            pieces = [decoder.push([v]) for v in values]
            pieces.append(decoder.push([], final=True))
            self.assertEqual(''.join(pieces), Codec().decode(values))
            self.assertNotIn('\ufffd', ''.join(pieces[:-1]))
            with self.assertRaises(RuntimeError):
                decoder.push([])

    def test_native_bytelevel_stream_uses_one_final_batch_decode(self):
        try:
            from tokenizers import Tokenizer, decoders, models
        except ImportError:
            self.skipTest('tokenizers optional; exercised on the deployment venv')
        byte_chars = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        table = {b: chr(b) for b in byte_chars}
        for extra, value in enumerate(b for b in range(256) if b not in table):
            table[value] = chr(256 + extra)
        backend = Tokenizer(models.BPE({table[b]: b for b in range(256)}, []))
        backend.decoder = decoders.ByteLevel()
        class Codec:
            output_end_token_ids = {999}
            tokenizer = SimpleNamespace(backend_tokenizer=backend)
            calls = 0
            def decode(self, tokens):
                self.calls += 1
                return backend.decode(tokens, skip_special_tokens=False)
        for raw in ('中文🙂 café\n'.encode(), 'x\ufffdy'.encode(), '👩🏽\u200d💻'.encode(), '中文🙂'.encode()[:-1]):
            codec = Codec()
            decoder = StreamingTextDecoder(codec)
            self.assertIsNotNone(decoder.stream)
            chunks = [decoder.push([b]) for b in raw]
            chunks.append(decoder.push([999, 65], final=True))
            self.assertEqual(''.join(chunks), backend.decode(list(raw), skip_special_tokens=False))
            self.assertEqual(codec.calls, 1)


if __name__ == "__main__":
    unittest.main()
