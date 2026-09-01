from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from q38_control_plane import (  # noqa: E402
    ControlPlaneError,
    Q38ControlPlane,
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
        del argument0, timeout_ms
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


if __name__ == "__main__":
    unittest.main()
