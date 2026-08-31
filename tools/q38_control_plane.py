#!/usr/bin/env python3
"""Built-in control-plane core for the Q38 ExecutorRPC Unix socket.

The control plane accepts token-native requests, validates exact-prefix
continuations, and consumes only committed token events.  q38_sidecar.py uses
this core to provide the repository's OpenAI-compatible HTTP/SSE service.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, Sequence

from q38_rpc_client import Client, OPCODES, Response, STATUSES


WRITER_OPCODES = {
    OPCODES["append"],
    OPCODES["seed"],
    OPCODES["decode"],
    OPCODES["spec"],
}
TOKEN = struct.Struct("<i")


class ControlPlaneError(RuntimeError):
    """An error with an HTTP-compatible status and stable machine code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, object]:
        return {
            "error": {
                "message": self.message,
                "type": "q38_control_plane_error",
                "code": self.code,
            }
        }


class ExecutorTransportProtocol(Protocol):
    @property
    def active_request_id(self) -> int | None: ...

    def call(
        self,
        opcode: int,
        *,
        tokens: Sequence[int] | None = None,
        argument0: int = 0,
        argument1: int = 0,
        timeout_ms: int = 0,
    ) -> Response: ...

    def cancel_active(self) -> Response | None: ...


class ExecutorTransport:
    """One-call-per-connection transport with an out-of-band cancel lane."""

    def __init__(self, socket_path: str, session_hash: int) -> None:
        self.socket_path = socket_path
        self.session_hash = session_hash
        self._lock = threading.Lock()
        self._next_request_id = max(1, time.time_ns())
        self._active_request_id: int | None = None

    def _reserve_request_id(self) -> int:
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            return request_id

    @property
    def active_request_id(self) -> int | None:
        with self._lock:
            return self._active_request_id

    def call(
        self,
        opcode: int,
        *,
        tokens: Sequence[int] | None = None,
        argument0: int = 0,
        argument1: int = 0,
        timeout_ms: int = 0,
    ) -> Response:
        request_id = self._reserve_request_id()
        writer = opcode in WRITER_OPCODES
        if writer:
            with self._lock:
                if self._active_request_id is not None:
                    raise ControlPlaneError(
                        409, "writer_busy", "another executor writer is active"
                    )
                self._active_request_id = request_id
        try:
            with Client(self.socket_path, self.session_hash, request_id) as client:
                response = client.call(
                    opcode,
                    tokens=list(tokens or ()),
                    argument0=argument0,
                    argument1=argument1,
                    timeout_ms=timeout_ms,
                )
        finally:
            if writer:
                with self._lock:
                    if self._active_request_id == request_id:
                        self._active_request_id = None
        return response

    def cancel_active(self) -> Response | None:
        request_id = self.active_request_id
        if request_id is None:
            return None
        # ExecutorRPC V1 deliberately uses the Cancel frame's request_id as
        # the target writer id.  A separate connection bypasses writer order.
        with Client(self.socket_path, self.session_hash, request_id) as client:
            return client.call(OPCODES["cancel"])


def _require_ok(response: Response, operation: str) -> Response:
    if response.status == 0:
        return response
    status_name = STATUSES.get(response.status, f"status_{response.status}")
    http_status = {
        1: 400,
        2: 409,
        3: 422,
        5: 503,
        6: 499,
        7: 504,
    }.get(response.status, 500)
    raise ControlPlaneError(
        http_status,
        f"executor_{status_name}",
        f"{operation} failed: {response.message or status_name}",
    )


def _token_bytes(tokens: Sequence[int]) -> bytes:
    return b"".join(TOKEN.pack(int(token)) for token in tokens)


def block_hashes(tokens: Sequence[int], block_tokens: int = 1024) -> tuple[str, ...]:
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    return tuple(
        hashlib.sha256(_token_bytes(tokens[offset : offset + block_tokens])).hexdigest()
        for offset in range(0, len(tokens), block_tokens)
    )


@dataclass
class SessionRecord:
    session_id: str
    context_limit: int
    model: str
    canonical_tokens: list[int] = field(default_factory=list)
    revision: int = 0
    cold_rebuild: bool = False
    hashes: tuple[str, ...] = ()

    def refresh_hashes(self) -> None:
        self.hashes = block_hashes(self.canonical_tokens)

    def public(self) -> dict[str, object]:
        committed = len(self.canonical_tokens)
        return {
            "id": self.session_id,
            "object": "q38.session",
            "model": self.model,
            "revision": self.revision,
            "committed_tokens": committed,
            "remaining_tokens": self.context_limit - committed,
            "context_limit": self.context_limit,
            "cold_rebuild": self.cold_rebuild,
        }


@dataclass(frozen=True)
class CommittedTokenEvent:
    token_ids: tuple[int, ...]
    revision: int
    committed_tokens: int
    remaining_tokens: int
    finish_reason: str | None = None


class Q38ControlPlane:
    """Single-session semantic owner above ExecutorRPC V1.

    The native executor is the authoritative state machine.  This class owns
    only canonical input bookkeeping needed for exact-prefix admission and API
    metadata.  It never publishes a token before an OK writer response.
    """

    def __init__(
        self,
        transport: ExecutorTransportProtocol,
        *,
        context_limit: int = 262_144,
        model: str = "Qwen3.8-Flash-Next",
        vocabulary: int = 248_320,
        stop_token_ids: Sequence[int] = (),
        default_timeout_ms: int = 3_600_000,
        mtp_enabled: bool = True,
        mtp_max_draft: int = 64,
    ) -> None:
        if (
            context_limit <= 0
            or vocabulary <= 0
            or default_timeout_ms <= 0
            or mtp_max_draft <= 0
        ):
            raise ValueError("invalid control-plane limits")
        self.transport = transport
        self.context_limit = context_limit
        self.model = model
        self.vocabulary = vocabulary
        self.stop_token_ids = frozenset(int(token) for token in stop_token_ids)
        self.default_timeout_ms = default_timeout_ms
        self.mtp_enabled = mtp_enabled
        self.mtp_max_draft = mtp_max_draft
        self._writer = threading.Lock()
        self._state = threading.Lock()
        self._session: SessionRecord | None = None

    def _read_state(self) -> Response:
        return _require_ok(self.transport.call(OPCODES["state"]), "state")

    def create_session(self, session_id: str | None = None) -> dict[str, object]:
        with self._writer:
            with self._state:
                if self._session is not None:
                    raise ControlPlaneError(
                        409,
                        "session_exists",
                        "this executor already owns a live session",
                    )
            response = self._read_state()
            if any(response.frontiers[:5]):
                raise ControlPlaneError(
                    409,
                    "executor_not_empty",
                    "executor state is not empty; cold rebuild or a new executor is required",
                )
            created = SessionRecord(
                session_id=session_id or f"q38-{uuid.uuid4().hex}",
                context_limit=self.context_limit,
                model=self.model,
            )
            with self._state:
                self._session = created
            return created.public()

    def _session_checked(self, session_id: str) -> SessionRecord:
        with self._state:
            if self._session is None:
                raise ControlPlaneError(404, "session_not_found", "session not found")
            if self._session.session_id != session_id:
                raise ControlPlaneError(404, "session_not_found", "session not found")
            return self._session

    def session(self, session_id: str) -> dict[str, object]:
        record = self._session_checked(session_id)
        return record.public()

    def active_session(self) -> dict[str, object] | None:
        with self._state:
            return None if self._session is None else self._session.public()

    def _validate_tokens(self, tokens: Sequence[int]) -> list[int]:
        result = [int(token) for token in tokens]
        if any(token < 0 or token >= self.vocabulary for token in result):
            raise ControlPlaneError(
                400, "invalid_token", "token id is outside the model vocabulary"
            )
        return result

    def _require_budget(self, record: SessionRecord, additional: int) -> None:
        if additional < 0 or len(record.canonical_tokens) + additional > self.context_limit:
            raise ControlPlaneError(
                422,
                "context_length_exceeded",
                "prompt plus requested generation exceeds the 262,144-token budget",
            )

    def _verify_executor_frontiers(
        self, response: Response, expected_canonical: int
    ) -> None:
        canonical, target, stage0, stage1, _draft, _epoch = response.frontiers
        if canonical != expected_canonical or stage0 != target or stage1 != target:
            raise ControlPlaneError(
                503,
                "frontier_divergence",
                "executor and control-plane frontiers diverged; session is isolated",
            )
        if target not in (canonical, canonical - 1 if canonical else 0):
            raise ControlPlaneError(
                503,
                "frontier_divergence",
                "executor pending-token range violates the Q38 contract",
            )

    def _append_locked(
        self,
        record: SessionRecord,
        tokens: Sequence[int],
        timeout_ms: int,
    ) -> None:
        values = self._validate_tokens(tokens)
        if not values:
            return
        self._require_budget(record, len(values))
        response = _require_ok(
            self.transport.call(
                OPCODES["append"], tokens=values, timeout_ms=timeout_ms
            ),
            "append",
        )
        expected = len(record.canonical_tokens) + len(values)
        self._verify_executor_frontiers(response, expected)
        record.canonical_tokens.extend(values)
        record.revision += 1
        record.refresh_hashes()

    def append_tokens(
        self,
        session_id: str,
        tokens: Sequence[int],
        *,
        timeout_ms: int | None = None,
    ) -> dict[str, object]:
        with self._writer:
            record = self._session_checked(session_id)
            self._append_locked(record, tokens, timeout_ms or self.default_timeout_ms)
            return record.public()

    def _append_full_history_locked(
        self,
        record: SessionRecord,
        history_tokens: Sequence[int],
        timeout_ms: int,
    ) -> dict[str, int]:
        started = time.perf_counter_ns()
        values = self._validate_tokens(history_tokens)
        canonical = record.canonical_tokens
        matching = len(values) >= len(canonical) and values[: len(canonical)] == canonical
        validation_ns = time.perf_counter_ns() - started
        if not matching:
            record.cold_rebuild = True
            raise ControlPlaneError(
                409,
                "cold_rebuild_required",
                "full-history tokens are not an exact extension of the live session",
            )
        suffix = values[len(canonical) :]
        self._append_locked(record, suffix, timeout_ms)
        return {
            "prefix_tokens": len(values) - len(suffix),
            "suffix_tokens": len(suffix),
            "prefix_validation_ns": validation_ns,
        }

    def append_full_history(
        self,
        session_id: str,
        history_tokens: Sequence[int],
        *,
        timeout_ms: int | None = None,
    ) -> tuple[dict[str, object], dict[str, int]]:
        with self._writer:
            record = self._session_checked(session_id)
            timing = self._append_full_history_locked(
                record, history_tokens, timeout_ms or self.default_timeout_ms
            )
            return record.public(), timing

    def _generate_locked(
        self,
        record: SessionRecord,
        max_new_tokens: int,
        *,
        mode: str,
        mtp_width: int,
        timeout_ms: int,
    ) -> Iterator[CommittedTokenEvent]:
        if max_new_tokens <= 0:
            raise ControlPlaneError(
                400, "invalid_max_tokens", "max_new_tokens must be positive"
            )
        if mode not in ("plain", "mtp") or not 1 <= mtp_width <= 64:
            raise ControlPlaneError(400, "invalid_generation_mode", "invalid mode/width")
        if mode == "mtp" and not self.mtp_enabled:
            raise ControlPlaneError(
                409, "mtp_disabled", "MTP is disabled for this executor"
            )
        if mode == "mtp" and mtp_width > self.mtp_max_draft:
            raise ControlPlaneError(
                400,
                "invalid_generation_mode",
                f"MTP draft width exceeds this checkpoint's {self.mtp_max_draft}-token limit",
            )
        self._require_budget(record, max_new_tokens)
        if not record.canonical_tokens:
            raise ControlPlaneError(409, "empty_prompt", "generation requires a prompt")

        generated = 0
        while generated < max_new_tokens:
            state = self._read_state()
            self._verify_executor_frontiers(state, len(record.canonical_tokens))
            canonical, target = state.frontiers[:2]
            remaining = max_new_tokens - generated
            if canonical == target:
                opcode = OPCODES["seed"]
                argument1 = 0
            elif mode == "mtp" and remaining > 1:
                opcode = OPCODES["spec"]
                # A speculative transaction can publish at most width + 1.
                argument1 = min(mtp_width, remaining - 1)
            else:
                opcode = OPCODES["decode"]
                argument1 = 0
            response = _require_ok(
                self.transport.call(
                    opcode,
                    argument0=1 if opcode in (OPCODES["decode"], OPCODES["spec"]) else 0,
                    argument1=argument1,
                    timeout_ms=timeout_ms,
                ),
                "generate",
            )
            if not response.tokens or len(response.tokens) > remaining:
                raise ControlPlaneError(
                    503,
                    "invalid_commit_event",
                    "executor returned an empty or oversized committed token event",
                )
            values = self._validate_tokens(response.tokens)
            expected = len(record.canonical_tokens) + len(values)
            self._verify_executor_frontiers(response, expected)
            record.canonical_tokens.extend(values)
            record.revision += 1
            record.refresh_hashes()
            generated += len(values)
            hit_stop = any(token in self.stop_token_ids for token in values)
            finish = "stop" if hit_stop else (
                "length" if generated == max_new_tokens else None
            )
            yield CommittedTokenEvent(
                token_ids=tuple(values),
                revision=record.revision,
                committed_tokens=len(record.canonical_tokens),
                remaining_tokens=self.context_limit - len(record.canonical_tokens),
                finish_reason=finish,
            )
            if hit_stop:
                return

    def generate(
        self,
        session_id: str,
        max_new_tokens: int,
        *,
        mode: str = "plain",
        mtp_width: int = 1,
        timeout_ms: int | None = None,
    ) -> Iterator[CommittedTokenEvent]:
        with self._writer:
            record = self._session_checked(session_id)
            yield from self._generate_locked(
                record,
                max_new_tokens,
                mode=mode,
                mtp_width=mtp_width,
                timeout_ms=timeout_ms or self.default_timeout_ms,
            )

    def complete_full_history(
        self,
        session_id: str,
        history_tokens: Sequence[int],
        max_new_tokens: int,
        *,
        mode: str = "plain",
        mtp_width: int = 1,
        timeout_ms: int | None = None,
    ) -> Iterator[tuple[CommittedTokenEvent, dict[str, int]]]:
        """Append an exact suffix and generate under one semantic writer lease."""

        with self._writer:
            record = self._session_checked(session_id)
            effective_timeout = timeout_ms or self.default_timeout_ms
            timing = self._append_full_history_locked(
                record, history_tokens, effective_timeout
            )
            for event in self._generate_locked(
                record,
                max_new_tokens,
                mode=mode,
                mtp_width=mtp_width,
                timeout_ms=effective_timeout,
            ):
                yield event, timing

    def cancel(self) -> dict[str, object]:
        target = self.transport.active_request_id
        if target is None:
            raise ControlPlaneError(409, "no_active_writer", "no writer is active")
        response = self.transport.cancel_active()
        if response is None:
            raise ControlPlaneError(409, "no_active_writer", "no writer is active")
        _require_ok(response, "cancel")
        return {"cancelled_request_id": target, "status": "cancel_requested"}

    def metrics(self) -> dict[str, object]:
        response = _require_ok(self.transport.call(OPCODES["stats"]), "stats")
        try:
            return json.loads(response.message)
        except json.JSONDecodeError as error:
            raise ControlPlaneError(
                503, "invalid_metrics", "executor returned invalid metrics JSON"
            ) from error


class HuggingFaceCodec:
    """Optional tokenizer codec for the built-in text-request endpoint."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "transformers is required for the OpenAI-compatible tokenizer path"
            ) from error
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        eos = self.tokenizer.eos_token_id
        self.stop_token_ids = () if eos is None else (int(eos),)

    def encode_chat(self, messages: Sequence[dict[str, object]]) -> list[int]:
        values = self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True
        )
        return [int(token) for token in values]

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(tokens), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
