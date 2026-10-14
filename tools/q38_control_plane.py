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
from collections.abc import Mapping
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator, Protocol, Sequence

from q38_rpc_client import Client, OPCODES, Response, STATUSES


WRITER_OPCODES = {
    OPCODES["append"],
    OPCODES["seed"],
    OPCODES["decode"],
    OPCODES["spec"],
    OPCODES["reset"],
    OPCODES["replace_pending_terminal"],
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
        self._connection_scope = threading.local()
        self._active_owner: threading.Event | None = None

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
        owner = getattr(self._connection_scope, "stop", None)
        if writer:
            with self._lock:
                if owner is not None and owner.is_set():
                    raise ControlPlaneError(499, "stream_cancelled", "stream was cancelled")
                if self._active_request_id is not None:
                    raise ControlPlaneError(
                        409, "writer_busy", "another executor writer is active"
                    )
                self._active_request_id = request_id
                self._active_owner = owner
        try:
            scoped = getattr(self._connection_scope, "client", None)
            connection = nullcontext(scoped) if scoped is not None else Client(self.socket_path, self.session_hash, request_id)
            with connection as client:
                client._next_request_id = request_id
                # Never retry a failed writer: it may already have committed.
                response = client.call(opcode, tokens=list(tokens or ()),
                    argument0=argument0, argument1=argument1, timeout_ms=timeout_ms)
        finally:
            if writer:
                with self._lock:
                    if self._active_request_id == request_id:
                        self._active_request_id = None
                        self._active_owner = None
        return response

    @contextmanager
    def writer_connection(self, stop_event: threading.Event | None = None):
        """Reuse one connection only within a single semantic writer lease."""
        if getattr(self._connection_scope, "client", None) is not None:
            yield
            return
        client = Client(self.socket_path, self.session_hash)
        self._connection_scope.client = client
        self._connection_scope.stop = stop_event
        try:
            yield
        finally:
            self._connection_scope.client = None
            self._connection_scope.stop = None
            client.close()

    def cancel_active(self) -> Response | None:
        with self._lock:
            request_id = self._active_request_id
            if self._active_owner is not None:
                self._active_owner.set()
        return self._cancel_request(request_id)

    def cancel_owned(self, owner: threading.Event) -> Response | None:
        """Stop this producer, and cancel only the RPC admitted for its owner."""
        with self._lock:
            owner.set()
            request_id = self._active_request_id if self._active_owner is owner else None
        return self._cancel_request(request_id)

    def _cancel_request(self, request_id: int | None) -> Response | None:
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

    _hashed_tokens: int = field(default=0, init=False, repr=False, compare=False)
    _complete_hashes: list[str] = field(default_factory=list, init=False, repr=False, compare=False)
    _partial_hash: object = field(default_factory=hashlib.sha256, init=False, repr=False, compare=False)

    def refresh_hashes(self, *, rebuild: bool = False) -> None:
        if rebuild or len(self.canonical_tokens) < self._hashed_tokens:
            self._hashed_tokens = 0
            self._complete_hashes.clear()
            self._partial_hash = hashlib.sha256()
        while self._hashed_tokens < len(self.canonical_tokens):
            boundary = (self._hashed_tokens // 1024 + 1) * 1024
            end = min(boundary, len(self.canonical_tokens))
            self._partial_hash.update(_token_bytes(self.canonical_tokens[self._hashed_tokens:end]))
            self._hashed_tokens = end
            if end == boundary:
                self._complete_hashes.append(self._partial_hash.hexdigest())
                self._partial_hash = hashlib.sha256()
        self.hashes = tuple(self._complete_hashes)
        if self._hashed_tokens % 1024:
            self.hashes += (self._partial_hash.hexdigest(),)

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
        reuse_terminal_prefix: bool = False,
    ) -> None:
        if (
            context_limit <= 0
            or vocabulary <= 0
            or default_timeout_ms <= 0
            or not 1 <= mtp_max_draft <= 64
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
        self.reuse_terminal_prefix = reuse_terminal_prefix
        self._writer = threading.Lock()
        self._state = threading.Lock()
        self._session: SessionRecord | None = None
        self._stream_guard = threading.Lock()
        self._stream_stop: threading.Event | None = None
        self._cache_counters = {
            "cold_starts": 0,
            "cold_rebuilds": 0,
            "incremental_hits": 0,
            "terminal_prefix_hits": 0,
        }

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

    def replace_session(self, session_id: str | None = None) -> dict[str, object]:
        """Atomically evict the resident logical session and activate a new one."""

        with self._writer:
            response = _require_ok(
                self.transport.call(
                    OPCODES["reset"], timeout_ms=self.default_timeout_ms
                ),
                "reset",
            )
            self._verify_reset_frontiers(response)
            replacement = SessionRecord(
                session_id=session_id or f"q38-{uuid.uuid4().hex}",
                context_limit=self.context_limit,
                model=self.model,
                revision=1,
                cold_rebuild=True,
            )
            with self._state:
                self._session = replacement
            return replacement.public()

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

    def _verify_reset_frontiers(self, response: Response) -> None:
        if any(response.frontiers[:5]):
            raise ControlPlaneError(
                503,
                "reset_divergence",
                "executor reset did not clear every semantic frontier",
            )

    def _reset_record_locked(
        self, record: SessionRecord, timeout_ms: int
    ) -> int:
        started = time.perf_counter_ns()
        response = _require_ok(
            self.transport.call(OPCODES["reset"], timeout_ms=timeout_ms),
            "reset",
        )
        self._verify_reset_frontiers(response)
        record.canonical_tokens.clear()
        record.refresh_hashes(rebuild=True)
        record.revision += 1
        record.cold_rebuild = True
        return time.perf_counter_ns() - started

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
    ) -> dict[str, int | bool | str]:
        started = time.perf_counter_ns()
        values = self._validate_tokens(history_tokens)
        if len(values) > self.context_limit:
            raise ControlPlaneError(
                422,
                "context_length_exceeded",
                "prompt exceeds the 262,144-token budget",
            )
        canonical = record.canonical_tokens
        empty_before = not canonical
        matching = len(values) >= len(canonical) and values[: len(canonical)] == canonical
        replaced = False
        if (not matching and self.reuse_terminal_prefix and canonical and len(values) > len(canonical)
            and canonical[-1] in self.stop_token_ids
            and values[:len(canonical)-1] == canonical[:-1]):
            state = self._read_state()
            self._verify_executor_frontiers(state, len(canonical))
            c, target, stage0, stage1, draft, epoch = state.frontiers
            if c == target + 1 and stage0 == stage1 == draft == target and epoch <= 0xffffffff:
                response = self.transport.call(OPCODES['replace_pending_terminal'],
                    tokens=[canonical[-1],values[len(canonical)-1]], argument0=c, argument1=epoch,
                    timeout_ms=timeout_ms)
                if response.status == 1:
                    # Sampled/unsupported sessions may decline this optional
                    # operation. Fallback only after an unchanged-state reply.
                    if response.frontiers != state.frontiers:
                        raise ControlPlaneError(503, 'frontier_divergence', 'declined pending replacement changed state')
                else:
                    _require_ok(response, 'replace_pending_terminal')
                    self._verify_executor_frontiers(response, len(canonical))
                    if response.frontiers != (*state.frontiers[:5], epoch + 1):
                        raise ControlPlaneError(503, 'frontier_divergence', 'pending replacement advanced unexpected frontiers')
                    canonical[-1] = values[len(canonical)-1]
                    record.revision += 1
                    record.refresh_hashes(rebuild=True)
                    matching = replaced = True
        validation_ns = time.perf_counter_ns() - started
        reset_ns = 0
        rebuilt = record.cold_rebuild and not canonical
        if not matching:
            reset_ns = self._reset_record_locked(record, timeout_ms)
            canonical = record.canonical_tokens
            rebuilt = True
        suffix = values[len(canonical) :]
        self._append_locked(record, suffix, timeout_ms)
        record.cold_rebuild = rebuilt
        cache_status = (
            "cold_rebuild"
            if rebuilt
            else "cold_start"
            if empty_before
            else "incremental_hit"
        )
        counter = {
            "cold_start": "cold_starts",
            "cold_rebuild": "cold_rebuilds",
            "incremental_hit": "incremental_hits",
        }[cache_status]
        with self._state:
            self._cache_counters[counter] += 1
            self._cache_counters['terminal_prefix_hits'] += int(replaced)
        return {
            "prefix_tokens": len(values) - len(suffix) - int(replaced),
            "suffix_tokens": len(suffix) + int(replaced),
            "prefix_validation_ns": validation_ns,
            "reset_ns": reset_ns,
            "cold_rebuild": rebuilt,
            "cache_status": cache_status,
            "terminal_prefix_reused": replaced,
        }

    def append_full_history(
        self,
        session_id: str,
        history_tokens: Sequence[int],
        *,
        timeout_ms: int | None = None,
    ) -> tuple[dict[str, object], dict[str, int | bool | str]]:
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
        stop_event: threading.Event | None = None,
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
                f"MTP draft width exceeds the configured {self.mtp_max_draft}-token limit",
            )
        self._require_budget(record, max_new_tokens)
        if not record.canonical_tokens:
            raise ControlPlaneError(409, "empty_prompt", "generation requires a prompt")

        generated = 0
        state = self._read_state()
        while generated < max_new_tokens:
            if stop_event is not None and stop_event.is_set():
                raise ControlPlaneError(499, "stream_cancelled", "stream was cancelled")
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
            state = response
            record.canonical_tokens.extend(values)
            record.revision += 1
            record.refresh_hashes()
            generated += len(values)
            # A cancellation can race an OK commit. Canonical bookkeeping must
            # reflect that commit even if its SSE event will not be delivered.
            if stop_event is not None and stop_event.is_set():
                raise ControlPlaneError(499, "stream_cancelled", "stream was cancelled")
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
        with self._writer, (self.transport.writer_connection() if hasattr(self.transport, "writer_connection") else nullcontext()):
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
        stop_event: threading.Event | None = None,
    ) -> Iterator[tuple[CommittedTokenEvent, dict[str, int | bool | str]]]:
        """Append an exact suffix and generate under one semantic writer lease."""

        connection = (self.transport.writer_connection(stop_event) if stop_event is not None
                      else self.transport.writer_connection()) if hasattr(self.transport, "writer_connection") else nullcontext()
        with self._writer, self._stream_lease(stop_event), connection:
            if stop_event is not None and stop_event.is_set():
                raise ControlPlaneError(499, "stream_cancelled", "stream was cancelled")
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
                stop_event=stop_event,
            ):
                yield event, timing

    @contextmanager
    def _stream_lease(self, stop_event: threading.Event | None):
        with self._stream_guard:
            self._stream_stop = stop_event
        try:
            yield
        finally:
            with self._stream_guard:
                self._stream_stop = None

    def stop_stream(self, owner: threading.Event) -> None:
        with self._stream_guard:
            owner.set()
            if self._stream_stop is owner and hasattr(self.transport, "cancel_owned"):
                self.transport.cancel_owned(owner)

    def cancel(self) -> dict[str, object]:
        with self._stream_guard:
            if self._stream_stop is not None:
                owner = self._stream_stop
                owner.set()
                response = self.transport.cancel_owned(owner) if hasattr(self.transport, "cancel_owned") else None
                # Status 2 is an already-finished writer racing cancellation;
                # the owner's stop flag still prevents another transaction.
                if response is not None and response.status != 2:
                    _require_ok(response, "cancel")
                return {"cancelled_request_id": None if response is None else response.request_id,
                        "status": "cancel_requested"}
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
            result = json.loads(response.message)
        except json.JSONDecodeError as error:
            raise ControlPlaneError(
                503, "invalid_metrics", "executor returned invalid metrics JSON"
            ) from error
        with self._state:
            result["session_cache"] = {
                **self._cache_counters,
                "resident_slots": 1,
                "resident_session_id": (
                    None if self._session is None else self._session.session_id
                ),
                "policy": "single_resident_exact_prefix",
            }
        return result


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
        vocab = self.tokenizer.get_vocab()
        self.output_end_token_ids = frozenset(self.stop_token_ids) | {
            vocab[tag] for tag in ("<|im_end|>", "<|endoftext|>") if tag in vocab
        }

    def encode_chat(
        self,
        messages: Sequence[dict[str, object]],
        *,
        tools: Sequence[dict[str, object]] | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> list[int]:
        template_options: dict[str, object] = {}
        if tools is not None:
            template_options["tools"] = list(tools)
        if enable_thinking is not None:
            template_options["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            template_options["reasoning_effort"] = reasoning_effort
        values = self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True,
            **template_options,
        )
        # transformers 5 returns BatchEncoding here, while transformers 4
        # returned the token list directly. Keep the codec independent of
        # that packaging change.
        if isinstance(values, Mapping):
            if "input_ids" not in values:
                raise RuntimeError("chat template result has no input_ids")
            values = values["input_ids"]
        return [int(token) for token in values]

    def decode(self, tokens: Sequence[int]) -> str:
        # Keep reasoning/tool delimiters for the response parser, but never
        # publish a turn terminator or text following it. This also keeps
        # cumulative streaming decodes stable after the first terminator.
        visible = []
        for token in tokens:
            if token in self.output_end_token_ids:
                break
            visible.append(token)
        return self.tokenizer.decode(
            visible, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def stream_decoder(self) -> "StreamingTextDecoder":
        return StreamingTextDecoder(self)


class StreamingTextDecoder:
    """Decode new IDs once; hold incomplete UTF-8 until its next token arrives.

    Use the tokenizer's own streaming decoder for a ByteLevel backend. Other
    tokenizers retain a cumulative fallback. A single final batch decode checks
    the emitted prefix and flushes a truncated byte sequence exactly as decode().
    """

    def __init__(self, codec: HuggingFaceCodec) -> None:
        self.codec = codec
        self.tokens: list[int] = []
        self.fragments: list[str] = []
        self.previous = ""
        self.ended = False
        self.finalized = False
        self.backend = getattr(getattr(codec, "tokenizer", None), "backend_tokenizer", None)
        self.stream = None
        if self.backend is not None:
            try:
                from tokenizers.decoders import DecodeStream
                decoder = json.loads(self.backend.decoder.__getstate__())
                if decoder.get("type") == "ByteLevel":
                    self.stream = DecodeStream(skip_special_tokens=False)
            except (ImportError, AttributeError, TypeError, ValueError):
                pass

    def push(self, tokens: Sequence[int], *, final: bool = False) -> str:
        if self.finalized:
            raise RuntimeError("stream decoder already finalized")
        end_ids = getattr(self.codec, "output_end_token_ids", ())
        visible: list[int] = []
        for token in tokens:
            if self.ended:
                break
            if token in end_ids:
                self.ended = True
                break
            visible.append(token)
        self.tokens.extend(visible)
        if self.stream is not None:
            fragment = (self.stream.step(self.backend, visible) or "") if visible else ""
        else:
            decoded = self.codec.decode(self.tokens)
            # A replacement at the end may be a split UTF-8 character rather
            # than actual output. Publishing it would require retracting SSE.
            stable = decoded if final or self.ended else decoded.rstrip("\ufffd")
            if not stable.startswith(self.previous):
                raise RuntimeError("tokenizer changed an already emitted prefix")
            fragment = stable[len(self.previous):]
            self.previous = stable
        self.fragments.append(fragment)
        if final:
            decoded = self.codec.decode(self.tokens)
            emitted = "".join(self.fragments)
            if not decoded.startswith(emitted):
                raise RuntimeError("streamed text differs from final tokenizer decode")
            tail = decoded[len(emitted):]
            self.fragments.append(tail)
            fragment += tail
            self.finalized = True
        return fragment
