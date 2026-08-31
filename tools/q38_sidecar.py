#!/usr/bin/env python3
"""Built-in HTTP/SSE control plane for the native Q38 executor.

Token-native deep-session endpoints are the primary interface.  The optional
OpenAI-compatible endpoint uses the official tokenizer codec.  Neither path
participates in model execution or owns model state.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from q38_control_plane import (  # noqa: E402
    CommittedTokenEvent,
    ControlPlaneError,
    ExecutorTransport,
    HuggingFaceCodec,
    Q38ControlPlane,
)


MAX_BODY_BYTES = 16 << 20


class SidecarApplication:
    def __init__(
        self,
        control: Q38ControlPlane,
        *,
        codec: HuggingFaceCodec | None,
        api_key: str | None,
    ) -> None:
        self.control = control
        self.codec = codec
        self.api_key = api_key


class Q38RequestHandler(BaseHTTPRequestHandler):
    server_version = "q38-sidecar/1"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> SidecarApplication:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(
            json.dumps(
                {
                    "event": "http",
                    "client": self.client_address[0],
                    "message": fmt % args,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    def _authorized(self) -> bool:
        if self.app.api_key is None:
            return True
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.app.api_key}"
        return secrets.compare_digest(supplied, expected)

    def _json(self, status: int, value: object) -> None:
        raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, error: Exception) -> None:
        if isinstance(error, ControlPlaneError):
            self._json(error.status, error.as_dict())
        else:
            self._json(
                500,
                {
                    "error": {
                        "message": str(error),
                        "type": "q38_sidecar_error",
                        "code": "internal_error",
                    }
                },
            )

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ControlPlaneError(400, "invalid_body", "invalid content length") from error
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ControlPlaneError(400, "invalid_body", "JSON body size is invalid")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ControlPlaneError(400, "invalid_json", "request body is not JSON") from error
        if not isinstance(value, dict):
            raise ControlPlaneError(400, "invalid_json", "request body must be an object")
        return value

    def _session_id_from_path(self, suffix: str = "") -> str | None:
        path = urlparse(self.path).path
        prefix = "/v1/q38/sessions/"
        if not path.startswith(prefix) or not path.endswith(suffix):
            return None
        value = path[len(prefix) : len(path) - len(suffix) if suffix else None]
        return value.rstrip("/") or None

    def do_GET(self) -> None:  # noqa: N802
        try:
            if not self._authorized():
                raise ControlPlaneError(401, "unauthorized", "invalid API key")
            path = urlparse(self.path).path
            if path == "/healthz":
                self._json(200, {"status": "ok", "session": self.app.control.active_session()})
                return
            if path == "/v1/q38/metrics":
                self._json(200, self.app.control.metrics())
                return
            session_id = self._session_id_from_path()
            if session_id:
                self._json(200, self.app.control.session(session_id))
                return
            raise ControlPlaneError(404, "not_found", "endpoint not found")
        except Exception as error:
            self._error(error)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if not self._authorized():
                raise ControlPlaneError(401, "unauthorized", "invalid API key")
            path = urlparse(self.path).path
            body = self._body()
            if path == "/v1/q38/sessions":
                self._json(201, self.app.control.create_session(body.get("session_id")))
                return
            if path == "/v1/q38/cancel":
                self._json(202, self.app.control.cancel())
                return
            if path == "/v1/chat/completions":
                self._chat_completion(body)
                return
            session_id = self._session_id_from_path("/append")
            if session_id:
                tokens = body.get("token_ids")
                if not isinstance(tokens, list):
                    raise ControlPlaneError(400, "invalid_tokens", "token_ids must be a list")
                result = self.app.control.append_tokens(
                    session_id, tokens, timeout_ms=body.get("timeout_ms")
                )
                self._json(200, result)
                return
            session_id = self._session_id_from_path("/execute")
            if session_id:
                self._token_execute(session_id, body)
                return
            raise ControlPlaneError(404, "not_found", "endpoint not found")
        except Exception as error:
            self._error(error)

    def _events(
        self,
        iterator: Iterator[CommittedTokenEvent],
        *,
        stream: bool,
        request_id: str,
    ) -> None:
        if not stream:
            events = list(iterator)
            tokens = [token for event in events for token in event.token_ids]
            last = events[-1] if events else None
            self._json(
                200,
                {
                    "id": request_id,
                    "object": "q38.token_completion",
                    "token_ids": tokens,
                    "finish_reason": None if last is None else last.finish_reason,
                    "revision": None if last is None else last.revision,
                    "committed_tokens": None if last is None else last.committed_tokens,
                    "remaining_tokens": None if last is None else last.remaining_tokens,
                },
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in iterator:
            value = {
                "id": request_id,
                "object": "q38.committed_token",
                "token_ids": list(event.token_ids),
                "revision": event.revision,
                "committed_tokens": event.committed_tokens,
                "remaining_tokens": event.remaining_tokens,
                "finish_reason": event.finish_reason,
            }
            self.wfile.write(
                b"data: "
                + json.dumps(value, separators=(",", ":")).encode()
                + b"\n\n"
            )
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _token_execute(self, session_id: str, body: dict[str, Any]) -> None:
        max_tokens = int(body.get("max_new_tokens", 1))
        timeout = body.get("timeout_ms")
        mode = str(body.get("mode", "plain"))
        width = int(body.get("mtp_width", 1))
        stream = bool(body.get("stream", False))
        if "full_token_ids" in body:
            iterator = (
                event
                for event, _timing in self.app.control.complete_full_history(
                    session_id,
                    body["full_token_ids"],
                    max_tokens,
                    mode=mode,
                    mtp_width=width,
                    timeout_ms=timeout,
                )
            )
        else:
            append = body.get("append_token_ids", [])
            if append:
                self.app.control.append_tokens(session_id, append, timeout_ms=timeout)
            iterator = self.app.control.generate(
                session_id,
                max_tokens,
                mode=mode,
                mtp_width=width,
                timeout_ms=timeout,
            )
        self._events(iterator, stream=stream, request_id=f"q38-{time.time_ns()}")

    def _chat_completion(self, body: dict[str, Any]) -> None:
        if self.app.codec is None:
            raise ControlPlaneError(
                503,
                "tokenizer_unavailable",
                "OpenAI message input requires --tokenizer; use token-native endpoints otherwise",
            )
        if body.get("tools"):
            raise ControlPlaneError(
                400, "unsupported_tools", "raw-token-v1 does not enable tool parsing"
            )
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ControlPlaneError(400, "invalid_messages", "messages must be a non-empty list")
        active = self.app.control.active_session()
        requested_session = body.get("session_id")
        if active is None:
            active = self.app.control.create_session(requested_session)
        elif requested_session and requested_session != active["id"]:
            raise ControlPlaneError(409, "session_mismatch", "another session is active")
        session_id = str(active["id"])
        encode_started = time.perf_counter_ns()
        history = self.app.codec.encode_chat(messages)
        tokenize_ns = time.perf_counter_ns() - encode_started
        max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 1)))
        mode = str(body.get("q38_mode", "plain"))
        width = int(body.get("q38_mtp_width", 1))
        timeout = body.get("q38_timeout_ms")
        pairs = self.app.control.complete_full_history(
            session_id,
            history,
            max_tokens,
            mode=mode,
            mtp_width=width,
            timeout_ms=timeout,
        )
        request_id = f"chatcmpl-q38-{time.time_ns()}"
        if body.get("stream", False):
            self._chat_stream(pairs, request_id, session_id, tokenize_ns)
            return
        events_and_timing = list(pairs)
        output_tokens = [
            token for event, _timing in events_and_timing for token in event.token_ids
        ]
        text = self.app.codec.decode(output_tokens)
        last = events_and_timing[-1][0]
        timing = events_and_timing[-1][1]
        self._json(
            200,
            {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.app.control.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": last.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(history),
                    "completion_tokens": len(output_tokens),
                    "total_tokens": len(history) + len(output_tokens),
                },
                "q38": {
                    "session_id": session_id,
                    "revision": last.revision,
                    "committed_tokens": last.committed_tokens,
                    "remaining_tokens": last.remaining_tokens,
                    "cold_rebuild": False,
                    "tokenize_ns": tokenize_ns,
                    **timing,
                },
            },
        )

    def _chat_stream(
        self,
        pairs: Iterator[tuple[CommittedTokenEvent, dict[str, int]]],
        request_id: str,
        session_id: str,
        tokenize_ns: int,
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        accumulated: list[int] = []
        previous = ""
        for event, timing in pairs:
            accumulated.extend(event.token_ids)
            decoded = self.app.codec.decode(accumulated)  # type: ignore[union-attr]
            fragment = decoded[len(previous) :] if decoded.startswith(previous) else decoded
            previous = decoded
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.app.control.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": fragment},
                        "finish_reason": event.finish_reason,
                    }
                ],
                "q38": {
                    "session_id": session_id,
                    "revision": event.revision,
                    "committed_tokens": event.committed_tokens,
                    "remaining_tokens": event.remaining_tokens,
                    "tokenize_ns": tokenize_ns,
                    **timing,
                },
            }
            self.wfile.write(
                b"data: " + json.dumps(chunk, separators=(",", ":"), ensure_ascii=False).encode() + b"\n\n"
            )
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


class Q38HttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: SidecarApplication) -> None:
        super().__init__(address, Q38RequestHandler)
        self.app = app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--session-hash", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--context-limit", type=int, default=262_144)
    parser.add_argument("--vocabulary", type=int, default=248_320)
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-ms", type=int, default=3_600_000)
    parser.add_argument("--stop-token-id", type=int, action="append", default=[])
    parser.add_argument("--enable-mtp", action="store_true")
    arguments = parser.parse_args()
    if arguments.host not in ("127.0.0.1", "::1", "localhost") and not arguments.api_key:
        parser.error("non-loopback binding requires --api-key")
    codec = HuggingFaceCodec(arguments.tokenizer) if arguments.tokenizer else None
    stop_tokens = set(arguments.stop_token_id)
    if codec is not None:
        stop_tokens.update(codec.stop_token_ids)
    control = Q38ControlPlane(
        ExecutorTransport(arguments.socket, arguments.session_hash),
        context_limit=arguments.context_limit,
        model=arguments.model,
        vocabulary=arguments.vocabulary,
        stop_token_ids=sorted(stop_tokens),
        default_timeout_ms=arguments.timeout_ms,
        mtp_enabled=arguments.enable_mtp,
        mtp_max_draft=1,
    )
    server = Q38HttpServer(
        (arguments.host, arguments.port),
        SidecarApplication(control, codec=codec, api_key=arguments.api_key),
    )
    print(
        json.dumps(
            {
                "event": "ready",
                "component": "q38-sidecar",
                "listen": f"http://{arguments.host}:{arguments.port}",
                "tokenizer": str(arguments.tokenizer) if arguments.tokenizer else None,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
