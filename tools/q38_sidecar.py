#!/usr/bin/env python3
"""Built-in HTTP/SSE control plane for the native Q38 executor.

Token-native deep-session endpoints are the primary interface.  The optional
OpenAI-compatible endpoint uses the official tokenizer codec.  Neither path
participates in model execution or owns model state.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

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
FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_.:-]+$")
OUTPUT_TOKEN_FIELDS = (
    "max_completion_tokens",
    "max_tokens",
    "max_new_tokens",
    "max_output_tokens",
)


def _resolve_output_tokens(
    body: dict[str, Any],
    *,
    default_tokens: int | None,
    maximum_tokens: int | None,
    available_tokens: int,
) -> int:
    """Resolve OpenAI and common local-runtime output-length aliases."""

    selected: str | None = None
    value: object = available_tokens if default_tokens is None else default_tokens
    for field in OUTPUT_TOKEN_FIELDS:
        if field in body and body[field] is not None:
            selected = field
            value = body[field]
            break
    if available_tokens <= 0:
        raise ControlPlaneError(
            422,
            "context_length_exceeded",
            "the prompt leaves no context capacity for output tokens",
        )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ControlPlaneError(
            400,
            "invalid_max_tokens",
            f"{selected or 'default output limit'} must be a positive integer",
        )
    if selected is not None and value > available_tokens:
        raise ControlPlaneError(
            422,
            "context_length_exceeded",
            f"{selected}={value} exceeds the {available_tokens} output tokens remaining in context",
        )
    server_limit = available_tokens if maximum_tokens is None else maximum_tokens
    return min(value, server_limit, available_tokens)


def _function_tools(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ControlPlaneError(400, "invalid_tools", "tools must be a list")
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for tool in value:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ControlPlaneError(
                400, "invalid_tools", "only function tools are supported"
            )
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ControlPlaneError(400, "invalid_tools", "tool.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not FUNCTION_NAME.fullmatch(name):
            raise ControlPlaneError(400, "invalid_tools", "tool function name is invalid")
        if name in names:
            raise ControlPlaneError(400, "invalid_tools", "tool function names must be unique")
        names.add(name)
        result.append(tool)
    return result


def _tool_choice(
    value: object, tools: list[dict[str, object]]
) -> tuple[list[dict[str, object]], str | None]:
    if value is None or value == "auto":
        return tools, None
    if value == "none":
        return [], None
    if value == "required":
        if not tools:
            raise ControlPlaneError(400, "invalid_tool_choice", "required needs tools")
        return tools, "You must call at least one available function in this response."
    if isinstance(value, dict) and value.get("type") == "function":
        function = value.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        selected = [tool for tool in tools if tool["function"]["name"] == name]  # type: ignore[index]
        if not selected:
            raise ControlPlaneError(
                400, "invalid_tool_choice", "selected tool is not available"
            )
        return selected, f"You must call the {name} function in this response."
    raise ControlPlaneError(400, "invalid_tool_choice", "tool_choice is invalid")


def _normalize_messages(
    value: object, required_instruction: str | None = None
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ControlPlaneError(400, "invalid_messages", "messages must be a non-empty list")
    messages: list[dict[str, object]] = []
    for message in value:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ControlPlaneError(400, "invalid_messages", "each message needs a role")
        normalized = dict(message)
        calls = normalized.get("tool_calls")
        if calls is not None:
            if normalized["role"] != "assistant" or not isinstance(calls, list):
                raise ControlPlaneError(400, "invalid_messages", "tool_calls are invalid")
            normalized_calls: list[dict[str, object]] = []
            for call in calls:
                if not isinstance(call, dict) or call.get("type", "function") != "function":
                    raise ControlPlaneError(400, "invalid_messages", "tool call is invalid")
                function = call.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    raise ControlPlaneError(400, "invalid_messages", "tool call function is invalid")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as error:
                        raise ControlPlaneError(
                            400, "invalid_messages", "tool call arguments are not JSON"
                        ) from error
                if not isinstance(arguments, dict):
                    raise ControlPlaneError(
                        400, "invalid_messages", "tool call arguments must be an object"
                    )
                normalized_calls.append(
                    {**call, "function": {**function, "arguments": arguments}}
                )
            normalized["tool_calls"] = normalized_calls
        messages.append(normalized)
    if required_instruction:
        if messages[0]["role"] == "system":
            content = messages[0].get("content")
            if content is not None and not isinstance(content, str):
                raise ControlPlaneError(
                    400, "invalid_messages", "system content must be text for tool_choice"
                )
            messages[0]["content"] = (
                required_instruction
                if not content
                else f"{content}\n\n{required_instruction}"
            )
        else:
            messages.insert(0, {"role": "system", "content": required_instruction})
    return messages


def _strip_reasoning(text: str) -> tuple[str | None, str]:
    value = text.strip()
    if value.startswith("<think>"):
        value = value[len("<think>") :]
    if "</think>" not in value:
        return None, value
    reasoning, visible = value.split("</think>", 1)
    return reasoning.strip() or None, visible.strip()


def _parameter_value(value: str) -> object:
    stripped = value.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _parse_tool_output(
    text: str, request_id: str
) -> tuple[str | None, str | None, list[dict[str, object]]]:
    reasoning, visible = _strip_reasoning(text)
    visible = visible.removesuffix("<|im_end|>").strip()
    calls: list[dict[str, object]] = []
    content_parts: list[str] = []
    cursor = 0
    while True:
        start = visible.find("<tool_call>", cursor)
        if start < 0:
            content_parts.append(visible[cursor:])
            break
        content_parts.append(visible[cursor:start])
        end = visible.find("</tool_call>", start)
        if end < 0:
            raise ControlPlaneError(502, "invalid_tool_call", "unterminated tool_call")
        block = visible[start + len("<tool_call>") : end].strip()
        function_start = block.find("<function=")
        function_header_end = block.find(">", function_start)
        function_end = block.rfind("</function>")
        if function_start != 0 or function_header_end < 0 or function_end < function_header_end:
            raise ControlPlaneError(502, "invalid_tool_call", "invalid function block")
        name = block[len("<function=") : function_header_end]
        if not FUNCTION_NAME.fullmatch(name):
            raise ControlPlaneError(502, "invalid_tool_call", "invalid function name")
        parameters = block[function_header_end + 1 : function_end]
        arguments: dict[str, object] = {}
        parameter_cursor = 0
        while parameter_cursor < len(parameters):
            if not parameters[parameter_cursor:].strip():
                break
            parameter_cursor += len(parameters[parameter_cursor:]) - len(
                parameters[parameter_cursor:].lstrip()
            )
            if not parameters.startswith("<parameter=", parameter_cursor):
                raise ControlPlaneError(502, "invalid_tool_call", "invalid parameter block")
            header_end = parameters.find(">", parameter_cursor)
            if header_end < 0:
                raise ControlPlaneError(502, "invalid_tool_call", "invalid parameter header")
            parameter_name = parameters[
                parameter_cursor + len("<parameter=") : header_end
            ]
            if not FUNCTION_NAME.fullmatch(parameter_name) or parameter_name in arguments:
                raise ControlPlaneError(502, "invalid_tool_call", "invalid parameter name")
            parameter_end = parameters.find("</parameter>", header_end)
            if parameter_end < 0:
                raise ControlPlaneError(502, "invalid_tool_call", "unterminated parameter")
            arguments[parameter_name] = _parameter_value(
                parameters[header_end + 1 : parameter_end]
            )
            parameter_cursor = parameter_end + len("</parameter>")
        calls.append(
            {
                "id": f"call_q38_{request_id.rsplit('-', 1)[-1]}_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, separators=(",", ":"), ensure_ascii=False
                    ),
                },
            }
        )
        cursor = end + len("</tool_call>")
    content = "".join(content_parts).strip() or None
    return reasoning, content, calls


def _validate_output_tools(
    calls: list[dict[str, object]],
    tools: list[dict[str, object]],
    required: bool,
) -> None:
    if required and not calls:
        raise ControlPlaneError(
            502, "tool_call_required", "model did not produce the required tool call"
        )
    available = {tool["function"]["name"] for tool in tools}  # type: ignore[index]
    for call in calls:
        name = call["function"]["name"]  # type: ignore[index]
        if name not in available:
            raise ControlPlaneError(
                502, "unknown_tool_call", f"model called unavailable function {name}"
            )


class SidecarApplication:
    def __init__(
        self,
        control: Q38ControlPlane,
        *,
        codec: HuggingFaceCodec | None,
        api_key: str | None,
        default_max_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if (
            (default_max_tokens is not None and default_max_tokens <= 0)
            or (max_output_tokens is not None and max_output_tokens <= 0)
            or (
                default_max_tokens is not None
                and max_output_tokens is not None
                and default_max_tokens > max_output_tokens
            )
        ):
            raise ValueError("invalid sidecar output-token limits")
        self.control = control
        self.codec = codec
        self.api_key = api_key
        self.default_max_tokens = default_max_tokens
        self.max_output_tokens = max_output_tokens


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
            model = self.app.control.model
            model_card = {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "q38",
            }
            if path in ("/v1/models", "/v1/models/"):
                self._json(200, {"object": "list", "data": [model_card]})
                return
            model_prefix = "/v1/models/"
            if path.startswith(model_prefix):
                requested = unquote(path[len(model_prefix) :]).rstrip("/")
                if requested == model:
                    self._json(200, model_card)
                    return
                raise ControlPlaneError(404, "model_not_found", "model not found")
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
        timeout = body.get("timeout_ms")
        mode = str(body.get("mode", "plain"))
        width = int(body.get("mtp_width", 1))
        stream = bool(body.get("stream", False))
        if "full_token_ids" in body:
            full_tokens = body["full_token_ids"]
            if not isinstance(full_tokens, list):
                raise ControlPlaneError(
                    400, "invalid_tokens", "full_token_ids must be a list"
                )
            available = self.app.control.context_limit - len(full_tokens)
        else:
            append = body.get("append_token_ids", [])
            if not isinstance(append, list):
                raise ControlPlaneError(
                    400, "invalid_tokens", "append_token_ids must be a list"
                )
            session = self.app.control.session(session_id)
            available = int(session["remaining_tokens"]) - len(append)
        max_tokens = _resolve_output_tokens(
            body,
            default_tokens=self.app.default_max_tokens,
            maximum_tokens=self.app.max_output_tokens,
            available_tokens=available,
        )
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
        tools = _function_tools(body.get("tools"))
        selected_tools, required_instruction = _tool_choice(
            body.get("tool_choice"), tools
        )
        messages = _normalize_messages(body.get("messages"), required_instruction)
        enable_thinking = body.get("enable_thinking", body.get("q38_enable_thinking"))
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ControlPlaneError(
                400, "invalid_reasoning", "enable_thinking must be a boolean"
            )
        reasoning_effort = body.get("reasoning_effort")
        if reasoning_effort is not None and not isinstance(reasoning_effort, str):
            raise ControlPlaneError(
                400, "invalid_reasoning", "reasoning_effort must be a string"
            )
        active = self.app.control.active_session()
        requested_session = body.get("session_id")
        if active is None:
            active = self.app.control.create_session(requested_session)
        elif requested_session and requested_session != active["id"]:
            active = self.app.control.replace_session(str(requested_session))
        session_id = str(active["id"])
        encode_started = time.perf_counter_ns()
        history = self.app.codec.encode_chat(
            messages,
            tools=selected_tools or None,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        tokenize_ns = time.perf_counter_ns() - encode_started
        max_tokens = _resolve_output_tokens(
            body,
            default_tokens=self.app.default_max_tokens,
            maximum_tokens=self.app.max_output_tokens,
            available_tokens=self.app.control.context_limit - len(history),
        )
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
            self._chat_stream(
                pairs, request_id, session_id, tokenize_ns,
                tool_parsing=bool(selected_tools),
                selected_tools=selected_tools,
                tool_call_required=required_instruction is not None,
            )
            return
        events_and_timing = list(pairs)
        output_tokens = [
            token for event, _timing in events_and_timing for token in event.token_ids
        ]
        text = self.app.codec.decode(output_tokens)
        reasoning: str | None = None
        content: str | None = text
        tool_calls: list[dict[str, object]] = []
        if selected_tools:
            reasoning, content, tool_calls = _parse_tool_output(text, request_id)
            _validate_output_tools(
                tool_calls, selected_tools, required_instruction is not None
            )
        last = events_and_timing[-1][0]
        timing = events_and_timing[-1][1]
        message: dict[str, object] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
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
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else last.finish_reason,
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
                    "tokenize_ns": tokenize_ns,
                    **timing,
                },
            },
        )

    def _chat_stream(
        self,
        pairs: Iterator[
            tuple[CommittedTokenEvent, dict[str, int | bool | str]]
        ],
        request_id: str,
        session_id: str,
        tokenize_ns: int,
        *,
        tool_parsing: bool = False,
        selected_tools: list[dict[str, object]] | None = None,
        tool_call_required: bool = False,
    ) -> None:
        if tool_parsing:
            self._chat_stream_tools(
                pairs,
                request_id,
                session_id,
                tokenize_ns,
                selected_tools or [],
                tool_call_required,
            )
            return
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

    def _chat_stream_tools(
        self,
        pairs: Iterator[
            tuple[CommittedTokenEvent, dict[str, int | bool | str]]
        ],
        request_id: str,
        session_id: str,
        tokenize_ns: int,
        selected_tools: list[dict[str, object]],
        tool_call_required: bool,
    ) -> None:
        events_and_timing = list(pairs)
        output_tokens = [
            token for event, _timing in events_and_timing for token in event.token_ids
        ]
        text = self.app.codec.decode(output_tokens)  # type: ignore[union-attr]
        reasoning, content, tool_calls = _parse_tool_output(text, request_id)
        _validate_output_tools(tool_calls, selected_tools, tool_call_required)
        last = events_and_timing[-1][0]
        timing = events_and_timing[-1][1]
        delta: dict[str, object] = {"role": "assistant", "content": content}
        if reasoning:
            delta["reasoning_content"] = reasoning
        if tool_calls:
            delta["tool_calls"] = [
                {"index": index, **call} for index, call in enumerate(tool_calls)
            ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.app.control.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            "q38": {
                "session_id": session_id,
                "revision": last.revision,
                "committed_tokens": last.committed_tokens,
                "remaining_tokens": last.remaining_tokens,
                "tokenize_ns": tokenize_ns,
                **timing,
            },
        }
        self.wfile.write(
            b"data: "
            + json.dumps(chunk, separators=(",", ":"), ensure_ascii=False).encode()
            + b"\n\n"
        )
        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.app.control.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool_calls else last.finish_reason,
                }
            ],
        }
        self.wfile.write(
            b"data: "
            + json.dumps(final, separators=(",", ":"), ensure_ascii=False).encode()
            + b"\n\n"
        )
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
    parser.add_argument(
        "--default-max-tokens",
        type=int,
        default=0,
        help="default output budget; 0 uses all context remaining after the prompt",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help="optional server output cap; 0 uses the model context limit",
    )
    parser.add_argument("--stop-token-id", type=int, action="append", default=[])
    parser.add_argument("--enable-mtp", action="store_true")
    arguments = parser.parse_args()
    if arguments.host not in ("127.0.0.1", "::1", "localhost") and not arguments.api_key:
        parser.error("non-loopback binding requires --api-key")
    if (
        arguments.default_max_tokens < 0
        or arguments.max_output_tokens < 0
        or (
            arguments.default_max_tokens
            and arguments.max_output_tokens
            and arguments.default_max_tokens > arguments.max_output_tokens
        )
    ):
        parser.error("output-token limits must be nonnegative and default <= maximum")
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
        SidecarApplication(
            control,
            codec=codec,
            api_key=arguments.api_key,
            default_max_tokens=arguments.default_max_tokens or None,
            max_output_tokens=arguments.max_output_tokens or None,
        ),
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
