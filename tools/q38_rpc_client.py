#!/usr/bin/env python3
"""Minimal client for the local Q38 ExecutorRPC V1 Unix socket."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from dataclasses import dataclass


MAGIC = int.from_bytes(b"Q38R", "little")
VERSION = 1
REQUEST = struct.Struct("<IHHQQIIIIIIQ")
RESPONSE = struct.Struct("<IHHQQIIIIQ6Q")
TOKEN = struct.Struct("<i")

OPCODES = {
    "ping": 1,
    "state": 2,
    "append": 3,
    "seed": 4,
    "decode": 5,
    "spec": 6,
    "stats": 7,
    "cancel": 8,
}

STATUSES = {
    0: "ok",
    1: "bad_request",
    2: "failed_precondition",
    3: "capacity_exhausted",
    4: "internal",
    5: "poisoned",
    6: "cancelled",
    7: "deadline_exceeded",
}


def parse_integer(value: str) -> int:
    return int(value, 0)


def read_exact(connection: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("ExecutorRPC connection ended inside a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class Response:
    request_id: int
    session_hash: int
    status: int
    opcode: int
    tokens: tuple[int, ...]
    message: str
    frontiers: tuple[int, int, int, int, int, int]

    def as_dict(self) -> dict[str, object]:
        canonical, target, stage0, stage1, draft, epoch = self.frontiers
        return {
            "request_id": self.request_id,
            "session_hash": self.session_hash,
            "status": STATUSES.get(self.status, f"unknown_{self.status}"),
            "opcode": self.opcode,
            "tokens": list(self.tokens),
            "message": self.message,
            "frontiers": {
                "canonical": canonical,
                "target": target,
                "stage0": stage0,
                "stage1": stage1,
                "draft": draft,
                "epoch": epoch,
            },
        }


class Client:
    def __init__(
        self, path: str, session_hash: int, request_id: int | None = None
    ) -> None:
        self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._connection.connect(path)
        self._session_hash = session_hash
        self._next_request_id = request_id or time.time_ns()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def call(
        self,
        opcode: int,
        *,
        tokens: list[int] | None = None,
        argument0: int = 0,
        argument1: int = 0,
        timeout_ms: int = 0,
    ) -> Response:
        token_values = tokens or []
        payload = b"".join(TOKEN.pack(token) for token in token_values)
        request_id = self._next_request_id
        self._next_request_id += 1
        header = REQUEST.pack(
            MAGIC,
            VERSION,
            REQUEST.size,
            request_id,
            self._session_hash,
            opcode,
            1 if timeout_ms else 0,
            len(token_values),
            argument0,
            argument1,
            timeout_ms,
            len(payload),
        )
        self._connection.sendall(header + payload)
        values = RESPONSE.unpack(read_exact(self._connection, RESPONSE.size))
        (
            magic,
            version,
            header_bytes,
            returned_request_id,
            returned_session_hash,
            status,
            returned_opcode,
            token_count,
            message_bytes,
            payload_bytes,
            canonical,
            target,
            stage0,
            stage1,
            draft,
            epoch,
        ) = values
        if magic != MAGIC or version != VERSION or header_bytes != RESPONSE.size:
            raise ValueError("invalid ExecutorRPC response header")
        if returned_request_id != request_id or returned_opcode != opcode:
            raise ValueError("ExecutorRPC response does not match its request")
        expected_payload = token_count * TOKEN.size + message_bytes
        if payload_bytes != expected_payload:
            raise ValueError("invalid ExecutorRPC response payload length")
        payload = read_exact(self._connection, payload_bytes)
        returned_tokens = tuple(
            TOKEN.unpack_from(payload, offset)[0]
            for offset in range(0, token_count * TOKEN.size, TOKEN.size)
        )
        message = payload[token_count * TOKEN.size :].decode("utf-8")
        return Response(
            request_id=returned_request_id,
            session_hash=returned_session_hash,
            status=status,
            opcode=returned_opcode,
            tokens=returned_tokens,
            message=message,
            frontiers=(canonical, target, stage0, stage1, draft, epoch),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--session-hash", type=parse_integer, default=1)
    parser.add_argument("--request-id", type=parse_integer)
    parser.add_argument("--timeout-ms", type=int, default=0)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("ping")
    subcommands.add_parser("state")
    subcommands.add_parser("stats")
    subcommands.add_parser("cancel")
    append = subcommands.add_parser("append")
    append.add_argument("tokens", nargs="+", type=parse_integer)
    subcommands.add_parser("seed")
    decode = subcommands.add_parser("decode")
    decode.add_argument("steps", nargs="?", type=int, default=1)
    spec = subcommands.add_parser("spec")
    spec.add_argument("steps", nargs="?", type=int, default=1)
    spec.add_argument("max_draft", nargs="?", type=int, default=1)
    arguments = parser.parse_args()

    opcode = OPCODES[arguments.command]
    call_arguments: dict[str, object] = {}
    if arguments.timeout_ms:
        call_arguments["timeout_ms"] = arguments.timeout_ms
    if arguments.command == "append":
        call_arguments["tokens"] = arguments.tokens
    elif arguments.command == "decode":
        call_arguments["argument0"] = arguments.steps
    elif arguments.command == "spec":
        call_arguments["argument0"] = arguments.steps
        call_arguments["argument1"] = arguments.max_draft
    with Client(
        arguments.socket, arguments.session_hash, arguments.request_id
    ) as client:
        response = client.call(opcode, **call_arguments)
    print(json.dumps(response.as_dict(), separators=(",", ":")))
    return 0 if response.status == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
