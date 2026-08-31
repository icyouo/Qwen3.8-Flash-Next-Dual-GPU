#!/usr/bin/env python3
"""Build the production SessionIdentityV1 manifest from immutable inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path


MAGIC = int.from_bytes(b"Q38I", "little")
VERSION = 1
HEADER_BYTES = 288
IGNORED_PARTS = {".git", "build", "__pycache__", ".pytest_cache", ".venv"}
RUNTIME_INPUTS = ("Makefile", "include", "src", "cuda", "tools")


def hash_path(path: Path) -> bytes:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as source:
            while block := source.read(8 << 20):
                digest.update(block)
        return digest.digest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and not any(part in IGNORED_PARTS for part in item.relative_to(path).parts)
        and not item.name.endswith((".pyc", ".q38i"))
    )
    if not files:
        raise ValueError(f"identity input directory is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as source:
            while block := source.read(8 << 20):
                digest.update(block)
    return digest.digest()


def hash_group(paths: list[Path]) -> bytes:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(hash_path(path))
    return digest.digest()


def hash_runtime(path: Path) -> bytes:
    """Hash only production build/runtime inputs, never mutable evidence."""
    root = path.resolve()
    digest = hashlib.sha256()
    for relative in RUNTIME_INPUTS:
        item = root / relative
        if not item.exists():
            raise FileNotFoundError(item)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hash_path(item))
    return digest.digest()


def fnv64(data: bytes) -> int:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


DEFAULT_SAMPLING: dict[str, object] = {
    "mode": "greedy",
    "parser": "raw-token-v1",
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "seed": 1,
    "stop_token_ids": [248_044],
}


def sampling_config(value: str | None) -> dict[str, object]:
    parsed: object = {}
    if value:
        stripped = value.lstrip()
        if stripped.startswith("{"):
            parsed = json.loads(value)
        else:
            source = Path(value)
            parsed = json.loads(source.read_text())
    if not isinstance(parsed, dict):
        raise ValueError("sampling/parser identity must be a JSON object")
    unknown = set(parsed) - set(DEFAULT_SAMPLING)
    if unknown:
        raise ValueError(f"unknown sampling identity fields: {sorted(unknown)}")
    result = {**DEFAULT_SAMPLING, **parsed}
    if result["mode"] not in ("greedy", "top-k-top-p"):
        raise ValueError("sampling mode must be greedy or top-k-top-p")
    if not isinstance(result["parser"], str) or not result["parser"]:
        raise ValueError("sampling parser must be a non-empty string")
    numeric = (
        "temperature",
        "top_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
    )
    for field in numeric:
        if isinstance(result[field], bool) or not isinstance(result[field], (int, float)):
            raise ValueError(f"sampling {field} must be numeric")
        result[field] = float(result[field])
        if not math.isfinite(result[field]):
            raise ValueError(f"sampling {field} must be finite")
    if not 0.0 < result["temperature"] or not 0.0 < result["top_p"] <= 1.0:
        raise ValueError("sampling temperature/top_p are invalid")
    if result["repetition_penalty"] < 1.0:
        raise ValueError("sampling repetition_penalty must be at least 1")
    for field in ("top_k", "seed"):
        if isinstance(result[field], bool) or not isinstance(result[field], int):
            raise ValueError(f"sampling {field} must be an integer")
    if result["top_k"] < 0 or result["top_k"] > 248_320:
        raise ValueError("sampling top_k is outside the vocabulary")
    if result["seed"] < 0 or result["seed"] >= 1 << 64:
        raise ValueError("sampling seed is outside uint64")
    stops = result["stop_token_ids"]
    if (
        not isinstance(stops, list)
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= 248_320
            for token in stops
        )
    ):
        raise ValueError("sampling stop_token_ids are invalid")
    result["stop_token_ids"] = sorted(set(stops))
    return result


def canonical_sampling(value: str | None) -> bytes:
    return json.dumps(
        sampling_config(value), sort_keys=True, separators=(",", ":")
    ).encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-hash", type=lambda text: int(text, 0), required=True)
    parser.add_argument("--context-limit", type=int, default=262_144)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, nargs="+", required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--kernels", type=Path)
    parser.add_argument("--ple-layout", type=Path, required=True)
    parser.add_argument("--stage0-index", type=Path, required=True)
    parser.add_argument("--stage1-index", type=Path, required=True)
    parser.add_argument("--sampling-parser")
    arguments = parser.parse_args()
    if not 0 < arguments.session_hash < 1 << 64:
        raise ValueError("session hash must be a nonzero uint64")
    if not 0 < arguments.context_limit < 1 << 32:
        raise ValueError("context limit must be a nonzero uint32")

    runtime_root = arguments.runtime_root.resolve()
    kernels = (arguments.kernels or runtime_root / "cuda").resolve()
    digests = [
        hash_path(arguments.model_metadata.resolve()),
        hash_group([path.resolve() for path in arguments.tokenizer]),
        hash_path(arguments.chat_template.resolve()),
        hash_runtime(runtime_root),
        hash_path(kernels),
        hash_path(arguments.ple_layout.resolve()),
        hash_group(
            [arguments.stage0_index.resolve(), arguments.stage1_index.resolve()]
        ),
        hashlib.sha256(canonical_sampling(arguments.sampling_parser)).digest(),
    ]
    raw = (
        struct.pack("<IHHQ", MAGIC, VERSION, HEADER_BYTES, arguments.session_hash)
        + b"".join(digests)
        + struct.pack("<IIQ", arguments.context_limit, 0, 0)
    )
    if len(raw) != HEADER_BYTES:
        raise AssertionError(f"identity ABI mismatch: {len(raw)}")
    checksum = fnv64(raw)
    labels = [
        "model_checkpoint_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "runtime_sha256",
        "kernels_sha256",
        "ple_layout_sha256",
        "stage_plan_sha256",
        "sampling_parser_sha256",
    ]
    lines = [
        "Q38_SESSION_IDENTITY_V1",
        f"session_hash={arguments.session_hash}",
        *(f"{label}={digest.hex()}" for label, digest in zip(labels, digests)),
        f"context_limit={arguments.context_limit}",
        "flags=0",
        f"identity_checksum={checksum}",
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=arguments.output.name + ".", dir=arguments.output.parent
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write("\n".join(lines) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, arguments.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({"output": str(arguments.output), "checksum": checksum}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
