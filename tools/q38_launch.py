#!/usr/bin/env python3
"""Fail-closed production launcher for a finalized Q38 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from q38_identity import (  # noqa: E402
    fnv64,
    hash_group,
    hash_path,
    hash_runtime,
    sampling_config,
)
from q38_rpc_client import Client, OPCODES  # noqa: E402


PINNED_REPO = "Qwen/Qwen3.8-Flash-Next"
PINNED_COMMIT = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PRODUCTION_CUT = 25
READY_SCHEMA = "Q38_PRODUCTION_ARTIFACT_READY_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact_path(ready_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("READY artifact path is invalid")
    path = Path(value)
    return (path if path.is_absolute() else ready_path.parent / path).resolve()


def parse_identity(path: Path) -> dict[str, str]:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "Q38_SESSION_IDENTITY_V1":
        raise ValueError("session identity has invalid magic/version")
    result: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise ValueError("session identity has invalid/duplicate fields")
        result[key] = value
    required = {
        "session_hash",
        "context_limit",
        "flags",
        "identity_checksum",
        "model_checkpoint_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "runtime_sha256",
        "kernels_sha256",
        "ple_layout_sha256",
        "stage_plan_sha256",
        "sampling_parser_sha256",
    }
    if set(result) != required:
        raise ValueError("session identity fields differ from V1")
    digest_names = (
        "model_checkpoint",
        "tokenizer",
        "chat_template",
        "runtime",
        "kernels",
        "ple_layout",
        "stage_plan",
        "sampling_parser",
    )
    try:
        raw = (
            struct.pack("<IHHQ", int.from_bytes(b"Q38I", "little"), 1, 288,
                        int(result["session_hash"], 0))
            + b"".join(bytes.fromhex(result[f"{name}_sha256"])
                       for name in digest_names)
            + struct.pack("<IIQ", int(result["context_limit"], 0),
                          int(result["flags"], 0), 0)
        )
    except (ValueError, struct.error) as error:
        raise ValueError("session identity field encoding is invalid") from error
    if len(raw) != 288 or fnv64(raw) != int(result["identity_checksum"], 0):
        raise ValueError("session identity checksum differs")
    return result


def parse_segment_records(index: Path) -> list[tuple[Path, int, str]]:
    lines = index.read_text().splitlines()
    if not lines or lines[0] != "Q38_DEVICE_INDEX_V1":
        raise ValueError(f"invalid device index: {index}")
    result = []
    for line in lines[1:]:
        fields = line.split("\t")
        if fields[0] != "segment":
            continue
        if len(fields) != 4:
            raise ValueError(f"invalid segment record in {index}")
        path = (index.parent / fields[1]).resolve()
        result.append((path, int(fields[2]), fields[3]))
    if not result:
        raise ValueError(f"device index has no segments: {index}")
    return result


def verify_segments(indexes: Sequence[Path], verify_hashes: bool) -> dict[str, int]:
    seen: set[Path] = set()
    total_bytes = 0
    for index in indexes:
        for path, expected_bytes, expected_sha in parse_segment_records(index):
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"artifact segment size differs: {path}")
            if verify_hashes and sha256_file(path) != expected_sha:
                raise ValueError(f"artifact segment digest differs: {path}")
            total_bytes += expected_bytes
    return {"segments": len(seen), "segment_bytes": total_bytes}


def validate_ready(
    ready_path: Path, *, verify_segment_hashes: bool = True
) -> dict[str, object]:
    ready_path = ready_path.resolve()
    ready = json.loads(ready_path.read_text())
    if not isinstance(ready, dict) or ready.get("schema") != READY_SCHEMA:
        raise ValueError("READY manifest has invalid schema")
    if (
        ready.get("source_repo") != PINNED_REPO
        or ready.get("source_commit") != PINNED_COMMIT
        or ready.get("cut") != PRODUCTION_CUT
        or ready.get("context_limit") != 262_144
        or ready.get("vocabulary") != 248_320
    ):
        raise ValueError("READY production identity differs from the pinned contract")
    partials = sorted(ready_path.parent.rglob("*.part"))
    if partials:
        raise ValueError(f"artifact contains partial files: {partials[:4]}")
    paths = {
        name: resolve_artifact_path(ready_path, ready.get(name))
        for name in (
            "stage0_index",
            "stage1_index",
            "ple_layout",
            "session_identity",
        )
    }
    digest_fields = {
        "stage0_index": "stage0_index_sha256",
        "stage1_index": "stage1_index_sha256",
        "ple_layout": "ple_layout_sha256",
        "session_identity": "identity_sha256",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = ready.get(digest_fields[name])
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"READY digest differs for {name}")
    identity = parse_identity(paths["session_identity"])
    if int(identity["flags"], 0) != 0:
        raise ValueError("production launch rejects development identity")
    if int(identity["context_limit"], 0) != ready["context_limit"]:
        raise ValueError("identity and READY context limits differ")
    session_hash = int(identity["session_hash"], 0)
    if not 0 < session_hash < 1 << 64:
        raise ValueError("identity session hash is invalid")
    sampling = sampling_config(json.dumps(ready.get("sampling", {})))
    external_paths = {
        "source_model_root": resolve_artifact_path(
            ready_path, ready.get("source_model_root")
        ),
        "model_metadata": resolve_artifact_path(
            ready_path, ready.get("model_metadata")
        ),
        "chat_template": resolve_artifact_path(
            ready_path, ready.get("chat_template")
        ),
    }
    tokenizer_values = ready.get("tokenizer_files")
    if not isinstance(tokenizer_values, list) or not tokenizer_values:
        raise ValueError("READY tokenizer file list is invalid")
    tokenizer_paths = [
        resolve_artifact_path(ready_path, value) for value in tokenizer_values
    ]
    digest_evidence = {
        "model_checkpoint_sha256": hash_path(external_paths["model_metadata"]).hex(),
        "tokenizer_sha256": hash_group(tokenizer_paths).hex(),
        "chat_template_sha256": hash_path(external_paths["chat_template"]).hex(),
        "runtime_sha256": hash_runtime(TOOLS.parent).hex(),
        "kernels_sha256": hash_path(TOOLS.parent / "cuda").hex(),
        "ple_layout_sha256": hash_path(paths["ple_layout"]).hex(),
        "stage_plan_sha256": hash_group(
            [paths["stage0_index"], paths["stage1_index"]]
        ).hex(),
        "sampling_parser_sha256": hashlib.sha256(
            json.dumps(sampling, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    for field, digest in digest_evidence.items():
        if identity[field] != digest:
            raise ValueError(f"session identity input differs: {field}")
    segment_census = verify_segments(
        [paths["stage0_index"], paths["stage1_index"]], verify_segment_hashes
    )
    return {
        "ready": ready,
        "paths": paths,
        "identity": identity,
        "session_hash": session_hash,
        "sampling": sampling,
        "external_paths": external_paths,
        "tokenizer_paths": tokenizer_paths,
        **segment_census,
    }


def sampling_arguments(config: dict[str, object]) -> list[str]:
    result = [
        "--sampling",
        str(config["mode"]),
        "--temperature",
        str(config["temperature"]),
        "--top-p",
        str(config["top_p"]),
        "--top-k",
        str(config["top_k"]),
        "--sampling-seed",
        str(config["seed"]),
        "--repetition-penalty",
        str(config["repetition_penalty"]),
        "--frequency-penalty",
        str(config["frequency_penalty"]),
        "--presence-penalty",
        str(config["presence_penalty"]),
    ]
    stops = config["stop_token_ids"]
    if not isinstance(stops, list):
        raise ValueError("sampling stop_token_ids must be a list")
    for token in stops:
        result.extend(["--stop-token-id", str(token)])
    return result


def runtime_command(
    validated: dict[str, object],
    runtime: Path,
    socket_path: Path,
    snapshot: Path | None,
    chunk: int,
    ple_cache_gib: int,
    enable_mtp: bool = False,
) -> list[str]:
    paths = validated["paths"]
    ready = validated["ready"]
    assert isinstance(paths, dict) and isinstance(ready, dict)
    command = [
        str(runtime.resolve()),
        "--stage0-index",
        str(paths["stage0_index"]),
        "--stage1-index",
        str(paths["stage1_index"]),
        "--ple-layout",
        str(paths["ple_layout"]),
        "--identity",
        str(paths["session_identity"]),
        "--session-hash",
        str(validated["session_hash"]),
        "--context-limit",
        str(ready["context_limit"]),
        "--chunk",
        str(chunk),
        "--ple-cache-gib",
        str(ple_cache_gib),
        "--ple-io",
        "direct",
        "--ple-queue-depth",
        "64",
        "--socket",
        str(socket_path.resolve()),
    ]
    if snapshot is not None:
        command.extend(["--snapshot-journal", str(snapshot.resolve())])
    if enable_mtp:
        command.append("--enable-mtp")
    command.extend(sampling_arguments(validated["sampling"]))  # type: ignore[arg-type]
    return command


def wait_for_executor(socket_path: Path, session_hash: int, process: subprocess.Popen, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"executor exited during startup with code {process.returncode}")
        try:
            with Client(str(socket_path), session_hash) as client:
                response = client.call(OPCODES["ping"])
            if response.status == 0:
                return
            last_error = RuntimeError(response.message)
        except (FileNotFoundError, ConnectionError, OSError) as error:
            last_error = error
        time.sleep(0.1)
    raise TimeoutError(f"executor socket did not become ready: {last_error}")


def terminate(process: subprocess.Popen, timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def interrupt_on_termination(_signum: int, _frame: object) -> None:
    """Route service-manager SIGTERM through the normal child cleanup path."""
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("build/q38-cuda-runtime"))
    parser.add_argument("--socket", type=Path, default=Path("/tmp/q38-executor.sock"))
    parser.add_argument("--snapshot", type=Path, default=Path("state/session.q38j"))
    parser.add_argument(
        "--durability",
        choices=("strict", "off"),
        default="strict",
        help="strict syncs a snapshot before each successful writer response; off disables crash recovery",
    )
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--ple-cache-gib", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--api-key")
    parser.add_argument("--skip-segment-hashes", action="store_true")
    parser.add_argument("--executor-only", action="store_true")
    parser.add_argument("--enable-mtp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.chunk <= 0 or arguments.ple_cache_gib <= 0:
        parser.error("chunk/cache values must be positive")
    if arguments.host not in ("127.0.0.1", "::1", "localhost") and not arguments.api_key:
        parser.error("non-loopback binding requires --api-key")
    if arguments.skip_segment_hashes and not arguments.dry_run:
        parser.error("production launch cannot skip artifact segment hashes")
    validated = validate_ready(
        arguments.ready, verify_segment_hashes=not arguments.skip_segment_hashes
    )
    command = runtime_command(
        validated,
        arguments.runtime,
        arguments.socket,
        arguments.snapshot if arguments.durability == "strict" else None,
        arguments.chunk,
        arguments.ple_cache_gib,
        arguments.enable_mtp,
    )
    sidecar = [
        sys.executable,
        str(TOOLS / "q38_sidecar.py"),
        "--socket",
        str(arguments.socket.resolve()),
        "--session-hash",
        str(validated["session_hash"]),
        "--context-limit",
        str(validated["ready"]["context_limit"]),  # type: ignore[index]
        "--vocabulary",
        str(validated["ready"]["vocabulary"]),  # type: ignore[index]
        "--model",
        str(validated["ready"].get("model", "Qwen3.8-Flash-Next")),  # type: ignore[union-attr]
        "--host",
        arguments.host,
        "--port",
        str(arguments.port),
    ]
    tokenizer_root = arguments.tokenizer or validated["external_paths"]["source_model_root"]  # type: ignore[index]
    if tokenizer_root:
        sidecar.extend(["--tokenizer", str(Path(tokenizer_root).resolve())])
    if arguments.api_key:
        sidecar.extend(["--api-key", arguments.api_key])
    stops = validated["sampling"]["stop_token_ids"]  # type: ignore[index]
    for token in stops:
        sidecar.extend(["--stop-token-id", str(token)])
    if arguments.enable_mtp:
        sidecar.append("--enable-mtp")
    evidence = {
        "schema": "Q38_LAUNCH_PLAN_V1",
        "session_hash": validated["session_hash"],
        "artifact_segments": validated["segments"],
        "artifact_bytes": validated["segment_bytes"],
        "segment_hashes_verified": not arguments.skip_segment_hashes,
        "mtp_enabled": arguments.enable_mtp,
        "durability": arguments.durability,
        "runtime_command": command,
        "sidecar_command": None if arguments.executor_only else sidecar,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    if arguments.dry_run:
        return 0
    if not arguments.runtime.is_file() or not os.access(arguments.runtime, os.X_OK):
        raise FileNotFoundError(f"runtime binary is not executable: {arguments.runtime}")
    if arguments.durability == "strict":
        arguments.snapshot.parent.mkdir(parents=True, exist_ok=True)
    arguments.socket.parent.mkdir(parents=True, exist_ok=True)
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_on_termination)
    executor: subprocess.Popen | None = None
    frontend: subprocess.Popen | None = None
    try:
        executor = subprocess.Popen(command)
        wait_for_executor(arguments.socket.resolve(), validated["session_hash"], executor)
        if arguments.executor_only:
            return executor.wait()
        frontend = subprocess.Popen(sidecar)
        while True:
            executor_code = executor.poll()
            frontend_code = frontend.poll()
            if executor_code is not None:
                raise RuntimeError(f"executor exited with code {executor_code}")
            if frontend_code is not None:
                raise RuntimeError(f"sidecar exited with code {frontend_code}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 130
    finally:
        if frontend is not None:
            terminate(frontend)
        if executor is not None:
            terminate(executor)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"q38_launch: {error}", file=sys.stderr)
        raise
