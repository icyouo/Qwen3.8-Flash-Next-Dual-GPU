#!/usr/bin/env python3
"""Capture and compare exact committed BF16 target-logit snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from q38_rpc_client import Client, OPCODES  # noqa: E402


SCHEMA = "q38.logits.bf16.v1"
CAPTURE_SCHEMA = "Q38_LOGIT_CAPTURE_V1"
COMPARISON_SCHEMA = "Q38_LOGIT_COMPARISON_V1"


def fnv1a64(raw: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in raw:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def validate_snapshot(snapshot: dict[str, Any], *, require_raw: bool) -> bytes | None:
    if snapshot.get("schema") != SCHEMA:
        raise ValueError("unsupported logit snapshot schema")
    if snapshot.get("dtype") != "bf16" or snapshot.get("byte_order") != "little":
        raise ValueError("logit snapshot is not little-endian BF16")
    elements = snapshot.get("element_count")
    if not isinstance(elements, int) or elements <= 0:
        raise ValueError("logit snapshot has an invalid element count")
    encoded = snapshot.get("raw_bf16_le_hex")
    if encoded is None:
        if require_raw:
            raise ValueError("logit snapshot did not include raw BF16 values")
        return None
    if not isinstance(encoded, str):
        raise ValueError("raw BF16 payload is not hex text")
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as error:
        raise ValueError("raw BF16 payload is malformed") from error
    if len(raw) != elements * 2:
        raise ValueError("raw BF16 payload length differs from element count")
    if snapshot.get("fnv1a64") != fnv1a64(raw):
        raise ValueError("raw BF16 payload checksum does not match the runtime")
    return raw


def manifest_path(output: Path) -> Path:
    return output if output.suffix == ".json" else Path(f"{output}.json")


def write_capture(snapshot: dict[str, Any], output: Path) -> dict[str, Any]:
    raw = validate_snapshot(snapshot, require_raw=True)
    assert raw is not None
    manifest = manifest_path(output)
    binary = manifest.with_suffix(".bf16")
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(raw)
    metadata = {
        key: value
        for key, value in snapshot.items()
        if key != "raw_bf16_le_hex"
    }
    metadata.update(
        {
            "capture_schema": CAPTURE_SCHEMA,
            "raw_file": binary.name,
            "raw_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_capture(path: Path) -> tuple[dict[str, Any], bytes]:
    metadata = json.loads(path.read_text())
    if metadata.get("capture_schema") != CAPTURE_SCHEMA:
        raise ValueError(f"{path} is not a Q38 logit capture")
    raw_name = metadata.get("raw_file")
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError(f"{path} has no raw BF16 file")
    raw = (path.parent / raw_name).read_bytes()
    if (
        len(raw) != metadata.get("raw_bytes")
        or len(raw) != metadata.get("element_count") * 2
    ):
        raise ValueError(f"{path} raw BF16 length does not match its manifest")
    if hashlib.sha256(raw).hexdigest() != metadata.get("sha256"):
        raise ValueError(f"{path} raw BF16 SHA-256 does not match its manifest")
    if fnv1a64(raw) != metadata.get("fnv1a64"):
        raise ValueError(f"{path} raw BF16 FNV-1a does not match its manifest")
    return metadata, raw


def compare_captures(left_path: Path, right_path: Path) -> dict[str, Any]:
    left, left_raw = load_capture(left_path)
    right, right_raw = load_capture(right_path)
    common_elements = min(len(left_raw), len(right_raw)) // 2
    mismatch_count = 0
    first_mismatch: dict[str, int] | None = None
    for index in range(common_elements):
        offset = index * 2
        left_value = int.from_bytes(left_raw[offset : offset + 2], "little")
        right_value = int.from_bytes(right_raw[offset : offset + 2], "little")
        if left_value != right_value:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "index": index,
                    "left_bf16_bits": left_value,
                    "right_bf16_bits": right_value,
                }
    mismatch_count += abs(len(left_raw) - len(right_raw)) // 2
    exact = left_raw == right_raw
    context_fields = (
        "session_hash",
        "transaction_epoch",
        "target_frontier",
        "transaction_kind",
        "element_count",
    )
    comparable_context = all(
        left.get(field) == right.get(field) for field in context_fields
    )
    selected_token_match = (
        left.get("selected_token") == right.get("selected_token")
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "parity_pass": exact and comparable_context and selected_token_match,
        "exact_bf16_match": exact,
        "comparable_context": comparable_context,
        "selected_token_match": selected_token_match,
        "mismatched_elements": mismatch_count,
        "first_mismatch": first_mismatch,
        "left": {
            "path": str(left_path.resolve()),
            "sha256": left["sha256"],
            "element_count": left["element_count"],
            "selected_token": left["selected_token"],
            "transaction_epoch": left["transaction_epoch"],
            "target_frontier": left["target_frontier"],
            "transaction_kind": left["transaction_kind"],
            "session_hash": left.get("session_hash"),
        },
        "right": {
            "path": str(right_path.resolve()),
            "sha256": right["sha256"],
            "element_count": right["element_count"],
            "selected_token": right["selected_token"],
            "transaction_epoch": right["transaction_epoch"],
            "target_frontier": right["target_frontier"],
            "transaction_kind": right["transaction_kind"],
            "session_hash": right.get("session_hash"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--socket", required=True)
    capture.add_argument(
        "--session-hash", type=lambda value: int(value, 0), required=True
    )
    capture.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if arguments.command == "capture":
        with Client(arguments.socket, arguments.session_hash) as client:
            response = client.call(OPCODES["logits"], argument0=1)
        if response.status != 0:
            raise RuntimeError(f"logit capture failed: {response.message}")
        snapshot = json.loads(response.message)
        snapshot["session_hash"] = response.session_hash
        snapshot["rpc_frontiers"] = response.as_dict()["frontiers"]
        result = write_capture(snapshot, arguments.output)
        print(json.dumps(result, sort_keys=True))
        return 0

    result = compare_captures(arguments.left, arguments.right)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["parity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
