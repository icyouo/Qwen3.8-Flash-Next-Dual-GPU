#!/usr/bin/env python3
"""Compile the verbose PLE JSON manifest into Q38_PLE_LAYOUT_V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_declared_path(manifest: Path, declared: str) -> Path:
    candidates = [manifest.parent / declared, manifest.parent.parent / declared]
    existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise ValueError(f"cannot uniquely resolve PLE file {declared}: {candidates}")
    return existing[0]


def compile_layout(
    manifest_path: Path, output_path: Path,
    unigram_vocab: int = 248320, eos_token: int = 248044,
) -> dict:
    data = json.loads(manifest_path.read_text())
    dtype_map = {"BF16": "bf16", "FP8_E4M3": "fp8_e4m3", "FP8": "fp8_e4m3"}
    try:
        dtype = dtype_map[data["storage_dtype"]]
    except KeyError as error:
        raise ValueError(f"unsupported PLE dtype {data.get('storage_dtype')}") from error
    files = sorted(data["physical_files"], key=lambda item: item["index"])
    parts = sorted(data["logical_parts"], key=lambda item: item["logical_part"])
    resolved = {item["index"]: resolve_declared_path(manifest_path, item["path"])
                for item in files}
    for item in files:
        actual = resolved[item["index"]].stat().st_size
        if actual != item["file_bytes"]:
            raise ValueError(f"PLE file size mismatch for {resolved[item['index']]}")

    lines = [
        "Q38_PLE_LAYOUT_V1",
        f"dtype={dtype}",
        f"alignment={data['alignment_bytes']}",
        f"row_stride={data['row_stride_bytes']}",
        f"row_dimension={data['embedding_row_dimension']}",
        f"usable_rows={data['usable_vocabulary_rows']}",
        f"padded_rows={data['padded_vocabulary_rows']}",
        f"unigram_vocab={unigram_vocab}",
        f"eos_token={eos_token}",
        "multipliers=" + ",".join(str(value) for value in data["layer_multipliers"]),
        "head_vocab_sizes=" + ",".join(str(value) for value in data["per_head_vocabulary_sizes"]),
        "head_offsets=" + ",".join(str(value) for value in data["per_head_offsets"]),
    ]
    for item in files:
        lines.append(
            "\t".join(
                ["file", str(item["index"]), str(resolved[item["index"]]),
                 str(item["file_bytes"]), str(item["payload_bytes"])]
            )
        )
    for item in parts:
        lines.append(
            "\t".join(
                ["part", str(item["logical_part"]), str(item["physical_file_index"]),
                 str(item["global_row_start"]), str(item["rows"]),
                 str(item["file_offset"]), str(item["payload_bytes"])]
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return {
        "output": str(output_path),
        "dtype": dtype,
        "files": len(files),
        "parts": len(parts),
        "usable_rows": data["usable_vocabulary_rows"],
        "row_stride": data["row_stride_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unigram-vocab", type=int, default=248320)
    parser.add_argument("--eos-token", type=int, default=248044)
    args = parser.parse_args()
    result = compile_layout(
        args.manifest, args.output, args.unigram_vocab, args.eos_token
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

