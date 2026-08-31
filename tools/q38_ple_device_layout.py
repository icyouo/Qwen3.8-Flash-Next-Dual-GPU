#!/usr/bin/env python3
"""Build a PLE SSD layout directly from a finalized stage0 device index."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import struct


@dataclass(frozen=True)
class Segment:
    path: Path
    size: int


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    format: str
    segment: int
    offset: int
    count: int
    scale_offset: int
    scale_count: int
    shape: tuple[int, ...]


def parse_index(path: Path) -> tuple[list[Segment], dict[str, Tensor]]:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "Q38_DEVICE_INDEX_V1":
        raise ValueError("invalid device index magic")
    segments: list[Segment] = []
    tensors: dict[str, Tensor] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if fields[0] == "segment":
            resolved = (path.parent / fields[1]).resolve()
            size = int(fields[2])
            if not resolved.is_file() or resolved.stat().st_size != size:
                raise ValueError(f"device segment size differs: {resolved}")
            segments.append(Segment(resolved, size))
        elif fields[0] == "tensor":
            if len(fields) != 14:
                raise ValueError("invalid device tensor record")
            tensor = Tensor(
                name=fields[1],
                dtype=fields[3],
                format=fields[4],
                segment=int(fields[6]),
                offset=int(fields[7]),
                count=int(fields[8]),
                scale_offset=int(fields[10]),
                scale_count=int(fields[11]),
                shape=tuple(int(value) for value in fields[13].split(",")),
            )
            if tensor.name in tensors or tensor.segment >= len(segments):
                raise ValueError(f"invalid device tensor identity: {tensor.name}")
            tensors[tensor.name] = tensor
    return segments, tensors


def read_i64(segments: list[Segment], tensor: Tensor, count: int) -> tuple[int, ...]:
    if tensor.dtype != "I64" or tensor.format != "preserve" or tensor.shape != (count,):
        raise ValueError(f"invalid PLE control tensor {tensor.name}")
    with segments[tensor.segment].path.open("rb") as source:
        source.seek(tensor.offset)
        data = source.read(tensor.count)
    if len(data) != 8 * count:
        raise ValueError(f"short PLE control tensor {tensor.name}")
    return struct.unpack(f"<{count}q", data)


def compile_layout(index_path: Path, output_path: Path,
                   unigram_vocab: int = 248320,
                   eos_token: int = 248044) -> dict:
    segments, tensors = parse_index(index_path)
    multipliers = read_i64(segments, tensors["blk.1.ple.layer_multipliers"], 3)
    sizes = read_i64(segments, tensors["blk.1.ple.head_vocab_sizes"], 16)
    offsets = read_i64(segments, tensors["blk.1.ple.head_offsets"], 16)
    expected = 0
    for size, offset in zip(sizes, offsets):
        if size <= 0 or offset != expected:
            raise ValueError("PLE head ranges are not contiguous")
        expected += size

    # PLE table shards are stage-global SSD payloads.  Unlike the small PLE
    # projection/control tensors they deliberately do not carry a decoder
    # layer prefix in the canonical device artifact contract.
    pattern = re.compile(r"^ple\.table\.part\.(\d{3})$")
    parts = []
    for tensor in tensors.values():
        match = pattern.match(tensor.name)
        if not match:
            continue
        if (tensor.format != "fp8_e4m3fn" or len(tensor.shape) != 2 or
                tensor.shape[1] != 160 or tensor.count != tensor.shape[0] * 160 or
                tensor.scale_count != tensor.shape[0] * 2):
            raise ValueError(f"invalid PLE table tensor {tensor.name}")
        parts.append((int(match.group(1)), tensor))
    parts.sort()
    logical_parts = [item[0] for item in parts]
    if not parts or logical_parts != list(range(len(parts))):
        missing = sorted(set(range(max(logical_parts, default=-1) + 1)) -
                         set(logical_parts))
        raise ValueError(
            "PLE table parts are incomplete: "
            f"found={len(parts)}, first={logical_parts[:4]}, "
            f"last={logical_parts[-4:]}, missing={missing[:8]}"
        )

    used_segments = sorted({tensor.segment for _, tensor in parts})
    file_index = {segment: index for index, segment in enumerate(used_segments)}
    payload_by_segment = {segment: 0 for segment in used_segments}
    for _, tensor in parts:
        payload_by_segment[tensor.segment] += tensor.count + tensor.scale_count
    padded_rows = sum(tensor.shape[0] for _, tensor in parts)
    if padded_rows < expected:
        raise ValueError("PLE table has fewer rows than the usable vocabulary")

    lines = [
        "Q38_PLE_LAYOUT_V1",
        "dtype=fp8_e4m3",
        "alignment=1",
        "row_stride=160",
        "row_dimension=160",
        f"usable_rows={expected}",
        f"padded_rows={padded_rows}",
        f"unigram_vocab={unigram_vocab}",
        f"eos_token={eos_token}",
        "multipliers=" + ",".join(str(value) for value in multipliers),
        "head_vocab_sizes=" + ",".join(str(value) for value in sizes),
        "head_offsets=" + ",".join(str(value) for value in offsets),
    ]
    for segment in used_segments:
        item = segments[segment]
        lines.append(
            f"file\t{file_index[segment]}\t{item.path}\t{item.size}\t"
            f"{payload_by_segment[segment]}"
        )
    row_start = 0
    for logical, tensor in parts:
        lines.append(
            f"part\t{logical}\t{file_index[tensor.segment]}\t{row_start}\t"
            f"{tensor.shape[0]}\t{tensor.offset}\t{tensor.count}\t"
            f"{file_index[tensor.segment]}\t{tensor.scale_offset}\t"
            f"{tensor.scale_count}"
        )
        row_start += tensor.shape[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".part")
    partial.write_text("\n".join(lines) + "\n")
    partial.replace(output_path)
    return {"parts": len(parts), "files": len(used_segments),
            "usable_rows": expected, "padded_rows": padded_rows,
            "output": str(output_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unigram-vocab", type=int, default=248320)
    parser.add_argument("--eos-token", type=int, default=248044)
    args = parser.parse_args()
    print(compile_layout(args.stage0_index, args.output,
                         args.unigram_vocab, args.eos_token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
