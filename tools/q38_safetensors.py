#!/usr/bin/env python3
"""Strict, dependency-free safetensors header reader."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import BinaryIO


MAX_HEADER_BYTES = 256 << 20


@dataclass(frozen=True)
class SafeTensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    relative_begin: int
    relative_end: int
    absolute_begin: int
    absolute_end: int

    @property
    def nbytes(self) -> int:
        return self.relative_end - self.relative_begin


@dataclass(frozen=True)
class SafeTensorFile:
    path: Path
    header_bytes: int
    data_begin: int
    metadata: dict[str, str]
    tensors: tuple[SafeTensorRecord, ...]


def _read_exact(source: BinaryIO, count: int) -> bytes:
    result = source.read(count)
    if len(result) != count:
        raise ValueError(f"short safetensors header: got {len(result)}, need {count}")
    return result


def read_safetensors_header(
    path: str | Path, *, verify_payload_extents: bool = True
) -> SafeTensorFile:
    path = Path(path)
    with path.open("rb") as source:
        raw_length = _read_exact(source, 8)
        (header_bytes,) = struct.unpack("<Q", raw_length)
        if header_bytes < 2 or header_bytes > MAX_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header length {header_bytes}")
        raw_header = _read_exact(source, header_bytes)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid safetensors JSON header") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")

    data_begin = 8 + header_bytes
    file_size = path.stat().st_size
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("safetensors metadata must contain string pairs")

    tensors: list[SafeTensorRecord] = []
    previous_end = 0
    for name, item in sorted(
        header.items(), key=lambda pair: pair[1].get("data_offsets", [-1])[0]
    ):
        if not isinstance(name, str) or not isinstance(item, dict):
            raise ValueError("invalid safetensors tensor record")
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if not isinstance(dtype, str):
            raise ValueError(f"tensor {name} has no dtype")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dim, int) or dim < 0 for dim in shape)
        ):
            raise ValueError(f"tensor {name} has an invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or value < 0 for value in offsets)
        ):
            raise ValueError(f"tensor {name} has invalid data offsets")
        begin, end = offsets
        if begin != previous_end or end <= begin:
            raise ValueError(f"tensor {name} has a gap, overlap, or empty extent")
        absolute_begin = data_begin + begin
        absolute_end = data_begin + end
        if verify_payload_extents and absolute_end > file_size:
            raise ValueError(f"tensor {name} exceeds safetensors file")
        tensors.append(
            SafeTensorRecord(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                relative_begin=begin,
                relative_end=end,
                absolute_begin=absolute_begin,
                absolute_end=absolute_end,
            )
        )
        previous_end = end
    if not tensors:
        raise ValueError("safetensors file has no tensors")
    return SafeTensorFile(
        path=path,
        header_bytes=header_bytes,
        data_begin=data_begin,
        metadata=metadata,
        tensors=tuple(tensors),
    )

