#!/usr/bin/env python3
"""Stream official Qwen3.8 safetensors into verified stage segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import BinaryIO, Iterable

try:
    import numpy as np
except ImportError:  # Header inspection remains usable without NumPy.
    np = None

from q38_quant_policy import (
    FP8_E4M3,
    PRESERVE,
    SKIP,
    W4A16_G128,
    W8A16_G128,
    decide,
    policy_identity,
)
from q38_safetensors import SafeTensorRecord, read_safetensors_header


ALIGNMENT = 4096
COPY_CHUNK = 64 << 20
WORKING_FLOAT_BYTES = 256 << 20


def align_file(destination: BinaryIO, alignment: int = ALIGNMENT) -> int:
    position = destination.tell()
    padding = (-position) % alignment
    if padding:
        destination.write(bytes(padding))
    return destination.tell()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(COPY_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def bf16_to_f32(values):
    if np is None:
        raise RuntimeError("NumPy is required for artifact quantization")
    words = np.asarray(values, dtype=np.uint16)
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16_rne(values):
    if np is None:
        raise RuntimeError("NumPy is required for artifact quantization")
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & 1)
    return (rounded >> np.uint32(16)).astype(np.uint16)


def f32_to_e4m3fn(values):
    """Encode float32 to IEEE-style E4M3FN with round-to-nearest-even."""
    if np is None:
        raise RuntimeError("NumPy is required for artifact quantization")
    source = np.asarray(values, dtype=np.float32)
    sign = np.signbit(source).astype(np.uint8) << np.uint8(7)
    magnitude = np.abs(source).astype(np.float32)
    result = np.zeros(source.shape, dtype=np.uint8)

    finite = np.isfinite(magnitude)
    nonzero = finite & (magnitude != 0)
    subnormal = nonzero & (magnitude < np.float32(2.0**-6))
    sub_mantissa = np.rint(magnitude[subnormal] * np.float32(512.0)).astype(
        np.int32
    )
    sub_mantissa = np.clip(sub_mantissa, 0, 8)
    sub_encoded = np.where(sub_mantissa == 8, 0x08, sub_mantissa).astype(np.uint8)
    result[subnormal] = sub_encoded

    normal = nonzero & ~subnormal
    if np.any(normal):
        value = np.minimum(magnitude[normal], np.float32(448.0))
        exponent = np.floor(np.log2(value)).astype(np.int32)
        exponent = np.clip(exponent, -6, 8)
        base = np.exp2(exponent).astype(np.float32)
        mantissa = np.rint((value / base - np.float32(1.0)) * 8).astype(
            np.int32
        )
        carry = mantissa == 8
        exponent = exponent + carry.astype(np.int32)
        mantissa = np.where(carry, 0, mantissa)
        exponent = np.clip(exponent, -6, 8)
        exponent_bits = exponent + 7
        mantissa = np.where(exponent_bits == 15, np.minimum(mantissa, 6), mantissa)
        result[normal] = ((exponent_bits << 3) | mantissa).astype(np.uint8)

    result[~finite] = np.uint8(0x7E)
    result[np.isnan(magnitude)] = np.uint8(0x7F)
    return result | sign


def e4m3fn_to_f32(values):
    if np is None:
        raise RuntimeError("NumPy is required for FP8 decoding")
    encoded = np.asarray(values, dtype=np.uint8)
    sign = np.where((encoded & np.uint8(0x80)) != 0, -1.0, 1.0).astype(
        np.float32
    )
    exponent = ((encoded >> np.uint8(3)) & np.uint8(0x0F)).astype(np.int32)
    mantissa = (encoded & np.uint8(0x07)).astype(np.float32)
    result = np.where(
        exponent == 0,
        mantissa * np.float32(2.0**-9),
        (np.float32(1.0) + mantissa / np.float32(8.0))
        * np.exp2(exponent - 7).astype(np.float32),
    )
    result = np.where((exponent == 15) & (mantissa == 7), 0.0, result)
    return result.astype(np.float32) * sign


def quantize_fp8_rows(values):
    if np is None:
        raise RuntimeError("NumPy is required for FP8 row quantization")
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] == 0:
        raise ValueError("FP8 row quantization requires a nonempty matrix")
    scales = np.max(np.abs(source), axis=1) / np.float32(448.0)
    scales = np.where(scales == 0, np.float32(1.0), scales)
    scale_words = f32_to_bf16_rne(scales)
    stored_scales = bf16_to_f32(scale_words)
    encoded = f32_to_e4m3fn(source / stored_scales[:, None])
    return encoded, scale_words


def tensor_source_array(path: Path, tensor: SafeTensorRecord):
    if np is None:
        raise RuntimeError("NumPy is required for artifact quantization")
    dtypes = {"BF16": np.uint16, "F32": np.float32, "I64": np.int64}
    dtype = dtypes.get(tensor.dtype)
    if dtype is None:
        raise ValueError(f"unsupported official source dtype {tensor.dtype}")
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=tensor.absolute_begin,
        shape=tensor.shape,
        order="C",
    )


def rows_per_chunk(columns: int) -> int:
    # float source + quant workspace can temporarily use about 2x this amount.
    return max(1, WORKING_FLOAT_BYTES // max(4 * columns, 1))


def as_float32(values, source_dtype: str):
    if source_dtype == "BF16":
        return bf16_to_f32(values)
    if source_dtype == "F32":
        return np.asarray(values, dtype=np.float32)
    raise ValueError(f"cannot quantize source dtype {source_dtype}")


def copy_extent(
    source_path: Path,
    tensor: SafeTensorRecord,
    destination: BinaryIO,
) -> tuple[int, int, str]:
    offset = align_file(destination)
    digest = hashlib.sha256()
    remaining = tensor.nbytes
    with source_path.open("rb") as source:
        source.seek(tensor.absolute_begin)
        while remaining:
            block = source.read(min(COPY_CHUNK, remaining))
            if not block:
                raise ValueError(f"short source tensor {tensor.name}")
            destination.write(block)
            digest.update(block)
            remaining -= len(block)
    return offset, tensor.nbytes, digest.hexdigest()


def quantize_groupwise(
    source_path: Path,
    tensor: SafeTensorRecord,
    destination: BinaryIO,
    bits: int,
    group_size: int,
) -> dict:
    if len(tensor.shape) < 2:
        raise ValueError(f"matrix tensor {tensor.name} has rank < 2")
    columns = tensor.shape[-1]
    if columns % group_size:
        raise ValueError(
            f"tensor {tensor.name} input width {columns} is not divisible by {group_size}"
        )
    rows = math.prod(tensor.shape[:-1])
    source = tensor_source_array(source_path, tensor).reshape(rows, columns)
    chunk_rows = rows_per_chunk(columns)
    data_offset = align_file(destination)
    data_digest = hashlib.sha256()
    scale_chunks: list[bytes] = []
    scale_digest = hashlib.sha256()
    quant_max = 7 if bits == 4 else 127

    for first in range(0, rows, chunk_rows):
        values = as_float32(source[first : first + chunk_rows], tensor.dtype)
        grouped = values.reshape(values.shape[0], columns // group_size, group_size)
        scales = np.max(np.abs(grouped), axis=2) / np.float32(quant_max)
        scales = np.where(scales == 0, np.float32(1.0), scales)
        quantized = np.rint(grouped / scales[..., None])
        quantized = np.clip(quantized, -quant_max, quant_max).astype(np.int8)
        quantized = quantized.reshape(values.shape[0], columns)
        if bits == 4:
            low = quantized[:, 0::2].astype(np.int16) & 0xF
            high = quantized[:, 1::2].astype(np.int16) & 0xF
            packed = (low | (high << 4)).astype(np.uint8).tobytes(order="C")
        else:
            packed = quantized.tobytes(order="C")
        destination.write(packed)
        data_digest.update(packed)
        encoded_scales = f32_to_bf16_rne(scales).tobytes(order="C")
        scale_chunks.append(encoded_scales)
        scale_digest.update(encoded_scales)

    data_bytes = destination.tell() - data_offset
    scale_offset = align_file(destination)
    for chunk in scale_chunks:
        destination.write(chunk)
    scale_bytes = destination.tell() - scale_offset
    expected_data = rows * columns * bits // 8
    expected_scales = rows * (columns // group_size) * 2
    if data_bytes != expected_data or scale_bytes != expected_scales:
        raise RuntimeError(f"quantized byte census mismatch for {tensor.name}")
    return {
        "data_offset": data_offset,
        "data_bytes": data_bytes,
        "data_sha256": data_digest.hexdigest(),
        "scale_offset": scale_offset,
        "scale_bytes": scale_bytes,
        "scale_sha256": scale_digest.hexdigest(),
    }


def quantize_fp8(
    source_path: Path,
    tensor: SafeTensorRecord,
    destination: BinaryIO,
) -> dict:
    if len(tensor.shape) < 2:
        raise ValueError(f"row-scaled FP8 tensor {tensor.name} has rank < 2")
    columns = tensor.shape[-1]
    rows = math.prod(tensor.shape[:-1])
    source = tensor_source_array(source_path, tensor).reshape(rows, columns)
    chunk_rows = rows_per_chunk(columns)
    data_offset = align_file(destination)
    data_digest = hashlib.sha256()
    scale_chunks: list[bytes] = []
    scale_digest = hashlib.sha256()
    for first in range(0, rows, chunk_rows):
        values = as_float32(source[first : first + chunk_rows], tensor.dtype)
        encoded_values, scale_words = quantize_fp8_rows(values)
        encoded = encoded_values.tobytes(order="C")
        destination.write(encoded)
        data_digest.update(encoded)
        encoded_scales = scale_words.tobytes(order="C")
        scale_chunks.append(encoded_scales)
        scale_digest.update(encoded_scales)
    data_bytes = destination.tell() - data_offset
    scale_offset = align_file(destination)
    for chunk in scale_chunks:
        destination.write(chunk)
    scale_bytes = destination.tell() - scale_offset
    if data_bytes != rows * columns or scale_bytes != rows * 2:
        raise RuntimeError(f"FP8 byte census mismatch for {tensor.name}")
    return {
        "data_offset": data_offset,
        "data_bytes": data_bytes,
        "data_sha256": data_digest.hexdigest(),
        "scale_offset": scale_offset,
        "scale_bytes": scale_bytes,
        "scale_sha256": scale_digest.hexdigest(),
    }


def compile_stage(
    source_path: Path,
    source_sha256: str,
    source_commit: str,
    tensors: Iterable[tuple[SafeTensorRecord, object]],
    root: Path,
    stage: str,
    cut: int,
) -> Path | None:
    selected = list(tensors)
    if not selected:
        return None
    segment_root = root / stage / "segments"
    segment_root.mkdir(parents=True, exist_ok=True)
    stem = source_path.name.removesuffix(".safetensors")
    segment_path = segment_root / f"{stem}.q38w"
    fragment_path = segment_root / f"{stem}.q38.json"
    segment_part = segment_path.with_suffix(segment_path.suffix + ".part")
    fragment_part = fragment_path.with_suffix(fragment_path.suffix + ".part")
    segment_part.unlink(missing_ok=True)
    fragment_part.unlink(missing_ok=True)

    records = []
    with segment_part.open("wb") as destination:
        for tensor, decision in selected:
            if decision.format == PRESERVE:
                offset, count, digest = copy_extent(source_path, tensor, destination)
                extents = {
                    "data_offset": offset,
                    "data_bytes": count,
                    "data_sha256": digest,
                    "scale_offset": 0,
                    "scale_bytes": 0,
                    "scale_sha256": None,
                }
            elif decision.format == W4A16_G128:
                extents = quantize_groupwise(
                    source_path, tensor, destination, 4, decision.group_size
                )
            elif decision.format == W8A16_G128:
                extents = quantize_groupwise(
                    source_path, tensor, destination, 8, decision.group_size
                )
            elif decision.format == FP8_E4M3:
                extents = quantize_fp8(source_path, tensor, destination)
            else:
                raise ValueError(f"cannot compile format {decision.format}")
            records.append(
                {
                    "source_name": tensor.name,
                    "name": decision.canonical_name,
                    "source_dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "format": decision.format,
                    "group_size": decision.group_size,
                    **extents,
                }
            )
        destination.flush()
        os.fsync(destination.fileno())

    segment_sha256 = sha256_path(segment_part)
    fragment = {
        "schema": "Q38_STAGE_FRAGMENT_V1",
        "source_repo": "Qwen/Qwen3.8-Flash-Next",
        "source_commit": source_commit,
        "source_shard": source_path.name,
        "source_sha256": source_sha256,
        "policy_sha256": policy_identity(cut),
        "stage": int(stage[-1]),
        "cut": cut,
        "segment": segment_path.name,
        "segment_bytes": segment_part.stat().st_size,
        "segment_sha256": segment_sha256,
        "tensors": records,
    }
    fragment_part.write_text(json.dumps(fragment, indent=2, sort_keys=True) + "\n")
    with fragment_part.open("rb") as source:
        os.fsync(source.fileno())
    os.replace(segment_part, segment_path)
    os.replace(fragment_part, fragment_path)
    return fragment_path


def compile_source_shard(args) -> list[Path]:
    source_path = args.source.resolve()
    source_sha256 = sha256_path(source_path)
    if args.source_sha256 and source_sha256 != args.source_sha256:
        raise ValueError(
            f"source sha256 {source_sha256} != expected {args.source_sha256}"
        )
    source = read_safetensors_header(source_path)
    by_stage: dict[str, list] = {"stage0": [], "stage1": []}
    skipped = 0
    for tensor in source.tensors:
        decision = decide(tensor.name, args.cut)
        if decision.format == SKIP:
            skipped += tensor.nbytes
            continue
        by_stage[decision.owner].append((tensor, decision))
        # MTP is owned by stage1 but reuses the base token embedding.  Sending
        # an embedding over the inter-stage boundary would double the decode
        # transport contract and couple MTP to stage0, so replicate this one
        # matrix in the stage1 artifact instead.
        if tensor.name == "model.language_model.embed_tokens.weight":
            by_stage["stage1"].append((tensor, decision))

    if args.plan_only:
        print(
            json.dumps(
                {
                    "source": str(source_path),
                    "source_sha256": source_sha256,
                    "stage0_tensors": len(by_stage["stage0"]),
                    "stage1_tensors": len(by_stage["stage1"]),
                    "skipped_source_bytes": skipped,
                },
                sort_keys=True,
            )
        )
        return []

    result = []
    for stage in ("stage0", "stage1"):
        fragment = compile_stage(
            source_path,
            source_sha256,
            args.source_commit,
            by_stage[stage],
            args.output,
            stage,
            args.cut,
        )
        if fragment:
            result.append(fragment)
    print(json.dumps({"source": str(source_path), "fragments": [str(p) for p in result]}))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--cut", type=int, default=25)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.cut < 48:
        parser.error("--cut must be in 1..47")
    compile_source_shard(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"q38_artifact_compiler: {error}", file=sys.stderr)
        raise
