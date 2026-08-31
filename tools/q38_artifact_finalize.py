#!/usr/bin/env python3
"""Verify stage fragments and publish the line-based device index ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


COPY_CHUNK = 64 << 20


def expected_stage_tensor_names(stage: int, cut: int) -> set[str]:
    if stage not in (0, 1) or not 0 < cut < 48:
        raise ValueError("invalid Q38 stage contract identity")
    names: set[str] = {"token_embd.weight"}
    hyper = ("norm.weight", "mix_down.weight", "mix_up.weight", "inject.weight")
    moe = (
        "ffn_gate_inp.weight",
        "ffn_gate_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
        "ffn_shexp_gate_inp.weight",
    )
    gdn = (
        "linear_attn.a_log",
        "linear_attn.dt_bias",
        "linear_attn.conv.weight",
        "linear_attn.qkv.weight",
        "linear_attn.z.weight",
        "linear_attn.in_b.weight",
        "linear_attn.in_a.weight",
        "linear_attn.norm.weight",
        "linear_attn.out.weight",
    )
    qsa = (
        "attn_index_qk.weight",
        "attn_index_q_norm.weight",
        "attn_index_k_norm.weight",
        "attn_q.weight",
        "attn_q_norm.weight",
        "attn_k.weight",
        "attn_k_norm.weight",
        "attn_v.weight",
        "attn_output.weight",
    )
    layer_range = range(0, cut) if stage == 0 else range(cut, 48)
    for layer in layer_range:
        prefix = f"blk.{layer}."
        names.update(prefix + "hc_attn." + suffix for suffix in hyper)
        names.update(prefix + "hc_ffn." + suffix for suffix in hyper)
        names.update(prefix + suffix for suffix in moe)
        names.update(prefix + suffix for suffix in (qsa if layer % 4 == 3 else gdn))
        if layer == 1:
            names.update(
                prefix + suffix
                for suffix in (
                    "ple.key.weight",
                    "ple.value.weight",
                    "ple.key_norm.weight",
                    "ple.query_norm.weight",
                    "ple.conv_norm.weight",
                    "ple.conv.weight",
                    "ple.layer_multipliers",
                    "ple.head_offsets",
                    "ple.head_vocab_sizes",
                )
            )
            names.update(f"ple.table.part.{part:03d}" for part in range(128))

    if stage == 1:
        names.update(
            {
                "output.weight",
                "hc_input.norm.weight",
                "hc_input.mix_down.weight",
                "hc_input.mix_up.weight",
                "mtp.fc_embedding.weight",
                "mtp.fc_hidden.weight",
                "mtp.fc_embedding_norm.weight",
                "mtp.fc_hidden_norm.weight",
                "mtp.hc_input.norm.weight",
                "mtp.hc_input.mix_down.weight",
                "mtp.hc_input.mix_up.weight",
            }
        )
        prefix = "mtp.blk.0."
        names.update(prefix + "hc_attn." + suffix for suffix in hyper)
        names.update(prefix + "hc_ffn." + suffix for suffix in hyper)
        names.update(prefix + suffix for suffix in moe)
        names.update(prefix + suffix for suffix in qsa)
    return names


def validate_stage_tensor_contract(names: set[str], stage: int, cut: int) -> None:
    expected = expected_stage_tensor_names(stage, cut)
    missing = sorted(expected - names)
    unexpected = sorted(names - expected)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing[:12]))
        if unexpected:
            detail.append("unexpected=" + ",".join(unexpected[:12]))
        raise ValueError("Q38 runtime tensor contract differs: " + " ".join(detail))


def hash_extent(path: Path, offset: int, count: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(offset)
        remaining = count
        while remaining:
            block = source.read(min(COPY_CHUNK, remaining))
            if not block:
                raise ValueError(f"short extent in {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def format_name(value: str) -> str:
    allowed = {
        "preserve",
        "w4a16_sym_g128",
        "w8a16_sym_g128",
        "fp8_e4m3fn",
    }
    if value not in allowed:
        raise ValueError(f"unknown device format {value}")
    return value


def finalize_stage(stage_root: Path, verify_hashes: bool = True) -> Path:
    fragments = []
    for path in sorted((stage_root / "segments").glob("*.q38.json")):
        fragment = json.loads(path.read_text())
        fragment["_path"] = path
        fragments.append(fragment)
    if not fragments:
        raise ValueError(f"no fragments under {stage_root}")

    identity_fields = ("source_repo", "source_commit", "policy_sha256", "stage", "cut")
    identity = {field: fragments[0][field] for field in identity_fields}
    if identity["stage"] not in (0, 1) or not 0 < identity["cut"] < 48:
        raise ValueError("invalid stage identity")
    for fragment in fragments:
        if fragment.get("schema") != "Q38_STAGE_FRAGMENT_V1":
            raise ValueError(f"invalid fragment schema: {fragment['_path']}")
        for field, value in identity.items():
            if fragment.get(field) != value:
                raise ValueError(f"fragment {field} differs: {fragment['_path']}")

    segments = []
    tensor_records = []
    tensor_names = set()
    for fragment in fragments:
        segment_path = fragment["_path"].parent / fragment["segment"]
        if not segment_path.is_file():
            raise ValueError(f"missing segment {segment_path}")
        if segment_path.stat().st_size != fragment["segment_bytes"]:
            raise ValueError(f"segment size differs: {segment_path}")
        segment_sha = hash_extent(segment_path, 0, segment_path.stat().st_size)
        if segment_sha != fragment["segment_sha256"]:
            raise ValueError(f"segment hash differs: {segment_path}")
        segment_index = len(segments)
        segments.append(
            (
                f"segments/{segment_path.name}",
                segment_path.stat().st_size,
                segment_sha,
            )
        )
        for tensor in fragment["tensors"]:
            name = tensor["name"]
            if name in tensor_names:
                raise ValueError(f"duplicate canonical tensor {name}")
            tensor_names.add(name)
            data_end = tensor["data_offset"] + tensor["data_bytes"]
            scale_end = tensor["scale_offset"] + tensor["scale_bytes"]
            if data_end > segment_path.stat().st_size or scale_end > segment_path.stat().st_size:
                raise ValueError(f"tensor exceeds segment: {name}")
            if verify_hashes:
                data_sha = hash_extent(
                    segment_path, tensor["data_offset"], tensor["data_bytes"]
                )
                if data_sha != tensor["data_sha256"]:
                    raise ValueError(f"tensor data hash differs: {name}")
                if tensor["scale_bytes"]:
                    scale_sha = hash_extent(
                        segment_path, tensor["scale_offset"], tensor["scale_bytes"]
                    )
                    if scale_sha != tensor["scale_sha256"]:
                        raise ValueError(f"tensor scale hash differs: {name}")
            tensor_records.append((segment_index, tensor))

    validate_stage_tensor_contract(
        tensor_names, int(identity["stage"]), int(identity["cut"])
    )

    body = [
        f"stage={identity['stage']}",
        f"cut={identity['cut']}",
        f"source_repo={identity['source_repo']}",
        f"source_commit={identity['source_commit']}",
        f"policy_sha256={identity['policy_sha256']}",
    ]
    for path, count, digest in segments:
        body.append(f"segment\t{path}\t{count}\t{digest}")
    for segment_index, tensor in sorted(tensor_records, key=lambda item: item[1]["name"]):
        scale_sha = tensor["scale_sha256"] or "-"
        shape = ",".join(str(value) for value in tensor["shape"])
        body.append(
            "\t".join(
                (
                    "tensor",
                    tensor["name"],
                    tensor["source_name"],
                    tensor["source_dtype"],
                    format_name(tensor["format"]),
                    str(tensor["group_size"]),
                    str(segment_index),
                    str(tensor["data_offset"]),
                    str(tensor["data_bytes"]),
                    tensor["data_sha256"],
                    str(tensor["scale_offset"]),
                    str(tensor["scale_bytes"]),
                    scale_sha,
                    shape,
                )
            )
        )
    artifact_sha = hashlib.sha256(("\n".join(body) + "\n").encode()).hexdigest()
    output = stage_root / "index.q38d"
    partial = output.with_suffix(output.suffix + ".part")
    contents = ["Q38_DEVICE_INDEX_V1", *body[:5], f"artifact_sha256={artifact_sha}", *body[5:]]
    with partial.open("w") as destination:
        destination.write("\n".join(contents) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(partial, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--skip-payload-hashes", action="store_true")
    args = parser.parse_args()
    outputs = []
    for stage in ("stage0", "stage1"):
        outputs.append(
            str(
                finalize_stage(
                    args.artifact_root / stage,
                    verify_hashes=not args.skip_payload_hashes,
                )
            )
        )
    print(json.dumps({"indexes": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"q38_artifact_finalize: {error}", file=sys.stderr)
        raise
