#!/usr/bin/env python3
"""Plan TP+EP sharding of a Q38 device artifact (offline, no byte rewrite).

Phase A/B planner: parses stage0/stage1 device indexes, decides how each tensor is
sharded across ranks, emits shard_plan.json with per-rank byte census, and
prints the plan summary. Byte-accounting invariants are checked; group-size
alignment is enforced. Byte rewriting lives in q38_shard_artifact.py.

Usage:
  q38_shard_plan.py --input MODEL_ROOT [--world-size 2] [--output plan.json]
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


# ------------- device index parsing (must mirror src/device_artifact.cc) -----

@dataclass
class Segment:
    relative_path: str
    bytes: int
    sha256: str

@dataclass
class Tensor:
    name: str
    source_name: str
    source_dtype: str
    fmt: str
    group_size: int
    segment: int
    data_offset: int
    data_bytes: int
    data_sha256: str
    scale_offset: int
    scale_bytes: int
    scale_sha256: str
    shape: tuple[int, ...]

@dataclass
class Stage:
    stage: int
    cut: int
    source_repo: str
    source_commit: str
    policy_sha256: str
    artifact_sha256: str
    segments: list[Segment] = field(default_factory=list)
    tensors: list[Tensor] = field(default_factory=list)


REQUIRED_HEADERS = {"stage", "cut", "source_repo", "source_commit",
                    "policy_sha256", "artifact_sha256"}


def load_stage(path: Path) -> Stage:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "Q38_DEVICE_INDEX_V1":
        raise SystemExit(f"{path}: bad magic")
    headers: dict[str, str] = {}
    st = Stage(stage=-1, cut=-1, source_repo="", source_commit="",
               policy_sha256="", artifact_sha256="")
    for ln in lines[1:]:
        if not ln or ln.startswith("#"):
            continue
        if "\t" not in ln and "=" in ln:
            k, _, v = ln.partition("=")
            if k not in REQUIRED_HEADERS:
                raise SystemExit(f"unknown header {k}")
            headers[k] = v
            continue
        parts = ln.split("\t")
        if parts[0] == "segment":
            if len(parts) != 4:
                raise SystemExit(f"bad segment line: {ln[:120]}")
            st.segments.append(Segment(parts[1], int(parts[2]), parts[3]))
        elif parts[0] == "tensor":
            if len(parts) != 14:
                raise SystemExit(f"bad tensor line: {ln[:120]}")
            st.tensors.append(Tensor(
                name=parts[1], source_name=parts[2], source_dtype=parts[3],
                fmt=parts[4], group_size=int(parts[5]),
                segment=int(parts[6]),
                data_offset=int(parts[7]), data_bytes=int(parts[8]),
                data_sha256=parts[9],
                scale_offset=int(parts[10]), scale_bytes=int(parts[11]),
                scale_sha256=("" if parts[12] == "-" else parts[12]),
                shape=tuple(int(x) for x in parts[13].split(",")),
            ))
        else:
            raise SystemExit(f"unknown record: {ln[:120]}")
    missing = REQUIRED_HEADERS - headers.keys()
    if missing:
        raise SystemExit(f"missing headers: {missing}")
    st.stage = int(headers["stage"])
    st.cut = int(headers["cut"])
    st.source_repo = headers["source_repo"]
    st.source_commit = headers["source_commit"]
    st.policy_sha256 = headers["policy_sha256"]
    st.artifact_sha256 = headers["artifact_sha256"]
    return st


# ------------------ shard plan rules ---------------------------------------

# Each rule: (pattern, kind, axis, comment)
#   kind: "row_shard"       - split shape[axis] across ranks (contiguous outer)
#         "expert_shard"    - split first dim (experts) across ranks
#         "col_shard"       - split shape[axis] across ranks (interior stride,
#                             quant-aware); v0 keeps as "replicated" and marks
#                             as TODO
#         "replicated"      - full copy on every rank (norms, biases, tiny)
#         "rank0_only"      - live only on rank 0 (e.g. embedding replica trick)
#         "rank1_only"      - live only on rank 1 (LM head / MTP head option)
#   axis:  index into shape or None
#
# We match against .name (short form like "blk.0.linear_attn.qkv.weight").
# Order matters: first match wins.
SHARD_RULES: list[tuple[str, str, int | None, str]] = [
    # ---------------- attention ----------------
    ("*.linear_attn.qkv.weight",        "row_shard", 0, "split by head; outer dim"),
    ("*.linear_attn.z.weight",          "row_shard", 0, "gate-like, row"),
    ("*.linear_attn.in_a.weight",       "row_shard", 0, ""),
    ("*.linear_attn.in_b.weight",       "row_shard", 0, ""),
    ("*.linear_attn.out.weight",        "col_shard", -1, "AllReduce output; needs interior slice"),
    ("*.linear_attn.conv.weight",       "replicated", None, "small conv, replicate"),
    ("*.linear_attn.a_log",             "replicated", None, ""),
    ("*.linear_attn.dt_bias",           "replicated", None, ""),
    ("*.linear_attn.norm.weight",       "replicated", None, ""),

    # ---------------- full-attention layers (every 4th layer in Qwen3.8-Next) --
    ("blk.*.attn_q.weight",             "row_shard", 0, "Q heads split by head"),
    ("blk.*.attn_k.weight",             "replicated", None, "KV heads too few to safely split; replicate"),
    ("blk.*.attn_v.weight",             "replicated", None, ""),
    ("blk.*.attn_index_qk.weight",      "replicated", None, "MLA indexer path; replicate"),
    ("blk.*.attn_index_k_norm.weight",  "replicated", None, ""),
    ("blk.*.attn_index_q_norm.weight",  "replicated", None, ""),
    ("blk.*.attn_k_norm.weight",        "replicated", None, ""),
    ("blk.*.attn_q_norm.weight",        "replicated", None, ""),
    ("blk.*.attn_output.weight",        "col_shard", -1, "output proj; AllReduce"),

    # ---------------- MoE experts (EP) --------
    ("*.ffn_gate_up_exps.weight",       "expert_shard", 0, "512 experts split; each expert intact"),
    ("*.ffn_down_exps.weight",          "expert_shard", 0, ""),

    # ---------------- MoE router (replicated) -
    ("*.ffn_gate_inp.weight",           "replicated", None, "router"),
    ("*.ffn_shexp_gate_inp.weight",     "replicated", None, "shexp gate select"),

    # ---------------- Shared expert (TP) ------
    ("*.ffn_gate_shexp.weight",         "row_shard", 0, "shexp gate: split output dim"),
    ("*.ffn_up_shexp.weight",           "row_shard", 0, "shexp up: split output dim"),
    ("*.ffn_down_shexp.weight",         "col_shard", -1, "shexp down: AllReduce"),

    # ---------------- Hyper connections -------
    ("*.hc_attn.inject.weight",         "row_shard", 0, ""),
    ("*.hc_attn.mix_down.weight",       "row_shard", 0, ""),
    ("*.hc_attn.mix_up.weight",         "row_shard", 0, ""),
    ("*.hc_ffn.inject.weight",          "row_shard", 0, ""),
    ("*.hc_ffn.mix_down.weight",        "row_shard", 0, ""),
    ("*.hc_ffn.mix_up.weight",          "row_shard", 0, ""),
    ("*.hc_input.mix_down.weight",      "row_shard", 0, ""),
    ("*.hc_input.mix_up.weight",        "row_shard", 0, ""),
    ("hc_input.mix_down.weight",        "row_shard", 0, "top-level hc_input"),
    ("hc_input.mix_up.weight",          "row_shard", 0, "top-level hc_input"),
    ("*.hc_*.norm.weight",              "replicated", None, ""),

    # ---------------- MTP (keep on rank1 for now) ---
    ("mtp.fc_embedding.weight",         "rank1_only", None, "MTP head lives on rank1"),
    ("mtp.fc_embedding_norm.weight",    "rank1_only", None, ""),
    ("mtp.fc_hidden.weight",            "rank1_only", None, ""),
    ("mtp.fc_hidden_norm.weight",       "rank1_only", None, ""),
    ("mtp.hc_input.*",                  "rank1_only", None, ""),
    ("mtp.blk.*.attn_index_*.weight",   "rank1_only", None, ""),
    ("mtp.blk.*.attn_k.weight",         "rank1_only", None, ""),
    ("mtp.blk.*.attn_q.weight",         "rank1_only", None, ""),
    ("mtp.blk.*.attn_v.weight",         "rank1_only", None, ""),
    ("mtp.blk.*.attn_*_norm.weight",    "rank1_only", None, ""),
    ("mtp.blk.*.attn_output.weight",    "rank1_only", None, ""),
    ("mtp.blk.*.ffn_gate_inp.weight",   "rank1_only", None, ""),
    ("mtp.blk.*.ffn_gate_up_exps.weight", "rank1_only", None, ""),
    ("mtp.blk.*.ffn_down_exps.weight",  "rank1_only", None, ""),
    ("mtp.blk.*.ffn_gate_shexp.weight", "rank1_only", None, ""),
    ("mtp.blk.*.ffn_up_shexp.weight",   "rank1_only", None, ""),
    ("mtp.blk.*.ffn_down_shexp.weight", "rank1_only", None, ""),
    ("mtp.blk.*.ffn_shexp_gate_inp.weight", "rank1_only", None, ""),
    ("mtp.blk.*.hc_*",                  "rank1_only", None, ""),

    # ---------------- Embedding / LM head -----
    ("token_embd.weight",               "replicated", None, "embedding replica"),
    ("output.weight",                   "col_shard", -1, "LM head: vocab shard, AllGather before sample"),
    ("output_norm.weight",              "replicated", None, ""),

    # ---------------- Norms / biases / PLE ----
    ("*.norm.weight",                   "replicated", None, ""),
    ("*_norm.weight",                   "replicated", None, ""),
    ("ple.*",                           "replicated", None, "PLE tables: replicate for now (revisit)"),
    ("*.ple.*",                         "replicated", None, "per-layer PLE fragments"),
    ("blk.*.ple_*",                     "replicated", None, ""),
    ("model.*",                         "replicated", None, "fallback for HF-style names"),
]


def match_rule(name: str) -> tuple[str, int | None, str]:
    for pat, kind, axis, note in SHARD_RULES:
        if fnmatch.fnmatchcase(name, pat):
            return kind, axis, note
    return "UNMATCHED", None, ""


# ------------------ per-tensor byte planner --------------------------------

def dtype_bytes(dtype: str) -> int:
    if dtype in ("BF16", "F16"): return 2
    if dtype in ("F32", "I32"):  return 4
    if dtype == "I64":            return 8
    if dtype in ("U8", "I8", "F8_E4M3"): return 1
    raise SystemExit(f"unknown dtype {dtype}")


def elements(shape: Sequence[int]) -> int:
    n = 1
    for d in shape: n *= d
    return n


def expected_data_bytes(t: Tensor) -> int:
    e = elements(t.shape)
    if t.fmt == "preserve":       return e * dtype_bytes(t.source_dtype)
    if t.fmt == "w4a16_sym_g128": return e // 2
    if t.fmt == "w8a16_sym_g128": return e
    if t.fmt == "fp8_e4m3fn":     return e
    raise SystemExit(f"unknown fmt {t.fmt}")


def expected_scale_bytes(t: Tensor) -> int:
    e = elements(t.shape)
    if t.fmt in ("w4a16_sym_g128", "w8a16_sym_g128"):
        return (e // t.group_size) * 2
    if t.fmt == "fp8_e4m3fn":
        return (e // t.shape[-1]) * 2
    return 0


@dataclass
class RankShard:
    rank: int
    data_bytes: int
    scale_bytes: int
    shape: tuple[int, ...]


def plan_tensor(t: Tensor, world: int) -> tuple[str, int | None, str, list[RankShard], list[str]]:
    """Return (kind, axis, note, per_rank_shards, warnings)."""
    kind, axis, note = match_rule(t.name)
    warns: list[str] = []
    shards: list[RankShard] = []

    if kind == "UNMATCHED":
        warns.append(f"unmatched name: {t.name} shape={t.shape} fmt={t.fmt}")
        # fallback: replicate
        kind = "replicated"

    if kind == "replicated":
        for r in range(world):
            shards.append(RankShard(r, t.data_bytes, t.scale_bytes, t.shape))
        return kind, axis, note, shards, warns

    if kind in ("rank0_only", "rank1_only"):
        owner = 0 if kind == "rank0_only" else 1
        for r in range(world):
            if r == owner:
                shards.append(RankShard(r, t.data_bytes, t.scale_bytes, t.shape))
            else:
                shards.append(RankShard(r, 0, 0, tuple(0 for _ in t.shape)))
        return kind, axis, note, shards, warns

    if kind == "expert_shard":
        # first dim is expert count; must divide by world
        E = t.shape[0]
        if E % world != 0:
            warns.append(f"{t.name}: E={E} not divisible by world={world}, falling back replicated")
            for r in range(world):
                shards.append(RankShard(r, t.data_bytes, t.scale_bytes, t.shape))
            return "replicated", None, note + " (fallback)", shards, warns
        per_rank_E = E // world
        # each expert is a contiguous block along dim 0
        one_expert_data = t.data_bytes // E
        one_expert_scale = t.scale_bytes // E if t.scale_bytes else 0
        new_shape = (per_rank_E,) + t.shape[1:]
        for r in range(world):
            shards.append(RankShard(r, one_expert_data * per_rank_E,
                                    one_expert_scale * per_rank_E, new_shape))
        return kind, 0, note, shards, warns

    if kind == "row_shard":
        # split shape[axis] across ranks; axis must be 0 for now (contiguous)
        assert axis == 0, "row_shard only supports axis=0 in v0"
        R = t.shape[0]
        if R % world != 0:
            warns.append(f"{t.name}: R={R} not divisible by world={world}, replicating")
            for r in range(world):
                shards.append(RankShard(r, t.data_bytes, t.scale_bytes, t.shape))
            return "replicated", None, note + " (fallback)", shards, warns
        per_rank_R = R // world
        # For each format, bytes scale linearly with R since data/scale layout is
        # contiguous C-order along axis 0.
        d_per = t.data_bytes // R
        s_per = t.scale_bytes // R if t.scale_bytes else 0
        new_shape = (per_rank_R,) + t.shape[1:]
        for r in range(world):
            shards.append(RankShard(r, d_per * per_rank_R, s_per * per_rank_R, new_shape))
        return kind, 0, note, shards, warns

    if kind == "col_shard":
        # Split last dim across ranks (quant-aware). Groups along last dim must
        # divide evenly; otherwise fall back to replicated.
        last = t.shape[-1]
        ax = -1 if axis is None else axis
        if ax not in (-1, len(t.shape) - 1):
            warns.append(f"{t.name}: col_shard only supports last-dim; replicating")
            for r in range(world):
                shards.append(RankShard(r, t.data_bytes, t.scale_bytes, t.shape))
            return "replicated", None, note + " (unsupported axis; replicated)", shards, warns
        if t.fmt in ("w4a16_sym_g128", "w8a16_sym_g128"):
            groups = last // t.group_size
            if groups % world != 0:
                warns.append(f"{t.name}: col_shard cannot align "
                             f"last_dim={last} gs={t.group_size} groups={groups} % world={world}; "
                             f"downgraded to replicated")
                for r in range(world):
                    shards.append(RankShard(r, t.data_bytes, t.scale_bytes, t.shape))
                return "replicated", None, note + " (col_shard alignment failed; replicated)", shards, warns
            per = last // world
            new_shape = t.shape[:-1] + (per,)
            # data/scale bytes scale with last dim
            d_per = t.data_bytes // world
            s_per = t.scale_bytes // world if t.scale_bytes else 0
            for r in range(world):
                shards.append(RankShard(r, d_per, s_per, new_shape))
            return kind, -1, note, shards, warns
        if t.fmt == "preserve":
            if last % world != 0:
                warns.append(f"{t.name}: preserve col_shard last={last} % world; replicating")
                for r in range(world):
                    shards.append(RankShard(r, t.data_bytes, t.scale_bytes, t.shape))
                return "replicated", None, note + " (preserve align fail)", shards, warns
            per = last // world
            new_shape = t.shape[:-1] + (per,)
            d_per = t.data_bytes // world
            for r in range(world):
                shards.append(RankShard(r, d_per, 0, new_shape))
            return kind, -1, note, shards, warns
        warns.append(f"{t.name}: col_shard unsupported fmt={t.fmt}; replicating")
        for r in range(world):
            shards.append(RankShard(r, t.data_bytes, t.scale_bytes, t.shape))
        return "replicated", None, note + " (fmt fallback)", shards, warns

    raise SystemExit(f"unknown kind {kind}")


# ------------------ main ----------------------------------------------------

def summarise(stages: list[Stage], world: int) -> dict:
    per_tensor: list[dict] = []
    warnings: list[str] = []
    census = {r: {"data_bytes": 0, "scale_bytes": 0} for r in range(world)}
    kinds_hist: dict[str, int] = {}
    source_data_bytes = 0
    source_scale_bytes = 0

    for stage in stages:
        for t in stage.tensors:
            # sanity: declared bytes match computed
            exp_d = expected_data_bytes(t)
            exp_s = expected_scale_bytes(t)
            if t.data_bytes != exp_d:
                warnings.append(f"{t.name}: data_bytes {t.data_bytes} != expected {exp_d}")
            if t.scale_bytes != exp_s:
                warnings.append(f"{t.name}: scale_bytes {t.scale_bytes} != expected {exp_s}")
            source_data_bytes += t.data_bytes
            source_scale_bytes += t.scale_bytes

            kind, axis, note, shards, warns = plan_tensor(t, world)
            kinds_hist[kind] = kinds_hist.get(kind, 0) + 1
            warnings.extend(warns)

            per_tensor.append({
                "stage": stage.stage,
                "name": t.name,
                "format": t.fmt,
                "group_size": t.group_size,
                "shape": list(t.shape),
                "source_data_bytes": t.data_bytes,
                "source_scale_bytes": t.scale_bytes,
                "kind": kind,
                "axis": axis,
                "note": note,
                "shards": [
                    {
                        "rank": s.rank, "shape": list(s.shape),
                        "data_bytes": s.data_bytes,
                        "scale_bytes": s.scale_bytes,
                    } for s in shards
                ],
            })
            for s in shards:
                census[s.rank]["data_bytes"] += s.data_bytes
                census[s.rank]["scale_bytes"] += s.scale_bytes

    return {
        "schema": "Q38_SHARD_PLAN_V1",
        "world_size": world,
        "kinds_histogram": kinds_hist,
        "source_totals": {
            "data_bytes": source_data_bytes,
            "scale_bytes": source_scale_bytes,
        },
        "per_rank_totals": census,
        "warnings": warnings,
        "tensors": per_tensor,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Model root (must have stage0/index.q38d, stage1/index.q38d)")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("shard_plan.json"))
    args = p.parse_args()

    stage0 = load_stage(args.input / "stage0" / "index.q38d")
    stage1 = load_stage(args.input / "stage1" / "index.q38d")
    plan = summarise([stage0, stage1], args.world_size)
    args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")

    # summary
    print(f"stages: 2  cut={stage0.cut}")
    print(f"tensors: stage0={len(stage0.tensors)} stage1={len(stage1.tensors)} "
          f"total={len(stage0.tensors)+len(stage1.tensors)}")
    print("kinds:")
    for k, n in sorted(plan["kinds_histogram"].items(), key=lambda x: -x[1]):
        print(f"  {k:15s} {n}")
    src = plan["source_totals"]
    total = src["data_bytes"] + src["scale_bytes"]
    print(f"source data  = {src['data_bytes']/1e9:8.2f} GB")
    print(f"source scale = {src['scale_bytes']/1e9:8.2f} GB")
    print(f"source total = {total/1e9:8.2f} GB")
    for r in range(args.world_size):
        d = plan["per_rank_totals"][r]["data_bytes"]
        s = plan["per_rank_totals"][r]["scale_bytes"]
        print(f"rank{r}: data={d/1e9:8.2f} GB  scale={s/1e9:5.2f} GB  total={(d+s)/1e9:8.2f} GB")
    print(f"warnings: {len(plan['warnings'])}")
    for w in plan["warnings"][:12]:
        print(f"  ! {w}")
    if len(plan["warnings"]) > 12:
        print(f"  ... {len(plan['warnings'])-12} more")
    print(f"plan written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
