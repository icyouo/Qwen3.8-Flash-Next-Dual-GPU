#!/usr/bin/env python3
"""Validate TP2 sharded artifact vs source: indexes, segments, one tensor reconstruct."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q38_shard_plan import load_stage, plan_tensor  # type: ignore


def sha256_file_slice(path: Path, off: int, n: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(off)
        left = n
        while left:
            chunk = f.read(min(8 << 20, left))
            if not chunk:
                raise EOFError(path)
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tp2", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()

    man = json.loads((args.tp2 / "shard_manifest.json").read_text())
    assert man["world_size"] == args.world_size

    errors: list[str] = []
    for stage_id in (0, 1):
        src = load_stage(args.source / f"stage{stage_id}" / "index.q38d")
        ranks = [
            load_stage(args.tp2 / f"rank{r}" / f"stage{stage_id}" / "index.q38d")
            for r in range(args.world_size)
        ]
        # segment files exist
        for r, st in enumerate(ranks):
            for seg in st.segments:
                p = args.tp2 / f"rank{r}" / f"stage{stage_id}" / seg.relative_path
                if not p.is_file():
                    errors.append(f"missing {p}")
                elif p.stat().st_size != seg.bytes:
                    errors.append(f"size mismatch {p}: {p.stat().st_size} != {seg.bytes}")

        src_by_name = {t.name: t for t in src.tensors}
        for r, st in enumerate(ranks):
            for t in st.tensors:
                if t.name not in src_by_name:
                    errors.append(f"rank{r} unknown tensor {t.name}")

        # Reconstruct one row_shard and one col_shard from stage0 if present
        probes = []
        for t in src.tensors:
            kind, _, _, shards, _ = plan_tensor(t, args.world_size)
            if kind in ("row_shard", "col_shard", "expert_shard") and t.data_bytes <= 64 << 20:
                probes.append((kind, t))
            if len(probes) >= 3:
                break

        for kind, t in probes:
            src_path = args.source / f"stage{stage_id}" / src.segments[t.segment].relative_path
            with src_path.open("rb") as f:
                f.seek(t.data_offset)
                full = f.read(t.data_bytes)
            parts = []
            for r in range(args.world_size):
                rt = next(x for x in ranks[r].tensors if x.name == t.name)
                rp = args.tp2 / f"rank{r}" / f"stage{stage_id}" / ranks[r].segments[rt.segment].relative_path
                with rp.open("rb") as f:
                    f.seek(rt.data_offset)
                    parts.append(f.read(rt.data_bytes))
            if kind == "row_shard" or kind == "expert_shard":
                recon = b"".join(parts)
            else:  # col_shard — interleave by row
                rows = 1
                for d in t.shape[:-1]:
                    rows *= d
                last = t.shape[-1]
                per = last // args.world_size
                if t.fmt == "w8a16_sym_g128":
                    recon = bytearray()
                    for row in range(rows):
                        for r in range(args.world_size):
                            recon.extend(parts[r][row * per : (row + 1) * per])
                    recon = bytes(recon)
                elif t.fmt == "w4a16_sym_g128":
                    packed_last = last // 2
                    packed_per = per // 2
                    recon = bytearray()
                    for row in range(rows):
                        for r in range(args.world_size):
                            recon.extend(
                                parts[r][row * packed_per : (row + 1) * packed_per]
                            )
                    recon = bytes(recon)
                else:
                    elem = 2 if t.source_dtype in ("BF16", "F16") else 1
                    recon = bytearray()
                    for row in range(rows):
                        for r in range(args.world_size):
                            b = per * elem
                            recon.extend(parts[r][row * b : (row + 1) * b])
                    recon = bytes(recon)
            if recon != full:
                errors.append(
                    f"stage{stage_id} {t.name} {kind} reconstruct mismatch "
                    f"({len(recon)} vs {len(full)})"
                )
            else:
                print(f"OK reconstruct stage{stage_id} {kind} {t.name} ({t.data_bytes} B)")

    if errors:
        print("FAIL")
        for e in errors[:30]:
            print(" ", e)
        if len(errors) > 30:
            print(f"  ... {len(errors)-30} more")
        return 1
    print("PASS tp2_artifact_smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
