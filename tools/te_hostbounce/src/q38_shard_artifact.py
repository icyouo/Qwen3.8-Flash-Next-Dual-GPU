#!/usr/bin/env python3
"""Phase B: rewrite a Q38 device artifact into per-rank sharded segments."""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from q38_shard_plan import Tensor, load_stage, plan_tensor, summarise  # type: ignore

ALIGN = 256


def align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def write_bytes_resilient(path: Path, data: bytes, retries: int = 8) -> None:
    """Chunked write with retries — SMB/NAS often times out on huge write_bytes()."""
    import time
    tmp = Path(str(path) + ".partial")
    chunk = 8 << 20  # 8 MiB
    last_err: BaseException | None = None
    for attempt in range(retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as f:
                for i in range(0, len(data), chunk):
                    f.write(data[i : i + chunk])
                    if (i // chunk) % 32 == 0:
                        f.flush()
                f.flush()
            tmp.replace(path)
            return
        except (TimeoutError, OSError, FileNotFoundError) as e:
            last_err = e
            print(f"  ! write retry {attempt+1}/{retries} {path.name}: {e}", flush=True)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            # SMB drop often unmounts parents — recreate before retry
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as mkdir_err:
                print(f"  ! mkdir failed: {mkdir_err}", flush=True)
            time.sleep(min(2 ** attempt, 60))
    assert last_err is not None
    raise last_err




def _row_count(shape: tuple[int, ...]) -> int:
    if len(shape) == 1:
        return 1
    n = 1
    for d in shape[:-1]:
        n *= d
    return n


def extract_row_shard_parts(data: bytes, scale: bytes, t: Tensor, rank: int, world: int):
    R = t.shape[0]
    per = R // world
    row0 = rank * per
    d_per_row = t.data_bytes // R
    s_per_row = (t.scale_bytes // R) if t.scale_bytes else 0
    d_out = data[row0 * d_per_row : (row0 + per) * d_per_row]
    s_out = scale[row0 * s_per_row : (row0 + per) * s_per_row] if s_per_row else b""
    return d_out, s_out, (per,) + t.shape[1:]


def extract_expert_shard_parts(data: bytes, scale: bytes, t: Tensor, rank: int, world: int):
    E = t.shape[0]
    per = E // world
    e0 = rank * per
    d_per = t.data_bytes // E
    s_per = (t.scale_bytes // E) if t.scale_bytes else 0
    d_out = data[e0 * d_per : (e0 + per) * d_per]
    s_out = scale[e0 * s_per : (e0 + per) * s_per] if s_per else b""
    return d_out, s_out, (per,) + t.shape[1:]


def extract_col_shard_parts(data: bytes, scale: bytes, t: Tensor, rank: int, world: int):
    last = t.shape[-1]
    per = last // world
    col0 = rank * per
    rows = _row_count(t.shape)
    new_shape = t.shape[:-1] + (per,)

    if t.fmt == "w8a16_sym_g128":
        G = t.group_size
        d_out = b"".join(data[r * last + col0 : r * last + col0 + per] for r in range(rows))
        groups = last // G
        g_per = groups // world
        g0 = rank * g_per
        s_out = b"".join(
            scale[r * groups * 2 + g0 * 2 : r * groups * 2 + (g0 + g_per) * 2]
            for r in range(rows)
        )
        return d_out, s_out, new_shape

    if t.fmt == "w4a16_sym_g128":
        G = t.group_size
        packed_last = last // 2
        packed_per = per // 2
        packed_col0 = col0 // 2
        d_out = b"".join(
            data[r * packed_last + packed_col0 : r * packed_last + packed_col0 + packed_per]
            for r in range(rows)
        )
        groups = last // G
        g_per = groups // world
        g0 = rank * g_per
        s_out = b"".join(
            scale[r * groups * 2 + g0 * 2 : r * groups * 2 + (g0 + g_per) * 2]
            for r in range(rows)
        )
        return d_out, s_out, new_shape

    if t.fmt == "preserve":
        elem = 2 if t.source_dtype in ("BF16", "F16") else (
            4 if t.source_dtype in ("F32", "I32") else 1
        )
        d_out = b"".join(
            data[r * last * elem + col0 * elem : r * last * elem + (col0 + per) * elem]
            for r in range(rows)
        )
        return d_out, b"", new_shape

    raise SystemExit(f"col_shard extract unsupported fmt={t.fmt} for {t.name}")


def shard_tensor_bytes(data: bytes, scale: bytes, t: Tensor, rank: int, world: int):
    kind, axis, note, shards, warns = plan_tensor(t, world)
    rs = shards[rank]
    if kind == "replicated":
        return kind, data, scale, t.shape, note
    if kind in ("rank0_only", "rank1_only"):
        owner = 0 if kind == "rank0_only" else 1
        if rank != owner:
            return kind, b"", b"", tuple(0 for _ in t.shape), note
        return kind, data, scale, t.shape, note
    if kind == "row_shard":
        d, s, sh = extract_row_shard_parts(data, scale, t, rank, world)
        assert len(d) == rs.data_bytes and len(s) == rs.scale_bytes
        return kind, d, s, sh, note
    if kind == "expert_shard":
        d, s, sh = extract_expert_shard_parts(data, scale, t, rank, world)
        assert len(d) == rs.data_bytes and len(s) == rs.scale_bytes
        return kind, d, s, sh, note
    if kind == "col_shard":
        d, s, sh = extract_col_shard_parts(data, scale, t, rank, world)
        assert len(d) == rs.data_bytes and len(s) == rs.scale_bytes
        return kind, d, s, sh, note
    raise SystemExit(f"unhandled kind {kind}")


def read_extent(mm: mmap.mmap, offset: int, nbytes: int) -> bytes:
    if nbytes == 0:
        return b""
    return mm[offset : offset + nbytes]


def write_stage_rank(
    stage_dir: Path,
    stage,
    rank: int,
    world: int,
    out_stage_dir: Path,
    only_segments: set[int] | None,
    dry_run: bool,
    resume: bool,
) -> dict:
    out_stage_dir.mkdir(parents=True, exist_ok=True)
    seg_out_dir = out_stage_dir / "segments"
    if not dry_run:
        seg_out_dir.mkdir(parents=True, exist_ok=True)

    by_seg: dict[int, list[Tensor]] = {}
    for t in stage.tensors:
        by_seg.setdefault(t.segment, []).append(t)

    out_seg_records: list[tuple[str, int, str]] = []
    tensor_lines: list[str] = []
    stats = {"tensors": 0, "data_bytes": 0, "scale_bytes": 0, "kinds": {}, "skipped_empty": 0, "resumed": 0}

    for seg_idx, seg in enumerate(stage.segments):
        if only_segments is not None and seg_idx not in only_segments:
            continue
        src_path = stage_dir / seg.relative_path
        if not src_path.exists():
            raise SystemExit(f"missing segment {src_path}")
        tensors = by_seg.get(seg_idx, [])

        if dry_run:
            for t in tensors:
                kind, _, _, shards, _ = plan_tensor(t, world)
                rs = shards[rank]
                stats["tensors"] += 1
                stats["data_bytes"] += rs.data_bytes
                stats["scale_bytes"] += rs.scale_bytes
                stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1
            continue

        out_name = Path(seg.relative_path).name
        out_rel = f"segments/{out_name}"
        out_path = seg_out_dir / out_name
        done_marker = Path(str(out_path) + ".done")
        side = Path(str(out_path) + ".tensors.json")

        if resume and done_marker.exists() and out_path.exists() and side.exists():
            meta = json.loads(side.read_text())
            out_seg_i = len(out_seg_records)
            out_seg_records.append((out_rel, meta["bytes"], meta["sha256"]))
            for line in meta["tensor_lines"]:
                parts = line.split("\t")
                if parts[0] == "tensor":
                    parts[6] = str(out_seg_i)
                    tensor_lines.append("\t".join(parts))
            stats["resumed"] += 1
            print(f"  rank{rank} stage{stage.stage} seg{seg_idx} {out_name}: RESUME skip", flush=True)
            continue

        with src_path.open("rb") as sf:
            mm = mmap.mmap(sf.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                if mm.size() < seg.bytes:
                    raise SystemExit(f"short segment {src_path}")
                parts: list[bytes] = []
                cursor = 0
                local_tensor_lines: list[str] = []

                def emit(buf: bytes) -> int:
                    nonlocal cursor
                    pad = align_up(cursor, ALIGN) - cursor
                    if pad:
                        parts.append(b"\0" * pad)
                        cursor += pad
                    off = cursor
                    if buf:
                        parts.append(buf)
                        cursor += len(buf)
                    return off

                for t in tensors:
                    data = read_extent(mm, t.data_offset, t.data_bytes)
                    scale = read_extent(mm, t.scale_offset, t.scale_bytes) if t.scale_bytes else b""
                    kind, d_out, s_out, new_shape, note = shard_tensor_bytes(
                        data, scale, t, rank, world
                    )
                    stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1
                    if len(d_out) == 0 and len(s_out) == 0:
                        stats["skipped_empty"] += 1
                        continue
                    stats["tensors"] += 1
                    stats["data_bytes"] += len(d_out)
                    stats["scale_bytes"] += len(s_out)
                    data_off = emit(d_out)
                    data_sha = sha256_bytes(d_out)
                    if s_out:
                        scale_off = emit(s_out)
                        scale_sha = sha256_bytes(s_out)
                        scale_bytes = len(s_out)
                    else:
                        scale_off, scale_sha, scale_bytes = 0, "-", 0
                    shape_s = ",".join(str(x) for x in new_shape)
                    local_tensor_lines.append(
                        "\t".join([
                            "tensor", t.name, t.source_name, t.source_dtype, t.fmt,
                            str(t.group_size), "<<SEG>>", str(data_off), str(len(d_out)),
                            data_sha, str(scale_off), str(scale_bytes), scale_sha, shape_s,
                        ])
                    )
                pad = align_up(cursor, ALIGN) - cursor
                if pad:
                    parts.append(b"\0" * pad)
                    cursor += pad
                blob = b"".join(parts)
            finally:
                mm.close()

        write_bytes_resilient(out_path, blob)
        digest = sha256_bytes(blob)
        out_seg_i = len(out_seg_records)
        out_seg_records.append((out_rel, len(blob), digest))
        fixed_lines = [ln.replace("<<SEG>>", str(out_seg_i), 1) for ln in local_tensor_lines]
        tensor_lines.extend(fixed_lines)
        side.write_text(json.dumps({"bytes": len(blob), "sha256": digest, "tensor_lines": fixed_lines}))
        done_marker.write_text("ok\n")
        print(
            f"  rank{rank} stage{stage.stage} seg{seg_idx} {out_name}: "
            f"{len(blob)/1e6:.1f} MB (src {seg.bytes/1e6:.1f} MB) tensors={len(local_tensor_lines)}",
            flush=True,
        )

    if dry_run:
        return stats

    art = hashlib.sha256()
    for _, _, dig in out_seg_records:
        art.update(dig.encode())
    lines = [
        "Q38_DEVICE_INDEX_V1",
        f"stage={stage.stage}",
        f"cut={stage.cut}",
        f"source_repo={stage.source_repo}",
        f"source_commit={stage.source_commit}",
        f"policy_sha256={stage.policy_sha256}",
        f"artifact_sha256={art.hexdigest()}",
    ]
    for rel, nbytes, dig in out_seg_records:
        lines.append(f"segment\t{rel}\t{nbytes}\t{dig}")
    lines.extend(tensor_lines)
    index_path = out_stage_dir / "index.q38d"
    index_path.write_text("\n".join(lines) + "\n")
    stats["index"] = str(index_path)
    stats["segments"] = len(out_seg_records)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--stages", type=str, default="0,1")
    ap.add_argument("--only-segments", type=str, default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    stages = [int(x) for x in args.stages.split(",") if x.strip() != ""]
    only = {int(x) for x in args.only_segments.split(",") if x.strip() != ""} if args.only_segments else None
    args.output.mkdir(parents=True, exist_ok=True)

    loaded = [load_stage(args.input / f"stage{s}" / "index.q38d") for s in stages]
    plan = summarise(loaded, args.world_size)
    (args.output / "shard_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")

    manifest = {
        "schema": "Q38_SHARD_ARTIFACT_V1",
        "input": str(args.input),
        "output": str(args.output),
        "world_size": args.world_size,
        "stages": stages,
        "only_segments": sorted(only) if only is not None else None,
        "ranks": {},
    }
    for s in stages:
        stage = load_stage(args.input / f"stage{s}" / "index.q38d")
        stage_dir = args.input / f"stage{s}"
        for rank in range(args.world_size):
            print(f"=== rank{rank} stage{s} ===", flush=True)
            out_stage = args.output / f"rank{rank}" / f"stage{s}"
            stats = write_stage_rank(
                stage_dir, stage, rank, args.world_size, out_stage, only, args.dry_run, args.resume
            )
            manifest["ranks"].setdefault(str(rank), {})[f"stage{s}"] = stats
    (args.output / "shard_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest: {args.output / 'shard_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
