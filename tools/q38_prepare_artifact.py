#!/usr/bin/env python3
"""Resumable end-to-end official BF16 -> production Q38 artifact pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import q38_artifact_compiler as compiler  # noqa: E402
import q38_artifact_finalize as finalizer  # noqa: E402
import q38_identity as identity  # noqa: E402
import q38_ple_device_layout as ple_layout  # noqa: E402
from q38_hf_fetch import fetch_one  # noqa: E402
from q38_quant_policy import SKIP, decide, policy_identity  # noqa: E402
from q38_safetensors import read_safetensors_header  # noqa: E402


PINNED_REPO = "Qwen/Qwen3.8-Flash-Next"
PINNED_COMMIT = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def required_shards(source: Path) -> list[Path]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing official tensor index: {index_path}")
    index = json.loads(index_path.read_text())
    names = sorted(set(index.get("weight_map", {}).values()))
    if not names:
        raise ValueError("official tensor index has no weight_map shards")
    shards = [source / name for name in names]
    missing = [path.name for path in shards if not path.is_file()]
    partial = sorted(path.name for path in source.glob("*.part*"))
    if missing or partial:
        raise RuntimeError(
            f"source download incomplete: missing={len(missing)}, "
            f"partial={len(partial)}, examples={(missing + partial)[:8]}"
        )
    return shards


def load_fetch_manifest(source: Path) -> dict:
    path = source / "q38-fetch-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned fetch manifest: {path}")
    manifest = json.loads(path.read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "Q38_HF_FETCH_V1"
        or manifest.get("repo") != PINNED_REPO
        or manifest.get("commit") != PINNED_COMMIT
    ):
        raise RuntimeError("source fetch manifest differs from pinned repo/commit")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError("source fetch manifest has no file records")
    names = [
        item.get("path")
        for item in records
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ]
    if len(names) != len(records) or len(set(names)) != len(names):
        raise RuntimeError("source fetch manifest has invalid/duplicate file records")
    return manifest


def validate_fetch_manifest(source: Path, shards: list[Path]) -> dict:
    manifest = load_fetch_manifest(source)
    records = manifest["files"]
    by_name = {
        item.get("path"): item
        for item in records
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    for shard in shards:
        record = by_name.get(shard.name)
        if (
            not isinstance(record, dict)
            or record.get("size") != shard.stat().st_size
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise RuntimeError(
                f"source shard is not bound by the pinned fetch manifest: {shard.name}"
            )
    return manifest


def expected_fragment_stages(shard: Path, cut: int) -> set[str]:
    result: set[str] = set()
    for tensor in read_safetensors_header(shard).tensors:
        decision = decide(tensor.name, cut)
        if decision.format == SKIP:
            continue
        result.add(decision.owner)
        if tensor.name == "model.language_model.embed_tokens.weight":
            result.add("stage1")
    return result


def fragment_set_complete(
    root: Path,
    shard_name: str,
    expected: set[str],
    source_sha256: str,
    cut: int,
) -> bool:
    stem = shard_name.removesuffix(".safetensors")
    if not expected:
        return True
    fragments = {
        stage: root / stage / "segments" / f"{stem}.q38.json"
        for stage in expected
    }
    if any(not path.is_file() for path in fragments.values()):
        return False
    for stage, fragment_path in fragments.items():
        try:
            fragment = json.loads(fragment_path.read_text())
            if not isinstance(fragment, dict):
                return False
            segment_name = fragment.get("segment")
            if (
                not isinstance(segment_name, str)
                or not segment_name
                or Path(segment_name).name != segment_name
            ):
                return False
            segment = fragment_path.parent / segment_name
            complete = (
                fragment.get("schema") == "Q38_STAGE_FRAGMENT_V1"
                and fragment.get("source_repo") == PINNED_REPO
                and fragment.get("source_shard") == shard_name
                and fragment.get("source_commit") == PINNED_COMMIT
                and fragment.get("source_sha256") == source_sha256
                and fragment.get("policy_sha256") == policy_identity(cut)
                and fragment.get("stage") == int(stage[-1])
                and fragment.get("cut") == cut
                and segment.is_file()
                and segment.stat().st_size == fragment.get("segment_bytes")
                and sha256_file(segment) == fragment.get("segment_sha256")
            )
        except (OSError, TypeError, ValueError, KeyError):
            return False
        if not complete:
            return False
    return True


def existing_fragment_complete(root: Path, shard: Path, cut: int) -> bool:
    return fragment_set_complete(
        root,
        shard.name,
        expected_fragment_stages(shard, cut),
        sha256_file(shard),
        cut,
    )


def expected_fragment_stages_from_index(
    index: dict, shard_name: str, cut: int
) -> set[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("official tensor index has no weight_map")
    result: set[str] = set()
    found = False
    for tensor_name, owner_shard in weight_map.items():
        if owner_shard != shard_name:
            continue
        found = True
        decision = decide(str(tensor_name), cut)
        if decision.format == SKIP:
            continue
        result.add(decision.owner)
        if tensor_name == "model.language_model.embed_tokens.weight":
            result.add("stage1")
    if not found:
        raise ValueError(f"shard is absent from official tensor index: {shard_name}")
    return result


def manifest_record(manifest: dict, name: str, require_sha: bool = True) -> dict:
    for item in manifest["files"]:
        if item["path"] == name:
            size = item.get("size")
            digest = item.get("sha256")
            blob_id = item.get("blob_id")
            if (
                not isinstance(size, int)
                or size <= 0
                or (
                    require_sha
                    and (not isinstance(digest, str) or len(digest) != 64)
                )
                or (
                    not require_sha
                    and not (
                        (isinstance(digest, str) and len(digest) == 64)
                        or (isinstance(blob_id, str) and len(blob_id) == 40)
                    )
                )
            ):
                raise RuntimeError(f"manifest record is not LFS-bound: {name}")
            return item
    raise RuntimeError(f"file is absent from fetch manifest: {name}")


def fetch_manifest_file(
    source: Path, manifest: dict, name: str, require_sha: bool = True
) -> tuple[str, str]:
    record = manifest_record(manifest, name, require_sha=require_sha)
    item = {
        "rfilename": name,
        "size": record["size"],
        "lfs": (
            {"sha256": record["sha256"]}
            if isinstance(record.get("sha256"), str)
            else {}
        ),
        "blobId": record.get("blob_id"),
    }
    return fetch_one(PINNED_REPO, PINNED_COMMIT, source, item)


def stream_compile_shards(
    source: Path,
    output: Path,
    cut: int,
    force: bool,
    prune_source_shards: bool,
) -> int:
    manifest = load_fetch_manifest(source)
    # Fetch all small repository metadata first.  This includes the tensor
    # index and tokenizer inputs required to derive ownership and publish the
    # final identity.  Source tensor shards are then handled one at a time.
    for item in manifest["files"]:
        name = item["path"]
        if not name.endswith(".safetensors"):
            fetch_manifest_file(source, manifest, name, require_sha=False)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("official tensor index has no weight_map shards")
    raw_shard_names = list(weight_map.values())
    if not raw_shard_names or any(
        not isinstance(name, str) or not name for name in raw_shard_names
    ):
        raise ValueError("official tensor index has invalid shard names")
    shard_names = sorted(set(raw_shard_names))

    for completed, shard_name in enumerate(shard_names, 1):
        record = manifest_record(manifest, shard_name)
        expected = expected_fragment_stages_from_index(index, shard_name, cut)
        reusable = not force and fragment_set_complete(
            output, shard_name, expected, record["sha256"], cut
        )
        shard = source / shard_name
        status = "reused"
        if not reusable:
            _name, fetch_status = fetch_manifest_file(
                source, manifest, shard_name
            )
            result = compile_one(shard, output, cut, force=True)
            status = f"{fetch_status}+{result['status']}"
        if not fragment_set_complete(
            output, shard_name, expected, record["sha256"], cut
        ):
            raise RuntimeError(
                f"compiled shard failed durable fragment verification: {shard_name}"
            )
        if prune_source_shards:
            shard.unlink(missing_ok=True)
            shard.with_name(shard.name + ".part").unlink(missing_ok=True)
            status += "+pruned"
        print(
            json.dumps(
                {
                    "event": "shard",
                    "completed": completed,
                    "total": len(shard_names),
                    "shard": shard_name,
                    "status": status,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return len(shard_names)


def compile_one(shard: Path, output: Path, cut: int, force: bool) -> dict:
    if not force and existing_fragment_complete(output, shard, cut):
        return {"shard": shard.name, "status": "reused"}
    arguments = SimpleNamespace(
        source=shard,
        output=output,
        source_commit=PINNED_COMMIT,
        source_sha256=None,
        cut=cut,
        plan_only=False,
    )
    fragments = compiler.compile_source_shard(arguments)
    return {
        "shard": shard.name,
        "status": "compiled",
        "fragments": [str(path) for path in fragments],
    }


def tokenizer_files(source: Path) -> list[Path]:
    candidates = [
        source / name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "vocab.json",
            "merges.txt",
        )
        if (source / name).is_file()
    ]
    if not candidates or not (source / "tokenizer_config.json").is_file():
        raise RuntimeError("official tokenizer files are incomplete")
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-hash", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--cut", type=int, default=25)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--context-limit", type=int, default=262_144)
    parser.add_argument("--sampling-parser")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stream-from-manifest", action="store_true")
    parser.add_argument("--prune-source-shards", action="store_true")
    arguments = parser.parse_args()
    if not 0 < arguments.cut < 48 or arguments.jobs not in (1, 2):
        parser.error("--cut must be 1..47 and --jobs must be 1 or 2")
    if arguments.prune_source_shards and not arguments.stream_from_manifest:
        parser.error("--prune-source-shards requires --stream-from-manifest")
    if arguments.stream_from_manifest and arguments.jobs != 1:
        parser.error("streaming conversion requires --jobs 1 for bounded storage")
    source = arguments.source.resolve()
    metadata = arguments.metadata.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arguments.stream_from_manifest:
        manifest = load_fetch_manifest(source)
        shard_count = sum(
            1 for item in manifest["files"]
            if item["path"].endswith(".safetensors")
        )
    else:
        shards = required_shards(source)
        validate_fetch_manifest(source, shards)
        shard_count = len(shards)
    print(
        json.dumps(
            {
                "event": "preflight",
                "repo": PINNED_REPO,
                "commit": PINNED_COMMIT,
                "shards": shard_count,
                "cut": arguments.cut,
                "jobs": arguments.jobs,
                "streaming": arguments.stream_from_manifest,
                "prune_source_shards": arguments.prune_source_shards,
            }
        ),
        flush=True,
    )

    if arguments.stream_from_manifest:
        stream_compile_shards(
            source,
            output,
            arguments.cut,
            arguments.force,
            arguments.prune_source_shards,
        )
    else:
        with ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
            futures = {
                pool.submit(
                    compile_one, shard, output, arguments.cut, arguments.force
                ): shard
                for shard in shards
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    json.dumps(
                        {"event": "shard", "completed": completed,
                         "total": len(shards), **result},
                        sort_keys=True,
                    ),
                    flush=True,
                )

    stage0_index = finalizer.finalize_stage(output / "stage0")
    stage1_index = finalizer.finalize_stage(output / "stage1")
    ple_path = output / "ple.q38p"
    ple_result = ple_layout.compile_layout(stage0_index, ple_path)
    identity_path = output / "session.q38i"
    tokenizer_inputs = tokenizer_files(source)
    identity_command = [
        sys.executable,
        str(TOOLS / "q38_identity.py"),
        "--output", str(identity_path),
        "--session-hash", str(arguments.session_hash),
        "--context-limit", str(arguments.context_limit),
        "--model-metadata", str(metadata),
        "--tokenizer", *(str(path) for path in tokenizer_inputs),
        "--chat-template", str(source / "tokenizer_config.json"),
        "--runtime-root", str(TOOLS.parent),
        "--ple-layout", str(ple_path),
        "--stage0-index", str(stage0_index),
        "--stage1-index", str(stage1_index),
    ]
    if arguments.sampling_parser:
        identity_command.extend(
            ["--sampling-parser", arguments.sampling_parser]
        )
    subprocess.run(identity_command, check=True)
    ready = {
        "schema": "Q38_PRODUCTION_ARTIFACT_READY_V1",
        "model": "Qwen3.8-Flash-Next",
        "source_repo": PINNED_REPO,
        "source_commit": PINNED_COMMIT,
        "cut": arguments.cut,
        "context_limit": arguments.context_limit,
        "vocabulary": 248_320,
        "sampling": identity.sampling_config(arguments.sampling_parser),
        "source_model_root": str(source),
        "model_metadata": str(metadata),
        "tokenizer_files": [str(path) for path in tokenizer_inputs],
        "chat_template": str(source / "tokenizer_config.json"),
        "stage0_index": str(stage0_index),
        "stage1_index": str(stage1_index),
        "ple_layout": str(ple_path),
        "session_identity": str(identity_path),
        "stage0_index_sha256": sha256_file(stage0_index),
        "stage1_index_sha256": sha256_file(stage1_index),
        "ple_layout_sha256": sha256_file(ple_path),
        "identity_sha256": sha256_file(identity_path),
        "ple": ple_result,
    }
    ready_path = output / "READY.json"
    partial = ready_path.with_suffix(".json.part")
    partial.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
    with partial.open("rb") as source_file:
        os.fsync(source_file.fileno())
    os.replace(partial, ready_path)
    print(json.dumps({"event": "ready", **ready}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"q38_prepare_artifact: {error}", file=sys.stderr)
        raise
