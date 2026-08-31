#!/usr/bin/env python3
"""Manifest-driven, resumable Hugging Face artifact fetcher.

The runtime build host has less free space than the upstream BF16 repository.
This helper selects files before transfer, resumes partial downloads, and
verifies the LFS SHA-256 before atomically publishing each file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request


def api_model(repo: str, revision: str) -> dict:
    base = "https://huggingface.co/api/models/" + urllib.parse.quote(
        repo, safe="/"
    )
    url = (
        base
        if revision == "main"
        else base + "/revision/" + urllib.parse.quote(revision, safe="")
    ) + "?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def matches(name: str, includes: list[str], excludes: list[str]) -> bool:
    return (not includes or any(fnmatch.fnmatch(name, item) for item in includes)) and not any(
        fnmatch.fnmatch(name, item) for item in excludes
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(8 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def fetch_one(repo: str, revision: str, root: Path, item: dict) -> tuple[str, str]:
    name = item["rfilename"]
    expected_size = int(item.get("size") or 0)
    lfs = item.get("lfs") or {}
    expected_sha = lfs.get("sha256")
    expected_blob = item.get("blobId")
    destination = root / name
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size == expected_size:
        if (
            (expected_sha and sha256_file(destination) == expected_sha)
            or (
                not expected_sha
                and isinstance(expected_blob, str)
                and git_blob_sha1(destination) == expected_blob
            )
        ):
            return name, "present"
        destination.replace(partial)

    url = (
        "https://huggingface.co/"
        + repo
        + "/resolve/"
        + urllib.parse.quote(revision, safe="")
        + "/"
        + urllib.parse.quote(name, safe="/")
    )
    command = [
        "wget",
        "--continue",
        "--tries=20",
        "--timeout=30",
        "--read-timeout=60",
        "--no-verbose",
        "--output-document",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True)
    if expected_size and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"{name}: size {partial.stat().st_size}, expected {expected_size}"
        )
    if expected_sha:
        actual_sha = sha256_file(partial)
        if actual_sha != expected_sha:
            raise RuntimeError(f"{name}: sha256 {actual_sha}, expected {expected_sha}")
    elif isinstance(expected_blob, str):
        actual_blob = git_blob_sha1(partial)
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"{name}: git blob {actual_blob}, expected {expected_blob}"
            )
    else:
        raise RuntimeError(f"{name}: upstream manifest has no content digest")
    os.replace(partial, destination)
    return name, "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--expected-commit")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 16:
        parser.error("--jobs must be in 1..16")

    model = api_model(args.repo, args.revision)
    resolved_commit = model.get("sha")
    if not isinstance(resolved_commit, str) or len(resolved_commit) != 40:
        raise RuntimeError("Hugging Face response has no resolved commit")
    if args.expected_commit and resolved_commit != args.expected_commit:
        raise RuntimeError(
            f"resolved commit {resolved_commit} differs from "
            f"--expected-commit {args.expected_commit}"
        )
    selected = [
        item
        for item in model.get("siblings", [])
        if matches(item["rfilename"], args.include, args.exclude)
    ]
    selected.sort(key=lambda item: item["rfilename"])
    if not selected:
        raise RuntimeError("selection matched no repository files")

    manifest = {
        "schema": "Q38_HF_FETCH_V1",
        "repo": args.repo,
        "revision": args.revision,
        "commit": resolved_commit,
        "total_bytes": sum(int(item.get("size") or 0) for item in selected),
        "files": [
            {
                "path": item["rfilename"],
                "size": int(item.get("size") or 0),
                "sha256": (item.get("lfs") or {}).get("sha256"),
                "blob_id": item.get("blobId"),
            }
            for item in selected
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "q38-fetch-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "files": len(selected),
                "bytes": manifest["total_bytes"],
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.manifest_only:
        return 0

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                fetch_one, args.repo, resolved_commit, args.output, item
            ): item["rfilename"]
            for item in selected
        }
        for future in concurrent.futures.as_completed(futures):
            name, status = future.result()
            completed += 1
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(selected),
                        "path": name,
                        "status": status,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # fail closed with a concise durable log record
        print(f"q38_hf_fetch: {error}", file=sys.stderr)
        raise
