#!/usr/bin/env python3
"""Generate the narrow CUDA 13.1/glibc rsqrt noexcept compatibility header."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    # nvcc resolves CUDA runtime headers relative to cuda_runtime.h, so a lone
    # patched crt/math_functions.h cannot override the toolkit copy.  Mirror
    # only the flat runtime headers plus crt/ (not CUB/Thrust/libcudacxx), then
    # patch the two glibc-compat declarations in that generated build tree.
    args.output_root.mkdir(parents=True, exist_ok=True)
    for source in args.source_root.iterdir():
        if source.is_file() and source.suffix in {".h", ".hpp", ".inc"}:
            shutil.copy2(source, args.output_root / source.name)
    shutil.copytree(
        args.source_root / "crt", args.output_root / "crt", dirs_exist_ok=True
    )
    crt_output = args.output_root / "crt" / "math_functions.h"
    text = crt_output.read_text()
    replacements = (
        (
            "double                 rsqrt(double x);",
            "double                 rsqrt(double x) noexcept(true);",
        ),
        (
            "float                  rsqrtf(float x);",
            "float                  rsqrtf(float x) noexcept(true);",
        ),
    )
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
        elif new not in text:
            raise RuntimeError(f"CUDA math header does not contain expected declaration: {old}")
    crt_output.write_text(text)
    (args.output_root / ".stamp").write_text("cuda-rsqrt-noexcept-v1\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
