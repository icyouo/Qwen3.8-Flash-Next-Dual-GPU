#!/usr/bin/env python3
"""Plan or execute isolated, fixed-output optimization comparisons on two GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from q38_launch import runtime_command, terminate, validate_ready, wait_for_executor
from q38_rpc_client import Client, OPCODES

CASES = {
    "baseline": ({}, {}),
    "hc": ({"Q38_CUDA_HC_GEMV": "1"}, {}),
    "hc-fused": ({"Q38_CUDA_HC_FUSED": "1"}, {}),
    "hc-grouped": ({"Q38_CUDA_HC_GROUPED": "1", "Q38_CUDA_HC_FUSED": "1"}, {}),
    "ple-fused": ({"Q38_CUDA_PLE_FUSED": "1"}, {}),
    "ple-norm": ({"Q38_CUDA_PLE_NORM_PAIR": "1"}, {}),
    "ple-residual": ({"Q38_CUDA_PLE_RESIDUAL": "1"}, {}),
    "ple-ops": ({"Q38_CUDA_PLE_NORM_PAIR": "1", "Q38_CUDA_PLE_RESIDUAL": "1"}, {}),
    "hc-ple": ({"Q38_CUDA_HC_GROUPED": "1", "Q38_CUDA_HC_FUSED": "1",
                "Q38_CUDA_PLE_FUSED": "1"}, {}),
    "fused": ({"Q38_CUDA_FUSED_PROJECTIONS": "1"}, {}),
    "graph": ({}, {"enable_piecewise_decode_graph": True}),
    "tile128": ({"Q38_CUDA_PREFILL_TILE": "128", "Q38_PLE_IO_TILE": "257"}, {}),
    "transport": ({"Q38_CUDA_PLE_COPY_STREAM": "1", "Q38_CUDA_ASYNC_BOUNDARY": "1"}, {}),
    "prefill": ({"Q38_CUDA_PREFILL_PREWARM": "1", "Q38_CUDA_PREFILL_SYNC_TILES": "4",
                 "Q38_PLE_IO_TILE": "2048", "Q38_CUDA_PLE_COPY_STREAM": "1",
                 "Q38_CUDA_ASYNC_BOUNDARY": "1"}, {}),
    "mtp-fixed": ({}, {"enable_mtp": True}),
    "mtp-adaptive": ({}, {"enable_mtp": True, "enable_adaptive_mtp": True}),
    "mtp-graph": ({"Q38_CUDA_MTP_GRAPH": "1"},
                  {"enable_mtp": True, "enable_piecewise_decode_graph": True}),
    "combined": ({"Q38_CUDA_FUSED_PROJECTIONS": "1", "Q38_CUDA_MTP_GRAPH": "1",
                  "Q38_CUDA_PREFILL_PREWARM": "1", "Q38_CUDA_PREFILL_SYNC_TILES": "4",
                  "Q38_PLE_IO_TILE": "2048", "Q38_CUDA_PLE_COPY_STREAM": "1",
                  "Q38_CUDA_ASYNC_BOUNDARY": "1"},
                 {"enable_mtp": True, "enable_adaptive_mtp": True,
                  "enable_piecewise_decode_graph": True}),
    "compact": ({}, {"enable_compact_sampling": True}),
}


def percentile(values: list[float], q: float) -> float:
    return sorted(values)[min(len(values) - 1, round((len(values) - 1) * q))]


def read_tokens(path: Path, output_tokens: int) -> list[int]:
    value = json.loads(path.read_text())
    tokens = value.get("tokens") if isinstance(value, dict) else value
    if (not isinstance(tokens, list) or not tokens or
        any(type(t) is not int or not 0 <= t < 248320 for t in tokens)):
        raise ValueError("fixture must contain integer model token IDs")
    if output_tokens < 2 or len(tokens) + output_tokens > 262144:
        raise ValueError("prompt plus output exceeds capacity or output is below two")
    return tokens


def measure(client: Client, tokens: list[int], count: int, width: int,
            stop_ids: set[int]) -> dict:
    def call(op: str, **kwargs):
        result = client.call(OPCODES[op], **kwargs)
        if result.status:
            raise RuntimeError(f"{op}: {result.message}")
        return result

    start = time.perf_counter()
    call("append", tokens=tokens)
    append_seconds = time.perf_counter() - start
    seed_start = time.perf_counter()
    seed = call("seed")
    seed_seconds = time.perf_counter() - seed_start
    output = list(seed.tokens)
    latencies: list[float] = []
    per_step: list[int] = []
    final = seed
    start = time.perf_counter()
    while len(output) < count and output[-1] not in stop_ids:
        remaining = count - len(output)
        # Width is bounded by the remaining publication budget, so variants
        # evaluate/publish the same number of tokens even when acceptance varies.
        actual_width = min(width, remaining - 1)
        step = time.perf_counter()
        final = call("spec" if actual_width else "decode", argument0=1,
                     argument1=actual_width)
        latencies.append(time.perf_counter() - step)
        if not final.tokens or len(final.tokens) > remaining:
            raise RuntimeError("invalid publication extent")
        output.extend(final.tokens)
        per_step.append(len(final.tokens))
    seconds = time.perf_counter() - start
    metrics = json.loads(call("stats").message)
    c, t, s0, s1, _, _ = final.frontiers
    if not (c == t + 1 and s0 == s1 == t and c == len(tokens) + len(output)):
        raise RuntimeError("frontier mismatch")
    if metrics["executor"]["failures"] or metrics["executor"]["rollbacks"]:
        raise RuntimeError("executor failed or rolled back")
    measured = len(output) - len(seed.tokens)
    return {
        "append_seconds": append_seconds,
        "prefill_tokens_per_second": len(tokens) / append_seconds,
        "ttft_seconds": append_seconds + seed_seconds,
        "decode_seconds": seconds,
        "measured_decode_tokens": measured,
        "decode_tokens_per_second": measured / seconds if seconds else 0,
        "transaction_latency_ms": {
            key: percentile(latencies, q) * 1000 if latencies else None
            for key, q in (("p50", .5), ("p95", .95), ("p99", .99))
        },
        "sample_count": len(latencies),
        "tokens_per_transaction": per_step,
        "output_tokens": output,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("build/q38-cuda-runtime"))
    parser.add_argument("--tokens-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--mtp-width", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["baseline", "fused", "graph", "prefill", "combined"])
    parser.add_argument("--run", action="store_true", help="start GPU executors; default prints the plan only")
    args = parser.parse_args()
    if not 1 <= args.mtp_width <= 64 or not 1 <= args.chunk <= 4096:
        parser.error("width must be 1..64 and chunk 1..4096")
    tokens = read_tokens(args.tokens_json, args.output_tokens)
    cases = list(dict.fromkeys(["baseline", *args.cases]))
    plan = {"schema": "Q38_OPTIMIZATION_GATE_V1", "cases": {
        name: {"environment": CASES[name][0], "flags": CASES[name][1]} for name in cases},
        "prompt_tokens": len(tokens), "output_tokens": args.output_tokens,
        "fixture_sha256": hashlib.sha256(args.tokens_json.read_bytes()).hexdigest(),
        "chunk": args.chunk, "mtp_width_cap": args.mtp_width, "durability": "off"}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    validated = validate_ready(args.ready)
    sampling = validated["sampling"]
    if sampling["mode"] != "greedy" and any(CASES[name][1].get("enable_mtp") for name in cases):
        parser.error("MTP cases require greedy sampling; use baseline/compact for sampled comparisons")
    if "compact" in cases and not (
        sampling["mode"] == "top-k-top-p" and sampling["top_k"] > 0 and
        sampling["repetition_penalty"] == 1 and sampling["frequency_penalty"] == 0 and
        sampling["presence_penalty"] == 0):
        parser.error("compact case requires sampled top-k with no history penalties")
    args.output.mkdir(parents=True, exist_ok=True)
    plan["runtime_sha256"] = hashlib.sha256(args.runtime.read_bytes()).hexdigest()
    base_env = {k: v for k, v in os.environ.items()
                if not k.startswith("Q38_CUDA_") and k != "Q38_PLE_IO_TILE"}
    results = {}
    for name in cases:
        changes, flags = CASES[name]
        with tempfile.TemporaryDirectory(prefix="q38-gate-") as tmp:
            sock = Path(tmp) / "executor.sock"
            cmd = runtime_command(validated, args.runtime, sock, None, args.chunk, 8, **flags)
            with (args.output / f"{name}.log").open("w") as log:
                process = subprocess.Popen(cmd, env={**base_env, **changes}, stdout=log, stderr=log)
                try:
                    wait_for_executor(sock, validated["session_hash"], process)
                    with Client(str(sock), validated["session_hash"]) as client:
                        result = measure(client, tokens, args.output_tokens,
                                         args.mtp_width if flags.get("enable_mtp") else 0,
                                         set(sampling["stop_token_ids"]))
                finally:
                    terminate(process)
            reference = results.get("baseline", result)["output_tokens"]
            result["token_parity"] = result["output_tokens"] == reference
            results[name] = result
            (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"{name}: parity={result['token_parity']}, "
                  f"prefill={result['prefill_tokens_per_second']:.2f}, "
                  f"decode={result['decode_tokens_per_second']:.2f}", flush=True)
    (args.output / "report.json").write_text(json.dumps({**plan, "results": results}, indent=2) + "\n")
    return 0 if all(v["token_parity"] for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
