#!/usr/bin/env python3
"""Measure plain decode over one persistent ExecutorRPC connection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from q38_rpc_client import Client, OPCODES  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--session-hash", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--append", nargs="*", type=lambda value: int(value, 0))
    parser.add_argument("--append-count", type=int, default=0)
    parser.add_argument("--append-token", type=lambda value: int(value, 0), default=42)
    arguments = parser.parse_args()
    if arguments.steps <= 0:
        parser.error("steps must be positive")
    if arguments.append_count < 0:
        parser.error("append-count cannot be negative")
    if arguments.append and arguments.append_count:
        parser.error("append and append-count are mutually exclusive")

    latencies_ms: list[float] = []
    output_tokens: list[int] = []
    append_seconds = 0.0
    with Client(arguments.socket, arguments.session_hash) as client:
        state = client.call(OPCODES["state"])
        if state.status != 0:
            raise RuntimeError(f"state failed: {state.message}")
        append_tokens = arguments.append
        if arguments.append_count:
            append_tokens = [arguments.append_token] * arguments.append_count
        if append_tokens:
            append_started = time.perf_counter()
            state = client.call(OPCODES["append"], tokens=append_tokens)
            append_seconds = time.perf_counter() - append_started
            if state.status != 0:
                raise RuntimeError(f"append failed: {state.message}")
        started = time.perf_counter()
        final = None
        for _ in range(arguments.steps):
            step_started = time.perf_counter()
            canonical, target = state.frontiers[:2]
            if canonical == target:
                response = client.call(OPCODES["seed"])
            else:
                response = client.call(OPCODES["decode"], argument0=1)
            latencies_ms.append((time.perf_counter() - step_started) * 1000.0)
            if response.status != 0:
                raise RuntimeError(f"decode failed: {response.message}")
            output_tokens.extend(response.tokens)
            final = response
            state = response
        elapsed = time.perf_counter() - started
        stats = client.call(OPCODES["stats"])
        if stats.status != 0:
            raise RuntimeError(f"stats failed: {stats.message}")
        metrics = json.loads(stats.message)
    assert final is not None
    result = {
        "schema": "Q38_DECODE_PROBE_V1",
        "steps": arguments.steps,
        "tokens": len(output_tokens),
        "append_tokens": len(append_tokens or []),
        "append_seconds": append_seconds,
        "wall_seconds": elapsed,
        "tokens_per_second": len(output_tokens) / elapsed,
        "itl_ms": {
            "minimum": min(latencies_ms),
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
            "maximum": max(latencies_ms),
        },
        "frontiers": final.as_dict()["frontiers"],
        "output_tokens": output_tokens,
        "metrics": metrics,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
