#!/usr/bin/env python3
"""Machine-readable direct-token 262,080 + 64 runtime acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from q38_rpc_client import Client, OPCODES  # noqa: E402


def percentile(values: list[int], percent: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = (len(ordered) * percent + 99) // 100
    return ordered[rank - 1]


def load_tokens(path: Path | None, count: int, vocabulary: int) -> list[int]:
    if path is None:
        # Deterministic transport/state stress input. Correctness runs should
        # provide tokenizer-produced marker tokens through --tokens-json.
        return [1 + ((index * 65537 + 17) % (vocabulary - 1)) for index in range(count)]
    parsed = json.loads(path.read_text())
    tokens = parsed["tokens"] if isinstance(parsed, dict) else parsed
    if not isinstance(tokens, list) or len(tokens) != count:
        raise ValueError(f"prompt must contain exactly {count} tokens")
    result = [int(token) for token in tokens]
    if any(token < 0 or token >= vocabulary for token in result):
        raise ValueError("prompt contains token outside vocabulary")
    return result


def host_metadata() -> dict[str, object]:
    result: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "uname": list(platform.uname()),
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
        result["gpus"] = [line for line in query.stdout.splitlines() if line]
    except Exception as error:
        result["gpus_error"] = str(error)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--session-hash", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-json", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=262_080)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--context-limit", type=int, default=262_144)
    parser.add_argument("--vocabulary", type=int, default=248_320)
    parser.add_argument("--mode", choices=("plain", "mtp"), default="plain")
    parser.add_argument("--mtp-width", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=3_600_000)
    arguments = parser.parse_args()
    if arguments.prompt_tokens + arguments.generation_tokens > arguments.context_limit:
        parser.error("prompt + generation exceeds context limit")
    if arguments.mode == "mtp" and not 1 <= arguments.mtp_width <= 64:
        parser.error("--mtp-width must be 1..64")

    prompt = load_tokens(
        arguments.tokens_json, arguments.prompt_tokens, arguments.vocabulary
    )
    prompt_digest = hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little", signed=True) for token in prompt)
    ).hexdigest()
    request_id = time.time_ns()
    generated: list[int] = []
    itl_ns: list[int] = []
    start = time.perf_counter_ns()
    with Client(arguments.socket, arguments.session_hash, request_id) as client:
        append_start = time.perf_counter_ns()
        append = client.call(
            OPCODES["append"], tokens=prompt, timeout_ms=arguments.timeout_ms
        )
        append_ns = time.perf_counter_ns() - append_start
        if append.status != 0:
            raise RuntimeError(f"append failed: {append.as_dict()}")
        seed_start = time.perf_counter_ns()
        seed = client.call(OPCODES["seed"], timeout_ms=arguments.timeout_ms)
        seed_ns = time.perf_counter_ns() - seed_start
        if seed.status != 0 or len(seed.tokens) != 1:
            raise RuntimeError(f"seed failed: {seed.as_dict()}")
        generated.extend(seed.tokens)
        while len(generated) < arguments.generation_tokens:
            step_start = time.perf_counter_ns()
            if arguments.mode == "plain":
                response = client.call(
                    OPCODES["decode"], argument0=1,
                    timeout_ms=arguments.timeout_ms,
                )
            else:
                response = client.call(
                    OPCODES["spec"], argument0=1,
                    argument1=arguments.mtp_width,
                    timeout_ms=arguments.timeout_ms,
                )
            itl_ns.append(time.perf_counter_ns() - step_start)
            if response.status != 0 or not response.tokens:
                raise RuntimeError(f"decode failed: {response.as_dict()}")
            remaining = arguments.generation_tokens - len(generated)
            if len(response.tokens) > remaining:
                raise RuntimeError(
                    "generation overshot the exact context gate; lower MTP width"
                )
            generated.extend(response.tokens)
        metrics_response = client.call(OPCODES["stats"])
        state_response = client.call(OPCODES["state"])
    total_ns = time.perf_counter_ns() - start
    if metrics_response.status != 0 or state_response.status != 0:
        raise RuntimeError("metrics/state query failed after generation")
    expected_canonical = arguments.prompt_tokens + arguments.generation_tokens
    expected_target = expected_canonical - 1
    canonical, target, stage0, stage1, draft, _epoch = state_response.frontiers
    if canonical != expected_canonical:
        raise RuntimeError(
            f"canonical frontier {canonical} != {expected_canonical}"
        )
    if target != expected_target or stage0 != target or stage1 != target:
        raise RuntimeError(
            "target/stage frontiers do not equal the exact final pending-token state"
        )
    if draft > target:
        raise RuntimeError("draft frontier advanced beyond committed target state")
    metrics = json.loads(metrics_response.message)
    if metrics.get("schema") != "q38.metrics.v1":
        raise RuntimeError("executor metrics schema differs from V1")
    executor_metrics = metrics.get("executor", {})
    if (
        executor_metrics.get("failures") != 0
        or executor_metrics.get("rollbacks") != 0
        or executor_metrics.get("cancellations") != 0
        or executor_metrics.get("deadline_exceeded") != 0
        or executor_metrics.get("published_tokens") != arguments.generation_tokens
        or executor_metrics.get("append_transactions") != 1
    ):
        raise RuntimeError("executor metrics violate the strict gate invariants")
    if arguments.mode == "plain":
        expected_decode = arguments.generation_tokens - 1
        expected_evaluated = arguments.prompt_tokens + expected_decode
        if (
            executor_metrics.get("decode_transactions") != expected_decode
            or executor_metrics.get("speculative_transactions") != 0
            or executor_metrics.get("transactions") != arguments.generation_tokens
            or executor_metrics.get("evaluated_tokens") != expected_evaluated
            or executor_metrics.get("state_committed_tokens") != expected_evaluated
        ):
            raise RuntimeError("plain-decode accounting differs from the exact gate")
    output_digest = hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little", signed=True) for token in generated)
    ).hexdigest()
    evidence = {
        "schema": "Q38_STRICT_GATE_EVIDENCE_V1",
        "passed": True,
        "session_hash": arguments.session_hash,
        "mode": arguments.mode,
        "mtp_width": arguments.mtp_width if arguments.mode == "mtp" else 0,
        "prompt_tokens": arguments.prompt_tokens,
        "generation_tokens": len(generated),
        "context_limit": arguments.context_limit,
        "prompt_sha256": prompt_digest,
        "output_sha256": output_digest,
        "first_output_tokens": generated[:16],
        "last_output_tokens": generated[-16:],
        "timing_ns": {
            "append": append_ns,
            "seed": seed_ns,
            "ttft": append_ns + seed_ns,
            "total": total_ns,
            "itl_count": len(itl_ns),
            "itl_mean": int(statistics.mean(itl_ns)) if itl_ns else 0,
            "itl_p50": percentile(itl_ns, 50),
            "itl_p95": percentile(itl_ns, 95),
            "itl_p99": percentile(itl_ns, 99),
        },
        "frontiers": state_response.as_dict()["frontiers"],
        "metrics": metrics,
        "host": host_metadata(),
        "recorded_unix_ns": time.time_ns(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=arguments.output.name + ".", dir=arguments.output.parent
    )
    try:
        with os.fdopen(descriptor, "w") as destination:
            json.dump(evidence, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, arguments.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({"passed": True, "output": str(arguments.output),
                      "output_sha256": output_digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"q38_strict_gate: {error}", file=sys.stderr)
        raise
