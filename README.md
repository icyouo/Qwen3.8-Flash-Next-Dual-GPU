English | [简体中文](README.zh-CN.md)

# Qwen3.8-Flash-Next-Dual-GPU

A purpose-built long-context inference runtime for **Qwen3.8-Flash-Next** on
two 64 GiB CMP 170HX GPUs (SM80). It is designed to fit the complete model on
two Ampere GPUs without GPU P2P, even when each card is limited to PCIe 2.0 x4,
and to extend a live session to a 256K context without recomputing its old
prefix.

This project is not a fork of vLLM, SGLang, or llama.cpp. The model data plane,
offline quantization, dual-GPU scheduling, KV and recurrent state, transactions,
SSD-PLE, artifact format, and CUDA kernels are implemented in this repository.
HTTP, tokenization, chat templates, and SSE form a replaceable control plane;
SGLang is not a runtime dependency.

The public project name is **Qwen3.8-Flash-Next-Dual-GPU**. `q38` remains the
compact internal engineering prefix for binaries, the ExecutorRPC ABI, artifact
schemas, environment variables, and tools. It does not constrain which dual-GPU
configurations may be supported in the future.

> **Project status: research preview.** A reproducible 8K decode baseline has
> been established on real dual-GPU hardware, but tokenizer golden parity,
> 32K/128K/strict-256K validation, MTP, and long-duration stability gates are
> not all complete. Current results are not a model-quality claim or a complete
> 256K release claim. Model weights are not distributed with this repository;
> users must comply with the upstream model license.

```text
OpenAI / deep-session client
            │
            ▼
SGLang or q38_sidecar.py
tokenize / template / SSE / cancel
            │  ExecutorRPC V1 over Unix socket
            ▼
single native process
GPU0 layers 0..24 + SSD-PLE ── BF16 4H host ring ── GPU1 layers 25..47 + LM/MTP
```

The strict product gate uses a 262,080-token prompt plus 64 committed generated
tokens for an exact final length of 262,144. Subsequent requests may only append
an exact suffix; the old prefix is never run through prefill again.

## Current baseline

The following results were measured on Ubuntu host `p3-ultra` with two 64 GiB
CMP 170HX GPUs, driver 610.43.02, CUDA 13.1, `sm_80`, the cut-25 custom mixed
W4/W8 artifact, batch size 1, and plain decode with MTP disabled. The two stages
execute serially for each token. Stage values in the table are their individual
p50 latencies; they do not imply that both GPUs compute the same token in
parallel.

| Run | Context + output | Stage 0 | Stage 1 + head | ITL p50 | Measured decode |
|---|---:|---:|---:|---:|---:|
| Current q38 baseline | 8,195 + 32 | 19.08 ms | 19.21 ms | 43.87 ms | **22.84 tok/s** |
| Current q38 baseline | 8 + 64 | 13.25 ms | 13.10 ms | 31.32 ms | **31.76 tok/s** |
| q38 before optimization | approximately 8K | 40.87 ms | 40.92 ms | 87.03 ms | approximately 11.4 tok/s |
| Historical DS4 reference | approximately 8K | 25.5 ms (0–23) | 26.0 ms (24–47 + head) | approximately 51.5 ms | approximately 19.4 tok/s |

The current 8K stage sum is 38.29 ms, corresponding to a GPU-stage-only ceiling
of approximately 26.1 tok/s; the end-to-end p50 is 43.87 ms. The difference
includes RPC, transaction commit, and the per-successful-write `fdatasync` when
the snapshot journal is enabled. The DS4 row is a historical engineering
reference from the same machine, not a strict apples-to-apples comparison with
the same artifact, quantization, and serving path.

These benchmarks use synthetic tokens to validate the real CUDA state machine,
transport, and performance. They do not establish text quality. Raw build,
fixture, profile, fallback A/B, and end-to-end JSON evidence is described in
[`READINESS.md`](READINESS.md).

## Model and quantization format

The artifact must be built from the official BF16 checkpoint at the pinned
commit:

```text
model          Qwen/Qwen3.8-Flash-Next
source commit  de4b8e4d43b917e7706784d8bb445c9af86a3540
policy         Q38_AMPERE_QUANT_POLICY_V5
stage split    GPU0: layers 0..24; GPU1: layers 25..47 + LM head/MTP
context limit  262,144 tokens
```

This is not NVFP4, AWQ, GPTQ, or GGUF. The compiler converts official
BF16/F32/I64 tensors directly into a stage-owned Q38 artifact. Quantized weights
are not fully dequantized or repacked into another framework format at runtime.

| Tensor class | Artifact format | Runtime precision and notes |
|---|---|---|
| Routed-expert gate/up/down in layers 2–45 | symmetric **W4A16 group-128** | Batch-1-specific W4 GEMV/MoE kernels with BF16 activations |
| Routed experts in layers 0/1/46/47 | symmetric **W8A16 group-128** | Higher precision for quality-sensitive edge experts |
| Always-active matrices, including embeddings, LM head, attention/GDN projections, and shared experts | symmetric **W8A16 group-128** | BF16 activations; one BF16 scale per group |
| MTP matrices and routed experts | symmetric **W8A16 group-128** | MTP is not loaded by default and must be enabled explicitly |
| SSD-PLE embedding table | row-scaled **FP8 E4M3FN** | One BF16 scale per row; the complete table remains on SSD instead of GPU/DRAM |
| Router, norm, gate, hyper-connection, convolution, and other critical control tensors | **preserve** | Preserve the upstream BF16/F32/I64 dtype |
| Main QSA K/V and compressed index | **BF16** | 256K-capacity baseline; INT8 KV is not enabled |
| GDN recurrent and accumulator state | **FP32** | Prevent long-sequence recurrent error accumulation |
| Vision tensors | **skip** | The current runtime is text-only |

The exact matching rules in
[`tools/q38_quant_policy.py`](tools/q38_quant_policy.py) are the single source of
truth. The policy digest is part of the artifact and session identity; the
launcher fails closed when policies differ.

## Implemented contracts

- One semantic writer; append, decode, and MTP share a
  prepare → execute → acknowledge → commit sequence.
- Request-atomic append across chunks, dual-stage rollback,
  cancellation/deadline handling, and request-ID idempotency.
- A stage-0/stage-1 prefill pipeline, a small decode ring, and a three-slot
  large prefill ring.
- BF16 QSA KV, provisional GDN/PLE/MTP state, and stop-token-aware MTP commit.
- A no-P2P pinned D2H/handoff/H2D boundary; GPU0's payload checksum is verified
  again against the pinned-host activation before GPU1 consumes it.
- A streaming and resumable official-BF16-to-Ampere
  W4A16/W8A16/FP8-PLE artifact compiler.
- A 4 KiB registered-buffer `io_uring READ_FIXED + O_DIRECT` PLE path with a
  hard-capacity cache.
- Transactional RNG and penalties plus an append-only cold-rebuild snapshot
  journal.
- `SessionIdentityV1`, `MetricsSchemaV1`, a strict direct-token gate, and an
  HTTP/SSE adapter.

An execution exception in the production CUDA backend causes the executor to
fail closed. Cancellation and deadline expiry still follow the complete
transaction rollback path. This prevents service from continuing with residual
device state after OOM, illegal access, Xid, or transport corruption.

See
[`docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md`](docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md)
for the architecture and performance gates. See
[`READINESS.md`](READINESS.md) for implementation evidence, unproven claims,
and the required order of the first hardware validation runs.

## Ubuntu build and validation

With the current driver, the production CMP 170HX reports as GA100 with compute
capability 8.0. The default target is therefore fixed to `sm_80`. Development
hosts with another CUDA target may override `CUDA_ARCH`:

```sh
make clean
make -j2 build/q38_runtime_tests build/q38-runtime
./build/q38_runtime_tests
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
make -j1 cuda-check cuda-runtime
```

Run the kernel fixtures after GPUs are connected:

```sh
make -j1 cuda-test
make -j1 cuda-bench
```

Batch-1 decode consumes BF16 activations and the production artifact's
group-128 W4/W8 layout directly. The general matrix path uses warp-per-row
GEMV. Routed MoE has dedicated top-10 gate/up, SiLU, and fused down/reduce
paths. The 512-expert router uses parallel bitonic top-k with a stable
tie-break. QSA computes softmax probabilities once in FP32 score scratch and
reuses them across all 256 value dimensions. None of these paths pads a
single-row decode into 16 WMMA rows.

For real-hardware A/B or emergency rollback, set
`Q38_CUDA_DECODE_GEMV=scalar`, `Q38_CUDA_DECODE_MOE=scalar`, or
`Q38_CUDA_DECODE_TOPK=scalar`. Optimized paths are enabled by default.
`cuda-bench` reports optimized-versus-reference results for production GEMV
shapes and routed MoE. Set `Q38_CUDA_PROFILE_DECODE=1` to emit a CUDA-event
decode profile grouped by hyper/GDN/QSA/MoE subpath. No profiling events are
created when the option is unset.

`make verify PYTHON=.venv/bin/python` runs the CPU, Python, and CUDA compilation
gates. All tests and runtime execution must occur on an Ubuntu host; the macOS
workspace is used only for source editing and synchronization.

## 1. Download the pinned official BF16 source

```sh
python3 tools/q38_hf_fetch.py \
  --repo Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --expected-commit de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --output /data/models/Qwen3.8-Flash-Next-official-source \
  --jobs 4 \
  --manifest-only
```

The production commit is pinned to
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. The downloader uses resolved commit
URLs instead of following `main`, `.part` files, LFS size/SHA-256 verification,
and atomic rename. This step publishes only the pinned manifest. The next step
streams each shard through download and conversion so the BF16 source and final
artifact do not fill the host's 468 GB root disk at the same time.

## 2. Build the production artifact

```sh
.venv/bin/python tools/q38_prepare_artifact.py \
  --source /data/models/Qwen3.8-Flash-Next-official-source \
  --metadata /data/models/Qwen3.8-Flash-Next-official-metadata \
  --output /data/models/Qwen3.8-Flash-Next-q38-cut25 \
  --session-hash 0x380025 \
  --cut 25 \
  --jobs 1 \
  --stream-from-manifest \
  --prune-source-shards
```

Each shard follows this sequence: pinned-commit download and LFS verification,
bounded-memory quantization, fragment/segment SHA-256 verification, and deletion
of that source shard. After interruption, the process can resume exactly from
the official tensor index, fetch manifest, and already verified fragments even
when source shards have been deleted. `--prune-source-shards` is an explicit
disk-reclamation switch; omit it when disk capacity is sufficient and retaining
the BF16 source is desired. `READY.json` is published atomically only after the
exact tensor census, PLE layout, and identity for both stages all pass.

Default precision policy:

- Routed experts: symmetric group-128 W4A16.
- Eligible always-active matrices: group-128 W8A16.
- PLE table: row-scaled FP8 E4M3FN.
- Router, norm, critical state, QSA KV, and related tensors remain BF16/FP32 as
  required by the contract.

## 3. Fail-closed launch

```sh
.venv/bin/python tools/q38_launch.py \
  --ready /data/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --runtime build/q38-cuda-runtime \
  --socket /tmp/q38-executor.sock \
  --snapshot /var/lib/q38/session.q38j \
  --host 127.0.0.1 \
  --port 30000
```

Before executing CUDA, the launcher verifies the pinned repository, commit and
cut; top-level digest; every segment SHA-256; the identity checksum and its
eight classes of input digest; and the context, sampling, parser, and stop-token
contracts. Production PLE requires `io_uring + O_DIRECT`. Startup fails when a
direct lane cannot be established; it does not silently fall back to the page
cache.

`GET /v1/q38/metrics` uses the fixed `q38.metrics.v1` ABI. For each stage, it
reports weight arena/uploaded/host-only bytes, QSA/GDN/PLE/MTP state, workspace,
prefill cache, boundary/workspace pinned bytes, and CUDA tracked
current/peak/free/total memory. It also reports executor RSS/peak/anon/file/swap,
faults, context switches, MemAvailable, swap, and cgroup
current/peak/max/OOM. PLE metrics additionally include cache current/capacity,
requested rows, per-batch deduplicated page requests, useful/physical bytes for
direct read-amplification calculation, read count/batch/QD,
io_uring/direct-I/O state, P50/P95/P99 latency, and resident FP8 row-scale
bytes. The raw `cudaMalloc` path has no caching allocator, so
`cuda_allocator_retries` and `cuda_graph_held_bytes` should remain zero in the
current implementation. `cuda_allocation_failures` separately records a
prefill optimization falling back because of insufficient VRAM.

Startup defaults to the design's plain/M1 lane. MTP weights are not uploaded,
and neither the draft QSA cache nor target-HC workspace is allocated. The saved
artifact bytes are reported as `weight_excluded_bytes`. Add `--enable-mtp` only
after the plain correctness and memory gates pass. This loads the official
single-layer MTP. The first release of this checkpoint permits only one draft
token, corresponding to width-2 target verification. Wider requests fail
closed instead of pretending to execute width-3.

Validate without launching:

```sh
.venv/bin/python tools/q38_launch.py --ready /path/READY.json --dry-run
```

## 4. Deep-session / SGLang adapter

SGLang should call the token-native endpoint so its tokenizer, chat template,
and parser can evolve without changing the executor ABI. Create the single live
session:

```sh
curl -sS -X POST http://127.0.0.1:30000/v1/q38/sessions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"deep-1"}'
```

Append an already-tokenized delta and generate:

```sh
curl -sS -N -X POST \
  http://127.0.0.1:30000/v1/q38/sessions/deep-1/execute \
  -H 'Content-Type: application/json' \
  -d '{"append_token_ids":[1,2,3],"max_new_tokens":64,"stream":true}'
```

When `full_token_ids` is supplied, the sidecar verifies that it is an exact
extension of the server's canonical tokens. A fork returns
`cold_rebuild_required` and does not overwrite live state. Responses stream
only token events that the executor has committed. Cancellation, status, and
metrics endpoints are:

```text
POST /v1/q38/cancel
GET  /v1/q38/sessions/{id}
GET  /v1/q38/metrics
```

When `transformers` is installed and the launcher loads the official tokenizer,
`POST /v1/chat/completions` is also available. This compatibility path reports
tokenization and exact-prefix validation latency separately.

## 5. Strict 262K gate

With the executor running:

```sh
.venv/bin/python tools/q38_strict_gate.py \
  --socket /tmp/q38-executor.sock \
  --session-hash 0x380025 \
  --output out/strict-262080-plus-64.json
```

The gate checks more than length. It requires:

- `canonical=262144` and `target=stage0=stage1=262143`.
- Exactly one append, 63 plain decode transactions, and 64 published tokens.
- An exact match between evaluated-token and state-committed-token censuses.
- Zero failures, rollbacks, cancellations, and deadline expirations.
- Prompt/output digests, TTFT/ITL P50/P95/P99, complete versioned metrics, and
  host/GPU metadata in the output.

Synthetic tokens validate only transport and state. The model-correctness gate
must receive tokenizer-generated `--tokens-json` and be combined with a
marker/sentinel reference fixture. Synthetic results must not be used to claim
model quality.

## Release boundary

The CPU/mock strict gate proves protocol, transaction, and 262K capacity paths.
It does not prove real-model performance or numerical correctness. A real
release must still pass 8K, 32K, 128K, 262K+64, near-256K suffix-only, reference
parity, VRAM/host-memory limits, and a 12-hour fault-injection soak on two CMP
170HX GPUs with the final artifact.

## License

Runtime code is released under the [MIT License](LICENSE). Model weights are not
covered by this license. Downloading, converting, and using
Qwen3.8-Flash-Next remains subject to the upstream model repository's license
terms.
