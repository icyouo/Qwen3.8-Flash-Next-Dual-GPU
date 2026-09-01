English | [简体中文](README.zh-CN.md)

# Qwen3.8-Flash-Next-Dual-GPU

A purpose-built, standalone runtime for running **Qwen3.8-Flash-Next** on two
Ampere GPUs. The current hardware target is 2 × 64 GiB CMP 170HX (SM80), with
no CUDA P2P requirement and as little as PCIe 2.0 x4 per card.

The runtime owns the complete inference path: official-checkpoint conversion,
mixed W4/W8 quantization, two-stage execution, QSA/GDN/PLE state, SSD-backed
PLE, CUDA kernels, transactions, recovery, and HTTP/SSE serving. It is not a
fork of vLLM, SGLang, llama.cpp, or DS4, and none of them is required at
runtime. `transformers` is optional and is used only for the official tokenizer
and chat template.

> **Status: research preview.** Real dual-GPU CUDA execution, the custom
> artifact, transactional serving, batch-1 decode, and the grouped-MMQ prefill
> path have been validated. Tokenizer/logit golden parity, 32K/128K/strict-256K
> model runs, MTP release gates, and long-duration fault testing are still in
> progress. The current numbers prove runtime mechanics and speed, not final
> model quality or a production-ready 256K release.

## Results at a glance

All q38 results below were measured on Ubuntu `p3-ultra` with two 64 GiB CMP
170HX GPUs, driver 610.43.02, CUDA 13.1, `sm_80`, the verified cut-25
`Q38_AMPERE_QUANT_POLICY_V5` artifact, batch size 1, and MTP disabled.

### Prefill

The new prefill data plane turns the model's irregular top-10-of-512 MoE work
into deterministic expert-grouped matrix work, then pipelines 512-token slabs
across both GPUs.

| Prefill path | Boundary chunk | 4,096-token wall time | Prefill |
|---|---:|---:|---:|
| Legacy route-wise atomic kernel | 4,096; internal tile 32 | 54.66 s | 74.93 tok/s |
| Grouped MMQ; internal tile 32 | 4,096 | 22.36 s | 183.15 tok/s |
| Grouped MMQ; 512-token slabs; serialized stages | 4,096 | 8.62 s | 475.15 tok/s |
| **Grouped MMQ; 512-token slabs; pipelined stages** | **512** | **5.02 s** | **815.78 tok/s** |
| Historical DS4 reference | n/a | 5.27 s for 4,102 tokens | 777.9 tok/s |

The final row is a historical measurement from the same machine, not a strict
artifact-for-artifact comparison. The optimized q38 result is **10.9×** the
original native prefill baseline and reaches the prior DS4 performance class.
The input was a repeated synthetic token sequence, so this is a real CUDA and
state-machine benchmark, not a text-quality result.

### Decode

Prefill and decode deliberately use different kernel families. After a
pipelined 4K prefill, a 64-token decode probe reached **27.89 tok/s** with
35.01 ms ITL p50.

| Run | Context + output | Stage 0 | Stage 1 + head | ITL p50 | Decode |
|---|---:|---:|---:|---:|---:|
| q38 high-throughput (`durability=off`) | 8,195 + 32 | 18.89 ms | 19.21 ms | 38.65 ms | **25.81 tok/s** |
| q38 strict durability | 8,195 + 32 | 19.08 ms | 19.21 ms | 43.87 ms | **22.84 tok/s** |
| q38 short-context baseline | 8 + 64 | 13.25 ms | 13.10 ms | 31.32 ms | **31.76 tok/s** |
| Native runtime before decode optimization | approximately 8K | 40.87 ms | 40.92 ms | 87.03 ms | approximately 11.4 tok/s |
| Historical DS4 reference | approximately 8K | approximately 25.5 ms | approximately 26.0 ms | approximately 51.5 ms | approximately 19.4 tok/s |

At 8K with durability disabled, GPU stages account for 38.10 ms of the 38.65
ms end-to-end p50. Strict mode is slower because every successful mutating RPC
waits for `fdatasync`. Benchmark results must therefore always name their
durability mode.

#### Long-context exact-QSA R2

The exact parallel selector and tiled attention path remove the earlier decode
collapse without changing the 512-block selection budget or tie semantics.

| Context | Previous decode | Exact-QSA R2 | Speedup | ITL p50, before → R2 |
|---:|---:|---:|---:|---:|
| 32,768 | 20.11 tok/s | **27.84 tok/s** | **1.38×** | 47.88 → 32.79 ms |
| 131,072 | 10.71 tok/s | **26.03 tok/s** | **2.43×** | 91.58 → 36.33 ms |
| 262,080 | 6.21 tok/s | **22.67 tok/s** | **3.65×** | 159.19 → 42.61 ms |

Conditions: two 64 GiB CMP 170HX cards, single concurrency, `durability=off`,
MTP disabled, official-tokenizer source-code corpus, model startup excluded,
one seed token for TTFT followed by exactly five measured GPU decode steps.
Five samples establish a directional throughput baseline, not p95/p99. The
128K and 256K fixtures repeat after 69,579 unique corpus tokens, so they do not
claim worst-case PLE locality or model-quality parity.

Raw evidence, exact commands, known limitations, and the remaining release
gates are tracked in [READINESS.md](READINESS.md).

## Architecture

```text
OpenAI-compatible or token-native client
                    │
                    ▼
       q38_sidecar.py — HTTP / SSE / cancel
       tokenizer and chat template are optional
                    │  ExecutorRPC V1 / Unix socket
                    ▼
          one native process, one semantic writer
                    │
        ┌───────────┴───────────────────────────┐
        │                                       │
 GPU0 / stage 0                         GPU1 / stage 1
 layers 0..24 + PLE                     layers 25..47 + LM head
        │                                       ▲
        └── BF16 4H via pinned-host ring ───────┘
              no NCCL and no P2P required
```

The layer cut is contiguous. A token is evaluated by stage 0 and then stage 1,
so single-sequence decode remains serial across the cards. The boundary payload
is small; PCIe bandwidth was not the decode bottleneck in profiling.

### Dedicated prefill lane

Qwen3.8-Flash-Next routes every token to 10 of 512 experts. The old path
launched work route by route, repeatedly loaded scales, and accumulated results
with FP32 atomics. The optimized lane instead:

1. builds a deterministic expert-major route plan on the GPU;
2. packs assignments by expert and executes direct W4/W8-A16 Tensor Core MMQ;
3. folds router weights into the intermediate activation;
4. writes one FP32 output per assignment; and
5. reduces each token's ten routes in a fixed order without atomics.

The request is divided into 512-token slabs. Three pinned-host boundary buffers
rotate through `free → GPU0 D2H → ready → GPU1 H2D/compute → free`. While GPU1
consumes slab *n*, GPU0 can produce slab *n + 1*; the third slot provides safe
ownership and transfer slack. This is a two-GPU pipeline with three buffers,
not a three-GPU design.

```text
time ─────────────────────────────────────────────────────────────►
GPU0   slab 0     slab 1     slab 2     slab 3     ...
GPU1              slab 0     slab 1     slab 2     ...
ring      A           B          C          A
```

`grouped` is the default prefill path. Set `Q38_CUDA_PREFILL_MOE=legacy` only
for diagnostic fallback, or `Q38_CUDA_PROFILE_PREFILL=1` for CUDA-event
profiling. The grouped path is a new numerical identity; do not reuse a session
or READY identity created for the old arithmetic path.

### Independent decode lane

Batch-1 decode keeps its purpose-built GEMV/MoE/top-k kernels instead of padding
one row into a prefill matrix. QSA reuses one FP32 probability vector across all
value dimensions, and the 512-expert router uses deterministic stable top-k.
The long-context QSA lane is also exact: histories within the 512-block budget
bypass scoring, while larger histories use a grid-parallel four-head score scan,
four byte-wide parallel radix passes, and a stable ascending gather. Attention
score and value work is split into four tiles per head so the 24 heads occupy up
to 96 CTAs instead of concentrating work on 24 SMs. These changes preserve the
existing FP32 reduction order and threshold-tie rule; they are not approximate
nearest-neighbor retrieval.
Diagnostic fallbacks are available through:

```sh
Q38_CUDA_DECODE_GEMV=scalar
Q38_CUDA_DECODE_MOE=scalar
Q38_CUDA_DECODE_TOPK=scalar
Q38_CUDA_PROFILE_DECODE=1
```

## Model artifact and memory placement

The source of truth is the official BF16 checkpoint at a pinned commit:

```text
repository      Qwen/Qwen3.8-Flash-Next
source commit   de4b8e4d43b917e7706784d8bb445c9af86a3540
policy          Q38_AMPERE_QUANT_POLICY_V5
stage split     GPU0: layers 0..24; GPU1: layers 25..47 + LM head/MTP
context target  262,144 tokens
```

This artifact is **not NVFP4, AWQ, GPTQ, or GGUF**. It is compiled directly
from the pinned official checkpoint into a versioned, content-addressed,
stage-owned layout for Ampere.

| Tensor class | Stored format | Runtime use |
|---|---|---|
| Routed experts, layers 2–45 | symmetric W4, group 128 | W4A16 grouped MMQ for prefill; batch-1 W4 kernels for decode |
| Routed experts, layers 0/1/46/47 | symmetric W8, group 128 | W8A16; edge experts retain more precision |
| Embedding, LM head, attention/GDN projections, shared experts, and other always-active matrices | symmetric W8, group 128 | BF16 activations with BF16 group scales |
| MTP matrices and experts | symmetric W8, group 128 | loaded only with `--enable-mtp` |
| PLE embedding table | row-scaled FP8 E4M3FN | approximately 47.68 GiB on SSD, one BF16 scale per row |
| Router, norm, HC, convolution, and other critical controls | preserved BF16/F32/I64 | no blanket quantization |
| Main QSA K/V and compressed index | BF16 | 256K-capacity baseline |
| GDN recurrent and accumulator state | FP32 | protects long-sequence recurrence |
| Vision tensors | skipped | current runtime is text-only |

Only the PLE table is SSD-resident. The default host PLE cache is 8 GiB and is
hard-bounded; the complete table is not copied into RAM or VRAM. Reads use
`io_uring READ_FIXED + O_DIRECT` into registered pinned host buffers, followed
by asynchronous host-to-device transfer. This implementation does **not** use
GPUDirect Storage. Stage-owned weights and active model state remain on their
respective GPUs.

The exact policy in
[tools/q38_quant_policy.py](tools/q38_quant_policy.py) is authoritative. Its
digest, the source commit, tensor hashes, stage cut, runtime hash, and state
layout are part of the artifact/session identity and are checked fail-closed.

## Runtime guarantees

- One semantic writer and one commit order for append, decode, and MTP.
- Request-atomic chunked append with dual-stage acknowledgement and rollback.
- Suffix-only continuation: an existing prefix is never silently recomputed or
  replaced; a fork requires a cold rebuild.
- Provisional QSA/GDN/PLE/MTP/RNG state is published only after commit.
- No-P2P pinned-host transport with position and payload-integrity checks.
- Cancellation, deadline, request-ID idempotency, and stop-token-aware commit.
- Fatal CUDA/device/transport errors invalidate the executor after rollback is
  attempted; execution never continues from uncertain device state.
- Streaming, resumable artifact construction with per-source and per-output
  hashes and atomic READY publication.
- Explicit durability policy: `strict` for crash-rebuild state, `off` for
  replayable benchmark/high-throughput workloads.

The strict capacity target is a 262,080-token prompt plus 64 committed output
tokens, for an exact final length of 262,144. This target is a release gate, not
a completed claim.

## Build and test on Ubuntu

All CUDA builds and executions are expected to run on Ubuntu. macOS can be used
as an editing workspace, but it is not a supported runtime host.

```sh
make clean
make -j2 build/q38_runtime_tests build/q38-runtime
./build/q38_runtime_tests
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
make -j1 cuda-check cuda-runtime cuda-test cuda-bench
```

`make verify PYTHON=.venv/bin/python` runs the CPU, Python, CUDA compile, and
runtime build gates. `cuda-test` requires the GPUs. The optimized grouped-MMQ
kernels have passed the real-SM80 fixtures; `compute-sanitizer` is not currently
installed on the validation host and must not be treated as a passed gate.

## Prepare the model

Model weights are not included in this repository.

### 1. Pin the official source manifest

```sh
python3 tools/q38_hf_fetch.py \
  --repo Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --expected-commit de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --output /data/models/Qwen3.8-Flash-Next-official-source \
  --jobs 4 \
  --manifest-only
```

### 2. Stream and compile the Q38 artifact

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

Each source shard is verified, converted with bounded memory, rehashed, and
published before it may be deleted. The process is resumable. Omit
`--prune-source-shards` if the host has enough capacity to retain the official
BF16 source. `READY.json` appears only after both stages, the PLE layout, tensor
census, and identity have passed validation.

## Launch the service

Strict durability is the default:

```sh
.venv/bin/python tools/q38_launch.py \
  --ready /data/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --runtime build/q38-cuda-runtime \
  --socket /tmp/q38-executor.sock \
  --snapshot /var/lib/q38/session.q38j \
  --durability strict \
  --host 127.0.0.1 \
  --port 30000
```

The launcher verifies the source commit, quantization policy, all artifact
segments, runtime identity, stage plan, state layout, and context/sampling
contracts before CUDA execution. Production PLE startup requires a working
`io_uring + O_DIRECT` lane and will not silently fall back to buffered I/O.

For a benchmark or a client that can replay the complete canonical token
history, disable only crash recovery:

```sh
.venv/bin/python tools/q38_launch.py \
  --ready /data/models/Qwen3.8-Flash-Next-q38-cut25/READY.json \
  --runtime build/q38-cuda-runtime \
  --durability off
```

`durability=off` removes the snapshot journal and its crash-rebuild guarantee;
it does not weaken in-process transaction, rollback, or committed-token
semantics. Use `--dry-run` to validate an artifact without launching. MTP is
off by default and should be enabled with `--enable-mtp` only after the plain
lane passes its correctness and memory gates.

## API

Create the single live session:

```sh
curl -sS -X POST http://127.0.0.1:30000/v1/q38/sessions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"deep-1"}'
```

Append token IDs and stream committed output:

```sh
curl -sS -N -X POST \
  http://127.0.0.1:30000/v1/q38/sessions/deep-1/execute \
  -H 'Content-Type: application/json' \
  -d '{"append_token_ids":[1,2,3],"max_new_tokens":64,"stream":true}'
```

`full_token_ids`, when supplied, must be an exact extension of the canonical
server prefix. Available operational endpoints include:

```text
POST /v1/q38/cancel
GET  /v1/q38/sessions/{id}
GET  /v1/q38/metrics
```

When the official tokenizer is configured, the sidecar also exposes
`POST /v1/chat/completions`. The codec does not enter the ExecutorRPC ABI or own
model state.

## Validation and roadmap

The immediate release sequence is:

1. publish the grouped-MMQ numerical identity and start from cold state;
2. freeze tokenizer-produced golden prompts and validate logits/tokens against
   a trusted official-BF16 reference;
3. run fresh 32K, 128K, and strict 262,080 + 64 gates with complete GPU, host,
   transport, and PLE metrics;
4. prove near-256K suffix continuation without prefix replay;
5. validate rollback, cancellation, duplicate requests, crash recovery, and
   fault injection at every context level;
6. validate opt-in MTP only after the plain lane passes; and
7. complete the long-duration stability and failure-injection soak.

For the full implementation/evidence boundary, see [READINESS.md](READINESS.md).
For the detailed design and release gates, see
[the runtime architecture document](docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md).

## License

Runtime code is released under the [MIT License](LICENSE). Model weights are
not covered by this license; downloading, converting, and using the model is
subject to the upstream repository's license terms.
