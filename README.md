English | [简体中文](README.zh-CN.md)

# Qwen3.8-Flash-Next-Dual-GPU

<details>
<summary>Optimization history (September–October 2026)</summary>

> **2026-10-12 GDN cache update:** The opt-in `mtp4-native-bf16-gdn-qk-candidate` precomputes normalized Q/K into FP32 storage while preserving the existing BF16 results and recurrent accumulation order, adding about 16 MiB/GPU. Two reversed-order rounds, six real 4K requests per binary, measure **3.071→3.035s TTFT (−1.19%)**, with identical complete 409-token answers; an installed three-request confirmation remains **1.62% faster**. The gap to the previous-round 2.771s vLLM reference is about **9.53%**; vLLM was not rerun this round. See [implementation, checks, numerical limits, launch and rollback](docs/gdn-normalized-prefill-20261012.md). Earlier BF16/FP24 binaries and profiles are retained. The 27B service and mining remain stopped; the two ComfyUI services were restored after testing.

> **2026-10-12 prefill repeat:** With background GPU workloads stopped, six cold-prefix 4K requests per engine measure **3.071s** for the installed BF16 runtime and **2.771s** for vLLM: a **300ms / 10.84% TTFT gap**. Two shared-expert overlap variants show no stable gain. The actual 1024-token slab passes same-BF16 full-logit checks but is **2.83% slower** in reversed-order A/B, so none of these candidates was installed. The 512-token BF16 runtime is retained. See [experiments, scope windows and evidence](docs/prefill-optimization-20261012.md).

> **2026-10-12 BF16 update:** The explicit `mtp4-native-bf16-candidate` now uses **16-bit BF16** MoE communication for prefill, decode, MTP and Graph execution. Two reversed-order A/B rounds, six 4K requests per version, measure **3.420→3.067s TTFT (−10.32%)** and **1198→1335 input tokens/TTFT (+11.50%)**, leaving a **10.46% gap** to the retained earlier 2.777s vLLM reference. Answers change to 409 tokens; quality, recovery and repeated full-logit checks pass, while the existing scalar/MTP full-logit difference remains. See [measurements, numerical limits, launch and rollback](docs/bf16-transport-20261012.md). FP24 is retained; no resident inference service was started, and the 27B service and mining are stopped.

> **2026-10-11 stability incident:** A later isolated MoE prefill test coincided with a Z4 whole-host reboot. That MoE candidate remains unqualified and was not installed; the earlier FP24 binary is unchanged. See [incident evidence](docs/moe-prefill-incident-20261011.md).

> **2026-10-11 exact-transport update:** The opt-in `mtp4-native-prefill-fp24-candidate` preserves exact FP32-sum/BF16 results with bounded 24-bit transport and full-FP32 escape reads. Two reversed-order A/B rounds, six requests per version, measure **3.506→3.346s TTFT** and **1168→1224 tok/s prefill (+4.77%)**, with identical complete 457-token answers and full logits/quality/recovery gates. The retained same-day vLLM reference is **2.777s**, leaving a **20.51% TTFT gap**. See [measurements, qualification and launch](docs/prefill-exact-transport-20261011.md).

> **2026-10-11 RSAG update:** The opt-in `mtp4-native-prefill-rsag-candidate` preserves FP32 summation and BF16 rounding while reducing MoE P2P traffic by 25%. Same-input three-repeat HTTP medians improve native TTFT **3.95→3.45s** and input/TTFT throughput **1037→1188 tok/s (+14.57%)**; complete 457-token answers, BF16/quality and recovery gates match the qualified baseline. vLLM measures **2.78s** and remains ahead. See [GPU scope distributions, evidence and launch](docs/prefill-distribution-20261011.md). Z4 NAS is now `/mnt/nas` with legacy compatibility; the 27B service stays stopped.

> **2026-10-11 prefill update:** The opt-in `mtp4-native-prefill-candidate` uses bulk P2P copies for large eager reductions. Three-repeat frozen 4K medians improve prefill **864.58→990.04 tok/s (+14.51%)** and TTFT **4.75→4.14s**; decode remains about **85 tok/s**, with identical 457-token answers. Full-model BF16/token, quality and recovery gates pass. vLLM's historical 1485.51 tok/s prefill is still ahead. See [measurements, evidence and launch](docs/native-vllm-parity-20261011.md).

> **2026-10-04 long-input update:** `mtp4-native-longinput` improves the frozen 4096-token request from **73.73 to 83.14 tok/s** in a paired run; final installed confirmation is **83.59 tok/s**, with the same 457 output tokens/text. Incremental tokenizer decoding and a bounded, request-owned SSE producer are included. Short web decode remains around **177–179 tok/s**; 4K TTFT remains about **4.7 seconds**. Experimental pending-terminal prefix replacement stays disabled because full-rebuild BF16 parity failed. See [validation and launch instructions](docs/native-longinput-optimization-20261004.md).

> **2026-10-04 llm_speedtest check:** Exact-usage random essay requests measured **96.00–98.31 tok/s at 128 input tokens** and **80.09 tok/s at 4K**. Two host reboots interrupted the repeat matrix; the earlier web/code rates are workload-specific. Streaming include_usage is now supported. See [measurements and reboot evidence](docs/llm-speedtest-20261004.md).


> **2026-10-04 native fixed-MTP4 fast profile:** `mtp4-native-fast` reaches official-launcher three-run mean decode rates of **172.81 / 174.70 tok/s** for web/code on dual 170HX at 200W per GPU, using the existing W4/W8 weights and RAM PLE. Six follow-up experiments retain shared-expert/P2P overlap, direct BF16 expert reduction, MoE/HC finish fusion and an MTP input cache (about 1.18 GiB extra per GPU). Full-model parity, answer-quality and cancellation-recovery gates pass. See [individual measurements, policy and rollback](docs/native-followup-optimization-20261004.md) and the [preceding 162 tok/s report](docs/native-fast-optimization-20261004.md).

> **2026-10-04 fixed MTP4 register update:** `mtp4-register-candidate` reaches 81.15/81.74 tok/s for web/code with exact operator and full-model parity checks. This register candidate did not meet 160 tok/s; the fast profile above is the later implementation. See [measurements and validation](docs/native-register-optimization-20261004.md).

> **2026-10-04 fixed MTP4 routing update:** `--execution-profile mtp4-route-candidate` enables exact parallel route planning and omits unused plan diagnostic hashes. Paired HTTP measurements reached 74.34/74.50 tok/s for web/code and 47.70 tok/s at 8K input, a 2.6–3.3% improvement over the existing integrated profile. See [validation, measurements and launch command](docs/native-route-optimization-20261004.md).

> **2026-09-21 native TP optimization plan:** The staged program covers measurement, persistent rank workers, stream-ordered P2P, multi-row MoE/target execution, TP graphs, exact speculative state, and shape-specific kernels. See the [complete plan and implementation status](docs/native-tp-optimization-plan-20260921.md). Planned capabilities are not enabled merely by selecting TP.

> **2026-09-20 Z4 TP2 update:** An opt-in eager TP2/EP runtime now serves HTTP/SSE on dual CMP 170HX. Producer-stream synchronization, reductions, multi-token slicing, and rank coordination were repaired. The native 256-step probe measured **32.43 tok/s**; TP has not surpassed the earlier PP baseline. PP remains the default. See [implementation, evidence, and launch instructions](docs/tp2-host-bounce-20260920.md).

> **P2P and PLE storage update:** The TP launcher now accepts `--tp-transport host|p2p`, and both launchers accept `--ple-storage ssd|ram`. RAM mode preloads one byte-preserving PLE table shared by both TP ranks. See [measured comparisons and commands](docs/p2p-ram-20260920.md) and the [updated SM80 reference assessment](docs/reference-sm80-170hx-20260920.md).

> **Fixed MTP4 integration:** `--execution-profile mtp4-candidate` selects the native W4/W8 graph configuration with GPU P2P ordering, device stage boundaries and fixed MTP4 HTTP defaults. Both `--ple-storage ram` and `ssd` remain available. See [implementation, validation and measurements](docs/native-mtp4-implementation-20260921.md).

> **2026-09-06 local changes:** Additional prefill/decode optimizations are implemented but await GPU validation. Historical results below do not measure this source revision. See [implementation and validation plan](docs/optimization-20260906.md).

</details>

A purpose-built, standalone runtime for running **Qwen3.8-Flash-Next** on two
Ampere GPUs. The current hardware target is 2 × 64 GiB CMP 170HX (SM80).
**Bidirectional CUDA P2P is verified and the latest optimized profile selects
asynchronous P2P.** Host transport remains available for hardware without P2P.
The measured PCIe 2.0 ×4 link sustains about 1.40 GB/s per direction under
bidirectional load; see the [P2P measurements](docs/p2p-diagnostic-20261011.md).

The runtime owns the complete inference path: official-checkpoint conversion,
mixed W4/W8 quantization, two-stage execution, QSA/GDN/PLE state, SSD-backed
PLE, CUDA kernels, transactions, recovery, and HTTP/SSE serving. It is not a
fork of vLLM, SGLang, llama.cpp, or DS4, and none of them is required at
runtime. `transformers` is optional and is used only for the official tokenizer
and chat template.

> **Status: research preview.** Real dual-GPU CUDA execution, the custom
> artifact, transactional serving, batch-1 decode, and the grouped-MMQ prefill
> path have been validated. The opt-in width-one MTP pipeline has passed
> directional 32K/128K/256K performance and state-consistency probes. Bounded
> width-N retained drafting supports 1..64 drafts with an initial operational
> cap of four; its online-prefix width-4 path has passed an initial 32K
> real-GPU token-parity and mixed accept/reject performance probe. Widths 2/3
> and the broader width-4 matrix still require validation. Full
> tokenizer/logit golden parity, strict 262,080 + 64 runs, broad MTP
> accept/reject quality gates, and long-duration fault testing are still in
> progress. The current numbers prove runtime mechanics and speed, not final
> model quality or a production-ready 256K release.

## Results at a glance

### Latest real-text measurements (2026-10-12)

HP Z4, dual 64 GiB CMP 170HX (SM80), 200W/GPU, CUDA 13.3; frozen 4096-token
real-text input, one request at a time, greedy/no-thinking, cold prefix, MTP4,
chunk512, and RAM PLE with a 64GiB budget. Native uses W4/W8 weights,
asynchronous P2P and BF16 MoE communication, with profiling disabled.
TTFT includes HTTP, tokenization, reset, prefill and the first token;
**it is not pure kernel prefill time**.

| Metric | Native BF16 baseline | Latest native BF16 + GDN Q/K cache | Previous-round vLLM reference |
|---|---:|---:|---:|
| Measured requests | 6 | 6 | 6 |
| Input tokens | 4096 | 4096 | 4096 |
| Median TTFT | 3.071s | **3.035s** | 2.771s |
| Input tokens / TTFT | 1333.66 tok/s | **1349.67 tok/s** | 1478.28 tok/s |
| Decode (N−1) | 92.90 tok/s | **92.86 tok/s** | No equal-answer comparison |
| Output tokens | 409, complete | 409, complete | 512, capped |

The current opt-in profile is `mtp4-native-bf16-gdn-qk-candidate`, using
`build/native-bf16-gdn-qk/q38-cuda-runtime`. Two reversed-order A/B rounds
improve TTFT by **1.19%**, with identical answers. A separate post-install
confirmation improves 1.62% and is not pooled into this table. Latest native
TTFT remains **264.02ms / 9.53%** above the previous-round vLLM reference.
vLLM was not rerun in the GDN round and answers differ across engines;
this is not a new synchronized cross-engine A/B or equal-answer decode test.

See the [GDN report](docs/gdn-normalized-prefill-20261012.md) for complete
measurements, numerical limits, and the current launch/rollback commands;
the [previous report](docs/prefill-optimization-20261012.md) contains the vLLM
reference. Same-path full BF16 logits, answers and recovery checks pass;
scalar/MTP logits differences and long-term stability remain open. Historical
short web/code rates around 173–179 tok/s do not represent this 4K workload.

<details>
<summary>Historical synthetic, ordinary-mode and long-context measurements (different builds and methods)</summary>


The historical q38 results below were measured on Ubuntu `p3-ultra` with two 64 GiB CMP
170HX GPUs, driver 610.43.02, CUDA 13.1, `sm_80`, the verified cut-25
`Q38_AMPERE_QUANT_POLICY_V5` artifact, and batch size 1. MTP is disabled unless
the table explicitly labels an MTP run.

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
| Historical: resident GDN state + tiled QSA reuse | **512** | **2.74 s** | **1,497.04 tok/s** |
| Historical DS4 reference | n/a | 5.27 s for 4,102 tokens | 777.9 tok/s |

The final row is a historical measurement from the same machine, not a strict
artifact-for-artifact comparison. That historical q38 result is **20.0×** the
original native prefill baseline and exceeds the prior DS4 performance class.
The input was a repeated synthetic token sequence, so this is a real CUDA and
state-machine benchmark, not a text-quality result.

That historical ordinary-mode checkpoint (MTP disabled) measured prefill and decode
together with the same runtime binary. Profiling was disabled for every row;
decode is five real GPU steps immediately after the listed cold prefill.

| Actual context | Append time | Prefill | Decode | ITL p50 |
|---:|---:|---:|---:|---:|
| 4,096 | 2.736 s | **1,497.04 tok/s** | **29.75 tok/s** | 31.60 ms |
| 32,768 | 19.313 s | **1,696.65 tok/s** | **29.65 tok/s** | 31.74 ms |
| 262,080 | 174.363 s | **1,503.07 tok/s** | **26.82 tok/s** | 34.24 ms |

Relative to the preceding published prefill checkpoint, the three rows
improved by 12.8%, 17.0%, and 13.3%, respectively. A separate detailed-profile
32K gate reached 1,668.40--1,668.72 tok/s while collecting per-kernel CUDA
events. Decode kernels were not changed by this prefill revision.

The same build also passed a five-turn coding-style continuation probe. The
first turn was a cold 4K prefill; each later turn reused the live recurrent and
QSA state and appended a fresh 1,024-token suffix. The extra evaluated token is
the previously generated continuation at the session frontier.

| Turn | New user tokens | Newly evaluated | Prefill / suffix | Decode |
|---:|---:|---:|---:|---:|
| 0, cold | 4,096 | 4,096 | **1,497.04 tok/s** | 29.75 tok/s |
| 1, incremental | 1,024 | 1,025 | **1,182.22 tok/s** | 30.69 tok/s |
| 2, incremental | 1,024 | 1,025 | **1,167.95 tok/s** | 30.84 tok/s |
| 3, incremental | 1,024 | 1,025 | **1,166.96 tok/s** | 30.56 tok/s |
| 4, incremental | 1,024 | 1,025 | **1,159.85 tok/s** | 30.70 tok/s |

Conditions: single concurrency, `durability=off`, MTP disabled, and five
measured decode steps per turn. These are live-session incremental hits, not
radix-cache or prompt-cache simulations.

### Decode

Prefill and decode deliberately use different kernel families. The historical
same-build, same-run figures are in the combined table above. The table below
is retained as a historical operating-mode comparison from the earlier decode
optimization milestone; it is not the current release checkpoint.

| Historical run | Context + output | Stage 0 | Stage 1 + head | ITL p50 | Decode |
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

#### Historical long-context exact-QSA optimization

The exact parallel selector and tiled attention path removed the earlier decode
collapse without changing the 512-block selection budget or tie semantics.
These rows record that optimization milestone; use the combined table above
for that historical binary's measurements.

| Context | Previous decode | Exact-QSA R2 | Speedup | ITL p50, before → R2 |
|---:|---:|---:|---:|---:|
| 32,768 | 20.11 tok/s | **27.84 tok/s** | **1.38×** | 47.88 → 32.79 ms |
| 131,072 | 10.71 tok/s | **26.03 tok/s** | **2.43×** | 91.58 → 36.33 ms |
| 262,080 (current ballot gather) | 6.21 tok/s | **27.79 tok/s** | **4.47×** | 159.19 → 33.91 ms |

Conditions: two 64 GiB CMP 170HX cards, single concurrency, `durability=off`,
MTP disabled, official-tokenizer source-code corpus, model startup excluded,
one seed token for TTFT followed by exactly five measured GPU decode steps.
Five samples establish a directional throughput baseline, not p95/p99. The
128K and 256K fixtures repeat after 69,579 unique corpus tokens, so they do not
claim worst-case PLE locality or model-quality parity.

#### Experimental piecewise CUDA Graph decode

`--enable-piecewise-decode-graph` enables a graph lane for ordinary one-token
decode and one-token speculative target verification. Each stage captures the
fixed-shape layer body as seven static fragments and runs the six position-dependent exact-QSA
layers eagerly between them. Dynamic input/output endpoints, PLE reads and H2D staging, inter-stage transport,
sampling, cancellation, and transaction commit/rollback stay outside capture.

CUDA graph nodes retain raw state pointers, while GDN and PLE transactions
alternate between two working banks. The runtime therefore builds at most one
lazy graph variant per bank pair and never patches a graph onto another bank.
Capture failure disables this lane for that stage and falls back to the existing
eager decode path. The feature is off by default and has no published speedup
claim until real-GPU token/logit parity, rollback, memory, and latency A/B gates
pass. The `cuda_graph_captures`, `cuda_graph_replays`,
`cuda_graph_fallbacks`, and `cuda_graph_nodes` metrics expose its behavior.

For strict eager/graph parity, launch the executor under test with
`--enable-logit-diagnostics`. This diagnostic-only flag retains the raw BF16
model-head output from the last committed append/decode transaction; it is
default-off and therefore adds no logits D2H copy to ordinary greedy serving.
Capture the identical committed step from separate eager and graph runs, then
compare every BF16 bit:

```bash
python3 tools/q38_logit_parity.py capture \
  --socket /tmp/q38-eager.sock --session-hash 368 --output eager-step
python3 tools/q38_logit_parity.py capture \
  --socket /tmp/q38-graph.sock --session-hash 368 --output graph-step
python3 tools/q38_logit_parity.py compare eager-step.json graph-step.json \
  --output parity.json
```

The capture writes a compact JSON manifest plus an exact little-endian `.bf16`
payload protected by both the runtime FNV-1a checksum and a tool-side SHA-256.
Comparison fails unless session/epoch/frontier/kind match, selected tokens match,
and every raw BF16 element is identical. Do not enable this flag for latency
benchmarks.

#### Measured width-one MTP baseline

The first measured MTP baseline retains one draft QSA row, verifies two target
rows as one-token microbatches, and pipelines stage 0 of the next row against
stage 1 of the current row. The runtime has since generalized this lane to a
bounded width N, but the table below remains width-one evidence rather than a
projected wider result.

| Context | Plain decode baseline | MTP width 1 | MTP effective ITL | Gain over plain |
|---:|---:|---:|---:|---:|
| 32,768 | 28.84 tok/s | **40.44 tok/s** | 24.73 ms | **40.2%** |
| 131,072 | 26.05 tok/s | **35.05 tok/s** | 28.53 ms | **34.5%** |
| 262,080 | 22.61 tok/s | **30.62 tok/s** | 32.66 ms | **35.4%** |

Conditions match the exact-QSA probe above: single concurrency,
`durability=off`, one seed token, and five measured transactions. All 5/5 draft
tokens happened to be accepted in each fixture. That small deterministic sample
validates the retained-row fast path and its throughput; it is not a general
acceptance-rate or model-quality claim. The plain figures were measured for the
preceding retained-row revision, where they stayed within 0.3% of the prior
baseline at 128K and 256K (and were 3.6% faster at 32K). The subsequent overlap
revision changes only the non-null retained-draft request branch; runtime tests
verify that ordinary decode never enters it. A redundant full 256K plain rerun
was intentionally skipped.

Raw evidence, exact commands, known limitations, and the remaining release
gates are tracked in [READINESS.md](READINESS.md).

#### Initial online-prefix width-four result

The wider lane now verifies target rows online and stops at the first mismatch.
GPU0 keeps at most one lookahead row, protected by one optional GDN/PLE
checkpoint, while GPU1 reconciles each accepted deeper MTP row against real
target HC. Commit therefore publishes the already-verified prefix instead of
verifying the rejected suffix and replaying accepted target/MTP rows.

On the frozen 32,768-token coding fixture, five width-4 transactions published
`5, 4, 4, 2, 5` tokens (excluding the seed), covering two all-accepted cases
and three early-stop depths. The run accepted 15/20 drafts and reached
**48.65 tok/s**; total backend commit time was **1.19 ms**, versus **124.17 ms**
in the previous retained-draft trace. All 21 published tokens exactly matched
the ordinary target decode prefix, and the run reported zero failures and zero
rollbacks. The simultaneously measured plain reference was 28.67 tok/s.

Acceptance differed from the older trace, so this is a directional end-to-end
result rather than a fixed-work kernel A/B. It validates the repaired control
path at 32K; it does not replace the remaining 128K/256K, logit-golden,
cancellation, long-generation, or width-2/3 comparison gates.

</details>

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

The recurrent GDN prefill kernel assigns each CTA one 128x32 state-column tile.
Every thread keeps four FP32 state cells in registers for the entire 512-token
slab, loads them once, reuses each normalized key for both recurrent phases,
and writes the cells back once. The last-slab GDN recurrent time fell from
69.47 to 34.95 ms on stage 0 and from 62.35 to 30.78 ms on stage 1.

Grouped QSA uses a 32-position score key tile to reduce score-CTA shared memory
from about 38 KiB to 22 KiB. Its value kernel stages 64 selected positions and
all twelve shared-query-head scores so 256 value-dimension threads reuse them;
last-slab `qsa_value` fell from 22.17 to about 13.15 ms. Both changes retain the
existing FP32 operation order. A full-vocabulary parity gate compared all
248,320 BF16 logits against the preceding path with zero mismatches and the
same selected token.

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

### Retained-draft MTP lane

For width N, GPU1 recurrently runs the checkpoint's multi-step-trained MTP
layer while GPU0 starts target row 0. The N+1 target rows then travel through
the existing scheduler as one-token chunks, allowing GPU0 row `n+1` to overlap
GPU1 row `n`. Verification stops at the first mismatch. Before each lookahead,
GPU0 checkpoints the current GDN/PLE prefix; an early stop restores that single
checkpoint while QSA commits its shorter append-only extent. The first retained
MTP QSA row is based on canonical target HC and can be committed directly.
Later draft rows used MTP-predicted HC, so every accepted deeper row is
reconciled online from the corresponding real target HC while GPU0 computes its
lookahead. Commit performs no serial prefix repair. Rejection, stop tokens,
cancellation, rollback, and context-tail capping remain request-atomic. MTP is
opt-in, and the plain scheduler takes none of this retained-state path.

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

In the default `--ple-storage ssd` mode, the PLE table stays on SSD. The host
PLE cache is bounded to 8 GiB per store (one store per TP rank). Reads use
`io_uring READ_FIXED + O_DIRECT` into registered pinned host buffers, followed
by asynchronous host-to-device transfer. This implementation does **not** use
GPUDirect Storage. Stage-owned weights and active model state remain on their
respective GPUs.

`--ple-storage ram --ple-ram-budget-gib 64` instead preloads all PLE payload and
BF16 row scales into owned host memory (approximately 48.28 GiB for this artifact).
The two TP ranks share one immutable table; their transaction state stays separate.
The budget includes payload and scales. Preload fails if the budget or available
host memory is insufficient, and there is no silent SSD fallback. RAM lookups do
not read the source files after preload. The allocation is not locked against OS
swap. RAM mode ignores the SSD page-cache capacity and does not pin the full table.

The exact policy in
[tools/q38_quant_policy.py](tools/q38_quant_policy.py) is authoritative. Its
digest, the source commit, tensor hashes, stage cut, runtime hash, and state
layout are part of the artifact/session identity and are checked fail-closed.

## Runtime guarantees

- One semantic writer and one commit order for append, decode, and MTP.
- Request-atomic chunked append with dual-stage acknowledgement and rollback.
- Exact-prefix continuation reuses the resident GPU state and evaluates only
  the suffix. A new conversation or history fork performs an atomic cold
  rebuild of mutable session state while retaining weights and non-semantic
  PLE/matrix caches.
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
contracts before CUDA execution. Default SSD PLE startup requires a working
`io_uring + O_DIRECT` lane and will not silently fall back to buffered I/O.
Explicit RAM mode uses bounded full-table preload and does not require this I/O lane.

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
off by default and should be enabled with `--enable-mtp --mtp-max-draft 4`
only after the plain lane passes its correctness and memory gates. Individual
token-native requests use `{"mode":"mtp","mtp_width":4}`; OpenAI-compatible
requests use `{"q38_mode":"mtp","q38_mtp_width":4}`. The configured maximum
is a capability ceiling, while each request still chooses its width. Piecewise
CUDA Graph decode is
also experimental and opt-in through `--enable-piecewise-decode-graph`; do not
combine it with benchmark claims until the graph-specific gates in
[READINESS.md](READINESS.md) pass.

## API

The current runtime has one GPU-resident session slot. Create its initial
logical session:

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

With `full_token_ids`, an exact extension of the resident canonical prefix is
an incremental cache hit. A non-extension is treated as a new conversation or
fork: the executor atomically resets QSA/GDN/PLE/MTP/sampler state and cold
prefills the supplied history instead of returning HTTP 409. Immutable weights,
the bounded PLE host cache, and the CUDA prefill matrix cache remain resident.
Responses report `q38.cache_status` as `cold_start`, `incremental_hit`, or
`cold_rebuild`, plus `cold_rebuild` and `reset_ns`. Passing a different
`session_id` to `/v1/chat/completions` explicitly evicts the resident logical
session under the same rule.

This is correct single-slot session/cache management, not multi-session KV
residency. Returning to an evicted conversation requires its client to replay
full history and cold-prefill it. Multi-slot KV state, LRU state swap, and a
radix prefix cache are future extensions; the API does not claim those hits
today. `GET /v1/q38/metrics` includes the session-cache policy and hit/rebuild
counters. Available operational endpoints include:

```text
POST /v1/q38/cancel
GET  /v1/q38/sessions/{id}
GET  /v1/q38/metrics
```

When the official tokenizer is configured, the sidecar also exposes
`GET /v1/models`, `GET /v1/models/{id}`, and
`POST /v1/chat/completions`. The codec does not enter the ExecutorRPC ABI or
own model state. Chat completions accept OpenAI function `tools`, `tool_choice`,
assistant `tool_calls`, and `role: tool` follow-ups. Qwen's official XML tool
syntax is parsed into OpenAI `tool_calls`; tool-enabled streaming buffers the
candidate until the complete call is structurally valid, then emits
`delta.tool_calls`.

Output length accepts `max_completion_tokens`, `max_tokens`,
`max_new_tokens`, and `max_output_tokens` in that precedence order. If none is
provided, the budget is the full model context remaining after the prompt;
EOS, stop tokens, and tool completion may still end generation earlier. An
explicit request larger than the remaining context returns HTTP 422. Optional
`--default-max-tokens` and `--max-output-tokens` server flags can impose a
smaller default or hard cap; zero (the default) means no limit below the model
context.

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
6. complete width 2/3 comparisons and the remaining width-4 accept-depth,
   cancellation, token/logit parity, long-generation, 128K/256K, and throughput
   gates before selecting the production width; and
7. complete the long-duration stability and failure-injection soak.

For the full implementation/evidence boundary, see [READINESS.md](READINESS.md).
For the detailed design and release gates, see
[the runtime architecture document](docs/qwen38-p3-ultra-dual-170hx-256k-optimal-runtime.md).

## License

Runtime code is released under the [MIT License](LICENSE). Model weights are
not covered by this license; downloading, converting, and using the model is
subject to the upstream repository's license terms.
