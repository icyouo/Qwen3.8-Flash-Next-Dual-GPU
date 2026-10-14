# Qwen3.8-Flash-Next-Dual-GPU test readiness

> **2026-10-12 current opt-in candidate:** `mtp4-native-bf16-gdn-qk-candidate` measures 3.035s TTFT on frozen 4K requests, about 9.53% behind the previous-round vLLM reference. Same-path logits/answer/recovery checks pass; scalar/MTP numerical differences and long-term stability remain open. See [current qualification and limits](docs/gdn-normalized-prefill-20261012.md).

> **2026-10-11 frozen 4K A/B:** The opt-in prefill profile passes the qualified full BF16/token oracle, answer-quality and recovery gates. Three-repeat median prefill rises **864.58→990.04 tok/s**, with identical 457-token answers and TTFT **4.75→4.14s**. Decode stays around 85 tok/s; vLLM parity and long-duration reliability are not established. See [measurements and installed profile](docs/native-vllm-parity-20261011.md).

> **2026-10-04 benchmark stability update:** llm_speedtest completed three exact-usage requests, then was interrupted by two whole-host reboots during 32K and 4K prefill. The three-repeat matrix is incomplete; general 160 tok/s and long-context stability are not qualified. See [the report](docs/llm-speedtest-20261004.md).


This file records the boundary between implementation evidence and the first
real dual-GPU/model run. It deliberately does not treat mock execution or a
successful CUDA compile as model-correctness or performance evidence.

## 2026-10-04 Z4 fixed-MTP4 native fast profile

The optional `mtp4-native-fast` profile reaches three-run mean HTTP decode
rates of 172.81 tok/s for web and 174.70 tok/s for code on dual 170HX, 200W per
GPU, existing W4/W8 and RAM PLE. It includes framework-free pinned Marlin MoE,
lossless vocabulary-row and GDN projection layouts, head-local GDN/QSA,
whole target/draft graphs, bounded peer-collective fusion and explicit BF16
rounding boundaries in the subsequent fusions. The qualified follow-up adds
shared-expert/P2P graph overlap, direct BF16 top10 reduction, MoE/HC finish
fusion and a normalized MTP embedding cache (about 1.18 GiB per GPU). Persistent
and wider Marlin scheduling and GDN conv/prepare fusion remain disabled after
individual measurements. This is a distinct scaled-BF16
Marlin expert numerical policy; it is not a bitwise-equivalence claim to the
older FP32-scale MoE backend or the unquantized official model.

C++ runtime/rank-worker and 88 Python tests pass. Model gates cover five prompts
including 8192-token input, full vocabulary logits, ten forced rejection
prefixes, nine complete quality answers and three cancellation/deadline recovery
paths. GPU fixtures cover local GDN/QSA state, MTP QSA graphs, output-head/argmax,
HC finish, Marlin tails and queued P2P generations; the fused peer path also
passes paired-graph mismatch, capture failure and reset/rebuild checks. New
fixtures exercise graph branch ordering, direct expert reduction, explicit HC
rounding, GDN state/snapshot parity and normalized MTP embedding gathers.
The official launcher rechecks every segment hash and default MTP4 requests;
HTTP content exactly matches the preceding qualified outputs. N−1 decode means
are 172.64/174.36 tok/s and end-to-end means are 171.53/171.54 tok/s.

This validates short-context decode and the tested quality/recovery paths.
Long-output and 128K/256K performance, context-edge release gates and continuous
load reliability still require separate validation. See the
[follow-up report and reproducible launch](docs/native-followup-optimization-20261004.md)
and the [preceding 162 tok/s qualification](docs/native-fast-optimization-20261004.md).
Historical results below retain their original contracts and do not describe
this profile.

## 2026-09-09 HC candidates (GPU validation pending)

Independent paired-row BF16 HC GEMV and activation fusion are opt-in via
`Q38_CUDA_HC_GEMV` / `Q38_CUDA_HC_FUSED`. Fusion explicitly retains the
intermediate BF16 rounding. GPU fixtures require bitwise equality with the
existing projection plus activation kernels, including CUDA Graph replay.
The next opt-in batch adds `Q38_CUDA_HC_GROUPED` for independent HC projections
and `Q38_CUDA_PLE_FUSED` for single-token PLE key/value projections. Scalar GEMV
selection is respected by the multi-projection fallback.
Further opt-in PLE paths pair key/query norms (`Q38_CUDA_PLE_NORM_PAIR`) and
fuse convolution with residual (`Q38_CUDA_PLE_RESIDUAL`) for decode/prefill;
fixtures require bitwise equality of both outputs and convolution state.
See [implementation and quality gates](docs/hc-optimization-20260909.md).

## 2026-09-06 local implementation (GPU validation pending)

New optimization paths and their switches are listed in [the implementation report](docs/optimization-20260906.md). macOS and Linux CPU tests pass (C++ runtime plus 63 Python tests). CUDA 12.8 on Linux ARM compiled and linked the SM80 runtime and GPU fixtures; GPU fixtures explicitly skip without hardware. Historical hardware evidence below predates these changes.

## Implemented runtime contracts

| Design contract | Implementation evidence |
|---|---|
| Versioned artifact, stage plan and tensor layout | `artifact.h`, `model_plan.h`, `device_artifact.h`; exact official tensor-census and cut-25 tests |
| Single native executor with two stage workers | `executor.cc`; independent stage threads, one semantic writer and fail-closed production backend |
| Request-atomic append/decode/MTP state | `transaction.cc`, `executor.cc`; provisional execution, dual acknowledge, commit/publish ordering and full-request rollback |
| No-P2P transport | `cuda_transport.cu`; separate small/large pinned rings, BF16 D2H/H2D and position-tagged payload checksum |
| QSA/GDN/PLE/HC/MoE/LM/MTP model data path | `cuda/q38_*.cu`, `cuda_backend.cu`; plain lane is default and MTP is explicit opt-in |
| Piecewise CUDA Graph decode | `cuda_backend.cu`; fixed-shape GPU regions only, exact-QSA and transaction/I/O boundaries remain eager, dual-bank variants are pointer-safe, and capture failure falls back to eager |
| Bounded SSD-PLE | `ple.cc`, `q38_ple.cu`; registered-buffer io_uring, O_DIRECT, bounded LRU, FP8 row scales and I/O metrics |
| Durable state and identity | `identity.cc`, `snapshot.cc`; identity-bound append-only journal with explicit cold-rebuild semantics |
| Stable local control plane | `rpc.cc`, `q38_control_plane.py`, `q38_sidecar.py`; concurrent Unix RPC, idempotence, cancel/deadline, token-native HTTP/SSE |
| Test and launch gates | `q38_launch.py`, `q38_strict_gate.py`; fail-closed READY validation and exact context/frontier/accounting evidence |

Production CUDA execution sets `backend_failure_is_fatal`: a CUDA, transport,
or device-state exception invalidates the executor after rollback is attempted.
Client cancel and deadline remain recoverable transaction outcomes.

## Latest implementation evidence

Executed on Ubuntu `p3-ultra`, not on the macOS editing workspace:

- Both 64 GiB CMP 170HX boards were active as SM80 devices under driver
  610.43.02 and CUDA 13.1.  Validation used the verified 128,193,992,032-byte
  cut-25 artifact in a separate source/identity/journal directory.
- C++ runtime tests passed; Python tests passed 59/59; CUDA
  compile-check, full runtime build, and real GPU fixtures passed.  Fixtures now
  cover W4/W8 production-layout decode GEMV, W4/W8 routed MoE, QSA parity, and
  stable 512-expert top-k including tied logits.
- Final short 8+64 plain decode: 31.76 tok/s, ITL p50 31.32 ms, stage p50
  13.25 + 13.10 ms.
- Final 8195+32 plain decode: 22.84 tok/s, ITL p50 43.87 ms, stage p50
  19.08 + 19.21 ms.  The original 8K native baseline was 11.4 tok/s.
- The identical 8195+32 run with explicit `durability=off` reached 25.81 tok/s,
  ITL p50 38.65 ms, and stage p50 18.89 + 19.21 ms.  Only about 0.55 ms
  remained outside GPU execution; strict per-token journal sync accounted for
  the remaining 5.22 ms difference.  These are separate durability contracts,
  not interchangeable benchmark labels.
- Both end-to-end runs reported zero executor failure/rollback/cancel/deadline,
  CUDA allocation failure, boundary wait, or PLE read error.  The ordered
  parallel top-k reproduced the scalar reference's initial short greedy tokens.
- Raw build, fixture, profile, fallback A/B and end-to-end evidence is archived
  beside this worktree in `validation-native-decode/VALIDATION.md`.
- The first grouped-MMQ prefill implementation passed W4/W8 fixtures on both
  SM80 boards, including deterministic route-plan construction, stable inverse
  mapping, exact task/workspace bounds, invalid-expert fail-closed behavior,
  hidden packing, and bitwise equality of grouped/safe dispatches. The host does
  not read the route plan between kernels.
- On a fresh no-journal runtime, 4,096 synthetic append tokens improved from
  54.66 s (74.93 tok/s) on the legacy tile-32 atomic path to 5.02 s
  (**815.78 tok/s**) with 512-token grouped-MMQ slabs and the existing dual-stage
  pipeline. A serialized-stage slab-512 run measured 8.62 s (475.15 tok/s),
  isolating the scheduler overlap gain. The following 4K-context 64-step decode
  probe measured 27.89 tok/s and 35.01 ms ITL p50.
- Raw JSON and CUDA-event logs for the 54.66 s, 22.36 s, 8.62 s, 5.05 s and
  clean 5.02 s runs are archived in the sibling
  `.artifacts/q38-prefill-grouped-mmq-r1` evidence directory. This grouped
  arithmetic still requires a new production identity and golden model-quality
  validation; testing used an explicitly manual development launch because the
  old READY correctly rejected the changed runtime hash.
- The repaired online-prefix width-4 MTP lane passed a frozen 32,768-token
  coding trace with exact plain-target token-prefix parity. Five transactions
  published `5, 4, 4, 2, 5` tokens, accepted 15/20 drafts, reached 48.65 tok/s,
  and reduced aggregate backend commit from 124.17 ms in the previous trace to
  1.19 ms. Metrics showed GPU0 only one physical row ahead of the logical GPU1
  prefix, with zero failures and rollbacks. Raw evidence is in the sibling
  `.artifacts/q38-mtp-prefix-r2` directory. This is a directional 32K control
  path gate, not a full model-quality or 256K release claim.

These synthetic-token runs prove real CUDA state mechanics and performance, but
they do **not** by themselves prove model quality or the strict 262K release
gate.

## Remaining real-hardware run order

Do not skip directly to the 262K production claim.

1. Run tokenizer-produced/frozen golden prompts and compare logits/tokens to a
   trusted reference before treating the kernel parity tests as model quality.
2. Run fresh executors at 32K, 128K and strict 262K. For the direct-token gate,
   the corresponding prompt lengths with 64 generated tokens are 32,704;
   131,008; and 262,080.
3. At every level inspect stage/device/host/PLE metrics, finite logits, exact
   prefix extension, cancel/timeout/duplicate/fault rollback, and memory reserve.
   Use tokenizer-produced marker inputs or frozen golden fixtures for model
   correctness; synthetic tokens are only a state/transport stress input.
4. Run near-256K suffix continuation from an existing session and prove the old
   prefix is not replayed.
5. Only after the plain lane passes may `--enable-mtp` be tested. The runtime
   now implements bounded retained drafting for widths 1..64, with launch
   capability capped by `--mtp-max-draft` (initially 4). Width one has the full
   directional 32K/128K/256K throughput set; the repaired width-four control
   path has initial 32K token-prefix and mixed accept/reject evidence. Widths
   2/3 and the remaining width-4 token/logit parity, cancellation/rollback,
   context-tail, 128K/256K, long-generation, and throughput gates must pass
   before a production default is selected.
6. Performance/release claims still require the design document's cold/warm PLE,
   latency percentiles, memory floors, growing-turn and soak gates.
7. The opt-in piecewise CUDA Graph lane must match eager token IDs and logits at
   short/32K/128K/256K contexts, survive cancel and injected rollback, report
   two or fewer bank variants per stage with zero steady-state captures, and
   demonstrate lower stage latency without increasing tracked CUDA allocations.
   Use `--enable-logit-diagnostics` plus `tools/q38_logit_parity.py` in isolated
   eager/graph runs: the gate requires matching session/epoch/frontier metadata,
   selected tokens, and bitwise-identical raw BF16 payloads. Disable diagnostics
   for the subsequent latency measurement because it intentionally adds a full
   logits D2H copy.
