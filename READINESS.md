# Qwen3.8-Flash-Next-Dual-GPU test readiness

This file records the boundary between implementation evidence and the first
real dual-GPU/model run. It deliberately does not treat mock execution or a
successful CUDA compile as model-correctness or performance evidence.

## Implemented runtime contracts

| Design contract | Implementation evidence |
|---|---|
| Versioned artifact, stage plan and tensor layout | `artifact.h`, `model_plan.h`, `device_artifact.h`; exact official tensor-census and cut-25 tests |
| Single native executor with two stage workers | `executor.cc`; independent stage threads, one semantic writer and fail-closed production backend |
| Request-atomic append/decode/MTP state | `transaction.cc`, `executor.cc`; provisional execution, dual acknowledge, commit/publish ordering and full-request rollback |
| No-P2P transport | `cuda_transport.cu`; separate small/large pinned rings, BF16 D2H/H2D and position-tagged payload checksum |
| QSA/GDN/PLE/HC/MoE/LM/MTP model data path | `cuda/q38_*.cu`, `cuda_backend.cu`; plain lane is default and MTP is explicit opt-in |
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
- C++ runtime tests passed; Python tests passed 42/42 with NumPy; CUDA
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
5. Only after the plain lane passes may `--enable-mtp` be tested. The pinned
   official checkpoint currently permits one draft token (width-2 target
   verification); wider requests intentionally fail closed.
6. Performance/release claims still require the design document's cold/warm PLE,
   latency percentiles, memory floors, growing-turn and soak gates.
