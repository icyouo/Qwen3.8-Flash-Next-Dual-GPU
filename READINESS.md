# Q38 runtime test readiness

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
- Both end-to-end runs reported zero executor failure/rollback/cancel/deadline,
  CUDA allocation failure, boundary wait, or PLE read error.  The ordered
  parallel top-k reproduced the scalar reference's initial short greedy tokens.
- Raw build, fixture, profile, fallback A/B and end-to-end evidence is archived
  beside this worktree in `validation-native-decode/VALIDATION.md`.

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
