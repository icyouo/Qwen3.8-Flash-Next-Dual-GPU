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

- `make -j2 verify PYTHON=.venv/bin/python`: C++ runtime tests passed,
  41 Python tests passed. The initial `sm_86` CUDA compile was superseded by
  the real boards reporting compute capability 8.0; production now targets
  `sm_80` and requires the real kernel fixtures below.
- `make -j1 cuda-test`: the expanded binary compiled; execution correctly
  skipped while no NVIDIA GPU was attached. When a GPU is present it covers
  transport checksum/D2H/H2D corruption rejection plus GDN, PLE and QSA fixtures.
- Unix executor socket → reference sidecar → HTTP create/execute/metrics passed
  with zero failures and metrics schema hash `0x7133386d65740136`.
- The final mock transport/state gate reached exactly `canonical=262144` and
  `target=stage0=stage1=262143`, publishing 64 tokens with zero failure or
  rollback. Evidence on Ubuntu:
  `/home/icy/src/q38-runtime/out/mock-strict-262080-plus-64-checksummed.json`,
  SHA-256 `0ccc0790bfaf8ae828a59b75cfbae7b1fb4c459872dd18125175cc7020b28bcc`.

The mock gate proves transport, RPC, transaction, frontier and bounded-context
behavior. It does **not** prove CUDA numerical correctness, checkpoint parity,
GPU memory fit, PLE SSD latency, or throughput.

## First real-hardware run order

Do not skip directly to the 262K production claim.

1. Attach both CMP 170HX boards and record `nvidia-smi -q`, `nvidia-smi topo -m`,
   PCIe link state, driver/CUDA versions and P2P-disabled behavior.
2. Run `make -j1 cuda-test`. All GPU fixtures, including transport checksum
   corruption rejection, must execute rather than report `skipped (no GPU)`.
3. Wait for the streaming artifact job to atomically publish `READY.json`, then
   run `q38_launch.py --dry-run`. Any digest, identity, tensor census, PLE direct
   lane or memory preflight failure is a hard stop.
4. Launch the default plain/M1 lane without `--enable-mtp`. First run frozen
   short numerical fixtures, then fresh executors at 8K, 32K, 128K and strict
   262K. For the direct-token gate, the corresponding prompt lengths with 64
   generated tokens are 8,128; 32,704; 131,008; and 262,080.
5. At every level inspect stage/device/host/PLE metrics, finite logits, exact
   prefix extension, cancel/timeout/duplicate/fault rollback, and memory reserve.
   Use tokenizer-produced marker inputs or frozen golden fixtures for model
   correctness; synthetic tokens are only a state/transport stress input.
6. Run near-256K suffix continuation from an existing session and prove the old
   prefix is not replayed.
7. Only after the plain lane passes may `--enable-mtp` be tested. The pinned
   official checkpoint currently permits one draft token (width-2 target
   verification); wider requests intentionally fail closed.
8. Performance/release claims still require the design document's cold/warm PLE,
   latency percentiles, memory floors, growing-turn and soak gates.

The runtime implementation is ready for this sequence once the two external
prerequisites exist: both GPUs and the verified production artifact.
