# Evaluated-prefix state snapshots

This opt-in P4 implementation preserves the recurrent state produced by the
evaluated target batch. It does not assert numerical equivalence between
batched and ordinary single-row arithmetic, and does not enable automatic MTP.

`--tp-prefix-snapshots` requires `--tp-batched-target` and contributes to the
launcher kernel identity. It covers complete target batches of 2..7 rows with
the precomputed GDN prefill policy. Other shapes/policies retain the existing
replay path. Ordinary append/decode and the existing PP checkpoint path keep
their behavior.

The GDN convolution and precomputed recurrence kernels write each requested
post-token state directly into a strided allocation. PLE convolution does the
same in both fused and unfused modes. Separate template instantiations keep
snapshot stores out of ordinary kernels. A snapshot view records pointer,
byte stride and requested count; extent/alignment/count validation precedes
launch. Storage ownership and non-overlap are caller responsibilities.

The backend lazily allocates six prefix banks, retained until teardown. Each
bank includes all local GDN layer states and, on the PLE-owning stage, its
convolution state. Partial commit restores the accepted row directly. QSA
commits the accepted extent and PLE hashes use the corresponding evaluated
prefix. Full acceptance keeps the existing complete working bank. Target HC
rows remain those produced by the target evaluation. Transaction start
invalidates previous prefix availability; cancelled/failed evaluation cannot
publish a prefix. Allocation bytes are included in state-memory metrics.

The expected additional GDN allocation is approximately 665 MiB per rank for
six prefixes across both logical stages, plus PLE state. Capture/restore trace
lines can be enabled with `Q38_TP_PREFIX_DIAGNOSTICS=1`; normal runs omit them.
GPU cancellation, long-context and automatic-MTP lifecycle qualification remain
required before promotion.

## Validation

`make prefix-snapshot-test CUDA_HOME=/usr/local/cuda CUDA_COMPAT=0` tests
GDN conv, GDN precomputed recurrence, PLE conv and fused PLE conv. Starting from
nonzero states, widths 1/2/3/5/7 compare each captured prefix byte-for-byte with
independent execution of that prefix. Snapshot-enabled output and final state
must equal the ordinary kernel; restoring a saved state must be exact. Strided
guard bytes and unrequested rows must remain unchanged. Invalid stride/count/
null/overflow views are rejected. All four paths passed on the 170HX.

The integrated runtime compiled, the complete CUDA kernel regression passed,
and CPU/runtime/worker plus 71 Python tests passed. Remote candidate header
predates a final local defensive check for zero caller-supplied alignment or
state size; kernel wrappers always supply nonzero constants.

Model harness: Homelab `.artifacts/q38-native-opt-prefix-20260921/run-prefix.py`.
Same binary compares replay and snapshot modes, both with grouped FP32 scales,
small GEMM, persistent workers, async P2P and RAM PLE. In addition to every
acceptance length at widths 0/1/2/4/6, it changes only candidate tokens after
the first rejected token and requires subsequent outputs/logits to be bitwise
invariant. Both modes completed all 54 acceptance cases and 27 additional
suffix variations. All tokens and frontiers are valid, and all suffix
variations preserve subsequent raw logits exactly. Snapshot-enabled versus
replay-mode logits match in all 15 full-acceptance/zero-width cases and in
27 of 39 rejection cases. The remaining 12 are precisely first-candidate
rejections: snapshots retain the evaluated batch's first-row state, while the
old path recomputes that row with the ordinary decode kernels. This distinction
remains part of the arithmetic-policy/quality gate.

The trace records 312 physical captures and 264 restores, matching 78 captured
transactions and 66 partial commits across four rank/stage backends. Each rank
allocates 698277888 additional bytes (665.93 MiB including PLE).

Median verification time across short diagnostic cases:

| Draft width | Rejection with replay | Rejection with snapshots | Full acceptance with snapshots |
| --- | ---: | ---: | ---: |
| 1 | 63.84 ms | 47.20 ms | 45.51 ms |
| 2 | 70.95 ms | 49.66 ms | 47.14 ms |
| 4 | 79.95 ms | 54.67 ms | 53.93 ms |
| 6 | 84.53 ms | 58.96 ms | 57.90 ms |

Width-six rejection is approximately 30.2% faster; full acceptance is similar
(57.52 ms replay mode versus 57.90 ms snapshots). These timings exclude draft
generation, use diagnostics, and do not establish end-to-end decode throughput.
No production policy was promoted. Original service restoration was initiated.

Runtime SHA256: `fb845bfea43bc14bd38c6973b8ec0235ac7a44e629178992de228c3faa123bb2`.
Evidence archive SHA256: `561d78c2cef5cf7803da5341035dab1271601a118170ad76e8d3bd41c5d753e1`.
