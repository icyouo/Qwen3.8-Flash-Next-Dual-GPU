# TP multirow target graphs

Opt-in `--tp-batched-graph` requires `--tp-piecewise-graph` and
`--tp-batched-target`. M=2/3/5/7 target transactions use the same GPU/collective
tape as single-row decode. Dynamic QSA, input/output transfer, PLE staging and
transaction publication remain outside capture. The first eligible transaction
executes eagerly to warm and seal small-GEMM scratch; it is not executed twice.
Cache variants include token count, working GDN/PLE pointers and prefix storage
address. Each rank/stage permits ten variants (two banks, five shapes including
M=1). Profiles and other shapes remain eager; capture failure is explicit.

## Model evidence

Both modes use the same binary, W4/W8 artifact, RAM PLE, persistent rank worker,
async P2P, grouped FP32 scales, small GEMM, prefix snapshots and M=1 graphs.
Only multirow graphs differ. Three prompts, each with two prompt append layouts
to exercise alternating banks, cover widths 0/1/2/4/6 and every acceptance
position. Each verification is followed by three ordinary decode tokens with
full-vocabulary BF16 snapshots. All 108 cases per mode match across modes in
candidate IDs, published IDs, tail IDs, raw logits and committed frontiers.
Each mode also passes 54 rejected-suffix perturbations exactly.

This proves graph/eager equivalence for these cases, not equivalence between
batched verification and ordinary single-row arithmetic: existing differences
against the single-row oracle remain visible in `tail_logits_bitwise`.

Every rank/stage logs actual batch replay counts of 11/24/48/72 for M=2/3/5/7.
One initial M=2 execution warms eagerly. Aggregated stage0 graph metrics are
1020 captures, 73338 replays and 49408 nodes; stage1 records 940/67586/44948.
Both have zero fallback. These counters include M=1 pieces.

| Draft width | M=1 graphs only, median verify ms | Multirow graphs, median verify ms |
|---|---:|---:|
| 0 | 24.647 | 24.559 |
| 1 | 51.698 | 37.061 |
| 2 | 52.642 | 40.553 |
| 4 | 55.787 | 48.251 |
| 6 | 59.468 | 53.484 |

Times include cold captures and diagnostics, and combine acceptance positions.
They exclude automatic draft production and are not end-to-end tok/s. GPU
cancellation, broader quality and long-context gates remain pending. Runtime
reported swap residency; RAM PLE is a storage policy, not proof every host page
remained resident throughout this diagnostic run. No default promotion occurred.

Evidence: Homelab `.artifacts/q38-native-opt-batch-graph-20260921/`, including
the harness, raw model snapshots, logs, metrics and independently rechecked
cross-mode results. Runtime SHA256:
`1531984ce9f8edcaa02504c8095efb9624dcdf8d5fc7f7612a6b1d081fa72f71`.
Archive SHA256 (verified after download):
`3e42285bafb4da3e7ff3bfac45a91f314b148dd265166790c6c46b7e1adcb8a1`.
