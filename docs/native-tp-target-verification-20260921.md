# Supplied-candidate target verification

Experimental P2/P4 foundation; not automatic TP MTP and not a promoted serving
policy. Existing W4/W8 artifacts and SSD/RAM PLE are unchanged.

`DualStageExecutor::verify_candidates` evaluates the pending canonical token
followed by supplied candidates in one bounded transaction. RPC V1 opcode 11
(`tools/q38_rpc_client.py ... verify TOKEN...`) accepts up to 64 candidate IDs;
both integer arguments must be zero. Empty candidates and sampled sessions use
ordinary target decoding. Greedy verification requires one pending token and
enough backend workspace for the evaluated rows; context-tail capacity caps the
width. An accepted EOS shortens the published prefix. Existing RPC cancellation,
deadline, writer serialization and request replay rules apply.

Without `--tp-batched-target`, CUDA evaluates these rows serially. The opt-in flag
selects existing causal prefill computation for a TP speculative transaction
whose rows fit a single tile. It computes each row's head prediction, retains
target HC when MTP state exists, and compares predictions across both ranks.
The flag contributes to the launcher kernel identity, preventing session/cache
reuse across arithmetic policies. Larger-than-tile inputs conservatively retain
the serial path. No automatic draft generation is enabled by this switch.

The default partial-acceptance path restores transaction-start GDN/PLE state and
replays the accepted prefix; QSA commits only that prefix. This conservative
mechanism remains a performance cost and requires model-level qualification.
In particular, batch shape can change arithmetic, and stage1 prefix replay uses
the original target boundary. Token agreement alone cannot establish exact
recursive-state equivalence. Per-prefix state capture and direct state checks
remain part of P4, before promotion. A subsequent opt-in implementation now
restores directly from evaluated-prefix snapshots; see
[snapshot implementation and evidence](native-tp-prefix-snapshots-20260921.md).

## Evidence

CPU tests cover every acceptance length at widths 1/2/4/6/16/64, continued decode,
EOS, context tail, invalid inputs, workspace capacity, pre-cancellation, both
stage execution failures and cancellation after either stage has executed with
successful retry, sampled fallback and RPC replay/conflicts.
The full CPU suite, persistent-worker test and 71 Python tests passed. The final
fault/cancellation additions also passed the runtime suite. CUDA runtime compiled
on the target SM80 toolchain.

Real-weight harness: `.artifacts/q38-native-opt-target-20260921/run-target.py`
under the Homelab root; remote `/home/icy/q38-native-opt-target-20260921`.
Same binary, three modes: serial, batch-row, batch-fp32-small. Each uses RAM PLE,
persistent rank worker and async P2P. Three complete chat prompts establish a
12-token ordinary-decode oracle. Widths 0/1/2/4/6 test every rejection position
and full acceptance, then compare three subsequent tokens and raw full-vocab
BF16 logits. There are 54 cases per mode. All 162 cases match published and
continued tokens, and all committed frontiers are valid. Serial verification
matches subsequent logits in all 54 cases. Both batched modes match logits in
only the three zero-width fallback cases; the other 51 cases differ. Therefore
the arithmetic/state-equivalence gate is still open, despite token agreement.
Across 162 subsequent full-vocabulary snapshots per mode, maximum absolute
logit error versus that mode's ordinary-decode oracle is 0.9375 for batch-row
and 1.3359375 for batch-fp32-small. Maximum probability total variation is
0.05719 and 0.04912 respectively (temperature-one softmax). All published and
continued tokens also match the serial mode. These short fixtures do not bound
long-context, sampled-generation or quality errors.

Median verification time across the three prompts, all candidates accepted:

| Draft width / evaluated rows | Serial | Batch row | Batch FP32 + small GEMM |
| --- | ---: | ---: | ---: |
| 1 / 2 | 50.77 ms | 53.49 ms | 48.68 ms |
| 2 / 3 | 76.06 ms | 71.63 ms | 48.96 ms |
| 4 / 5 | 126.95 ms | 108.43 ms | 54.00 ms |
| 6 / 7 | 177.65 ms | 145.25 ms | 57.15 ms |

These are short diagnostic runs, not interleaved repeated performance trials.
They exclude draft generation, include the current publication/commit path,
and are not end-to-end MTP throughput. The width-six verification improvement
is approximately 3.11x. Partial-rejection cases still incur prefix replay.

Runtime SHA256:
`6cfac0b2898ac19b8b7c588842907b2383d761d2f2056f74516e12fe29bfe95d`.
Raw evidence is archived alongside the harness. Original service restoration
was started at the end of the run; no experimental policy was promoted.
Evidence archive SHA256:
`ee4289da6dfe3021cb1a6062f8ff92f52fc97901c6ddf44f1a5bf9f85767cf51`.
