# Automatic TP MTP: initial diagnostic run

Experimental `--tp-batched-mtp` requires batched target verification and a
complete MTP artifact view. Paired rank drafts must agree. Draft QSA state is
ephemeral and rolled back before target verification; accepted MTP state is
rebuilt from committed target HC at commit. Existing retained PP execution is
unchanged. Sampled requests fall back to ordinary target sampling. Adaptive
MTP now has a separate opt-in batch-cost policy; real performance qualification
is in progress.

The `mtp` artifact-view profile imports full stage1 MTP controls, FC, QSA,
shared MLP and embedding tensors from the original checkpoint while retaining
routed expert partitions and the original main-model layout. Tensor extents
are hashed before manifest publication. QSA output now reduces only when
column-sharded; replicated QSA must not be summed twice.

CPU coverage includes widths1/2/4/6 across repeated decode/reset with ordinary
token oracle; draft failure, partial/invalid output, post-draft cancellation,
fatal versus retry handling, unchanged target frontier, EOS/context-tail,
workspace limits, pre-cancellation and sampled fallback. These pass. Artifact
tests and prior 73 Python regressions pass. Real GPU cancellation remains open.

## First GPU run

Each case generates 96 tokens after the seed, using the same MTP-enabled
runtime with RAM PLE, W4/W8, async P2P, worker, grouped FP32 scales, small GEMM,
prefix snapshots and target graphs. Logit diagnostics are enabled. Timings
include RPC overhead and cold graph setup and are not a serving acceptance run.

| Prompt | Ordinary | MTP1 | MTP2 | MTP4 | MTP6 |
|---|---:|---:|---:|---:|---:|
| Math | 38.14 | 42.34 | 51.99 | 52.36 | 52.05 |
| Chinese narrative | 38.86 | 40.25 | 44.81 | 41.83 | 36.51 |
| Insertion sort | 39.04 | 46.13 | 56.50 | 60.73 | 65.26 |

Units: tok/s. Code matches ordinary token IDs at all four widths. Math diverges
at zero-based token34; narrative at37, consistently across widths. This is not
a passed quality gate. Need identical-prefix target-only/candidate-batch/MTP
comparisons to distinguish existing batched arithmetic differences from state
commit bugs before drawing conclusions about quality or promotion.

Code MTP6 accepts 73/138 draft tokens; narrative MTP6 accepts54/246. Narrative
draft overhead grows from156ms at width2 to401ms at width6 and target work also
grows. Fixed maximum width is therefore inappropriate; measured adaptive
selection remains required. No optimized runtime has replaced production.

Evidence: Homelab `.artifacts/q38-native-opt-mtp-20260921/`.
Runtime SHA256: `42f9210b4411a79586952c44d2c3fda0e84f6c8bf4e94dd2a0f635f660e8e6d3`.
Downloaded archive SHA256: `b49e66d70fd1601302e5ad771436f9fe5f23d2325cdfd88132962c021c6e5340`.

## State isolation follow-up and adaptive policy

On the same three prompts at 48 generated tokens, actual automatic drafts were
recorded and replayed with MTP enabled but no draft execution, then with MTP
fully disabled. All 12 cases per mode match automatic publications, semantic
frontiers and three tail full-vocabulary BF16 snapshots exactly. Ordinary
outputs/logits also match between modes. These observed divergences therefore
reproduce in the target batch path without MTP state involvement. General
quality and long-context state safety still need qualification.
Evidence: Homelab `.artifacts/q38-native-opt-mtp-replay-20260921/`, archive hash
`65f17df7fabb2c093a1e33f036b85f3a4ceda82abe3de0536cd538d079649470`.

Experimental `--tp-adaptive-mtp` measures wall time per published token for
ordinary decode and widths1/2/4/6. It initializes each eligible shape twice,
replaces initial capture timing with the second sample, then uses an EWMA and
revisits the stalest alternative every eight observations. It respects caps,
ignores stopped/invalid observations, updates only after successful commits and
resets with the session. PP keeps its retained verifier cost model. CPU tests
cover workload transitions, caps/reset/invalid observations and complete mock
token/frontier equivalence. Full runtime/worker and73 Python tests pass. GPU
performance comparison is pending; no adaptive performance claim yet.

## 256-token timing and first adaptive rejection

With raw logit diagnostics disabled, fixed width6 reached61.58/57.36/70.97
tok/s on math/narrative/code; fixed width2 reached55.30/52.88/57.80; ordinary
decode reached39.36/39.82/39.82. Adaptive reached48.76/50.29/44.02 and **fails
the performance gate**. Keep it opt-in; do not promote it as an optimization.
The adaptive code case needed155 transactions versus55 at fixed6, drafting157
versus329 candidates and accepting101 versus201. Per-arm cost observations
are too conservative on this run; graph warmup and changing acceptance require
trace evidence before a precise causal conclusion or policy revision.

These longer batched outputs all differ from ordinary decode, including code
at token163 (the earlier96-token code sample matched). Fixed token budgets are
performance samples, not complete-answer quality results. Complete-answer
mathematical, structured-output and executed-code fixtures are being added.

Evidence: Homelab `.artifacts/q38-native-opt-mtp-adaptive-20260921/`.
Runtime SHA256 `a22f2c4368603d5ea8b809be90cd540e35f61d100adc38b30a493a871d870f69`.
Archive SHA256, checked after download:
`a1752a5ffc15b697e8c5ab2f6ee5a406e685a97717b7bb42bc2a00e4c6ac9811`.
