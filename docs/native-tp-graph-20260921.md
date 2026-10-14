# TP GPU graph fragments

Experimental P3 implementation for ordinary single-row decode. It preserves
existing W4/W8 arithmetic and uses the configured host/synchronous-P2P/async-P2P
collectives between GPU graph fragments. It is not a graph containing CPU
rendezvous and does not yet graph multi-row target verification.

`CudaTeGraphTape` records an ordered sequence of GPU graph executable objects
and collective metadata. A thread-local TE recorder intercepts bridge calls
during construction, closes the current GPU capture, stores the collective,
and starts a fresh capture. Construction performs no collective rendezvous
and launches no captured GPU work, so it cannot advance recurrent state twice.
Adjacent collectives produce empty GPU segments, which are omitted without
changing collective order. Source graphs are released after instantiation;
executables are destroyed before caller-owned workspace/tensors.

Replay launches each GPU piece and executes the ordinary collective bridge at
its recorded boundary. The stream must match its recorded stream. Cancellation
checks occur between pieces; lifecycle qualification of real GPU cancellation
remains pending. A failed capture clears its recorder and destroys partial
graphs. Nested recording is rejected.

The existing single-row piecewise backend selects this tape for TP fragments;
PP retains its original graph implementation. QSA, variable input/output
offsets, PLE I/O/H2D, and transaction clones/commits stay outside capture.
Variants are local to each rank/stage and keyed by the actual working GDN/PLE
bank pointers. Reset reuses stable allocations and selects the corresponding
bank variant. GPU graph capture/node/replay counts refer to actual GPU pieces,
not the outer tape containers. Capture failures remain explicit fallbacks.

Launcher `--tp-piecewise-graph` sets `Q38_TP_PIECEWISE_GRAPH=1` and changes the
kernel identity. Defaults remain eager; the readiness graph field reflects the
TP setting, while replay counters establish whether execution actually used it.

## Validation

The standalone `make te-graph-test` fixture passed 768 paired changing-input
iterations across host staging, synchronous P2P, and async P2P, on both groups.
It tests FP32 sum, token-major BF16 gather and BF16 sum with adjacent
collectives. Sequential rank construction proves no peer rendezvous occurred;
unchanged device values and bridge counters prove no execution during capture.
Expected graph/collective/node counts are asserted. Injected capture failures
are followed by successful ordinary paired execution. The full runtime compiled
and CPU/runtime/worker plus 71 Python regressions passed.

Model harness: Homelab `.artifacts/q38-native-opt-graph-20260921/run-graph.py`.
Both modes use grouped FP32 scales, small GEMM, RAM PLE, persistent rank worker,
async P2P, batched target and prefix snapshots. Only the graph flag differs.
Cases cover three repeated 256-token webpage generations, thinking, 8192-token
input, a complete short math answer, shorter prefill, and all seven acceptance
positions of a six-candidate target transaction followed by ordinary decode.
Token IDs and sampled full-vocabulary BF16 logit snapshots must match, and both
stages must show nonzero graph replay counts with zero fallbacks.

Model A/B completed: each mode ran 14 cases, published 1135 tokens and captured
115 full-vocabulary BF16 logit snapshots. All compared token IDs, prompts and
logit snapshots match exactly, including continuation after partial prefix
restoration. Stage0 records 204 GPU graph captures, 112200 replays and 3744
nodes; stage1 records 188 captures, 103400 replays and 3396 nodes. Both report
zero fallbacks. Baseline graph counters are zero.

Warm webpage generation (median repeats 1/2): 37.0438→39.8502 tok/s, about 7.6%.
Thinking short run: 36.9110→38.4711 tok/s. The 8192-token context run:
32.0715→33.0451 tok/s. The first graph webpage run, including lazy capture,
reached 38.5150 tok/s. These are diagnostic single-request RPC measurements;
they should not be directly compared with the earlier reference SSE benchmark.

Timings include logit diagnostics and initial capture;
normal serving performance, larger target graph shapes, GPU cancellation and
remaining long-context/quality gates are not yet qualified. Graph executable
driver allocation size is not separately measured by the current tracked-byte
counter; whole-device free/total memory measurements remain available.
The final free-memory observation was 27341357056 bytes in baseline and
27253276672 bytes with graphs (84 MiB lower). This is a whole-device difference,
not an exact graph-driver allocation accounting.

Runtime SHA256: `4dbe9d7290a99590155faff15b09ca2c992854f5b42ae0709998675b43e2a8f9`.
Evidence archive SHA256: `d56a04986ec60dd8ed048183480f7f86386e6a496b5e2ce45fd6b1d17c06b695`.
Original service restoration started at the end of the experiment. The graph
policy remains opt-in and was not promoted to the original serving command.
