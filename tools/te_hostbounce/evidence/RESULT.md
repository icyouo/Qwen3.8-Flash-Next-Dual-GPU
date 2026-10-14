# TE host-bounce milestone (2026-09-20)

## Policy
P2P/BAR1 unlock deferred. AllReduce uses pinned host staging (D2H → H2D), not cudaMemcpyPeer.

## Environment
- Host: hp-z4-g4, kernel 7.0.0-31-generic
- 2× CMP 170HX, 65536 MiB, Gen2 x4, BAR1 8GB
- Artifact: cut-25 `model-00001-of-00131.q38w` offset 6656000, shape O=10240 I=320 BF16

## Results (`tp_golden_hb`)
| Test | max_abs | mean_abs | Verdict |
|---|---:|---:|---|
| ColumnParallel (row shard, concat) | 0 | 0 | **PASS** bit-exact |
| RowParallel + host-bounce AllReduce | 2.861e-06 | 5.871e-08 | **PASS** (<1e-3) |

## Next
- S3.2 W4/W8 quantized TP parity (host-bounce AR)
- S3.4 one attention block TP (QKV col / O row + AR)
- S2B real shard byte cutting later
