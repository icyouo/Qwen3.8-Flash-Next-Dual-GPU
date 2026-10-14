# TE runtime smoke (TP2 artifact + host-bounce)

Date: 2026-09-20 CST · Host: Z4 (dual CMP 170HX)

## Artifact
- Path: `~/q38-tp2-local` on Z4 (`shard_manifest.json`, rank0/1 × stage0/1)
- Size: ~85G × 2 ranks

## Smokes
| Test | Result |
|------|--------|
| `tp2_artifact_smoke.py` segment extents + row/expert reconstruct vs source | **PASS** |
| `tp2_dual_upload_smoke.cu` `blk.0.linear_attn.out.weight` col-shard dual upload + host AR | **PASS** max_abs≈1.1e-6 |

## Runtime wiring
Each TE rank is a normal Q38 device artifact root:
```
rank0/ → stage0/index.q38d + stage1/index.q38d
rank1/ → stage0/index.q38d + stage1/index.q38d
```
Load with existing `DeviceWeightStore` / `CudaDeviceWeightStore` per GPU (`device=rank`). Collectives for RowParallel use host-bounce AR (see `tp2_dual_upload_smoke.cu`), not CUDA P2P.

## Deferred
- Full end-to-end decode loop with TP across all layers
- NAS mirror of `q38-tp2-local` (SMB write previously unstable)
- Real P2P / 64GB BAR1

## All-layer RowParallel hostAR (2026-09-20)
`tp2_all_colshard_smoke.cu`: every `linear_attn.out` / `attn_output` col_shard across stage0+stage1 (48 matrices) dual-GPU GEMV + host AR vs full source — **48/48 PASS** (max_abs ~1e-6). Skipped `output.weight` (vocab-sized).
