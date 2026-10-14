# S3.4b Real GDN + TP + host-bounce AR

Date: 2026-09-20 (CST)

## Path
1. ColumnParallel W8 gemv for qkv / z / in_a / in_b (host AllGather)
2. Real `cuda_gdn_conv_decode_bf16` + `cuda_gdn_recurrent_decode_bf16` + `cuda_gdn_output_norm_bf16` on GPU0
3. RowParallel `out.weight` + host-bounce AllReduce

## Result
| Test | max_abs | mean_abs | verdict |
|------|---------|----------|---------|
| realGDN+TP+hostAR | 1.49e-7 | 1.10e-8 | **PASS** |

Source linked: `q38_gdn.cu` from Dual-GPU tree. Harness: `src/tp_gdn_block_hb.cu`.
