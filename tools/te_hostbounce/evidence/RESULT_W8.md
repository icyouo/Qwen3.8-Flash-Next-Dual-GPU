# S3.2 W8 TP golden (host-bounce AllReduce)

Date: 2026-09-20 (CST)

## Tensor
- `blk.0.linear_attn.out.weight` from segment `model-00001-of-00131.q38w`
- layout: `w8a16_sym_g128` (int8 × BF16 scale / group 128)
- shape: O=2560, I=6144, G=128
- data_off=56156160, scale_off=71884800

## Results
| Test | max_abs | mean_abs | verdict |
|------|---------|----------|---------|
| ColumnParallel (row split) | 0 | 0 | **PASS** (bit-exact) |
| RowParallel + host-bounce AR | 1.55e-6 | 9.91e-8 | **PASS** |

## Notes
- No P2P / BAR1 peer path used; AR is D2H + H2D + GPU0 add.
- Col split I/2=3072 is group-aligned (3072 % 128 == 0).
- Source: `src/tp_golden_w8.cu`; log: `tp_golden_w8.run.log`
