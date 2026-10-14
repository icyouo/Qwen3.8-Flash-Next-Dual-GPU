# S3.4 Attention-block TP golden (host-bounce AllReduce)

Date: 2026-09-20 (CST)

## Scope
`blk.0.linear_attn` projection sandwich on dual GPU with host-bounce AR (no P2P).

| Tensor | shape | format | role |
|--------|-------|--------|------|
| qkv.weight | 10240×2560 | w8a16_sym_g128 | ColumnParallel on V heads (rows 4096:) |
| z.weight | 6144×2560 | w8a16_sym_g128 | ColumnParallel (head split) |
| out.weight | 2560×6144 | w8a16_sym_g128 | RowParallel + host AR |

QKV packing assumed: Q(2048)|K(2048)|V(6144). V/Z split 3072+3072 (group-aligned).

## Middle
Stand-in `core = V * sigmoid(Z)` (elementwise, head-local). Real `cuda_gdn_*` conv/recurrent not linked yet — TP cut points match GDN head sharding.

## Result
| Test | max_abs | mean_abs | verdict |
|------|---------|----------|---------|
| AttnBlock TP + host AR | 7.15e-7 | 4.85e-8 | **PASS** |

Source: `src/tp_attn_block_hb.cu` · log: `tp_attn_block_hb.run.log`
