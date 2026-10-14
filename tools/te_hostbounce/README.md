# TE / TP host-bounce prototypes (Q38)

Unattended TE path for dual CMP 170HX without working BAR1 P2P.

## Status (2026-09-20)
| Gate | Result |
|------|--------|
| S3.1 BF16 Column/RowParallel + host AR | PASS |
| S3.2 W8 TP + host AR | PASS |
| S3.3 P2P data plane @ 8GB BAR1 | FAIL (ghost BW) — deferred |
| S3.4 Attn proj sandwich + host AR | PASS (stand-in middle) |
| S3.4b Real `cuda_gdn_*` + TP + host AR | PASS |
| S2B `q38_shard_artifact.py` | smoke seg0 PASS; full rewrite → `…-tp2` |

## Tools
- `../q38_shard_plan.py` — offline plan / census
- `../q38_shard_artifact.py` — Phase B byte rewrite
- `src/tp_*.cu` — Z4 goldens

P2P remains deferred: Z4 host MMIO windows cannot assign 2×64GB BAR1.

## TE runtime smoke (2026-09-20)
- Artifact on Z4: `~/q38-tp2-local`
- `tp2_artifact_smoke.py` + `tp2_dual_upload_smoke.cu` both **PASS**
- Env helper: `source q38_te_rank_env.sh ~/q38-tp2-local`
- Details: `RESULT_TE_RUNTIME.md`
