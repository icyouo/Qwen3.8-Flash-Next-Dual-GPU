# S2B q38_shard_artifact — Phase B byte rewrite

## Tool
- `Qwen3.8-Flash-Next-Dual-GPU/tools/q38_shard_plan.py` — col_shard census now real (no Phase B deferral)
- `Qwen3.8-Flash-Next-Dual-GPU/tools/q38_shard_artifact.py` — row/col/expert/rank-only rewrite + `--resume`

## Smoke (segment 0)
- Output: `Homelab/.artifacts/q38-shard-smoke-seg0/`
- out.weight col_shard numeric match **PASS**
- qkv row_shard shapes 5120×2560 per rank

## Full rewrite (in progress)
- Dest: `/Volumes/datacenter/ai/models/Qwen3.8-Flash-Next-q38-cut25-tp2/`
- Runner: tmux session `q38shard` + `--resume`
- Log: `Homelab/.artifacts/q38-te-hostbounce/shard_full.run.log`
