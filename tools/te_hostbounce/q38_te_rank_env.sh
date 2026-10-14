#!/usr/bin/env bash
# Export per-rank TE artifact roots for Dual-GPU / host-bounce runs on Z4.
# Usage: source q38_te_rank_env.sh [/path/to/q38-tp2-local]
set -euo pipefail
ROOT=${1:-$HOME/q38-tp2-local}
export Q38_TE_WORLD_SIZE=2
export Q38_TE_TP2_ROOT="$ROOT"
export Q38_TE_RANK0_ARTIFACT="$ROOT/rank0"
export Q38_TE_RANK1_ARTIFACT="$ROOT/rank1"
export Q38_TE_COLLECTIVE=host_bounce
echo "Q38_TE_TP2_ROOT=$Q38_TE_TP2_ROOT"
echo "Q38_TE_RANK0_ARTIFACT=$Q38_TE_RANK0_ARTIFACT"
echo "Q38_TE_RANK1_ARTIFACT=$Q38_TE_RANK1_ARTIFACT"
echo "Q38_TE_COLLECTIVE=$Q38_TE_COLLECTIVE"
for r in 0 1; do
  for s in 0 1; do
    idx="$ROOT/rank$r/stage$s/index.q38d"
    [[ -f "$idx" ]] || { echo "missing $idx" >&2; return 1 2>/dev/null || exit 1; }
  done
done
echo "TE rank roots OK"
