#!/bin/bash
# Sweep DetBS over temperature and kernel-weight on a small subset.
# Each run takes ~10-25 min on a single GPU at limit=8 prompts.
set -u
cd "$(dirname "$0")/../.."
mkdir -p logs

LIMIT=${LIMIT:-8}
GPU=${GPU:-0}

# (T, w) pairs to scan. Include the two we already have for reference
# (w=0.1 T=1.0 and w=1.0 T=1.0; subsetting them again is fast).
SWEEP=(
  "1.0 0.05"
  "1.0 0.3"
  "1.0 0.5"
  "1.0 2.0"
  "1.3 0.1"
  "1.3 0.3"
  "1.3 1.0"
  "0.7 0.3"
  "0.7 1.0"
)

for pair in "${SWEEP[@]}"; do
  read -r T W <<<"$pair"
  TAG="sweep_detbs_T${T}_w${W}"
  LOG="logs/${TAG}.log"
  echo "=== T=$T w=$W (tag=$TAG) ===" | tee -a logs/sweep_detbs_summary.log
  CUDA_VISIBLE_DEVICES=$GPU python baselines/detbs_ssk/run_intrinsic_baselines.py \
    --method detbs --beam-size 10 \
    --kernel-weight $W --temperature $T \
    --max-new-tokens 50 --limit $LIMIT \
    --tag "$TAG" \
    > "$LOG" 2>&1
  tail -1 "$LOG" | tee -a logs/sweep_detbs_summary.log
done
echo "ALL_DONE" | tee -a logs/sweep_detbs_summary.log
