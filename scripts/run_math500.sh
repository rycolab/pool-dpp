#!/bin/bash
# MATH-500 saturation check (app:math500, fig:math500): 100 problems, 10 pools of N=100. Resume-safe.
#
#   bash scripts/run_math500.sh                 # sample and embed on a GPU, verify and score on CPU
#   STAGES="verify score" bash scripts/run_math500.sh
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

for stage in ${STAGES:-sample verify embed score}; do
    python analysis/math500.py "$stage" --n-pools "${N_POOLS:-10}"
done
