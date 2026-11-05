#!/bin/bash
# GSM8K selections on the instructed kernel (tab:gsm8k-kernels, tab:significance), under
# cache-gsm8k/pool{i}/est10qwenins. Embed first: python analysis/embed_instructed.py
# --pool-start 0 --pool-end 20. Resume-safe.
#
#   J=16 bash scripts/run_gsm8k_instructed.sh
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

N_POOLS=${N_POOLS:-20}
J=${J:-1}
export EMN="google__gemma-2-9b-it_qwenins" K=10 N_RUNS=10 ANCHOR_ST=1.2
export TAUS="0.003 0.01 0.03 0.1 0.3 1 3 10 30"
export CAL="cache-gsm8k/calibration_${EMN}.json"

estimate_pool() {
    local CD="cache-gsm8k/pool$1"
    local E="dpp-estimation --output_folder $CD/est${K}qwenins --model_name $EMN --cache_dir $CD --k $K"
    $E --N "$N_RUNS" --ancestral --naive --kmeans
    $E --N "$N_RUNS" --samp --taus $TAUS --sampling_T "$ANCHOR_ST" --calibration "$CAL"
    $E --N 1 --map --taus $TAUS --sampling_T "$ANCHOR_ST" --calibration "$CAL"
    echo "POOL_DONE qwenins $1"
}
export -f estimate_pool

[ -f "$CAL" ] || dpp-estimation --calibrate --calibration "$CAL" --model_name "$EMN" --cache_dir cache-gsm8k/pool0
seq 0 $((N_POOLS - 1)) | xargs -P "$J" -I{} bash -c 'set -e; estimate_pool {}'
