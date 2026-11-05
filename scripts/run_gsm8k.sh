#!/bin/bash
# GSM8K pools and selections (fig:gsm8k_pareto): Gemma-2-9B-Instruct, Qwen3-Embedding-0.6B,
# K=10, N=100, 20 pools under cache-gsm8k/pool{i}. Resume-safe.
#
#   bash scripts/run_gsm8k.sh                        # sample and embed (GPU), then estimate (CPU)
#   STAGES=estimate J=16 bash scripts/run_gsm8k.sh   # estimate only, J pools in parallel
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

N_POOLS=${N_POOLS:-20}
STAGES=${STAGES:-"sample estimate"}
J=${J:-1}

LM="google/gemma-2-9b-it"; MN="google__gemma-2-9b-it"
EMB="Qwen/Qwen3-Embedding-0.6B"; ES="qwen"
export EMN="${MN}_${ES}" K=10 N_RUNS=10 ANCHOR_ST=1.2
export TAUS="0.003 0.01 0.03 0.1 0.3 1 3 10 30"
export SAMP_TS="0.428571 0.666667 0.818182 1 1.03 1.05 1.07 1.1 1.22222"
export MAP_TS="500 100 50 25 10 3 1"
export CAL="cache-gsm8k/calibration_${EMN}.json"

estimate_pool() {
    local CD="cache-gsm8k/pool$1"
    local E="dpp-estimation --output_folder $CD/est${K}${EMN##*_} --model_name $EMN --cache_dir $CD --k $K"
    $E --N "$N_RUNS" --ancestral --naive --kmeans
    $E --N "$N_RUNS" --samp --target_Ts $SAMP_TS --sampling_T "$ANCHOR_ST"
    $E --N "$N_RUNS" --samp --taus $TAUS --sampling_T "$ANCHOR_ST" --calibration "$CAL"
    $E --N 1 --map --target_Ts $MAP_TS --sampling_T "$ANCHOR_ST"
    $E --N 1 --map --taus $TAUS --sampling_T "$ANCHOR_ST" --calibration "$CAL"
    echo "POOL_DONE gsm8k $1"
}
export -f estimate_pool

case " $STAGES " in *" sample "*)
    for i in $(seq 0 $((N_POOLS - 1))); do
        CD="cache-gsm8k/pool$i"
        dpp-sample --mode gsm8k --n-samples 100 --temperatures 0.5 0.8 1.0 1.2 \
            --max-new-tokens 4000 --max-model-len 4000 --gpu-mem-util 0.90 --sample-concurrency 40 \
            --model "$LM" --cache-dir "$CD" --seed "$i"
        dpp-embeddings --samples_file "$CD/${MN}.pkl" --model_name "$EMB" \
            --output_file "$CD/${EMN}_embeddings.pkl"
    done ;;
esac

case " $STAGES " in *" estimate "*)
    # sigma_hat comes from pool 0 and is shared by all pools
    [ -f "$CAL" ] || dpp-estimation --calibrate --calibration "$CAL" --model_name "$EMN" --cache_dir cache-gsm8k/pool0
    seq 0 $((N_POOLS - 1)) | xargs -P "$J" -I{} bash -c 'set -e; estimate_pool {}' ;;
esac
