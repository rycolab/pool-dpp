#!/bin/bash
# Ambrosia pools and selections (fig:ambrosia_pareto): Gemma-2-9B-Instruct, MiniLM, K=3, N=200,
# 30 pools under cache-ambrosia/pool{i}. Requires ambrosia.csv in the repo root. Resume-safe.
#
#   bash scripts/run_ambrosia.sh                        # sample and embed (GPU), then estimate (CPU)
#   STAGES=estimate J=16 bash scripts/run_ambrosia.sh   # estimate only, J pools in parallel
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

N_POOLS=${N_POOLS:-30}
STAGES=${STAGES:-"sample estimate"}
J=${J:-1}

LM="google/gemma-2-9b-it"; MN="google__gemma-2-9b-it"
EMB="sentence-transformers/all-MiniLM-L6-v2"; ES="minilm"
export EMN="${MN}_${ES}" K=3 N_RUNS=10
export TAUS="0.003 0.01 0.03 0.1 0.3 1 3 10 30"
export SAMP_TS="0.428571 0.538462 0.666667 0.818182 1"   # PoolDPP-Samp at sT=1.0
export MAP_TS="10 7.5 3 1"                               # PoolDPP-MAP at sT=1.5
export CAL="cache-ambrosia/calibration_${EMN}.json"

estimate_pool() {
    local CD="cache-ambrosia/pool$1"
    local E="dpp-estimation --output_folder $CD/est${K}${EMN##*_} --model_name $EMN --cache_dir $CD --k $K"
    $E --N "$N_RUNS" --ancestral --naive --kmeans
    $E --N "$N_RUNS" --samp --target_Ts $SAMP_TS --sampling_T 1.0
    $E --N "$N_RUNS" --samp --taus $TAUS --sampling_T 1.0 --calibration "$CAL"
    $E --N 1 --map --target_Ts $MAP_TS --sampling_T 1.5
    $E --N 1 --map --taus $TAUS --calibration "$CAL"
    echo "POOL_DONE ambrosia $1"
}
export -f estimate_pool

case " $STAGES " in *" sample "*)
    for i in $(seq 0 $((N_POOLS - 1))); do
        CD="cache-ambrosia/pool$i"
        dpp-sample --mode ambrosia-top100 --chat-template --n-samples 200 \
            --temperatures 0.6 0.7 1.0 1.2 1.25 1.3 1.5 --max-new-tokens 800 \
            --gpu-mem-util 0.90 --sample-concurrency 50 --model "$LM" --cache-dir "$CD" --seed "$i"
        dpp-embeddings --samples_file "$CD/${MN}.pkl" --model_name "$EMB" \
            --output_file "$CD/${EMN}_embeddings.pkl"
    done ;;
esac

case " $STAGES " in *" estimate "*)
    # sigma_hat comes from pool 0 and is shared by all pools
    [ -f "$CAL" ] || dpp-estimation --calibrate --calibration "$CAL" --model_name "$EMN" --cache_dir cache-ambrosia/pool0
    seq 0 $((N_POOLS - 1)) | xargs -P "$J" -I{} bash -c 'set -e; estimate_pool {}' ;;
esac
