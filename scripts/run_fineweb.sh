#!/bin/bash
# FineWeb pools and selections (fig:fineweb_pareto, fig:abl_lm_emb, fig:abl_multimodel), K=10, N=400.
#   cache-fineweb/              pool 0: Gemma-2-9B and Qwen3-8B x MiniLM and Qwen3-Embedding,
#                               Llama-3.1-8B x MiniLM
#   cache-fineweb-pools/pool{i} pools 1..9: Gemma-2-9B x MiniLM
# Resume-safe.
#
#   bash scripts/run_fineweb.sh                        # sample and embed (GPU), then estimate (CPU)
#   STAGES=estimate J=16 bash scripts/run_fineweb.sh   # estimate only, J jobs in parallel
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

N_EXTRA=${N_EXTRA:-9}
STAGES=${STAGES:-"sample estimate"}
J=${J:-1}

GEMMA="google/gemma-2-9b-it"; QWEN="Qwen/Qwen3-8B"; LLAMA="meta-llama/Llama-3.1-8B-Instruct"
MINILM="sentence-transformers/all-MiniLM-L6-v2"; QWEN_EMB="Qwen/Qwen3-Embedding-0.6B"
export N_RUNS=10
export TAUS="0.003 0.01 0.03 0.1 0.3 1 3 10 30"
export SAMP_TS="0.428571 0.666667 0.818182 1 1.22222"    # PoolDPP-Samp at sT=1.0
export MAP_TS="500 100 50 25 10"                         # PoolDPP-MAP at sT=1.0

fs() { echo "$1" | sed 's|/|__|g'; }

sample_lm() { # sample_lm <cache> <lm> <seed>
    dpp-sample --mode fineweb --n-samples 400 --temperatures 0.6 0.8 1.0 1.2 1.5 --max-new-tokens 50 \
        --gpu-mem-util 0.90 --sample-concurrency 50 --model "$2" --cache-dir "$1" --seed "$3"
}
embed() { # embed <cache> <lm> <embedder> <suffix>
    dpp-embeddings --samples_file "$1/$(fs "$2").pkl" --model_name "$3" \
        --output_file "$1/$(fs "$2")_$4_embeddings.pkl"
}

estimate_cache() { # estimate_cache <cache> <lm_embedder> <calibration>
    local CD=$1 EMN=$2 CAL=$3
    local E="dpp-estimation --output_folder $CD/est10${EMN##*_} --model_name $EMN --cache_dir $CD --k 10"
    $E --N "$N_RUNS" --ancestral --naive --kmeans
    $E --N "$N_RUNS" --samp --target_Ts $SAMP_TS --sampling_T 1.0
    $E --N "$N_RUNS" --samp --taus $TAUS --sampling_T 1.0 --calibration "$CAL"
    $E --N 1 --map --target_Ts $MAP_TS --sampling_T 1.0
    $E --N 1 --map --taus $TAUS --calibration "$CAL"
    echo "EST_DONE $CD $EMN"
}
export -f estimate_cache

case " $STAGES " in *" sample "*)
    for LM in "$GEMMA" "$QWEN"; do
        sample_lm cache-fineweb "$LM" 0
        embed cache-fineweb "$LM" "$MINILM" minilm
        embed cache-fineweb "$LM" "$QWEN_EMB" qwen
    done
    sample_lm cache-fineweb "$LLAMA" 0
    embed cache-fineweb "$LLAMA" "$MINILM" minilm
    for i in $(seq 1 "$N_EXTRA"); do
        sample_lm "cache-fineweb-pools/pool$i" "$GEMMA" "$i"
        embed "cache-fineweb-pools/pool$i" "$GEMMA" "$MINILM" minilm
    done ;;
esac

case " $STAGES " in *" estimate "*)
    # sigma_hat comes from pool 0
    EMNS="$(fs "$GEMMA")_minilm $(fs "$GEMMA")_qwen $(fs "$QWEN")_minilm $(fs "$QWEN")_qwen $(fs "$LLAMA")_minilm"
    for EMN in $EMNS; do
        CAL="cache-fineweb/calibration_${EMN}.json"
        [ -f "$CAL" ] || dpp-estimation --calibrate --calibration "$CAL" --model_name "$EMN" --cache_dir cache-fineweb
    done
    {
        for EMN in $EMNS; do echo "cache-fineweb $EMN cache-fineweb/calibration_${EMN}.json"; done
        G="$(fs "$GEMMA")_minilm"
        for i in $(seq 1 "$N_EXTRA"); do echo "cache-fineweb-pools/pool$i $G cache-fineweb/calibration_${G}.json"; done
    } | xargs -P "$J" -L 1 bash -c 'set -e; estimate_cache "$0" "$1" "$2"' ;;
esac
