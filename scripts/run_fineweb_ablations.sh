#!/bin/bash
# FineWeb ablations (fig:abl_pool_size, fig:abl_k_size), CPU only, on Gemma-2-9B x MiniLM:
#   pool size N in {25, 50, 100, 150, 200} at K=10  -> cache-fineweb/est10minilm_pool{N}/
#   subset size K in {3, 5, 15, 20} at N=400        -> cache-fineweb/est{K}minilm/
# PoolDPP-Samp at T=0.82 and PoolDPP-MAP at T=50, at sT=1. Run after scripts/run_fineweb.sh. Resume-safe.
#
#   J=9 bash scripts/run_fineweb_ablations.sh
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

J=${J:-1}
export SAMP_T=0.818182 MAP_T=50
export N_RUNS=10 EMN="google__gemma-2-9b-it_minilm"

suite() { # suite <out> <k> [extra estimation args...]
    local OUT=$1 K=$2; shift 2
    local E="dpp-estimation --output_folder cache-fineweb/$OUT --model_name $EMN --cache_dir cache-fineweb --k $K"
    $E --N "$N_RUNS" --ancestral --kmeans "$@"
    $E --N "$N_RUNS" --samp --target_Ts "$SAMP_T" --sampling_T 1.0 "$@"
    $E --N 1 --map --target_Ts "$MAP_T" --sampling_T 1.0 "$@"
    echo "ABL_DONE $OUT"
}
export -f suite

{
    for N in 25 50 100 150 200; do echo "est10minilm_pool$N 10 --pool-size $N"; done
    for K in 3 5 15 20; do echo "est${K}minilm $K"; done
} | xargs -P "$J" -L 1 bash -c 'set -e; suite "$0" "$@"'
