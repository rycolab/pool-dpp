#!/bin/bash
# GPU: the larger pools behind tab:weights and tab:ndial. Resume-safe.
#   cache-gsm8k-n2500-{a,b}    GSM8K, 1,250 draws per question each, sT=1.2, seeds 303 and 404
#   cache-fineweb-n8000-{a,b}  FineWeb, 4,000 draws per prompt each, sT=1.0, seeds 101 and 202
# The Ambrosia N'=6000 instance merges pools 0-29 of scripts/run_ambrosia.sh.
set -eu
cd "$(dirname "$0")/.."

GEMMA="google/gemma-2-9b-it"; MN="google__gemma-2-9b-it"
for tag_seed in a:303 b:404; do
    tag=${tag_seed%%:*}; seed=${tag_seed##*:}; CD="cache-gsm8k-n2500-$tag"
    dpp-sample --mode gsm8k --n-samples 1250 --temperatures 1.2 --max-new-tokens 4000 --max-model-len 4000 \
        --gpu-mem-util 0.90 --sample-concurrency 40 --model "$GEMMA" --cache-dir "$CD" --seed "$seed"
    dpp-embeddings --samples_file "$CD/${MN}.pkl" --model_name Qwen/Qwen3-Embedding-0.6B \
        --output_file "$CD/${MN}_qwen_embeddings.pkl"
done
for tag_seed in a:101 b:202; do
    tag=${tag_seed%%:*}; seed=${tag_seed##*:}; CD="cache-fineweb-n8000-$tag"
    dpp-sample --mode fineweb --n-samples 4000 --temperatures 1.0 --max-new-tokens 50 \
        --gpu-mem-util 0.90 --sample-concurrency 50 --model "$GEMMA" --cache-dir "$CD" --seed "$seed"
    dpp-embeddings --samples_file "$CD/${MN}.pkl" --model_name sentence-transformers/all-MiniLM-L6-v2 \
        --output_file "$CD/${MN}_minilm_embeddings.pkl"
done
