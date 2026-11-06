#!/bin/bash
# PoolDPP — decoding-time SWOR baselines: Stochastic Beam Search (SBS; Kool et
# al. 2019) and Conditional Poisson SBS (CPSBS; Meister et al. 2021).
#
# Unlike the pool-selection methods, each run of these decoders directly
# produces one set of K sequences sampled without replacement from the model,
# so there is no pool / estimation step. Settings mirror the corresponding
# pool experiments (same models, prompts, K, and temperature sweeps; the
# sampling temperature is each method's diversity knob, annealing the local
# conditionals exactly as in the reference implementations).
#
#   METHOD=sbs bash scripts/run_beam_baselines.sh     (or METHOD=cps, METHOD=detbs, METHOD=divbs)
#
# METHOD=detbs runs Determinantal Beam Search (Meister et al. 2021) instead:
# deterministic, so its diversity knob is the interaction weight w (w=0 is
# standard beam search at T=1), swept in one batched decode per prompt.
#
# METHOD=divbs runs Diverse Beam Search (Vijayakumar et al. 2018), the DivBS row of tab:beam_baselines:
# deterministic, on FineWeb with the chat template.
#
# Every step is resume-safe. Score with: python analysis/beam_baselines.py
set -e
cd "$(dirname "$0")/.."

METHOD=${METHOD:-sbs}
N_RUNS=${N_RUNS:-10}
GEMMA="google/gemma-2-9b-it"
QWEN="Qwen/Qwen3-8B"

echo "=== SWOR baselines: method=$METHOD ==="

if [ "$METHOD" = "divbs" ]; then
    python -m lm_dpp.sample_divbs --mode fineweb --model "$GEMMA" --chat-template \
        --k 10 --penalties 1.0 --max-new-tokens 50 --cache-dir cache-fineweb-swor
    echo "=== SWOR baselines done: method=divbs ==="
    exit 0
fi

if [ "$METHOD" = "detbs" ]; then
    # Paper (k=10 MT) sweeps w in [0.01, 0.15]; w=0 recovers beam search.
    python -m lm_dpp.sample_detbs --mode gsm8k --model "$GEMMA" \
        --k 10 --weights 0.0 0.01 0.03 0.06 0.1 0.15 \
        --max-new-tokens 2000 --cache-dir cache-gsm8k-swor
    python -m lm_dpp.sample_detbs --mode ambrosia-top100 --model "$GEMMA" \
        --chat-template --k 3 --weights 0.0 0.05 0.1 0.2 0.4 \
        --max-new-tokens 800 --cache-dir cache-ambrosia-swor
    for LM in "$GEMMA" "$QWEN"; do
        python -m lm_dpp.sample_detbs --mode fineweb --model "$LM" \
            --k 10 --weights 0.0 0.01 0.03 0.06 0.1 0.15 \
            --max-new-tokens 50 --cache-dir cache-fineweb-swor
    done
    echo "=== SWOR baselines done: method=detbs ==="
    exit 0
fi

# GSM8K: K=10, no chat template, pool temperature sweep. max-new-tokens 2000
# covers the pool length distribution (pool max 1063, p99.9 = 803).
python -m lm_dpp.sample_sbs --mode gsm8k --method "$METHOD" --model "$GEMMA" \
    --k 10 --n-runs "$N_RUNS" --temperatures 0.5 0.8 1.0 1.2 \
    --max-new-tokens 2000 --cache-dir cache-gsm8k-swor --seed 0

# Ambrosia: K=3, chat template (instruction-tuned SQL), subset of the pool
# temperature sweep.
python -m lm_dpp.sample_sbs --mode ambrosia-top100 --method "$METHOD" --model "$GEMMA" \
    --chat-template --k 3 --n-runs "$N_RUNS" --temperatures 0.6 1.0 1.2 1.5 \
    --max-new-tokens 800 --cache-dir cache-ambrosia-swor --seed 0

# FineWeb: K=10, raw continuation, both LMs, full pool temperature sweep.
for LM in "$GEMMA" "$QWEN"; do
    python -m lm_dpp.sample_sbs --mode fineweb --method "$METHOD" --model "$LM" \
        --k 10 --n-runs "$N_RUNS" --temperatures 0.6 0.8 1.0 1.2 1.5 \
        --max-new-tokens 50 --cache-dir cache-fineweb-swor --seed 0
done

echo "=== SWOR baselines done: method=$METHOD ==="
