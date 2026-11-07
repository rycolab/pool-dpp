#!/bin/bash
# Figures and tables of the paper from existing pools and selections. CPU only, except that the
# GSM8K steps download the dataset and the embedder probe is faster with --device cuda.
# Outputs go to analysis/.
#
#   bash scripts/make_figures.sh                    # everything
#   bash scripts/make_figures.sh frontiers bounds   # some groups
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

CPUS=${CPUS:-8}
GROUPS_ALL="frontiers ablations bounds ndial weights pools significance baselines beam cost probe checks"

frontiers() {
    python analysis/plot_intrinsic.py --ci-mode pool_t
    python analysis/plot_ambrosia.py --n-pools 100 --ci-mode pool_t
    python analysis/plot_gsm8k.py --n-pools 100 --ci-mode pool_t
    python analysis/plot_gsm8k.py --n-pools 100 --ci-mode pool_t --kernel instructed
}
ablations() { python analysis/plot_ablations.py; }
bounds() {
    python analysis/bound_empirical.py --task ambrosia --stage all --Ts 0.666667 1 --n-pools 100 --workers "$CPUS"
    python analysis/bound_empirical.py --task fineweb --stage all --Ts 0.818182 1 --workers "$CPUS"
    python analysis/bound_empirical.py --task gsm8k --stage all --Ts 1 1.05 --workers "$CPUS"
    python analysis/bound_required_n.py
    python analysis/bound_tables.py
}
ndial() {  # after bounds
    python analysis/gsm8k_n2500.py
    python analysis/plot_ndial.py
}
weights() {
    python analysis/weight_diagnostics.py
    python analysis/ess_fineweb_n8000.py
    python analysis/ess_ambrosia_n6000.py
}
significance() {  # after pools: reuses the execution table of pool_ceiling.py
    python analysis/significance.py --task ambrosia --phase collect --n-pools 100
    python analysis/significance.py --task ambrosia --phase analyze
    python analysis/significance.py --task gsm8k --phase collect --n-pools 100
    python analysis/significance.py --task gsm8k --phase analyze
    GSM_VARIANT=qwenins python analysis/significance.py --task gsm8k --phase collect --n-pools 100
    GSM_VARIANT=qwenins python analysis/significance.py --task gsm8k --phase analyze
}
baselines() {
    python analysis/baselines.py --task ambrosia --n-pools 100
    python analysis/baselines.py --task fineweb --fineweb-pools cache-fineweb \
        $(for i in $(seq 1 9); do echo "cache-fineweb-pools/pool$i"; done)
    python analysis/baselines.py --task gsm8k
    python analysis/baselines_gsm8k.py
    python analysis/baselines_gsm8k.py --variant qwenins
    python analysis/heldout_diversity.py
}
beam() { for task in fineweb ambrosia gsm8k; do python analysis/beam_baselines.py --task "$task"; done; }
pools() {
    python analysis/pool_ceiling.py --part A --n-pools 100 --workers "$CPUS"
    python analysis/pool_ceiling.py --part B --n-pools 100
    python analysis/length_stratified.py
    python analysis/length_analysis.py
    python analysis/length_cap.py cap
}
cost() { python analysis/dual_vs_primal.py; }
probe() { python analysis/embedder_probe.py --n-items 600 --device "${DEVICE:-cpu}"; }
checks() {
    python analysis/zhat_zero_counts.py --workers "$CPUS"
    python analysis/pool_checks.py
}

for g in ${@:-$GROUPS_ALL}; do
    echo "=== $g"
    "$g"
done
