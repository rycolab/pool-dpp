"""
GSM8K Math Reasoning Extrinsic Evaluation Plot
================================================
Accuracy vs. diversity frontier for GSM8K math reasoning (Gemma-2-9B-IT, Qwen
embeddings, k=10 selected).

Headline methods (mirrors the paper figure):
  - Ancestral, Naive-DPP, k-Means : sweep the pool sampling temperature sT.
  - PoolDPP-Samp (stochastic k-DPP): hold sT=1.2, sweep the tilt tau.
  - PoolDPP-MAP  (greedy MAP DPP)  : hold sT=1.2, sweep the tilt tau.

Accuracy = pass@K: fraction of questions in which at least one of the K=10
selected chain-of-thought solutions is correct.  Diversity = mean log-determinant
of the selected-solution embedding Gram matrix.

Multi-pool confidence intervals: each pool is an independent re-sample of the LM
(``cache-gsm8k`` = pool 0, ``cache-gsm8k-pool1`` = pool 1, …).  Within each pool
we average the metric over its run dirs ``0/``..``9/``; across pools we report a
CI.  This is what gives the deterministic methods (k-Means, PoolDPP-MAP) error
bars — they vary pool-to-pool even though they are deterministic within a pool.

Usage::

    python analysis/plot_gsm8k.py [--n-pools N] [--ci-mode bootstrap|pool_t]

Output: ``analysis/gsm8k_plot.pdf``
"""

import argparse
import os
import pickle
from math import comb

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from lm_dpp.utils.math_utils import build_math_prompt, is_correct, load_gsm8k_dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
EST_SUBDIR = "est10qwen"
MODEL = "google__gemma-2-9b-it_qwen"
N_QUESTIONS = 50
K_SELECTED = 10
# pass@PASSK_TARGET from the K_SELECTED chosen solutions. None -> pass@K (any of
# the K selected correct), the headline metric.
PASSK_TARGET = None

ST_SWEEP = [0.5, 0.8, 1.0, 1.2]                   # Ancestral / k-Means
NAIVE_ST_SWEEP = [0.5, 0.8, 1.0]                  # Naive-DPP (collapses at hot sT)
SAMP_ST_HELD = 1.2                                # PoolDPP-Samp: same anchor as MAP
MAP_ST_HELD = 1.2                                 # PoolDPP-MAP: accuracy peak
# PoolDPP-Samp / -MAP quality knob: the dimensionless tilt tau (shared grid;
# beta = tau / sigma_hat, see cache-gsm8k/calibration_*.json).
SAMP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
MAP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


def fmt(x):
    return f"{float(x):g}"

COLORS = {"Ancestral": "#1f77b4", "$k$-Means": "#9467bd", "Naive-DPP": "#2ca02c",
          "PoolDPP-Samp": "#ff7f0e", "PoolDPP-MAP": "#d62728"}
MARKERS = {"Ancestral": "^", "$k$-Means": "s", "Naive-DPP": "v",
           "PoolDPP-Samp": "o", "PoolDPP-MAP": "D"}


# ---------------------------------------------------------------------------
# Pool discovery + metrics
# ---------------------------------------------------------------------------
def pool_dirs(n_pools, est_subdir=EST_SUBDIR):
    """est dirs across pools: cache-gsm8k/pool0, cache-gsm8k/pool1, ..."""
    dirs = []
    for i in range(n_pools):
        d = os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{i}", est_subdir)
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def compute_logdet(emb):
    F = np.asarray(emb)
    K = F @ F.T + 1e-6 * np.eye(len(F))
    s, ld = np.linalg.slogdet(K)
    return ld if s > 0 else np.nan


def discover_runs(est_dir):
    return sorted([d for d in os.listdir(est_dir)
                   if d.isdigit() and os.path.isdir(os.path.join(est_dir, d))],
                  key=int)


def pass_at_k(n_correct, n_total, k):
    """Probability a random k-subset of the n_total selections contains >=1
    correct one: 1 - C(n-c, k)/C(n, k).  k=None -> binary pass@n (any correct)."""
    if n_correct == 0:
        return 0.0
    if k is None or n_total - n_correct < k:
        return 1.0
    return 1.0 - comb(n_total - n_correct, k) / comb(n_total, k)


def make_metrics(prompt_to_gold):
    def metrics(data):
        """Per-run mean (accuracy=pass@PASSK_TARGET, diversity=mean log-det)."""
        accs, lds = [], []
        for item in data:
            if len(item) != 4:
                continue
            prompt, sents, _, emb = item
            if prompt not in prompt_to_gold:
                continue
            gold = prompt_to_gold[prompt]
            c = sum(1 for s in sents if is_correct(s, gold))
            accs.append(pass_at_k(c, len(sents), PASSK_TARGET))
            lds.append(compute_logdet(emb))
        return (np.mean(accs) if accs else np.nan,
                np.nanmean(lds) if lds else np.nan)
    return metrics


def collect_pool_level(pdirs, method_subfolder, sweep, key_fn, metrics, ci_mode="bootstrap"):
    """For each sweep value: average within each pool (over run dirs), then take
    mean + CI across pools.  x = diversity (log-det), y = accuracy (pass@K)."""
    out = {}
    for v in sweep:
        sub = method_subfolder(v) if callable(method_subfolder) else method_subfolder
        pool_xs, pool_ys = [], []
        for est_dir in pdirs:
            run_xs, run_ys = [], []
            for r in discover_runs(est_dir):
                p = os.path.join(est_dir, r, sub, key_fn(v))
                if not os.path.exists(p):
                    continue
                yval, xval = metrics(load_pkl(p))
                if not (np.isnan(yval) or np.isnan(xval)):
                    run_xs.append(xval)
                    run_ys.append(yval)
            if run_xs:
                pool_xs.append(np.mean(run_xs))
                pool_ys.append(np.mean(run_ys))
        n = len(pool_xs)
        if n == 0:
            continue
        x_mean, y_mean = np.mean(pool_xs), np.mean(pool_ys)
        if n < 2:
            x_ci = y_ci = 0.0
        elif ci_mode == "pool_t":
            t = scipy_stats.t.ppf(0.975, n - 1)
            x_ci = t * np.std(pool_xs, ddof=1) / np.sqrt(n)
            y_ci = t * np.std(pool_ys, ddof=1) / np.sqrt(n)
        else:  # bootstrap percentile interval over pool means
            rng = np.random.default_rng(0)
            idx = rng.integers(0, n, size=(10000, n))
            xb = np.asarray(pool_xs)[idx].mean(axis=1)
            yb = np.asarray(pool_ys)[idx].mean(axis=1)
            x_ci = (np.quantile(xb, 0.975) - np.quantile(xb, 0.025)) / 2.0
            y_ci = (np.quantile(yb, 0.975) - np.quantile(yb, 0.025)) / 2.0
        out[v] = {"x_mean": x_mean, "y_mean": y_mean, "x_ci": x_ci, "y_ci": y_ci, "n": n}
    return out


def plot_curve(ax, name, pts, sort_by_x=False, annotate=True, label_fn=str):
    if not pts:
        return
    keys = list(pts.keys())
    xs = np.array([pts[k]["x_mean"] for k in keys])
    ys = np.array([pts[k]["y_mean"] for k in keys])
    xerr = np.array([pts[k]["x_ci"] for k in keys])
    yerr = np.array([pts[k]["y_ci"] for k in keys])
    if sort_by_x:
        order = np.argsort(xs)
        xs, ys, xerr, yerr = xs[order], ys[order], xerr[order], yerr[order]
        keys = [keys[i] for i in order]
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr,
                color=COLORS[name], marker=MARKERS[name], markersize=4,
                capsize=2, capthick=0.5, elinewidth=0.6, linewidth=1.2, label=name)
    if annotate and len(keys) <= 10:
        for x, y, k in zip(xs, ys, keys):
            ax.annotate(label_fn(k), (x, y), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6, color=COLORS[name])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
# Panels: (est subdir, model name, panel title). The instructed-kernel panel is
# drawn only when its estimation dirs exist (run scripts/run_gsm8k_instructed.sh
# after analysis/embed_instructed.py).
PANELS = [
    ("est10qwen", "google__gemma-2-9b-it_qwen", "baseline kernel (Qwen3-Emb-0.6B)"),
    ("est10qwenins", "google__gemma-2-9b-it_qwenins", "instructed kernel"),
]


def draw_panel(ax, pdirs, model, metrics, ci_mode):
    plot_curve(ax, "Ancestral",
               collect_pool_level(pdirs, f"{model}-ancestral", ST_SWEEP,
                                  lambda v: f"s{fmt(v)}.pkl", metrics, ci_mode))
    plot_curve(ax, "Naive-DPP",
               collect_pool_level(pdirs, f"{model}-naive", NAIVE_ST_SWEEP,
                                  lambda v: f"s{fmt(v)}.pkl", metrics, ci_mode))
    plot_curve(ax, "$k$-Means",
               collect_pool_level(pdirs, f"{model}-kmeans", ST_SWEEP,
                                  lambda v: f"s{fmt(v)}.pkl", metrics, ci_mode))
    plot_curve(ax, "PoolDPP-Samp",
               collect_pool_level(pdirs, f"{model}-samp", SAMP_TAUS,
                                  lambda t: f"tau{fmt(t)}_s{fmt(SAMP_ST_HELD)}.pkl", metrics, ci_mode),
               sort_by_x=True, label_fn=lambda t: f"{t:.3g}")
    plot_curve(ax, "PoolDPP-MAP",
               collect_pool_level(pdirs, f"{model}-map", MAP_TAUS,
                                  lambda t: f"tau{fmt(t)}_s{fmt(MAP_ST_HELD)}.pkl", metrics, ci_mode),
               sort_by_x=True, label_fn=lambda t: f"{t:.3g}")
    ax.set_xlabel("Diversity (log-determinant of Gram matrix)")
    ax.grid(True, alpha=0.3)


def main(n_pools, ci_mode):
    print("Loading GSM8K dataset...")
    df_gsm8k = load_gsm8k_dataset().head(N_QUESTIONS)
    prompt_to_gold = {build_math_prompt(r["question"]): r["answer"]
                      for _, r in df_gsm8k.iterrows()}
    metrics = make_metrics(prompt_to_gold)

    panels = []
    for est_subdir, model, title in PANELS:
        pdirs = pool_dirs(n_pools, est_subdir)
        if pdirs:
            panels.append((pdirs, model, title))
    print(f"Panels: {[(t, len(p)) for p, _, t in panels]} ; ci_mode={ci_mode}")

    plt.rcParams.update({
        "text.usetex": False, "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 300,
    })

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.0),
                             sharey=True, squeeze=False)
    for ax, (pdirs, model, title) in zip(axes[0], panels):
        draw_panel(ax, pdirs, model, metrics, ci_mode)
        if n > 1:
            ax.set_title(title)
    if PASSK_TARGET is None:
        axes[0][0].set_ylabel(f"Accuracy (pass@$K{{=}}{K_SELECTED}$)")
    else:
        axes[0][0].set_ylabel(f"Accuracy (pass@${PASSK_TARGET}$ from $K{{=}}{K_SELECTED}$)")
    axes[0][0].legend(loc="lower left", frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.4)

    output_path = os.path.join(SCRIPT_DIR, "gsm8k_plot.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-pools", type=int, default=20)
    p.add_argument("--ci-mode", choices=["bootstrap", "pool_t"], default="bootstrap")
    args = p.parse_args()
    main(args.n_pools, args.ci_mode)
