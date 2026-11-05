"""
FineWeb Intrinsic Evaluation Plot
==================================
Quality vs. diversity frontier on the full sampling pool (Gemma-2-9B-IT, MiniLM
embeddings).  Reads estimation results from ``cache-fineweb/est10minilm/``
(pool 0, ``scripts/run_fineweb.sh``) and, when present, from the independent
pools ``cache-fineweb-pools/pool{1..9}/est10minilm/``.

Headline methods (mirrors the paper figure):
  - Ancestral, Naive-DPP, k-Means : sweep the pool sampling temperature sT.
  - PoolDPP-Samp                  : hold sT=1.0, sweep T (kappa = plm^(1/T)).
  - PoolDPP-MAP (greedy MAP DPP)  : hold sT=1.0, sweep T (beta = 1/(2T)).
--knob tau plots the tau grid instead of T.

Each ``.pkl`` is a list of ``(prompt, sample_sents, sample_logps,
sample_embeddings)`` tuples.

Confidence intervals (--ci-mode):
  pool_t (default with >= 2 pools): average each method within a pool over
      its run dirs, then a 95% t-interval across pools. Pools are independent
      LM samples, so deterministic methods (k-Means per run seed, PoolDPP-MAP)
      get error bars too.
  bootstrap: percentile interval of the mean over pool means.
  run (single pool): 95% t-interval across run dirs of one pool.

Usage::

    python analysis/plot_intrinsic.py [--est-dirs D ...] [--ci-mode pool_t]

Output: ``analysis/intrinsic_plot.pdf`` and ``analysis/intrinsic_points.json``
"""

import argparse
import glob
import json
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
EST_DIR = os.path.join(ROOT_DIR, "cache-fineweb", "est10minilm")
MODEL = "google__gemma-2-9b-it_minilm"

ST_SWEEP = [0.6, 0.8, 1.0, 1.2, 1.5]              # Ancestral / Naive / k-Means
SAMP_ST = 1.0                                     # PoolDPP-Samp: fixed pool temp
MAP_ST = 1.0                                      # PoolDPP-MAP: fixed pool temp
SAMP_TS = [0.428571, 0.666667, 0.818182, 1, 1.22222]
MAP_TS = [500, 100, 50, 25, 10]                   # beta = 1/(2T)
# --knob tau: the dimensionless tilt (beta = tau / sigma_hat).
SAMP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
MAP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


def fmt(x):
    return f"{float(x):g}"

COLORS = {"Ancestral": "#1f77b4", "$k$-Means": "#9467bd", "Naive-DPP": "#2ca02c",
          "PoolDPP-Samp": "#ff7f0e", "PoolDPP-MAP": "#d62728"}
MARKERS = {"Ancestral": "^", "$k$-Means": "s", "Naive-DPP": "v",
           "PoolDPP-Samp": "o", "PoolDPP-MAP": "D"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ci95(std, n):
    if n <= 1:
        return 0.0
    return scipy_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)


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


def metrics(data):
    """Per-pool mean (quality=mean log-prob, diversity=mean log-det)."""
    qs, ds = [], []
    for item in data:
        if len(item) != 4:
            continue
        _, sents, lp, emb = item
        if len(sents) < 2:
            continue
        qs.append(np.mean(lp))
        ds.append(compute_logdet(emb))
    return (np.nanmean(qs) if qs else np.nan,
            np.nanmean(ds) if ds else np.nan)


def default_est_dirs():
    dirs = [EST_DIR]
    extra = sorted(glob.glob(os.path.join(ROOT_DIR, "cache-fineweb-pools", "pool*",
                                          "est10minilm")),
                   key=lambda d: int(re.sub(r"\D", "", d.split(os.sep)[-2]) or 0))
    return [d for d in dirs + extra if os.path.isdir(d)]


def collect_pools(est_dirs, method_subfolder, sweep, key_fn, ci_mode):
    """Average within each pool over its run dirs, then mean + 95% CI across
    pools. Returns x/y means and CI half-widths (x = log-det, y = quality)."""
    out = {}
    for v in sweep:
        sub = method_subfolder(v) if callable(method_subfolder) else method_subfolder
        pool_xs, pool_ys = [], []
        for est_dir in est_dirs:
            xs, ys = [], []
            for r in discover_runs(est_dir):
                p = os.path.join(est_dir, r, sub, key_fn(v))
                if not os.path.exists(p):
                    continue
                yval, xval = metrics(load_pkl(p))
                if not (np.isnan(yval) or np.isnan(xval)):
                    xs.append(xval)
                    ys.append(yval)
            if xs:
                pool_xs.append(np.mean(xs))
                pool_ys.append(np.mean(ys))
        n = len(pool_xs)
        if n == 0:
            continue
        if n < 2:
            xci = yci = 0.0
        elif ci_mode == "bootstrap":
            rng = np.random.default_rng(0)
            idx = rng.integers(0, n, size=(10000, n))
            xb = np.asarray(pool_xs)[idx].mean(axis=1)
            yb = np.asarray(pool_ys)[idx].mean(axis=1)
            xci = (np.quantile(xb, 0.975) - np.quantile(xb, 0.025)) / 2.0
            yci = (np.quantile(yb, 0.975) - np.quantile(yb, 0.025)) / 2.0
        else:
            xci = ci95(np.std(pool_xs, ddof=1), n)
            yci = ci95(np.std(pool_ys, ddof=1), n)
        out[v] = {"x_mean": float(np.mean(pool_xs)), "y_mean": float(np.mean(pool_ys)),
                  "x_ci": float(xci), "y_ci": float(yci), "n": n}
    return out


def collect(est_dir, method_subfolder, sweep, key_fn):
    """For each sweep value, average the per-run metric across run dirs and
    return run-level mean/std/n (so CIs are run-level, not prompt-level)."""
    runs = discover_runs(est_dir)
    out = {}
    for v in sweep:
        sub = method_subfolder(v) if callable(method_subfolder) else method_subfolder
        xs, ys = [], []
        for r in runs:
            p = os.path.join(est_dir, r, sub, key_fn(v))
            if not os.path.exists(p):
                continue
            yval, xval = metrics(load_pkl(p))
            if not (np.isnan(yval) or np.isnan(xval)):
                xs.append(xval)
                ys.append(yval)
        if xs:
            out[v] = {"x_mean": np.mean(xs), "x_std": np.std(xs),
                      "y_mean": np.mean(ys), "y_std": np.std(ys), "n": len(xs)}
    return out


def plot_curve(ax, name, pts, annotate=True, label_fn=str):
    if not pts:
        return
    keys = list(pts.keys())
    xs = np.array([pts[k]["x_mean"] for k in keys])
    ys = np.array([pts[k]["y_mean"] for k in keys])
    if "x_ci" in pts[keys[0]]:
        xerr = np.array([pts[k]["x_ci"] for k in keys])
        yerr = np.array([pts[k]["y_ci"] for k in keys])
    else:
        xerr = np.array([ci95(pts[k]["x_std"], pts[k]["n"]) for k in keys])
        yerr = np.array([ci95(pts[k]["y_std"], pts[k]["n"]) for k in keys])
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
def main(est_dirs=None, ci_mode=None, out=None, points_out=None, knob="T"):
    est_dirs = est_dirs or default_est_dirs()
    if ci_mode is None:
        ci_mode = "pool_t" if len(est_dirs) >= 2 else "run"
    print(f"est dirs ({len(est_dirs)}): {est_dirs} ; ci_mode={ci_mode}")
    plt.rcParams.update({
        "text.usetex": False, "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    if ci_mode == "run":
        get = lambda sub, sweep, kf: collect(est_dirs[0], sub, sweep, kf)  # noqa: E731
    else:
        get = lambda sub, sweep, kf: collect_pools(est_dirs, sub, sweep, kf, ci_mode)  # noqa: E731
    tag = "T" if knob == "T" else "tau"
    curves = {
        "Ancestral": get(f"{MODEL}-ancestral", ST_SWEEP, lambda v: f"s{fmt(v)}.pkl"),
        "$k$-Means": get(f"{MODEL}-kmeans", ST_SWEEP, lambda v: f"s{fmt(v)}.pkl"),
        "PoolDPP-Samp": get(f"{MODEL}-samp", SAMP_TS if knob == "T" else SAMP_TAUS,
                            lambda t: f"{tag}{fmt(t)}_s{fmt(SAMP_ST)}.pkl"),
        "PoolDPP-MAP": get(f"{MODEL}-map", MAP_TS if knob == "T" else MAP_TAUS,
                           lambda t: f"{tag}{fmt(t)}_s{fmt(MAP_ST)}.pkl"),
    }
    plot_curve(ax, "Ancestral", curves["Ancestral"])
    plot_curve(ax, "$k$-Means", curves["$k$-Means"])
    plot_curve(ax, "PoolDPP-Samp", curves["PoolDPP-Samp"],
               label_fn=lambda t: f"{t:.2g}" if t < 10 else f"{t:g}")
    plot_curve(ax, "PoolDPP-MAP", curves["PoolDPP-MAP"],
               label_fn=lambda t: f"{t:.2g}" if t < 10 else f"{t:g}")

    ax.set_xlabel(r"Diversity (log-determinant of Gram matrix) $\longrightarrow$")
    ax.set_ylabel(r"Quality (mean log-probability) $\longrightarrow$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.4)

    output_path = out or os.path.join(SCRIPT_DIR, "intrinsic_plot.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")

    pts = {name.replace("$", ""): {fmt(k): v for k, v in c.items()}
           for name, c in curves.items()}
    pts["_meta"] = {"est_dirs": [os.path.relpath(d, ROOT_DIR) for d in est_dirs],
                    "ci_mode": ci_mode, "x": "mean log-det (MiniLM Gram)",
                    "y": "mean log p_t1", "knob": f"sT for baselines, {knob} for PoolDPP"}
    points_out = points_out or os.path.join(SCRIPT_DIR, "intrinsic_points.json")
    with open(points_out, "w") as f:
        json.dump(pts, f, indent=1)
    print(f"Saved {points_out}")
    for name, c in curves.items():
        for k, v in c.items():
            print(f"  {name:<14s} {fmt(k):>6s}: logdet {v['x_mean']:8.3f} +- "
                  f"{v.get('x_ci', float('nan')):.3f}  quality {v['y_mean']:8.3f} +- "
                  f"{v.get('y_ci', float('nan')):.3f}  (n={v['n']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--est-dirs", nargs="+", default=None,
                   help="est10minilm dirs, one per pool (default: cache-fineweb + "
                        "cache-fineweb-pools/pool*)")
    p.add_argument("--ci-mode", choices=["pool_t", "bootstrap", "run"], default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--points-out", default=None)
    p.add_argument("--knob", choices=["T", "tau"], default="T")
    a = p.parse_args()
    main([os.path.join(ROOT_DIR, d) for d in a.est_dirs] if a.est_dirs else None,
         a.ci_mode, a.out, a.points_out, a.knob)
