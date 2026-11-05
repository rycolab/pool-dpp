"""
Ambrosia SQL Generation Extrinsic Evaluation Plot
==================================================
Precision (golden accuracy) vs. recall (golden-query coverage) for the Ambrosia
SQL ambiguity benchmark (Gemma-2-9B-IT, MiniLM embeddings, k=3).

Headline methods (mirrors the paper figure):
  - Ancestral, Naive DPP, k-Means : sweep the pool sampling temperature sT.
  - PoolDPP-Samp                  : hold sT=1.0, sweep T (kappa = plm^(1/T)).
  - PoolDPP-MAP (greedy MAP DPP)  : hold sT=1.5, sweep T (beta = 1/(2T)).
--knob tau plots the tau grid instead of T.

Metrics (per generated set, after de-duplicating the SQL queries):
  - Precision = golden_hits / #unique queries          (y; "Golden Accuracy")
  - Recall    = covered_golden_sigs / total_golden_sigs (x; "Golden Query Coverage", ∈[0,1])

Multi-pool confidence intervals: each pool is an independent sample of the LM
(``cache-ambrosia/pool{i}``).  Within each
pool we average over its run dirs ``0/``..``9/``; across pools we report a CI.
This is what gives the deterministic methods (k-Means, PoolDPP-MAP) error bars —
they vary pool-to-pool even though they are deterministic within a pool.

Usage::

    python analysis/plot_ambrosia.py [--n-pools N] [--ci-mode bootstrap|pool_t]

Output: ``analysis/ambrosia_plot.pdf`` and the plotted points with their CIs in
``analysis/ambrosia_points.json``
"""

import argparse
import json
import os
import pickle
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from lm_dpp.utils.ambrosia_utils import load_ambrosia_df


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
AMBROSIA_CSV = os.path.join(ROOT_DIR, "ambrosia.csv")
MODEL = "google__gemma-2-9b-it_minilm"
EST_SUBDIR = "est3minilm"

ST_SWEEP = [0.6, 0.7, 1.0, 1.2, 1.25, 1.3]        # Ancestral / k-Means (skips temps not sampled)
ST_SWEEP_NAIVE = [0.6, 0.7, 1.0, 1.2, 1.25]       # Naive DPP collapses at sT>=1.3
SAMP_ST_HELD = 1.0                                # PoolDPP-Samp anchor temperature
MAP_ST_HELD = 1.5                                 # PoolDPP-MAP: hottest pool
SAMP_TS = [0.428571, 0.538462, 0.666667, 0.818182, 1]
MAP_TS = [10, 7.5, 3, 1]                          # beta = 1/(2T)
# --knob tau: the dimensionless tilt (beta = tau / sigma_hat).
SAMP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
MAP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


def fmt(x):
    return f"{float(x):g}"

COLORS = {"Ancestral": "#1f77b4", "$k$-Means": "#9467bd", "Naive DPP": "#2ca02c",
          "PoolDPP-Samp": "#ff7f0e", "PoolDPP-MAP": "#d62728"}
MARKERS = {"Ancestral": "^", "$k$-Means": "s", "Naive DPP": "v",
           "PoolDPP-Samp": "o", "PoolDPP-MAP": "D"}

DF = load_ambrosia_df(AMBROSIA_CSV)


# ---------------------------------------------------------------------------
# Pool discovery + metrics
# ---------------------------------------------------------------------------
def pool_dirs(n_pools, subdir=EST_SUBDIR):
    """Estimate dirs across pools: cache-ambrosia/pool{i}/{subdir}."""
    dirs = []
    for i in range(n_pools):
        d = os.path.join(ROOT_DIR, "cache-ambrosia", f"pool{i}", subdir)
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def parse_question(prompt_text):
    s = prompt_text.find("and the following question:\n")
    if s < 0:
        return None
    s += len("and the following question:\n")
    e = prompt_text.find("\ngenerate the SQL query", s)
    return prompt_text[s:e if e != -1 else len(prompt_text)].strip()


_PROMPT_ENV = {}   # prompt -> (cursor, golden_sigs, {query -> signature-or-None})


def _get_env(pt, row):
    """Per-prompt in-memory DB, golden result signatures, and a query cache."""
    if pt not in _PROMPT_ENV:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.executescript(row["db_dump"])
        gs = set()
        for gq in eval(row["ambig_queries"]):
            try:
                cur.execute(gq)
                gs.add(tuple(cur.fetchall()))
            except Exception:
                pass
        _PROMPT_ENV[pt] = (cur, gs, {})
    return _PROMPT_ENV[pt]


def _exec_sig(cur, cache, query):
    if query not in cache:
        try:
            cur.execute(query)
            cache[query] = tuple(cur.fetchall())
        except Exception:
            cache[query] = None
    return cache[query]


def per_run_prompt_metrics(items):
    """Return {prompt -> (precision, recall)} for one run's pkl.

    Both axes use whole-result matching: a generated query scores only if its
    entire execution result equals a golden query's result, the metric the
    paper text defines.
    """
    per_prompt = {}
    for item in items:
        if len(item) != 4:
            continue
        pt, sents, _, _ = item
        q = parse_question(pt)
        if q is None:
            continue
        m = DF[DF["question"].str.strip() == q]
        if m.empty:
            continue
        row = m.iloc[0]
        deduped = list(set(sents))
        if not deduped:
            continue
        cur, gs, cache = _get_env(pt, row)
        if not gs:
            continue
        normed = [s.strip().replace("```sql\n", "").replace("\n```", "")
                  for s in deduped]
        sigs = [_exec_sig(cur, cache, nq) for nq in normed]
        hits = [s for s in sigs if s is not None and s in gs]
        per_prompt[pt] = (len(hits) / len(deduped), len(set(hits)) / len(gs))
    return per_prompt


def collect_pool_level(pdirs, method_subfolder, sweep, key_fn, ci_mode="bootstrap"):
    """For each sweep value: average within each pool (over run dirs), then take
    mean + CI across pools.  x = recall, y = precision."""
    out = {}
    for v in sweep:
        sub = method_subfolder(v) if callable(method_subfolder) else method_subfolder
        pool_xs, pool_ys = [], []
        for est_dir in pdirs:
            run_dirs = sorted([d for d in os.listdir(est_dir)
                               if d.isdigit() and os.path.isdir(os.path.join(est_dir, d))],
                              key=int)
            run_xs, run_ys = [], []
            for r in run_dirs:
                p = os.path.join(est_dir, r, sub, key_fn(v))
                if not os.path.exists(p):
                    continue
                with open(p, "rb") as f:
                    pp = per_run_prompt_metrics(pickle.load(f))
                if not pp:
                    continue
                run_ys.append(np.mean([m[0] for m in pp.values()]))
                run_xs.append(np.mean([m[1] for m in pp.values()]))
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
    xe = np.array([pts[k]["x_ci"] for k in keys])
    ye = np.array([pts[k]["y_ci"] for k in keys])
    if sort_by_x:
        order = np.argsort(xs)
        xs, ys, xe, ye = xs[order], ys[order], xe[order], ye[order]
        keys = [keys[i] for i in order]
    ax.errorbar(xs, ys, xerr=xe, yerr=ye,
                color=COLORS[name], marker=MARKERS[name], markersize=4,
                capsize=2, capthick=0.5, elinewidth=0.6, linewidth=1.2, label=name)
    if annotate and len(keys) <= 10:
        for x, y, k in zip(xs, ys, keys):
            ax.annotate(label_fn(k), (x, y), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6, color=COLORS[name])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def main(n_pools, ci_mode, out=None, points_out=None, knob="T"):
    pdirs = pool_dirs(n_pools)
    print(f"Pools found: {len(pdirs)} ; ci_mode={ci_mode}")

    plt.rcParams.update({
        "text.usetex": False, "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    curves = {
        "Ancestral": collect_pool_level(pdirs, f"{MODEL}-ancestral", ST_SWEEP,
                                        lambda v: f"s{fmt(v)}.pkl", ci_mode),
        "Naive DPP": collect_pool_level(pdirs, f"{MODEL}-naive", ST_SWEEP_NAIVE,
                                        lambda v: f"s{fmt(v)}.pkl", ci_mode),
        "$k$-Means": collect_pool_level(pdirs, f"{MODEL}-kmeans", ST_SWEEP,
                                        lambda v: f"s{fmt(v)}.pkl", ci_mode),
        "PoolDPP-Samp": collect_pool_level(
            pdirs, f"{MODEL}-samp", SAMP_TS if knob == "T" else SAMP_TAUS,
            lambda t: f"{'T' if knob == 'T' else 'tau'}{fmt(t)}_s{fmt(SAMP_ST_HELD)}.pkl", ci_mode),
        "PoolDPP-MAP": collect_pool_level(
            pdirs, f"{MODEL}-map", MAP_TS if knob == "T" else MAP_TAUS,
            lambda t: f"{'T' if knob == 'T' else 'tau'}{fmt(t)}_s{fmt(MAP_ST_HELD)}.pkl", ci_mode),
    }
    for name in ("Ancestral", "Naive DPP", "$k$-Means"):
        plot_curve(ax, name, curves[name])
    for name in ("PoolDPP-Samp", "PoolDPP-MAP"):
        plot_curve(ax, name, curves[name], sort_by_x=True, label_fn=lambda t: f"{t:.2g}")

    ax.set_xlabel("Recall: Golden Query Coverage")
    ax.set_ylabel("Precision: Golden Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.4)

    output_path = out or os.path.join(SCRIPT_DIR, "ambrosia_plot.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")

    pts = {name.replace("$", ""): {fmt(k): {kk: float(vv) for kk, vv in v.items()}
                                   for k, v in c.items()}
           for name, c in curves.items()}
    pts["_meta"] = {"n_pools": len(pdirs), "ci_mode": ci_mode,
                    "x": "recall (golden query coverage)", "y": "precision (golden accuracy)",
                    "knob": f"sT for baselines, {knob} for PoolDPP (Samp sT=1.0, MAP sT=1.5)"}
    points_out = points_out or os.path.join(SCRIPT_DIR, "ambrosia_points.json")
    with open(points_out, "w") as f:
        json.dump(pts, f, indent=1)
    print(f"Saved {points_out}")
    for name, c in curves.items():
        for k, v in c.items():
            print(f"  {name:<14s} {fmt(k):>6s}: recall {v['x_mean']:.4f} +- {v['x_ci']:.4f}  "
                  f"precision {v['y_mean']:.4f} +- {v['y_ci']:.4f}  (pools={v['n']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-pools", type=int, default=30)
    p.add_argument("--ci-mode", choices=["bootstrap", "pool_t"], default="bootstrap")
    p.add_argument("--out", default=None)
    p.add_argument("--points-out", default=None)
    p.add_argument("--knob", choices=["T", "tau"], default="T")
    args = p.parse_args()
    main(args.n_pools, args.ci_mode, args.out, args.points_out, args.knob)
