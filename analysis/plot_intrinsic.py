"""
FineWeb Intrinsic Evaluation Plot
==================================
Quality vs. diversity frontier on the full sampling pool (Gemma-2-9B-IT, MiniLM
embeddings).  Reads estimation results from ``cache-fineweb/est10minilm/``
(pool 0, ``scripts/run_fineweb.sh``) and from the independent pools
``cache-fineweb-pools/pool{1..9}/est10minilm/`` (N_EXTRA=9 in run_fineweb.sh).

Headline methods (mirrors the paper figure):
  - Ancestral, Naive-DPP, k-Means : sweep the pool sampling temperature sT.
  - PoolDPP-Samp                  : hold sT=1.0, sweep T.
  - PoolDPP-MAP (greedy MAP DPP)  : hold sT=1.0, sweep T.
--knob tau plots the tau grid instead of T.

T sets the target quality kappa = plm^(1/T) in both PoolDPP modes. Samp draws from the pool, so its
weight corrects for the proposal: w = kappa/q. MAP maximizes over the pool's strings, where draw
frequencies play no part: w = kappa. MAP's T therefore runs larger for the same kernel: at sT=1,
w = plm^(1/T_map) equals w = plm^(1/T_samp - 1) when T_samp = T_map / (1 + T_map).

Each ``.pkl`` is a list of ``(prompt, sample_sents, sample_logps,
sample_embeddings)`` tuples.

Confidence intervals (--ci-mode):
  pool_t (default with >= 2 pools): average each method within a pool over
      its run dirs, then a 95% t-interval across pools. Pools are independent
      LM samples, so deterministic methods (k-Means per run seed, PoolDPP-MAP)
      get error bars too.
  bootstrap: percentile interval of the mean over pool means.
  run (single pool): 95% t-interval across run dirs of one pool.

Missing data stops the script (no pool, run, prompt, or point is skipped): it raises if any of
the 10 default est dirs (or a --est-dirs entry) is missing, or if cache-fineweb-pools holds a pool
outside pool1..pool9; if a stochastic method's selection file is missing from any run dir (or pools
have different numbers of run dirs); if PoolDPP-MAP's file is missing from run dir 0; if a
selection file does not cover exactly the 50 FineWeb prompts with K=10 strings each; if a
log-determinant is not finite; or if fewer than 2 pools (2 runs in run mode) are left for a CI.
In run mode PoolDPP-MAP, which is deterministic, has no interval: its points carry
"deterministic": 1 and a zero half-width. Run-mode intervals use the sample standard deviation
(ddof=1; ddof=0 before). The log-determinant adds a ridge of 1e-6 to the Gram matrix (unchanged).
For PoolDPP-Samp and PoolDPP-MAP points, the points JSON records "n_redrawn" and
"n_prompt_pools", summed over pools from the run-0 sidecars X.redraws.json that dpp-estimation
writes next to each selection file X.pkl (FineWeb runs without --redraw_dirs, so a pool with
Zhat = 0 stops dpp-estimation and n_redrawn is 0).

Usage::

    python analysis/plot_intrinsic.py [--est-dirs D ...] [--ci-mode pool_t]

Output: ``analysis/intrinsic_plot.pdf`` and ``analysis/intrinsic_points.json``
"""

import argparse
import glob
import json
import os
import pickle

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
N_EXTRA_POOLS = 9                                 # cache-fineweb-pools/pool{1..9}, N_EXTRA of run_fineweb.sh
K_SELECTED = 10

ST_SWEEP = [0.6, 0.8, 1.0, 1.2, 1.5]              # Ancestral / Naive / k-Means
SAMP_ST = 1.0                                     # PoolDPP-Samp: fixed pool temp
MAP_ST = 1.0                                      # PoolDPP-MAP: fixed pool temp
SAMP_TS = [0.428571, 0.666667, 0.818182, 1, 1.22222]  # w = kappa/q
MAP_TS = [500, 100, 50, 25, 10]                       # w = kappa
# --knob tau: w = plm^(2 tau / sigma_hat), with no q.
SAMP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
MAP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


def fmt(x):
    return f"{float(x):g}"

COLORS = {"Ancestral": "#1f77b4", "$k$-Means": "#9467bd", "Naive-DPP": "#2ca02c",
          "PoolDPP-Samp": "#ff7f0e", "PoolDPP-MAP": "#d62728"}
# Every point has the same small marker, drawn under its interval, so short intervals stay visible.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ci95(std, n):
    """Half-width of the 95% t-interval; std is the sample standard deviation (ddof=1)."""
    if n < 2:
        raise ValueError(f"a CI needs at least 2 values, got {n}")
    return scipy_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)


def load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def compute_logdet(emb, where="selection"):
    """log det(F F^T + 1e-6 I). Raises if the determinant is not positive or not finite."""
    F = np.asarray(emb)
    K = F @ F.T + 1e-6 * np.eye(len(F))
    s, ld = np.linalg.slogdet(K)
    if not (s > 0 and np.isfinite(ld)):
        raise ValueError(f"{where}: log-determinant failed (sign {s}, value {ld})")
    return ld


def discover_runs(est_dir):
    """Run dirs of est_dir, which must be exactly 0..N-1 (N >= 1)."""
    runs = sorted(int(d) for d in os.listdir(est_dir)
                  if d.isdigit() and os.path.isdir(os.path.join(est_dir, d)))
    if not runs or runs != list(range(len(runs))):
        raise FileNotFoundError(f"{est_dir}: run dirs are {runs}, expected 0..N-1")
    return [str(r) for r in runs]


def run_files(est_dir, sub, fname, deterministic):
    """Selection files of one method in one pool: run dir 0 only for a deterministic method
    (PoolDPP-MAP), every run dir otherwise. Raises FileNotFoundError on any missing file."""
    runs = ["0"] if deterministic else discover_runs(est_dir)
    paths = [os.path.join(est_dir, r, sub, fname) for r in runs]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} run files missing, e.g. {missing[0]}")
    return paths


def expected_prompts():
    from lm_dpp.utils.fineweb_prompts import prompts
    return set(prompts)


def check_selection(data, expected, k, where):
    """Every record is (prompt, sents, logps_t1, embs) with k strings, and the prompts are
    exactly `expected`, each once. Raises ValueError otherwise."""
    seen = []
    for item in data:
        if len(item) != 4:
            raise ValueError(f"{where}: a record has {len(item)} fields, expected 4")
        prompt, sents, lps, emb = item
        if not (len(sents) == len(lps) == len(emb) == k):
            raise ValueError(f"{where}: prompt {prompt[:60]!r} has {len(sents)} strings, {len(lps)} "
                             f"log-probs and {len(emb)} embeddings; expected K={k}")
        seen.append(prompt)
    dup = len(seen) - len(set(seen))
    missing, extra = set(expected) - set(seen), set(seen) - set(expected)
    if dup or missing or extra:
        ex = next(iter(missing or extra), "")
        raise ValueError(f"{where}: {len(missing)} expected prompts missing, {len(extra)} unexpected, "
                         f"{dup} duplicated (e.g. {ex[:60]!r})")


def redraw_counts(est_dirs, sub, fname, k, n_prompts):
    """(n_redrawn, n_prompt_pools, sampler_tier_paths) summed over pools from the run-0 sidecars X.redraws.json that
    dpp-estimation writes next to each selection file X.pkl (src/lm_dpp/estimation.py).
    sampler_tier_paths (None for greedy MAP) counts the (prompt, pool) kernels by the tier path of
    dpp.stable.StableKDPP ("flat" exact, "drop", "tiered", "unknown") and keeps the largest
    log10 bound on the probability mass the "drop" path leaves out."""
    n_redrawn = n_pairs = 0
    tiers = None
    for est_dir in est_dirs:
        p = os.path.join(est_dir, "0", sub, os.path.splitext(fname)[0] + ".redraws.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing: rerun dpp-estimation "
                                    f"(STAGES=estimate bash scripts/run_fineweb.sh) to write it")
        with open(p) as f:
            rec = json.load(f)
        if (rec["selection_file"] != fname or rec["k"] != k or rec["n_prompts"] != n_prompts
                or rec["n_redrawn"] != len(rec["redrawn"])):
            raise ValueError(f"{p}: does not match {fname} (k={k}, {n_prompts} prompts): "
                             f"selection_file={rec['selection_file']!r} k={rec['k']} "
                             f"n_prompts={rec['n_prompts']} n_redrawn={rec['n_redrawn']}")
        n_redrawn += rec["n_redrawn"]
        n_pairs += rec["n_prompts"]
        smp = rec["sampler"]
        if smp is not None:
            tiers = tiers or dict(n_flat=0, n_drop=0, n_tiered=0, n_unknown=0, max_log10_drop_bound=None)
            for c in ("n_flat", "n_drop", "n_tiered", "n_unknown"):
                tiers[c] += smp[c]
            if smp["max_log10_drop_bound"] is not None:
                tiers["max_log10_drop_bound"] = max(smp["max_log10_drop_bound"],
                                                    tiers["max_log10_drop_bound"] or -np.inf)
    return n_redrawn, n_pairs, tiers


def metrics(data, where="selection"):
    """Per-run mean (quality=mean log-prob, diversity=mean log-det) over the 50 prompts."""
    check_selection(data, expected_prompts(), K_SELECTED, where)
    qs, ds = [], []
    for prompt, sents, lp, emb in data:
        qs.append(np.mean(lp))
        ds.append(compute_logdet(emb, f"{where}, prompt {prompt[:60]!r}"))
    if not np.all(np.isfinite(qs)):
        raise ValueError(f"{where}: non-finite log-probabilities")
    return np.mean(qs), np.mean(ds)


def default_est_dirs():
    """cache-fineweb/est10minilm and cache-fineweb-pools/pool{1..N_EXTRA_POOLS}/est10minilm.
    Raises if one is missing, or if cache-fineweb-pools holds another pool."""
    extra = [os.path.join(ROOT_DIR, "cache-fineweb-pools", f"pool{i}", "est10minilm")
             for i in range(1, N_EXTRA_POOLS + 1)]
    dirs = [EST_DIR] + extra
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"fineweb: {len(missing)} of {len(dirs)} est dirs missing: {missing}")
    found = glob.glob(os.path.join(ROOT_DIR, "cache-fineweb-pools", "pool*", "est10minilm"))
    other = sorted(set(map(os.path.normpath, found)) - set(map(os.path.normpath, extra)))
    if other:
        raise ValueError(f"fineweb: est dirs outside pool1..pool{N_EXTRA_POOLS}: {other}; "
                         f"pass --est-dirs or change N_EXTRA_POOLS")
    return dirs


def collect_pools(est_dirs, method_subfolder, sweep, key_fn, ci_mode, deterministic=False,
                  redraws=False):
    """Average within each pool over its run dirs, then mean + 95% CI across
    pools. Returns x/y means and CI half-widths (x = log-det, y = quality).
    deterministic: read run dir 0 only (PoolDPP-MAP). redraws: add the redraw counts of the
    run-0 sidecars (PoolDPP)."""
    out = {}
    for v in sweep:
        sub = method_subfolder(v) if callable(method_subfolder) else method_subfolder
        pool_xs, pool_ys, n_runs = [], [], set()
        for est_dir in est_dirs:
            xs, ys = [], []
            paths = run_files(est_dir, sub, key_fn(v), deterministic)
            n_runs.add(len(paths))
            for p in paths:
                yval, xval = metrics(load_pkl(p), p)
                xs.append(xval)
                ys.append(yval)
            pool_xs.append(np.mean(xs))
            pool_ys.append(np.mean(ys))
        if len(n_runs) > 1:
            raise ValueError(f"{sub}/{key_fn(v)}: pools have different numbers of run dirs: {sorted(n_runs)}")
        n = len(pool_xs)
        if n < 2:
            raise ValueError(f"{sub}/{key_fn(v)}: {n} pool(s); a CI across pools needs at least 2")
        if ci_mode == "bootstrap":
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
                  "x_ci": float(xci), "y_ci": float(yci), "n": n, "n_runs": n_runs.pop()}
        if redraws:
            out[v]["n_redrawn"], out[v]["n_prompt_pools"], out[v]["sampler_tier_paths"] = redraw_counts(
                est_dirs, sub, key_fn(v), K_SELECTED, len(expected_prompts()))
    return out


def collect(est_dir, method_subfolder, sweep, key_fn, deterministic=False, redraws=False):
    """For each sweep value, average the per-run metric across run dirs and
    return run-level mean/std/n and the 95% t-interval half-width (so CIs are
    run-level, not prompt-level; std is the sample standard deviation, ddof=1).
    A deterministic method (PoolDPP-MAP) has one run: no interval, flagged in the output."""
    out = {}
    for v in sweep:
        sub = method_subfolder(v) if callable(method_subfolder) else method_subfolder
        xs, ys = [], []
        for p in run_files(est_dir, sub, key_fn(v), deterministic):
            yval, xval = metrics(load_pkl(p), p)
            xs.append(xval)
            ys.append(yval)
        n = len(xs)
        if deterministic:
            out[v] = {"x_mean": float(xs[0]), "y_mean": float(ys[0]), "x_ci": 0.0, "y_ci": 0.0,
                      "n": n, "deterministic": 1}
        else:
            x_std, y_std = float(np.std(xs, ddof=1)), float(np.std(ys, ddof=1))
            out[v] = {"x_mean": float(np.mean(xs)), "x_std": x_std,
                      "y_mean": float(np.mean(ys)), "y_std": y_std, "n": n,
                      "x_ci": float(ci95(x_std, n)), "y_ci": float(ci95(y_std, n))}
        if redraws:
            out[v]["n_redrawn"], out[v]["n_prompt_pools"], out[v]["sampler_tier_paths"] = redraw_counts(
                [est_dir], sub, key_fn(v), K_SELECTED, len(expected_prompts()))
    return out


def plot_curve(ax, name, pts, annotate=True, label_fn=str):
    if not pts:
        return
    keys = list(pts.keys())
    xs = np.array([pts[k]["x_mean"] for k in keys])
    ys = np.array([pts[k]["y_mean"] for k in keys])
    xerr = np.array([pts[k]["x_ci"] for k in keys])
    yerr = np.array([pts[k]["y_ci"] for k in keys])
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, color=COLORS[name], linewidth=1.5, marker="o", markersize=2.5,
                capsize=2.5, capthick=0.8, elinewidth=0.8, barsabove=True, label=name)
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

    missing = [d for d in est_dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"fineweb: est dirs missing: {missing}")
    if ci_mode == "run":
        if len(est_dirs) != 1:
            raise ValueError(f"--ci-mode run reads one pool; got {len(est_dirs)} est dirs")
        get = lambda sub, sweep, kf, **kw: collect(est_dirs[0], sub, sweep, kf, **kw)  # noqa: E731
    else:
        get = lambda sub, sweep, kf, **kw: collect_pools(est_dirs, sub, sweep, kf, ci_mode, **kw)  # noqa: E731
    tag = "T" if knob == "T" else "tau"
    curves = {
        "Ancestral": get(f"{MODEL}-ancestral", ST_SWEEP, lambda v: f"s{fmt(v)}.pkl"),
        "$k$-Means": get(f"{MODEL}-kmeans", ST_SWEEP, lambda v: f"s{fmt(v)}.pkl"),
        "PoolDPP-Samp": get(f"{MODEL}-samp", SAMP_TS if knob == "T" else SAMP_TAUS,
                            lambda t: f"{tag}{fmt(t)}_s{fmt(SAMP_ST)}.pkl", redraws=True),
        "PoolDPP-MAP": get(f"{MODEL}-map", MAP_TS if knob == "T" else MAP_TAUS,
                           lambda t: f"{tag}{fmt(t)}_s{fmt(MAP_ST)}.pkl",
                           deterministic=True, redraws=True),
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
                  f"{v['x_ci']:.3f}  quality {v['y_mean']:8.3f} +- "
                  f"{v['y_ci']:.3f}  (n={v['n']})")


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
