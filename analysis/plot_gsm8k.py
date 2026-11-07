"""
GSM8K Math Reasoning Extrinsic Evaluation Plot
================================================
Accuracy vs. diversity frontier for GSM8K math reasoning (Gemma-2-9B-IT, Qwen
embeddings, k=10 selected).

Headline methods (mirrors the paper figure):
  - Ancestral, Naive-DPP, k-Means : sweep the pool sampling temperature sT.
  - PoolDPP-Samp (stochastic k-DPP): hold sT=1.2, sweep T.
  - PoolDPP-MAP  (greedy MAP DPP)  : hold sT=1.2, sweep T.
--knob tau plots the tau grid instead of T.

T sets the target quality kappa = plm^(1/T) in both PoolDPP modes. Samp draws from the pool, so its
weight corrects for the proposal: w = kappa/q. MAP maximizes over the pool's strings, where draw
frequencies play no part: w = kappa. MAP's T therefore runs larger for the same kernel: at sT=1,
w = plm^(1/T_map) equals w = plm^(1/T_samp - 1) when T_samp = T_map / (1 + T_map).

Accuracy = pass@K: fraction of questions in which at least one of the K=10
selected chain-of-thought solutions is correct.  Diversity = mean log-determinant
of the selected-solution embedding Gram matrix.

Multi-pool confidence intervals: each pool is an independent re-sample of the LM
(``cache-gsm8k`` = pool 0, ``cache-gsm8k-pool1`` = pool 1, …).  Within each pool
we average the metric over its run dirs ``0/``..``9/``; across pools we report a
CI.  This is what gives the deterministic methods (k-Means, PoolDPP-MAP) error
bars — they vary pool-to-pool even though they are deterministic within a pool.

Missing data stops the script (no pool, run, prompt, or point is skipped): it raises if any of
pools 0..N-1 lacks its est dir, if a stochastic method's selection file is missing from any run dir
(or pools have different numbers of run dirs), if PoolDPP-MAP's file is missing from run dir 0, if a
selection file does not cover exactly the 50 questions with K strings each, if a log-determinant is
not finite, or if fewer than 2 pools are left for a CI. The log-determinant adds a ridge of 1e-6 to
the Gram matrix (unchanged). For PoolDPP-Samp, PoolDPP-MAP and Naive-DPP points, the points JSON
records "n_redrawn" and "n_prompt_pools": the number of (question, pool) pairs whose pool spans fewer
than K dimensions (Zhat = 0) and was redrawn, and the number of pairs, summed over pools from the
run-0 sidecars X.redraws.json that dpp-estimation writes next to each selection file X.pkl.

Usage::

    python analysis/plot_gsm8k.py [--n-pools N] [--ci-mode bootstrap|pool_t]
                                  [--kernel baseline|instructed] [--knob T|tau]

Output: ``analysis/gsm8k_plot.pdf`` and ``analysis/gsm8k_points.json``, or
``analysis/gsm8k_instructed_plot.pdf`` and ``analysis/gsm8k_instructed_points.json``
with ``--kernel instructed``.
"""

import argparse
import json
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
# kernel -> (est subdir, model name, output stem)
KERNELS = {
    "baseline": ("est10qwen", "google__gemma-2-9b-it_qwen", "gsm8k"),
    "instructed": ("est10qwenins", "google__gemma-2-9b-it_qwenins", "gsm8k_instructed"),
}
N_QUESTIONS = 50
K_SELECTED = 10
# pass@PASSK_TARGET from the K_SELECTED chosen solutions. None -> pass@K (any of
# the K selected correct), the headline metric.
PASSK_TARGET = None

ST_SWEEP = [0.5, 0.8, 1.0, 1.2]                   # Ancestral / k-Means
NAIVE_ST_SWEEP = [0.5, 0.8, 1.0]                  # Naive-DPP (collapses at hot sT)
SAMP_ST_HELD = 1.2                                # PoolDPP-Samp: same anchor as MAP
MAP_ST_HELD = 1.2                                 # PoolDPP-MAP: accuracy peak
SAMP_TS = [0.428571, 0.666667, 0.818182, 1, 1.03, 1.05, 1.07, 1.1, 1.22222]  # w = kappa/q
MAP_TS = [2.5, 5, 10, 15, 20, 25, 50, 100, 500]                               # w = kappa
# --knob tau: w = plm^(2 tau / sigma_hat), with no q (see cache-gsm8k/calibration_*.json).
SAMP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
MAP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


def fmt(x):
    return f"{float(x):g}"

COLORS = {"Ancestral": "#1f77b4", "$k$-Means": "#9467bd", "Naive-DPP": "#2ca02c",
          "PoolDPP-Samp": "#ff7f0e", "PoolDPP-MAP": "#d62728"}
# Every point has the same small marker, drawn under its interval, so short intervals stay visible.


# ---------------------------------------------------------------------------
# Pool discovery + metrics
# ---------------------------------------------------------------------------
def pool_ranges(idx):
    """[30, 31, ..., 99] -> 'pool30..pool99'."""
    out, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(f"pool{idx[i]}" + (f"..pool{idx[j]}" if j > i else ""))
        i = j + 1
    return ", ".join(out)


def pool_dirs(n_pools, est_subdir):
    """est dirs of pools 0..n_pools-1: cache-gsm8k/pool0, cache-gsm8k/pool1, ... Raises
    FileNotFoundError listing every missing pool; a missing pool is never skipped."""
    dirs = [os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{i}", est_subdir) for i in range(n_pools)]
    missing = [i for i, d in enumerate(dirs) if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(
            f"gsm8k: {len(missing)} of {n_pools} pools missing (no {est_subdir} dir): "
            f"{pool_ranges(missing)}, e.g. {dirs[missing[0]]}")
    return dirs


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


def redraw_counts(pdirs, sub, fname, k, n_prompts):
    """(n_redrawn, n_prompt_pools, sampler_tier_paths) summed over pools from the run-0 sidecars X.redraws.json that
    dpp-estimation writes next to each selection file X.pkl (src/lm_dpp/estimation.py).
    sampler_tier_paths (None for greedy MAP) counts the (prompt, pool) kernels by the tier path of
    dpp.stable.StableKDPP ("flat" exact, "drop", "tiered", "unknown") and keeps the largest
    log10 bound on the probability mass the "drop" path leaves out."""
    n_redrawn = n_pairs = 0
    tiers = None
    for est_dir in pdirs:
        p = os.path.join(est_dir, "0", sub, os.path.splitext(fname)[0] + ".redraws.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing: rerun dpp-estimation (STAGES=estimate bash "
                                    f"scripts/run_gsm8k.sh, or scripts/run_gsm8k_instructed.sh) to write it")
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


def pass_at_k(n_correct, n_total, k):
    """Probability a random k-subset of the n_total selections contains >=1
    correct one: 1 - C(n-c, k)/C(n, k).  k=None -> binary pass@n (any correct)."""
    if n_correct == 0:
        return 0.0
    if k is None or n_total - n_correct < k:
        return 1.0
    return 1.0 - comb(n_total - n_correct, k) / comb(n_total, k)


def make_metrics(prompt_to_gold):
    def metrics(data, where="selection"):
        """Per-run mean (accuracy=pass@PASSK_TARGET, diversity=mean log-det). The selection must
        cover exactly the questions of prompt_to_gold with K_SELECTED strings each."""
        check_selection(data, prompt_to_gold, K_SELECTED, where)
        accs, lds = [], []
        for prompt, sents, _, emb in data:
            gold = prompt_to_gold[prompt]
            c = sum(1 for s in sents if is_correct(s, gold))
            accs.append(pass_at_k(c, len(sents), PASSK_TARGET))
            lds.append(compute_logdet(emb, f"{where}, prompt {prompt[:60]!r}"))
        return np.mean(accs), np.mean(lds)
    return metrics


def collect_pool_level(pdirs, method_subfolder, sweep, key_fn, metrics, ci_mode="bootstrap",
                       deterministic=False, redraws=None):
    """For each sweep value: average within each pool (over run dirs), then take
    mean + CI across pools.  x = diversity (log-det), y = accuracy (pass@K).
    deterministic: read run dir 0 only (PoolDPP-MAP). redraws: (k, n_prompts) to add the
    redraw counts of the run-0 sidecars (PoolDPP and Naive-DPP)."""
    out = {}
    for v in sweep:
        sub = method_subfolder(v) if callable(method_subfolder) else method_subfolder
        pool_xs, pool_ys, n_runs = [], [], set()
        for est_dir in pdirs:
            run_xs, run_ys = [], []
            paths = run_files(est_dir, sub, key_fn(v), deterministic)
            n_runs.add(len(paths))
            for p in paths:
                yval, xval = metrics(load_pkl(p), p)
                run_xs.append(xval)
                run_ys.append(yval)
            pool_xs.append(np.mean(run_xs))
            pool_ys.append(np.mean(run_ys))
        if len(n_runs) > 1:
            raise ValueError(f"{sub}/{key_fn(v)}: pools have different numbers of run dirs: {sorted(n_runs)}")
        n = len(pool_xs)
        if n < 2:
            raise ValueError(f"{sub}/{key_fn(v)}: {n} pool(s); a CI across pools needs at least 2")
        x_mean, y_mean = np.mean(pool_xs), np.mean(pool_ys)
        if ci_mode == "pool_t":
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
        out[v] = {"x_mean": x_mean, "y_mean": y_mean, "x_ci": x_ci, "y_ci": y_ci, "n": n,
                  "n_runs": n_runs.pop()}
        if redraws is not None:
            out[v]["n_redrawn"], out[v]["n_prompt_pools"], out[v]["sampler_tier_paths"] = redraw_counts(pdirs, sub, key_fn(v), *redraws)
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
def main(n_pools, ci_mode, kernel, knob, out=None, points_out=None):
    print("Loading GSM8K dataset...")
    df_gsm8k = load_gsm8k_dataset().head(N_QUESTIONS)
    prompt_to_gold = {build_math_prompt(r["question"]): r["answer"]
                      for _, r in df_gsm8k.iterrows()}
    if len(prompt_to_gold) != N_QUESTIONS:
        raise ValueError(f"GSM8K: {len(prompt_to_gold)} distinct prompts, expected {N_QUESTIONS}")
    metrics = make_metrics(prompt_to_gold)

    est_subdir, model, stem = KERNELS[kernel]
    pdirs = pool_dirs(n_pools, est_subdir)
    print(f"Kernel: {kernel} ; pools: {len(pdirs)} ; knob={knob} ; ci_mode={ci_mode}")

    plt.rcParams.update({
        "text.usetex": False, "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    redraws = (K_SELECTED, len(prompt_to_gold))

    def baseline(method, sweep, redraws=None):
        return collect_pool_level(pdirs, f"{model}-{method}", sweep,
                                  lambda v: f"s{fmt(v)}.pkl", metrics, ci_mode, redraws=redraws)

    def pooldpp(method, sweep, st, deterministic=False):
        return collect_pool_level(pdirs, f"{model}-{method}", sweep,
                                  lambda v: f"{knob}{fmt(v)}_s{fmt(st)}.pkl", metrics, ci_mode,
                                  deterministic=deterministic, redraws=redraws)

    curves = {
        "Ancestral": baseline("ancestral", ST_SWEEP),
        "Naive-DPP": baseline("naive", NAIVE_ST_SWEEP, redraws),
        "$k$-Means": baseline("kmeans", ST_SWEEP),
        "PoolDPP-Samp": pooldpp("samp", SAMP_TS if knob == "T" else SAMP_TAUS, SAMP_ST_HELD),
        "PoolDPP-MAP": pooldpp("map", MAP_TS if knob == "T" else MAP_TAUS, MAP_ST_HELD,
                               deterministic=True),
    }
    for name in ("Ancestral", "Naive-DPP", "$k$-Means"):
        plot_curve(ax, name, curves[name])
    for name in ("PoolDPP-Samp", "PoolDPP-MAP"):
        plot_curve(ax, name, curves[name], label_fn=lambda t: f"{t:.3g}")

    arrow = r" $\longrightarrow$"
    ax.set_xlabel("Diversity (log-determinant of Gram matrix)" + arrow)
    if PASSK_TARGET is None:
        ax.set_ylabel(f"Accuracy (pass@$K{{=}}{K_SELECTED}$)" + arrow)
    else:
        ax.set_ylabel(f"Accuracy (pass@${PASSK_TARGET}$ from $K{{=}}{K_SELECTED}$)" + arrow)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.4)

    output_path = out or os.path.join(SCRIPT_DIR, f"{stem}_plot.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")

    pts = {name.replace("$", ""): {fmt(k): {kk: float(vv) for kk, vv in v.items()}
                                   for k, v in c.items()}
           for name, c in curves.items()}
    pts["_meta"] = {"kernel": kernel, "n_pools": len(pdirs), "n_questions": len(prompt_to_gold),
                    "ci_mode": ci_mode,
                    "x": "diversity (log-det)", "y": "accuracy (pass@K)",
                    "n_redrawn": ("(question, pool) pairs with Zhat = 0 whose pool was redrawn, "
                                  "of n_prompt_pools, from dpp-estimation's .redraws.json sidecars"),
                    "knob": f"sT for baselines, {knob} for PoolDPP (sT=1.2)"}
    points_out = points_out or os.path.join(SCRIPT_DIR, f"{stem}_points.json")
    with open(points_out, "w") as f:
        json.dump(pts, f, indent=1)
    print(f"Saved {points_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-pools", type=int, default=100)
    p.add_argument("--ci-mode", choices=["bootstrap", "pool_t"], default="bootstrap")
    p.add_argument("--kernel", choices=sorted(KERNELS), default="baseline")
    p.add_argument("--knob", choices=["T", "tau"], default="T")
    p.add_argument("--out", default=None)
    p.add_argument("--points-out", default=None)
    args = p.parse_args()
    main(args.n_pools, args.ci_mode, args.kernel, args.knob, args.out, args.points_out)
