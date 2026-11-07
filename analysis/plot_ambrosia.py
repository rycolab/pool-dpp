"""
Ambrosia SQL Generation Extrinsic Evaluation Plot
==================================================
Precision (golden accuracy) vs. recall (golden-query coverage) for the Ambrosia
SQL ambiguity benchmark (Gemma-2-9B-IT, MiniLM embeddings, k=3).

Headline methods (mirrors the paper figure):
  - Ancestral, k-Means            : sweep the pool sampling temperature sT.
  - PoolDPP-Samp                  : hold sT=1.0, sweep T.
  - PoolDPP-MAP (greedy MAP DPP)  : hold sT=1.5, sweep T.
The naive DPP is not run on Ambrosia: at sT 0.6 and 0.7, about 2.7% and 1.9% of (question, pool) cells
have Zhat = 0, against 0.24% for PoolDPP-Samp at sT=1, so neither a redraw nor a zero score compares fairly.
--knob tau plots the tau grid instead of T.

T sets the target quality kappa = plm^(1/T) in both PoolDPP modes. Samp draws from the pool, so its
weight corrects for the proposal: w = kappa/q. MAP maximizes over the pool's strings, where draw
frequencies play no part: w = kappa. MAP's T therefore runs larger for the same kernel: at sT=1,
w = plm^(1/T_map) equals w = plm^(1/T_samp - 1) when T_samp = T_map / (1 + T_map).

Metrics (per generated set, after de-duplicating the SQL queries):
  - Precision = golden_hits / #unique queries          (y; "Golden Accuracy")
  - Recall    = covered_golden_sigs / total_golden_sigs (x; "Golden Query Coverage", ∈[0,1])

Multi-pool confidence intervals: each pool is an independent sample of the LM
(``cache-ambrosia/pool{i}``).  Within each
pool we average over its run dirs ``0/``..``9/``; across pools we report a CI.
This is what gives the deterministic methods (k-Means, PoolDPP-MAP) error bars —
they vary pool-to-pool even though they are deterministic within a pool.

Missing data stops the script (no pool, run, question, or point is skipped): it raises if any of
pools 0..N-1 lacks its est dir, if a stochastic method's selection file is missing from any run dir
(or pools have different numbers of run dirs), if PoolDPP-MAP's file is missing from run dir 0, if a
selection file does not cover exactly the 100 questions of --mode ambrosia-top100 (the mode of
scripts/run_ambrosia.sh) with K strings each, or if fewer than 2 pools are left for a CI. Each
selection is scored against the dataset row its prompt was built from. A golden query that fails
to run, or a question with no golden query, is an error (it used to be dropped from the golden
set, which shrank the recall denominator). A generated query that fails to run is not an error:
it scores as a miss, which is the metric. Generated queries run on the same in-memory database,
one per question, as the golden ones (unchanged). For PoolDPP-Samp and PoolDPP-MAP points, the
points JSON records "n_redrawn" and "n_prompt_pools": the number of (question, pool) pairs whose
pool spans fewer than K dimensions (Zhat = 0) and was redrawn from another pool, and the number of
pairs, summed over pools from the run-0 sidecars X.redraws.json that dpp-estimation writes next to
each selection file X.pkl.

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

from lm_dpp.utils.ambrosia_utils import get_ambrosia_prompts, load_ambrosia_df


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
AMBROSIA_CSV = os.path.join(ROOT_DIR, "ambrosia.csv")
MODEL = "google__gemma-2-9b-it_minilm"
EST_SUBDIR = "est3minilm"
PROMPT_MODE = "ambrosia-top100"                   # dpp-sample --mode of scripts/run_ambrosia.sh
K_SELECTED = 3

ST_SWEEP = [0.6, 0.7, 1.0, 1.2, 1.25, 1.3]        # Ancestral / k-Means
SAMP_ST_HELD = 1.0                                # PoolDPP-Samp anchor temperature
MAP_ST_HELD = 1.5                                 # PoolDPP-MAP: hottest pool
SAMP_TS = [0.428571, 0.538462, 0.666667, 0.818182, 1]  # w = kappa/q
MAP_TS = [10, 7.5, 3, 1]                               # w = kappa
# --knob tau: w = plm^(2 tau / sigma_hat), with no q.
SAMP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
MAP_TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


def fmt(x):
    return f"{float(x):g}"

COLORS = {"Ancestral": "#1f77b4", "$k$-Means": "#9467bd",
          "PoolDPP-Samp": "#ff7f0e", "PoolDPP-MAP": "#d62728"}
# Every point has the same small marker, drawn under its interval, so short intervals stay visible.

DF = load_ambrosia_df(AMBROSIA_CSV)


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


def pool_dirs(n_pools, subdir=EST_SUBDIR):
    """Estimate dirs of pools 0..n_pools-1: cache-ambrosia/pool{i}/{subdir}. Raises
    FileNotFoundError listing every missing pool; a missing pool is never skipped."""
    dirs = [os.path.join(ROOT_DIR, "cache-ambrosia", f"pool{i}", subdir) for i in range(n_pools)]
    missing = [i for i, d in enumerate(dirs) if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(
            f"ambrosia: {len(missing)} of {n_pools} pools missing (no {subdir} dir): "
            f"{pool_ranges(missing)}, e.g. {dirs[missing[0]]}")
    return dirs


_PROMPT_ROWS = None   # prompt -> the dataset row it was built from


def prompt_rows():
    """{prompt: dataset row} for the questions of PROMPT_MODE, built as dpp-sample builds them."""
    global _PROMPT_ROWS
    if _PROMPT_ROWS is None:
        pairs = get_ambrosia_prompts(PROMPT_MODE, DF)
        rows = {pt: DF.loc[name] for pt, name in pairs}
        if len(rows) != len(pairs):
            raise ValueError(f"{PROMPT_MODE}: {len(pairs)} questions give {len(rows)} distinct prompts")
        _PROMPT_ROWS = rows
    return _PROMPT_ROWS


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
            raise FileNotFoundError(f"{p} missing: rerun dpp-estimation "
                                    f"(STAGES=estimate bash scripts/run_ambrosia.sh) to write it")
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


def parse_question(prompt_text):
    s = prompt_text.find("and the following question:\n")
    if s < 0:
        return None
    s += len("and the following question:\n")
    e = prompt_text.find("\ngenerate the SQL query", s)
    return prompt_text[s:e if e != -1 else len(prompt_text)].strip()


_PROMPT_ENV = {}   # prompt -> (cursor, golden_sigs, {query -> signature-or-None})


def _get_env(pt, row):
    """Per-prompt in-memory DB, golden result signatures, and a query cache. Raises if a
    golden query fails to run or the question has none."""
    if pt not in _PROMPT_ENV:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.executescript(row["db_dump"])
        gs = set()
        for gq in eval(row["ambig_queries"]):
            try:
                cur.execute(gq)
            except Exception as e:
                raise ValueError(f"Ambrosia question {row['question']!r}: golden query {gq!r} "
                                 f"failed to run: {e}") from e
            gs.add(tuple(cur.fetchall()))
        if not gs:
            raise ValueError(f"Ambrosia question {row['question']!r}: no golden queries")
        _PROMPT_ENV[pt] = (cur, gs, {})
    return _PROMPT_ENV[pt]


def _exec_sig(cur, cache, query):
    """Result signature of a generated query, or None if it fails to run (a miss, by the metric)."""
    if query not in cache:
        try:
            cur.execute(query)
            cache[query] = tuple(cur.fetchall())
        except Exception:
            cache[query] = None
    return cache[query]


def per_run_prompt_metrics(items, where="selection"):
    """Return {prompt -> (precision, recall)} for one run's pkl, which must cover exactly the
    questions of PROMPT_MODE with K_SELECTED strings each.

    Both axes use whole-result matching: a generated query scores only if its
    entire execution result equals a golden query's result, the metric the
    paper text defines.
    """
    rows = prompt_rows()
    check_selection(items, rows, K_SELECTED, where)
    per_prompt = {}
    for pt, sents, _, _ in items:
        row = rows[pt]
        deduped = list(set(sents))
        cur, gs, cache = _get_env(pt, row)
        normed = [s.strip().replace("```sql\n", "").replace("\n```", "")
                  for s in deduped]
        sigs = [_exec_sig(cur, cache, nq) for nq in normed]
        hits = [s for s in sigs if s is not None and s in gs]
        per_prompt[pt] = (len(hits) / len(deduped), len(set(hits)) / len(gs))
    return per_prompt


def collect_pool_level(pdirs, method_subfolder, sweep, key_fn, ci_mode="bootstrap",
                       deterministic=False, redraws=False):
    """For each sweep value: average within each pool (over run dirs), then take
    mean + CI across pools.  x = recall, y = precision.
    deterministic: read run dir 0 only (PoolDPP-MAP). redraws: add the redraw counts of the
    run-0 sidecars (PoolDPP)."""
    out = {}
    for v in sweep:
        sub = method_subfolder(v) if callable(method_subfolder) else method_subfolder
        pool_xs, pool_ys, n_runs = [], [], set()
        for est_dir in pdirs:
            run_xs, run_ys = [], []
            paths = run_files(est_dir, sub, key_fn(v), deterministic)
            n_runs.add(len(paths))
            for p in paths:
                with open(p, "rb") as f:
                    pp = per_run_prompt_metrics(pickle.load(f), p)
                run_ys.append(np.mean([m[0] for m in pp.values()]))
                run_xs.append(np.mean([m[1] for m in pp.values()]))
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
        if redraws:
            out[v]["n_redrawn"], out[v]["n_prompt_pools"], out[v]["sampler_tier_paths"] = redraw_counts(
                pdirs, sub, key_fn(v), K_SELECTED, len(prompt_rows()))
    return out


def plot_curve(ax, name, pts, annotate=True, label_fn=str):
    if not pts:
        return
    keys = list(pts.keys())
    xs = np.array([pts[k]["x_mean"] for k in keys])
    ys = np.array([pts[k]["y_mean"] for k in keys])
    xe = np.array([pts[k]["x_ci"] for k in keys])
    ye = np.array([pts[k]["y_ci"] for k in keys])
    ax.errorbar(xs, ys, xerr=xe, yerr=ye, color=COLORS[name], linewidth=1.5, marker="o", markersize=2.5,
                capsize=2.5, capthick=0.8, elinewidth=0.8, barsabove=True, label=name)
    if annotate and len(keys) <= 10:
        for x, y, k in zip(xs, ys, keys):
            ax.annotate(label_fn(k), (x, y), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=6, color=COLORS[name])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def main(n_pools, ci_mode, out=None, points_out=None, knob="T"):
    pdirs = pool_dirs(n_pools)
    print(f"Pools: {len(pdirs)} ; ci_mode={ci_mode}")

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
        "$k$-Means": collect_pool_level(pdirs, f"{MODEL}-kmeans", ST_SWEEP,
                                        lambda v: f"s{fmt(v)}.pkl", ci_mode),
        "PoolDPP-Samp": collect_pool_level(
            pdirs, f"{MODEL}-samp", SAMP_TS if knob == "T" else SAMP_TAUS,
            lambda t: f"{'T' if knob == 'T' else 'tau'}{fmt(t)}_s{fmt(SAMP_ST_HELD)}.pkl", ci_mode,
            redraws=True),
        "PoolDPP-MAP": collect_pool_level(
            pdirs, f"{MODEL}-map", MAP_TS if knob == "T" else MAP_TAUS,
            lambda t: f"{'T' if knob == 'T' else 'tau'}{fmt(t)}_s{fmt(MAP_ST_HELD)}.pkl", ci_mode,
            deterministic=True, redraws=True),
    }
    for name in ("Ancestral", "$k$-Means"):
        plot_curve(ax, name, curves[name])
    for name in ("PoolDPP-Samp", "PoolDPP-MAP"):
        plot_curve(ax, name, curves[name], label_fn=lambda t: f"{t:.2g}")

    ax.set_xlabel(r"Recall: Golden Query Coverage $\longrightarrow$")
    ax.set_ylabel(r"Precision: Golden Accuracy $\longrightarrow$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right" if knob == "T" else "lower right", frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.4)

    output_path = out or os.path.join(SCRIPT_DIR, "ambrosia_plot.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")

    pts = {name.replace("$", ""): {fmt(k): {kk: float(vv) for kk, vv in v.items()}
                                   for k, v in c.items()}
           for name, c in curves.items()}
    pts["_meta"] = {"n_pools": len(pdirs), "n_questions": len(prompt_rows()), "ci_mode": ci_mode,
                    "n_redrawn": ("(question, pool) pairs with Zhat = 0 whose pool was redrawn, "
                                  "of n_prompt_pools, from dpp-estimation's .redraws.json sidecars"),
                    "x": "recall (golden query coverage)", "y": "precision (golden accuracy)",
                    "knob": f"sT for baselines, {knob} for PoolDPP (Samp sT=1.0, MAP sT=1.5)"}
    points_out = points_out or os.path.join(SCRIPT_DIR, "ambrosia_points.json")
    with open(points_out, "w") as f:
        json.dump(pts, f, indent=1)
    print(f"Saved {points_out}")
    for name, c in curves.items():
        for k, v in c.items():
            print(f"  {name:<14s} {fmt(k):>6s}: recall {v['x_mean']:.4f} +- {v['x_ci']:.4f}  "
                  f"precision {v['y_mean']:.4f} +- {v['y_ci']:.4f}  (pools={v['n']})"
                  + (f"  redrawn {v['n_redrawn']}/{v['n_prompt_pools']}" if "n_redrawn" in v else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-pools", type=int, default=100)
    p.add_argument("--ci-mode", choices=["bootstrap", "pool_t"], default="bootstrap")
    p.add_argument("--out", default=None)
    p.add_argument("--points-out", default=None)
    p.add_argument("--knob", choices=["T", "tau"], default="T")
    args = p.parse_args()
    main(args.n_pools, args.ci_mode, args.out, args.points_out, args.knob)
