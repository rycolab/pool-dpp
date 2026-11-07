"""
Paired significance tests and the coverage/accuracy trade-off curve (tab:significance)
=========================================================================
Overlapping error bars are not a significance test: the figure's bars are
marginal CIs across pools, and every method is evaluated on the SAME pools and
the SAME questions, so the paired comparison has far less variance than the
marginals suggest. This script runs the paired test.

Two pairings, both reported:
  question-level : average each method over pools -> one value per question;
                   pair across methods on the question. n = 50 (GSM8K) / 100
                   (Ambrosia).
  pool-level     : average each method over questions -> one value per pool;
                   pair across methods on the pool. n = 100. This is the
                   unit the figure's error bars are computed over.

For each pair: mean paired difference, bootstrap 95% CI on that difference, and
a paired permutation test (random sign flips, 100k draws).

Also emitted, for C4: the trade-off curve. For every method and knob we report
the set-level metric (Ambrosia golden-query coverage / GSM8K pass@K) against the
per-item metric (Ambrosia golden accuracy / GSM8K mean per-chain accuracy), so
the per-chain drop that buys the coverage gain is visible and attributable.

Phase 1 caches per-(method, knob, pool, run, question) metrics; phase 2 runs the
tests from the cache.

Missing data stops the script (no pool, run, question, or comparison is skipped). Phase 1 raises
if any of pools 0..N-1 lacks its est dir, if a stochastic method's selection file is missing from
any run dir (or pools have different numbers of run dirs), if PoolDPP-MAP's file is missing from
run dir 0, if a selection file does not cover exactly the task's questions (Ambrosia: those of
pool_ceiling.py's signature table; GSM8K: the first 50) with K strings each, or if a selected
Ambrosia query is not in the signature table. A query the table marks as not runnable scores as a
miss (the metric). Phase 2 raises if a compared cell is missing, if the two methods of a
comparison do not share exactly the same questions or pools, or if fewer than 2 are shared. The
permutation p-value is floored at 1/n_perm, and "p_at_resolution_floor" flags where the floor
was hit; the test counts |null| >= |observed| - 1e-15 (both unchanged).

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/significance.py \
        --task ambrosia --phase collect
    OPENBLAS_NUM_THREADS=1 python analysis/significance.py \
        --task gsm8k --phase collect
    OPENBLAS_NUM_THREADS=1 python analysis/significance.py \
        --task both --phase analyze
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

AMB_MODEL = "google__gemma-2-9b-it_minilm"
AMB_EST = "est3minilm"
GSM_VARIANT = os.environ.get("GSM_VARIANT", "qwen")  # qwen or qwenins (instructed)
GSM_MODEL = f"google__gemma-2-9b-it_{GSM_VARIANT}"
GSM_EST = f"est10{GSM_VARIANT}"
SUFFIX = "" if GSM_VARIANT == "qwen" else f"_{GSM_VARIANT}"

# ---- method tables, mirroring the frontier figures ----
AMB_SAMP_TS = [0.428571, 0.538462, 0.666667, 0.818182, 1]
AMB_MAP_TS = [10, 7.5, 3, 1]
GSM_SAMP_TS = [0.428571, 0.666667, 0.818182, 1, 1.03, 1.05, 1.07, 1.1, 1.22222]
GSM_MAP_TS = [2.5, 5, 10, 15, 20, 25, 50, 100, 500]


AMB_K, GSM_K = 3, 10


def fmt(x):
    return f"{float(x):g}"


# methods run once (dpp-estimation --N 1): only run dir 0 holds their selections
DETERMINISTIC = {"PoolDPP-MAP"}

AMB_METHODS = (
    [("Ancestral", sT, f"{AMB_MODEL}-ancestral", f"s{fmt(sT)}.pkl")
     for sT in [0.6, 0.7, 1.0, 1.2, 1.25, 1.3]]
    + [("k-Means", sT, f"{AMB_MODEL}-kmeans", f"s{fmt(sT)}.pkl")
       for sT in [0.6, 0.7, 1.0, 1.2, 1.25, 1.3]]
    + [("PoolDPP-Samp", T, f"{AMB_MODEL}-samp", f"T{fmt(T)}_s1.pkl") for T in AMB_SAMP_TS]
    + [("PoolDPP-MAP", T, f"{AMB_MODEL}-map", f"T{fmt(T)}_s1.5.pkl") for T in AMB_MAP_TS]
)

GSM_METHODS = (
    [("Ancestral", sT, f"{GSM_MODEL}-ancestral", f"s{fmt(sT)}.pkl")
     for sT in [0.5, 0.8, 1.0, 1.2]]
    + [("k-Means", sT, f"{GSM_MODEL}-kmeans", f"s{fmt(sT)}.pkl")
       for sT in [0.5, 0.8, 1.0, 1.2]]
    + [("Naive-DPP", sT, f"{GSM_MODEL}-naive", f"s{fmt(sT)}.pkl")
       for sT in [0.5, 0.8, 1.0]]
    + [("PoolDPP-Samp", T, f"{GSM_MODEL}-samp", f"T{fmt(T)}_s1.2.pkl") for T in GSM_SAMP_TS]
    + [("PoolDPP-MAP", T, f"{GSM_MODEL}-map", f"T{fmt(T)}_s1.2.pkl") for T in GSM_MAP_TS]
)


def normalize(q):
    return q.strip().replace("```sql\n", "").replace("\n```", "")


def parse_question_amb(pt):
    s = pt.find("and the following question:\n")
    if s < 0:
        return None
    s += len("and the following question:\n")
    e = pt.find("\ngenerate the SQL query", s)
    return pt[s:e if e != -1 else len(pt)].strip()


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


def est_dirs(task, cache, est_sub, n_pools):
    """est dirs of pools 0..n_pools-1; raises FileNotFoundError listing every missing pool."""
    dirs = [os.path.join(ROOT_DIR, cache, f"pool{pi}", est_sub) for pi in range(n_pools)]
    missing = [pi for pi, d in enumerate(dirs) if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"{task}: {len(missing)} of {n_pools} pools missing (no {est_sub} dir): "
                                f"{pool_ranges(missing)}, e.g. {dirs[missing[0]]}")
    return dirs


def run_files(est, sub, fname, deterministic):
    """[(run, path)] of one method in one pool: run dir 0 only for a deterministic method, every
    run dir otherwise (which must be exactly 0..N-1). Raises FileNotFoundError on a missing file."""
    if deterministic:
        runs = [0]
    else:
        runs = sorted(int(x) for x in os.listdir(est) if x.isdigit())
        if not runs or runs != list(range(len(runs))):
            raise FileNotFoundError(f"{est}: run dirs are {runs}, expected 0..N-1")
    out = [(r, os.path.join(est, str(r), sub, fname)) for r in runs]
    missing = [p for _, p in out if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(out)} run files missing, e.g. {missing[0]}")
    return out


def check_selection(items, expected, k, where, key=lambda pt: pt):
    """Every record is (prompt, sents, logps_t1, embs) with k strings, and key(prompt) runs over
    exactly `expected`, each once. Raises ValueError otherwise."""
    seen = []
    for item in items:
        if len(item) != 4:
            raise ValueError(f"{where}: a record has {len(item)} fields, expected 4")
        pt, sents, lps, emb = item
        if not (len(sents) == len(lps) == len(emb) == k):
            raise ValueError(f"{where}: prompt {pt[:60]!r} has {len(sents)} strings, {len(lps)} "
                             f"log-probs and {len(emb)} embeddings; expected K={k}")
        kk = key(pt)
        if kk is None:
            raise ValueError(f"{where}: cannot parse the question of prompt {pt[:60]!r}")
        seen.append(kk)
    dup = len(seen) - len(set(seen))
    missing, extra = set(expected) - set(seen), set(seen) - set(expected)
    if dup or missing or extra:
        ex = next(iter(missing or extra), "")
        raise ValueError(f"{where}: {len(missing)} expected questions missing, {len(extra)} unexpected, "
                         f"{dup} duplicated (e.g. {ex[:60]!r})")


# ---------------------------------------------------------------------------
# Phase 1: collect
# ---------------------------------------------------------------------------
def collect_ambrosia(n_pools):
    """Uses the signature table of pool_ceiling.py, so no SQL re-execution."""
    with open(os.path.join(SCRIPT_DIR, "pool_ceiling_sigtable.pkl"), "rb") as f:
        d = pickle.load(f)
    golden, sig_tables = d["golden"], d["sig_tables"]

    dirs = est_dirs("ambrosia", "cache-ambrosia", AMB_EST, n_pools)
    rows = []
    for name, knob, sub, fname in AMB_METHODS:
        n_runs = set()
        for pi, est in enumerate(dirs):
            files = run_files(est, sub, fname, name in DETERMINISTIC)
            n_runs.add(len(files))
            for r, p in files:
                with open(p, "rb") as f:
                    items = pickle.load(f)
                check_selection(items, golden, AMB_K, p, key=parse_question_amb)
                for pt, sents, _, _ in items:
                    q = parse_question_amb(pt)
                    gs, tbl = golden[q], sig_tables[q]
                    ded = list(set(sents))
                    absent = [s for s in ded if normalize(s) not in tbl]
                    if absent:
                        raise KeyError(f"{p}: question {q[:60]!r}: {len(absent)} selected queries are not in "
                                       f"the signature table of pool_ceiling.py (e.g. {absent[0][:80]!r}); "
                                       f"rerun pool_ceiling.py --part A on the same pools")
                    sigs = [tbl[normalize(s)] for s in ded]
                    covered = set(sigs) & gs
                    n_gold_hits = sum(1 for s in sigs if s in gs)
                    rows.append((name, str(knob), pi, int(r), q,
                                 len(covered) / len(gs),        # set: coverage
                                 n_gold_hits / len(ded)))       # item: precision
        if len(n_runs) > 1:
            raise ValueError(f"ambrosia {name} knob={knob}: pools have different numbers of run dirs: "
                             f"{sorted(n_runs)}")
        print(f"  collected {name} knob={knob}", flush=True)
    return rows


def collect_gsm8k(n_pools):
    from lm_dpp.utils.math_utils import build_math_prompt, is_correct, load_gsm8k_dataset
    # mirrors analysis/plot_gsm8k.py: the headline uses the first 50 questions
    df = load_gsm8k_dataset().head(50)
    prompt_to_gold = {build_math_prompt(r["question"]): r["answer"]
                      for _, r in df.iterrows()}
    if len(prompt_to_gold) != 50:
        raise ValueError(f"GSM8K: {len(prompt_to_gold)} distinct prompts, expected 50")

    dirs = est_dirs("gsm8k", "cache-gsm8k", GSM_EST, n_pools)
    rows = []
    for name, knob, sub, fname in GSM_METHODS:
        n_runs = set()
        for pi, est in enumerate(dirs):
            files = run_files(est, sub, fname, name in DETERMINISTIC)
            n_runs.add(len(files))
            for r, p in files:
                with open(p, "rb") as f:
                    items = pickle.load(f)
                check_selection(items, prompt_to_gold, GSM_K, p)
                for pt, sents, _, _ in items:
                    gold = prompt_to_gold[pt]
                    c = sum(1 for s in sents if is_correct(s, gold))
                    rows.append((name, str(knob), pi, int(r), pt,
                                 1.0 if c > 0 else 0.0,          # set: pass@K
                                 c / len(sents)))                # item: per-chain
        if len(n_runs) > 1:
            raise ValueError(f"gsm8k {name} knob={knob}: pools have different numbers of run dirs: "
                             f"{sorted(n_runs)}")
        print(f"  collected {name} knob={knob}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Phase 2: analyze
# ---------------------------------------------------------------------------
def to_cells(rows):
    """-> {(method, knob): {(pool, question): (set_metric, item_metric)}}
    averaging over run dirs."""
    acc = defaultdict(lambda: defaultdict(list))
    for name, knob, pi, r, q, set_m, item_m in rows:
        acc[(name, knob)][(pi, q)].append((set_m, item_m))
    out = {}
    for mk, d in acc.items():
        out[mk] = {pq: (float(np.mean([v[0] for v in vs])),
                        float(np.mean([v[1] for v in vs])))
                   for pq, vs in d.items()}
    return out


def paired_test(a, b, n_perm=100000, seed=0):
    """a, b are aligned arrays. Returns mean diff, bootstrap CI, permutation p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 2:
        raise ValueError(f"a paired test needs at least 2 pairs, got {n}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(10000, n))
    bs = d[idx].mean(axis=1)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= abs(obs) - 1e-15).mean())
    # a permutation p-value cannot resolve below 1/n_perm; report the floor
    # rather than a literal 0, and flag it
    p_floored = max(p, 1.0 / n_perm)
    return {"n": int(n), "mean_diff": float(obs),
            "ci_lo": float(np.quantile(bs, 0.025)),
            "ci_hi": float(np.quantile(bs, 0.975)),
            "p": p_floored,
            "p_at_resolution_floor": bool(p == 0.0),
            "n_perm": n_perm}


def analyze(task, cells, comparisons, matched_note):
    # per (method, knob): question-level and pool-level aggregates
    q_level, p_level, summary = {}, {}, {}
    for mk, d in cells.items():
        by_q, by_p = defaultdict(list), defaultdict(list)
        for (pi, q), (s, it) in d.items():
            by_q[q].append((s, it))
            by_p[pi].append((s, it))
        q_level[mk] = {q: (float(np.mean([v[0] for v in vs])),
                           float(np.mean([v[1] for v in vs])))
                       for q, vs in by_q.items()}
        p_level[mk] = {pi: (float(np.mean([v[0] for v in vs])),
                            float(np.mean([v[1] for v in vs])))
                       for pi, vs in by_p.items()}
        allv = list(d.values())
        summary[f"{mk[0]}|{mk[1]}"] = {
            "set_metric": float(np.mean([v[0] for v in allv])),
            "item_metric": float(np.mean([v[1] for v in allv])),
            "n_cells": len(allv),
        }

    def best_knob(name):
        ks = [mk for mk in cells if mk[0] == name]
        if not ks:
            return None
        return max(ks, key=lambda mk: np.mean([v[0] for v in cells[mk].values()]))

    units = {mk: set(d) for mk, d in cells.items()}
    if len({frozenset(u) for u in units.values()}) != 1:
        raise ValueError(f"{task}: methods cover different (pool, question) cells")
    pq = next(iter(units.values()))
    out = {"matched_note": matched_note, "per_method_knob": summary,
           "n_pools": len({pi for pi, _ in pq}), "n_questions": len({q for _, q in pq}),
           "tests": {}}
    for label, mk_a, mk_b in comparisons:
        absent = [mk for mk in (mk_a, mk_b) if mk not in cells]
        if absent:
            raise KeyError(f"{task}: {label}: no cells for {absent}")
        res = {}
        for lvl, tbl in (("question_level", q_level), ("pool_level", p_level)):
            if set(tbl[mk_a]) != set(tbl[mk_b]):
                raise ValueError(f"{task}: {label}: {lvl}: {mk_a} and {mk_b} cover different units "
                                 f"({len(tbl[mk_a])} vs {len(tbl[mk_b])}, "
                                 f"{len(set(tbl[mk_a]) ^ set(tbl[mk_b]))} not shared)")
            keys = sorted(tbl[mk_a])
            for mi, mname in ((0, "set_metric"), (1, "item_metric")):
                a = [tbl[mk_a][k][mi] for k in keys]
                b = [tbl[mk_b][k][mi] for k in keys]
                res[f"{lvl}|{mname}"] = paired_test(a, b)
        out["tests"][label] = res
        q = res["question_level|set_metric"]
        pl = res["pool_level|set_metric"]
        qi = res["question_level|item_metric"]
        print(f"  {label}")
        print(f"      set  metric: question-level d={q['mean_diff']:+.4f} "
              f"[{q['ci_lo']:+.4f},{q['ci_hi']:+.4f}] "
              f"p={q['p']:.4g} (n={q['n']}) | "
              f"pool-level d={pl['mean_diff']:+.4f} "
              f"p={pl['p']:.4g} (n={pl['n']})")
        print(f"      item metric: question-level d={qi['mean_diff']:+.4f} "
              f"[{qi['ci_lo']:+.4f},{qi['ci_hi']:+.4f}] "
              f"p={qi['p']:.4g}")
    out["best_knob"] = {n: list(best_knob(n)) if best_knob(n) else None
                        for n in {mk[0] for mk in cells}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ambrosia", "gsm8k", "both"], required=True)
    ap.add_argument("--phase", choices=["collect", "analyze"], required=True)
    ap.add_argument("--n-pools", type=int, default=None)
    a = ap.parse_args()

    tasks = ["ambrosia", "gsm8k"] if a.task == "both" else [a.task]

    for task in tasks:
        cache = os.path.join(SCRIPT_DIR, f"significance_cells_{task}{SUFFIX if task == 'gsm8k' else ''}.pkl")
        if a.phase == "collect":
            n = a.n_pools or 100
            print(f"=== collecting {task} ({n} pools) ===", flush=True)
            rows = (collect_ambrosia(n) if task == "ambrosia"
                    else collect_gsm8k(n))
            with open(cache, "wb") as f:
                pickle.dump(rows, f)
            print(f"Saved {cache} ({len(rows)} rows)")
            continue

        with open(cache, "rb") as f:
            rows = pickle.load(f)
        cells = to_cells(rows)
        print(f"\n=== {task}: paired tests ===")
        if task == "ambrosia":
            comparisons = [
                # each baseline at its own best-recall local temperature (1.0)
                ("Samp(T=0.666667,sT=1.0) vs Ancestral(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "0.666667"), ("Ancestral", "1.0")),
                ("Samp(T=0.666667,sT=1.0) vs k-Means(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "0.666667"), ("k-Means", "1.0")),
                ("Samp(T=0.818182,sT=1.0) vs Ancestral(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "0.818182"), ("Ancestral", "1.0")),
                ("Samp(T=0.818182,sT=1.0) vs k-Means(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "0.818182"), ("k-Means", "1.0")),
                ("MAP(T=10,sT=1.5) vs k-Means(sT=1.0) [k-means best recall]",
                 ("PoolDPP-MAP", "10"), ("k-Means", "1.0")),
                ("MAP(T=10,sT=1.5) vs Ancestral(sT=1.0) [ancestral best recall]",
                 ("PoolDPP-MAP", "10"), ("Ancestral", "1.0")),
                ("MAP(T=10,sT=1.5) vs Samp(T=0.818182,sT=1.0)",
                 ("PoolDPP-MAP", "10"), ("PoolDPP-Samp", "0.818182")),
            ]
            note = ("PoolDPP-MAP is anchored at sT=1.5 and PoolDPP-Samp at sT=1.0, as in the "
                    "figure, so 'matched' applies to the Samp comparisons only; MAP is compared "
                    "against each baseline at that baseline's best sT.")
        else:
            comparisons = [
                ("Samp(T=1.05) vs Ancestral(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-Samp", "1.05"), ("Ancestral", "1.2")),
                ("Samp(T=1) vs Ancestral(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-Samp", "1"), ("Ancestral", "1.2")),
                ("Samp(T=1.05) vs k-Means(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-Samp", "1.05"), ("k-Means", "1.2")),
                ("MAP(T=500) vs Ancestral(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-MAP", "500"), ("Ancestral", "1.2")),
                ("MAP(T=500) vs k-Means(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-MAP", "500"), ("k-Means", "1.2")),
                ("MAP(T=10) vs Ancestral(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-MAP", "10"), ("Ancestral", "1.2")),
                ("MAP(T=10) vs k-Means(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-MAP", "10"), ("k-Means", "1.2")),
                ("MAP(T=500) vs MAP(T=10)",
                 ("PoolDPP-MAP", "500"), ("PoolDPP-MAP", "10")),
            ]
            note = ("All PoolDPP variants are anchored at sT=1.2, so every comparison against a "
                    "baseline at sT=1.2 is exactly matched in sampling temperature.")
        res = analyze(task, cells, comparisons, note)
        dest = os.path.join(SCRIPT_DIR, f"significance_{task}{SUFFIX if task == 'gsm8k' else ''}.json")
        with open(dest, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved {dest}")

        print(f"\n--- {task}: trade-off curve (set metric vs item metric) ---")
        for k, v in sorted(res["per_method_knob"].items()):
            print(f"  {k:<28s} set={v['set_metric']:.4f}  item={v['item_metric']:.4f}")


if __name__ == "__main__":
    main()
