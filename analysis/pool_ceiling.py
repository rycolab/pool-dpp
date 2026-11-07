"""
Ambrosia pool ceiling and recall against the achievable ceiling (tab:pool-ceiling)
=====================================================================
How many gold interpretations appear anywhere in the pool, and how much recall
does that leave the selection methods (tab:pool-ceiling)?

Pool items are matched to gold interpretations by **execution equivalence**: a
sample covers a gold interpretation iff executing it against the question's
SQLite dump yields exactly the gold query's result signature.

Part A  pool coverage at N in {25, 100, 200}
Part B  selected-set coverage as a fraction of pool coverage, per method:
        recall against the achievable ceiling rather than against 1.0.

Both parts share one signature table: every unique SQL string is executed
exactly once per question and cached to disk.

Failures stop the run with an error that names the pool, question, or file:
a missing pool, est dir, run dir, or selection file; a pool without a
question of the ambrosia-top100 set, or without one of the sweep's sT; a
question that is not in ambrosia.csv or appears there twice; a database dump
or a gold query that does not execute; a pool smaller than N; a selection
that is not K strings or does not cover the 100 questions; a selected string
missing from the signature table; fewer than 2 pools for an interval. A
generated SQL string that does not execute is part of the metric (it covers
no gold interpretation); their number is written as n_sql_not_runnable.
recall_vs_ceiling is undefined on a question whose pool ceiling is 0; those
questions are left out of that ratio only, and their number is written as
n_questions_zero_ceiling. The ceiling of a question is its N=200 ceiling
averaged over pools (per_question_full_pool), not the ceiling of the pool
the selection came from.

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/pool_ceiling.py \
        --part A [--n-pools 100] [--workers 4]
    OPENBLAS_NUM_THREADS=1 python analysis/pool_ceiling.py \
        --part B [--n-pools 100]
"""

import argparse
import json
import os
import pickle
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from lm_dpp.utils.ambrosia_utils import load_ambrosia_df

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
AMBROSIA_CSV = os.path.join(ROOT_DIR, "ambrosia.csv")
RAW_PKL = "google__gemma-2-9b-it.pkl"
MODEL = "google__gemma-2-9b-it_minilm"
EST_SUBDIR = "est3minilm"
SIGTABLE = os.path.join(SCRIPT_DIR, "pool_ceiling_sigtable.pkl")

ST_SWEEP = [0.6, 0.7, 1.0, 1.2, 1.25, 1.3, 1.5]
N_SWEEP = [25, 100, 200]          # 200 = full Ambrosia pool
N_SUBSAMPLE_DRAWS = 10            # random draws averaged for N < full pool

# Method sweeps, mirroring analysis/plot_ambrosia.py
SAMP_TS = [0.428571, 0.538462, 0.666667, 0.818182, 1]
MAP_TS = [10, 7.5, 3, 1]
SAMP_ST_HELD = 1.0
MAP_ST_HELD = 1.5
ST_SWEEP_BASE = [0.6, 0.7, 1.0, 1.2, 1.25, 1.3]
CACHE = os.environ.get("E1_CACHE", "cache-ambrosia")
K = 3                             # selected strings per set (run_ambrosia.sh)
N_RUNS = 10                       # run dirs of the stochastic methods (run_ambrosia.sh)


def fmt(x):
    return f"{float(x):g}"


def parse_question(prompt_text):
    s = prompt_text.find("and the following question:\n")
    if s < 0:
        return None
    s += len("and the following question:\n")
    e = prompt_text.find("\ngenerate the SQL query", s)
    return prompt_text[s:e if e != -1 else len(prompt_text)].strip()


def normalize(q):
    return q.strip().replace("```sql\n", "").replace("\n```", "")


def pool_path(pool_idx):
    return os.path.join(ROOT_DIR, CACHE, f"pool{pool_idx}", RAW_PKL)


def require_pools(paths, what):
    """Raise FileNotFoundError listing every missing path of {pool index: path}."""
    missing = [i for i, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"ambrosia: {len(missing)} of {len(paths)} pools have no {what}: "
            + ", ".join(f"pool{i}" for i in missing)
            + f" (first missing path: {paths[missing[0]]})")


def load_pool(pool_idx):
    """-> {question: {sT: [sql strings]}} for one pool, at every sT of ST_SWEEP."""
    path = pool_path(pool_idx)
    with open(path, "rb") as f:
        raw = pickle.load(f)
    missing_T = [sT for sT in ST_SWEEP if sT not in raw]
    if missing_T:
        raise ValueError(f"ambrosia pool{pool_idx}: no draws at sT={missing_T} in {path}")
    out = defaultdict(dict)
    for sT, per_prompt in raw.items():
        if sT not in ST_SWEEP:   # temperatures this table does not report
            continue
        for prompt_text, entry in per_prompt.items():
            q = parse_question(prompt_text)
            if q is None:
                raise ValueError(f"ambrosia pool{pool_idx} sT={sT}: prompt without a question: "
                                 f"{prompt_text[:60]!r}")
            out[q][sT] = list(entry[0])
    return dict(out)


def signatures_for_question(args):
    """Execute every unique SQL string for one question exactly once."""
    question, db_dump, ambig_queries_str, unique_sqls = args
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    try:
        cur.executescript(db_dump)
    except Exception as e:
        raise RuntimeError(f"ambrosia question {question!r}: the database dump does not load: {e}") from e

    golden_sigs = set()
    for gq in eval(ambig_queries_str):
        try:
            cur.execute(gq)
        except Exception as e:
            raise RuntimeError(f"ambrosia question {question!r}: gold query does not execute "
                               f"({e}): {gq!r}") from e
        golden_sigs.add(tuple(cur.fetchall()))
    if not golden_sigs:
        raise ValueError(f"ambrosia question {question!r}: no gold queries")

    sig_of = {}
    for s in unique_sqls:
        try:
            cur.execute(s)
            sig_of[s] = tuple(cur.fetchall())
        except Exception:
            sig_of[s] = None   # not runnable: covers no gold interpretation (the metric)
    conn.close()
    return question, golden_sigs, sig_of


# ---------------------------------------------------------------------------
# Part A: pool coverage
# ---------------------------------------------------------------------------
def part_a(n_pools, workers):
    df = load_ambrosia_df(AMBROSIA_CSV)
    by_question = {str(r["question"]).strip(): r for _, r in df.iterrows()}
    expected = expected_questions(df)

    require_pools({i: pool_path(i) for i in range(n_pools)}, RAW_PKL)
    print(f"Loading {n_pools} pools ...", flush=True)
    pools = {}
    for i in range(n_pools):
        pools[i] = load_pool(i)
        for sT in ST_SWEEP:
            got = {q for q, per_t in pools[i].items() if sT in per_t}
            if got != expected:
                raise ValueError(f"ambrosia pool{i} sT={sT}: {len(expected - got)} of the "
                                 f"{len(expected)} ambrosia-top100 questions missing, "
                                 f"{len(got - expected)} unexpected")
    print(f"Pools loaded: {len(pools)}", flush=True)

    unique_by_q = defaultdict(set)
    for pool in pools.values():
        for q, per_t in pool.items():
            for sents in per_t.values():
                unique_by_q[q].update(normalize(s) for s in sents)

    jobs = []
    for q, sqls in unique_by_q.items():
        row = by_question[q]   # expected_questions checked that q is in ambrosia.csv once
        jobs.append((q, row["db_dump"], row["ambig_queries"], sorted(sqls)))
    print(f"Questions: {len(jobs)}; unique SQL strings: "
          f"{sum(len(j[3]) for j in jobs)}", flush=True)

    sig_tables, golden = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (q, gs, tbl) in enumerate(ex.map(signatures_for_question, jobs), 1):
            golden[q], sig_tables[q] = gs, tbl
            if n % 20 == 0 or n == len(jobs):
                print(f"  executed {n}/{len(jobs)} questions", flush=True)
    print(f"Questions with usable golden queries: {len(golden)}", flush=True)

    with open(SIGTABLE, "wb") as f:
        pickle.dump({"golden": golden, "sig_tables": sig_tables}, f)
    print(f"Cached signature table -> {SIGTABLE}", flush=True)

    # ceiling[N][sT][pool] = mean over questions of covered/total
    rng = np.random.default_rng(0)
    ceiling = {N: defaultdict(lambda: defaultdict(list)) for N in N_SWEEP}
    hist = {N: defaultdict(lambda: {"all": 0, "partial": 0, "none": 0, "n": 0})
            for N in N_SWEEP}
    per_q_full = defaultdict(lambda: defaultdict(list))

    for pool_idx, pool in pools.items():
        for q, per_t in pool.items():
            gs, tbl = golden[q], sig_tables[q]
            for sT, sents in per_t.items():
                sigs = [tbl[normalize(s)] for s in sents]
                for N in N_SWEEP:
                    if N > len(sigs):
                        raise ValueError(f"ambrosia pool{pool_idx} sT={sT} question {q!r}: "
                                         f"{len(sigs)} draws, fewer than N={N}")
                    if N == len(sigs):
                        fracs = [len(set(sigs) & gs) / len(gs)]
                    else:
                        fracs = []
                        for _ in range(N_SUBSAMPLE_DRAWS):
                            idx = rng.choice(len(sigs), size=N, replace=False)
                            sub = {sigs[i] for i in idx}
                            fracs.append(len(sub & gs) / len(gs))
                    frac = float(np.mean(fracs))
                    ceiling[N][sT][pool_idx].append(frac)
                    h = hist[N][sT]
                    h["n"] += 1
                    if frac >= 1.0:
                        h["all"] += 1
                    elif frac > 0.0:
                        h["partial"] += 1
                    else:
                        h["none"] += 1
                    if N == max(N_SWEEP):
                        per_q_full[sT][q].append(frac)

    boot = np.random.default_rng(0)
    summary = {}
    for N in N_SWEEP:
        summary[str(N)] = {}
        for sT in sorted(ceiling[N]):
            pm = np.array([np.mean(v) for v in ceiling[N][sT].values()])
            n = len(pm)
            if n < 2:
                raise ValueError(f"ambrosia N={N} sT={sT}: {n} pool(s), need at least 2 for an interval")
            idx = boot.integers(0, n, size=(10000, n))
            bs = pm[idx].mean(axis=1)
            ci = float((np.quantile(bs, .975) - np.quantile(bs, .025)) / 2)
            h = hist[N][sT]
            summary[str(N)][str(sT)] = {
                "ceiling_recall_mean": float(pm.mean()),
                "ceiling_recall_ci95": ci,
                "n_pools": int(n),
                "frac_questions_all_modes": h["all"] / max(h["n"], 1),
                "frac_questions_no_mode": h["none"] / max(h["n"], 1),
            }

    n_gold = {q: len(g) for q, g in golden.items()}
    out = {
        "n_pools": len(pools),
        "n_questions": len(golden),
        "n_unique_sql": int(sum(len(t) for t in sig_tables.values())),
        "n_sql_not_runnable": int(sum(v is None for t in sig_tables.values() for v in t.values())),
        "n_golden_sigs_distribution": {
            str(k): int(sum(1 for v in n_gold.values() if v == k))
            for k in sorted(set(n_gold.values()))},
        "pool_coverage_by_N_and_sT": summary,
        "per_question_full_pool": {
            str(sT): {q: {"ceiling_mean": float(np.mean(v)),
                          "n_golden_sigs": n_gold[q]}
                      for q, v in d.items()}
            for sT, d in per_q_full.items()},
    }
    dest = os.path.join(SCRIPT_DIR, "pool_ceiling.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Part A: Ambrosia pool coverage (execution equivalence) ===")
    print(f"pools={len(pools)}  questions={len(golden)}  "
          f"gold sigs/question={out['n_golden_sigs_distribution']}")
    for N in N_SWEEP:
        print(f"\n  pool size N={N}")
        print(f"  {'sT':>6} {'pool coverage':>20} {'all modes':>11} {'no mode':>9}")
        for sT, s in summary[str(N)].items():
            print(f"  {sT:>6} {s['ceiling_recall_mean']:>11.4f} +/- "
                  f"{s['ceiling_recall_ci95']:.4f} "
                  f"{s['frac_questions_all_modes']:>10.3f} "
                  f"{s['frac_questions_no_mode']:>9.3f}")
    print(f"\nSaved {dest}")


# ---------------------------------------------------------------------------
# Part B: selected-set coverage as a fraction of pool coverage
# ---------------------------------------------------------------------------
def _ratio_for_pkl(path, golden, sig_tables, ceil_lookup, sT):
    """-> (mean selected recall, mean ceiling, mean ratio, #questions with ceiling 0)
    over the questions. Raises on a malformed selection file (see the module docstring)."""
    with open(path, "rb") as f:
        items = pickle.load(f)
    sel, cei, rat, n_zero, seen = [], [], [], 0, set()
    for item in items:
        if len(item) != 4:
            raise ValueError(f"{path}: record of length {len(item)}, expected 4")
        pt, sents, _, _ = item
        q = parse_question(pt)
        if q not in golden:
            raise ValueError(f"{path}: prompt {pt[:60]!r} is not an ambrosia question of the signature table")
        if q in seen:
            raise ValueError(f"{path}: question {q!r} selected twice")
        seen.add(q)
        if len(sents) != K:
            raise ValueError(f"{path}: question {q!r} has {len(sents)} selected strings, expected K={K}")
        gs, tbl = golden[q], sig_tables[q]
        missing = [s for s in set(sents) if normalize(s) not in tbl]
        if missing:
            raise ValueError(f"{path}: question {q!r}: selected string not in the signature table "
                             f"(not drawn by any of the Part A pools): {missing[0][:60]!r}")
        covered = {tbl[normalize(s)] for s in set(sents)} & gs
        r_sel = len(covered) / len(gs)
        if (q, sT) not in ceil_lookup:
            raise ValueError(f"{path}: no pool ceiling for question {q!r} at sT={sT} in pool_ceiling.json")
        r_ceil = ceil_lookup[(q, sT)]
        sel.append(r_sel)
        cei.append(r_ceil)
        # ratio is defined only where the pool contains at least one gold mode
        if r_ceil > 0:
            rat.append(r_sel / r_ceil)
        else:
            n_zero += 1
    if seen != set(golden):
        raise ValueError(f"{path}: {len(set(golden) - seen)} of {len(golden)} questions have no selection")
    if not rat:
        raise ValueError(f"{path}: every question has pool ceiling 0; recall_vs_ceiling is undefined")
    return float(np.mean(sel)), float(np.mean(cei)), float(np.mean(rat)), n_zero


def expected_questions(df):
    """The questions of the ambrosia-top100 prompts that run_ambrosia.sh samples. Raises if one
    of them appears more than once in ambrosia.csv (its database dump would be ambiguous)."""
    from lm_dpp.utils.ambrosia_utils import get_ambrosia_prompts
    qs = {parse_question(pt) for pt, _ in get_ambrosia_prompts("ambrosia-top100", df)}
    counts = df["question"].astype(str).str.strip().value_counts()
    dup = sorted(q for q in qs if counts.get(q, 0) != 1)
    if dup:
        raise ValueError(f"{len(dup)} ambrosia-top100 questions are not in ambrosia.csv exactly once, "
                         f"e.g. {dup[0]!r} ({counts.get(dup[0], 0)} rows)")
    return qs


def part_b(n_pools):
    with open(SIGTABLE, "rb") as f:
        d = pickle.load(f)
    golden, sig_tables = d["golden"], d["sig_tables"]

    with open(os.path.join(SCRIPT_DIR, "pool_ceiling.json")) as f:
        ceil_json = json.load(f)
    ceil_lookup = {}
    for sT_s, per_q in ceil_json["per_question_full_pool"].items():
        for q, v in per_q.items():
            ceil_lookup[(q, float(sT_s))] = v["ceiling_mean"]

    # (name, knob, method dir, file, sT, deterministic): deterministic methods are in run dir 0 only
    methods = []
    for sT in ST_SWEEP_BASE:
        methods.append(("Ancestral", sT, f"{MODEL}-ancestral", f"s{fmt(sT)}.pkl", sT, False))
        methods.append(("k-Means", sT, f"{MODEL}-kmeans", f"s{fmt(sT)}.pkl", sT, False))
    for T in SAMP_TS:
        methods.append(("PoolDPP-Samp", fmt(T), f"{MODEL}-samp",
                        f"T{fmt(T)}_s{fmt(SAMP_ST_HELD)}.pkl", SAMP_ST_HELD, False))
    for T in MAP_TS:
        methods.append(("PoolDPP-MAP", fmt(T), f"{MODEL}-map",
                        f"T{fmt(T)}_s{fmt(MAP_ST_HELD)}.pkl", MAP_ST_HELD, True))

    ests = {i: os.path.join(ROOT_DIR, CACHE, f"pool{i}", EST_SUBDIR) for i in range(n_pools)}
    require_pools(ests, f"est dir {EST_SUBDIR}")
    for i, est in ests.items():
        runs = sorted(int(r) for r in os.listdir(est) if r.isdigit() and os.path.isdir(os.path.join(est, r)))
        if runs != list(range(N_RUNS)):
            raise ValueError(f"ambrosia pool{i}: run dirs {runs} in {est}, expected 0..{N_RUNS - 1}")

    results = defaultdict(list)
    for name, knob, sub, fname, sT, deterministic in methods:
        runs = [0] if deterministic else list(range(N_RUNS))
        paths = {(i, r): os.path.join(est, str(r), sub, fname) for i, est in ests.items() for r in runs}
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"ambrosia {name} {knob}: {len(missing)} of {len(paths)} selection files missing, "
                f"e.g. pool{missing[0][0]} run {missing[0][1]}: {paths[missing[0]]}")
        pool_sel, pool_cei, pool_rat, zero = [], [], [], set()
        for i in ests:
            rs, rc, rr = [], [], []
            for r in runs:
                got = _ratio_for_pkl(paths[(i, r)], golden, sig_tables, ceil_lookup, sT)
                rs.append(got[0]); rc.append(got[1]); rr.append(got[2]); zero.add(got[3])
            pool_sel.append(np.mean(rs))
            pool_cei.append(np.mean(rc))
            pool_rat.append(np.mean(rr))
        n = len(pool_sel)
        if n < 2:
            raise ValueError(f"ambrosia {name} {knob}: {n} pool(s), need at least 2 for an interval")
        (n_zero,) = zero   # the ceiling is per question and sT, so every file leaves out the same questions
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=(10000, n))
        def ci(a):
            b = np.asarray(a)[idx].mean(axis=1)
            return float((np.quantile(b, .975) - np.quantile(b, .025)) / 2)
        results[name].append({
            "knob": knob, "sT": sT, "n_pools": n, "n_runs": len(runs),
            "n_questions_zero_ceiling": int(n_zero),
            "selected_recall": float(np.mean(pool_sel)),
            "selected_recall_ci95": ci(pool_sel),
            "pool_ceiling": float(np.mean(pool_cei)),
            "recall_vs_ceiling": float(np.mean(pool_rat)),
            "recall_vs_ceiling_ci95": ci(pool_rat),
        })
        print(f"  {name:14s} knob={str(knob):>5s} sT={sT:<4} "
              f"sel={np.mean(pool_sel):.4f} ceil={np.mean(pool_cei):.4f} "
              f"ratio={np.mean(pool_rat):.4f}  (pools={n})", flush=True)

    dest = os.path.join(SCRIPT_DIR, "recall_vs_ceiling.json")
    with open(dest, "w") as f:
        json.dump(dict(results), f, indent=2)

    print("\n=== Part B: selected-set coverage as a fraction of pool coverage ===")
    for name, rows in results.items():
        best = max(rows, key=lambda r: r["recall_vs_ceiling"])
        print(f"{name:14s} best knob={str(best['knob']):>5s}  "
              f"selected recall={best['selected_recall']:.4f}  "
              f"ceiling={best['pool_ceiling']:.4f}  "
              f"fraction of ceiling={best['recall_vs_ceiling']:.4f} "
              f"+/- {best['recall_vs_ceiling_ci95']:.4f}")
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["A", "B"], required=True)
    p.add_argument("--n-pools", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.part == "A":
        part_a(a.n_pools, a.workers)
    else:
        part_b(a.n_pools)
