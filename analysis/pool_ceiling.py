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


def load_pool(pool_idx):
    """-> {question: {sT: [sql strings]}} for one pool."""
    path = os.path.join(ROOT_DIR, CACHE, f"pool{pool_idx}", RAW_PKL)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out = defaultdict(dict)
    for sT, per_prompt in raw.items():
        if sT not in ST_SWEEP:
            continue
        for prompt_text, entry in per_prompt.items():
            q = parse_question(prompt_text)
            if q is not None:
                out[q][sT] = list(entry[0])
    return dict(out)


def signatures_for_question(args):
    """Execute every unique SQL string for one question exactly once."""
    question, db_dump, ambig_queries_str, unique_sqls = args
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    try:
        cur.executescript(db_dump)
    except Exception:
        conn.close()
        return question, None, None

    golden_sigs = set()
    try:
        for gq in eval(ambig_queries_str):
            try:
                cur.execute(gq)
                golden_sigs.add(tuple(cur.fetchall()))
            except Exception:
                pass
    except Exception:
        conn.close()
        return question, None, None
    if not golden_sigs:
        conn.close()
        return question, None, None

    sig_of = {}
    for s in unique_sqls:
        try:
            cur.execute(s)
            sig_of[s] = tuple(cur.fetchall())
        except Exception:
            sig_of[s] = None
    conn.close()
    return question, golden_sigs, sig_of


# ---------------------------------------------------------------------------
# Part A: pool coverage
# ---------------------------------------------------------------------------
def part_a(n_pools, workers):
    df = load_ambrosia_df(AMBROSIA_CSV)
    by_question = {str(r["question"]).strip(): r for _, r in df.iterrows()}

    print(f"Loading {n_pools} pools ...", flush=True)
    pools = {}
    for i in range(n_pools):
        p = load_pool(i)
        if p is not None:
            pools[i] = p
    print(f"Pools loaded: {len(pools)}", flush=True)

    unique_by_q = defaultdict(set)
    for pool in pools.values():
        for q, per_t in pool.items():
            for sents in per_t.values():
                unique_by_q[q].update(normalize(s) for s in sents)

    jobs = []
    for q, sqls in unique_by_q.items():
        row = by_question.get(q)
        if row is not None:
            jobs.append((q, row["db_dump"], row["ambig_queries"], sorted(sqls)))
    print(f"Questions: {len(jobs)}; unique SQL strings: "
          f"{sum(len(j[3]) for j in jobs)}", flush=True)

    sig_tables, golden = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (q, gs, tbl) in enumerate(ex.map(signatures_for_question, jobs), 1):
            if gs is not None:
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
            if q not in golden:
                continue
            gs, tbl = golden[q], sig_tables[q]
            for sT, sents in per_t.items():
                sigs = [tbl.get(normalize(s)) for s in sents]
                for N in N_SWEEP:
                    if N >= len(sigs):
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
            if n >= 2:
                idx = boot.integers(0, n, size=(10000, n))
                bs = pm[idx].mean(axis=1)
                ci = float((np.quantile(bs, .975) - np.quantile(bs, .025)) / 2)
            else:
                ci = 0.0
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
    """-> (mean selected recall, mean ceiling, mean ratio) over questions."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        items = pickle.load(f)
    sel, cei, rat = [], [], []
    for item in items:
        if len(item) != 4:
            continue
        pt, sents, _, _ = item
        q = parse_question(pt)
        if q is None or q not in golden:
            continue
        gs, tbl = golden[q], sig_tables[q]
        covered = {tbl.get(normalize(s)) for s in set(sents)} & gs
        r_sel = len(covered) / len(gs)
        r_ceil = ceil_lookup.get((q, sT))
        if r_ceil is None:
            continue
        sel.append(r_sel)
        cei.append(r_ceil)
        # ratio is defined only where the pool contains at least one gold mode
        if r_ceil > 0:
            rat.append(r_sel / r_ceil)
    if not sel:
        return None
    return float(np.mean(sel)), float(np.mean(cei)), \
        (float(np.mean(rat)) if rat else float("nan"))


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

    methods = []
    for sT in ST_SWEEP_BASE:
        methods.append(("Ancestral", sT, f"{MODEL}-ancestral", f"s{fmt(sT)}.pkl", sT))
        methods.append(("k-Means", sT, f"{MODEL}-kmeans", f"s{fmt(sT)}.pkl", sT))
    for T in SAMP_TS:
        methods.append(("PoolDPP-Samp", fmt(T), f"{MODEL}-samp",
                        f"T{fmt(T)}_s{fmt(SAMP_ST_HELD)}.pkl", SAMP_ST_HELD))
    for T in MAP_TS:
        methods.append(("PoolDPP-MAP", fmt(T), f"{MODEL}-map",
                        f"T{fmt(T)}_s{fmt(MAP_ST_HELD)}.pkl", MAP_ST_HELD))

    results = defaultdict(list)
    for name, knob, sub, fname, sT in methods:
        pool_sel, pool_cei, pool_rat = [], [], []
        for i in range(n_pools):
            est = os.path.join(ROOT_DIR, CACHE, f"pool{i}", EST_SUBDIR)
            if not os.path.isdir(est):
                continue
            runs = sorted([r for r in os.listdir(est)
                           if r.isdigit() and os.path.isdir(os.path.join(est, r))],
                          key=int)
            rs, rc, rr = [], [], []
            for r in runs:
                got = _ratio_for_pkl(os.path.join(est, r, sub, fname),
                                     golden, sig_tables, ceil_lookup, sT)
                if got:
                    rs.append(got[0]); rc.append(got[1]); rr.append(got[2])
            if rs:
                pool_sel.append(np.mean(rs))
                pool_cei.append(np.mean(rc))
                pool_rat.append(np.nanmean(rr))
        if not pool_sel:
            continue
        n = len(pool_sel)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=(10000, n)) if n >= 2 else None
        def ci(a):
            if idx is None:
                return 0.0
            b = np.asarray(a)[idx].mean(axis=1)
            return float((np.quantile(b, .975) - np.quantile(b, .025)) / 2)
        results[name].append({
            "knob": knob, "sT": sT, "n_pools": n,
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
