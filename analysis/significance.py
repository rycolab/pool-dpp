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
                   pair across methods on the pool. n = 20 / 30. This is the
                   unit the figure's error bars are computed over.

For each pair: mean paired difference, bootstrap 95% CI on that difference, and
a paired permutation test (random sign flips, 100k draws).

Also emitted, for C4: the trade-off curve. For every method and knob we report
the set-level metric (Ambrosia golden-query coverage / GSM8K pass@K) against the
per-item metric (Ambrosia golden accuracy / GSM8K mean per-chain accuracy), so
the per-chain drop that buys the coverage gain is visible and attributable.

Phase 1 caches per-(method, knob, pool, run, question) metrics; phase 2 runs the
tests from the cache.

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

# ---- method tables, mirroring the headline plots (unified tau layout) ----
# Old target-temperature anchors map to tau = sigma_hat / (2*T_old); the
# nearest shared-grid points (log scale) are: Ambrosia Samp T=500 -> tau 0.004
# -> 0.003; Ambrosia MAP T=10 -> 2.8 -> 3; GSM8K Samp/MAP T=500 -> 0.024 ->
# 0.03; GSM8K MAP T=10 -> 1.2 -> 1; T=2.5 -> 4.8 -> 3.
TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


def fmt(x):
    return f"{float(x):g}"


AMB_METHODS = (
    [("Ancestral", sT, f"{AMB_MODEL}-ancestral", f"s{fmt(sT)}.pkl")
     for sT in [0.6, 0.7, 1.0, 1.2, 1.25, 1.3]]
    + [("k-Means", sT, f"{AMB_MODEL}-kmeans", f"s{fmt(sT)}.pkl")
       for sT in [0.6, 0.7, 1.0, 1.2, 1.25, 1.3]]
    + [("Naive DPP", sT, f"{AMB_MODEL}-naive", f"s{fmt(sT)}.pkl")
       for sT in [0.6, 0.7, 1.0, 1.2, 1.25]]
    + [("PoolDPP-Samp", tau, f"{AMB_MODEL}-samp", f"tau{fmt(tau)}_s1.pkl")
       for tau in TAUS]
    + [("PoolDPP-MAP", tau, f"{AMB_MODEL}-map", f"tau{fmt(tau)}_s1.5.pkl")
       for tau in TAUS]
)

GSM_METHODS = (
    [("Ancestral", sT, f"{GSM_MODEL}-ancestral", f"s{fmt(sT)}.pkl")
     for sT in [0.5, 0.8, 1.0, 1.2]]
    + [("k-Means", sT, f"{GSM_MODEL}-kmeans", f"s{fmt(sT)}.pkl")
       for sT in [0.5, 0.8, 1.0, 1.2]]
    + [("Naive-DPP", sT, f"{GSM_MODEL}-naive", f"s{fmt(sT)}.pkl")
       for sT in [0.5, 0.8, 1.0]]
    + [("PoolDPP-Samp", tau, f"{GSM_MODEL}-samp", f"tau{fmt(tau)}_s1.2.pkl")
       for tau in TAUS]
    + [("PoolDPP-MAP", tau, f"{GSM_MODEL}-map", f"tau{fmt(tau)}_s1.2.pkl")
       for tau in TAUS]
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


# ---------------------------------------------------------------------------
# Phase 1: collect
# ---------------------------------------------------------------------------
def collect_ambrosia(n_pools):
    """Uses the signature table of pool_ceiling.py, so no SQL re-execution."""
    with open(os.path.join(SCRIPT_DIR, "pool_ceiling_sigtable.pkl"), "rb") as f:
        d = pickle.load(f)
    golden, sig_tables = d["golden"], d["sig_tables"]

    rows = []
    for name, knob, sub, fname in AMB_METHODS:
        for pi in range(n_pools):
            est = os.path.join(ROOT_DIR, "cache-ambrosia", f"pool{pi}", AMB_EST)
            if not os.path.isdir(est):
                continue
            for r in sorted([x for x in os.listdir(est) if x.isdigit()], key=int):
                p = os.path.join(est, r, sub, fname)
                if not os.path.exists(p):
                    continue
                with open(p, "rb") as f:
                    items = pickle.load(f)
                for item in items:
                    if len(item) != 4:
                        continue
                    pt, sents, _, _ = item
                    q = parse_question_amb(pt)
                    if q is None or q not in golden:
                        continue
                    gs, tbl = golden[q], sig_tables[q]
                    ded = list(set(sents))
                    if not ded:
                        continue
                    sigs = [tbl.get(normalize(s)) for s in ded]
                    covered = set(sigs) & gs
                    n_gold_hits = sum(1 for s in sigs if s in gs)
                    rows.append((name, str(knob), pi, int(r), q,
                                 len(covered) / len(gs),        # set: coverage
                                 n_gold_hits / len(ded)))       # item: precision
        print(f"  collected {name} knob={knob}", flush=True)
    return rows


def collect_gsm8k(n_pools):
    from lm_dpp.utils.math_utils import build_math_prompt, is_correct, load_gsm8k_dataset
    # mirrors analysis/plot_gsm8k.py: the headline uses the first 50 questions
    df = load_gsm8k_dataset().head(50)
    prompt_to_gold = {build_math_prompt(r["question"]): r["answer"]
                      for _, r in df.iterrows()}

    rows = []
    for name, knob, sub, fname in GSM_METHODS:
        for pi in range(n_pools):
            est = os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{pi}", GSM_EST)
            if not os.path.isdir(est):
                continue
            for r in sorted([x for x in os.listdir(est) if x.isdigit()], key=int):
                p = os.path.join(est, r, sub, fname)
                if not os.path.exists(p):
                    continue
                with open(p, "rb") as f:
                    items = pickle.load(f)
                for item in items:
                    if len(item) != 4:
                        continue
                    pt, sents, _, _ = item
                    gold = prompt_to_gold.get(pt)
                    if gold is None or not sents:
                        continue
                    c = sum(1 for s in sents if is_correct(s, gold))
                    rows.append((name, str(knob), pi, int(r), pt,
                                 1.0 if c > 0 else 0.0,          # set: pass@K
                                 c / len(sents)))                # item: per-chain
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
        return {"n": n, "mean_diff": float(d.mean()) if n else float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "p": float("nan")}
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

    out = {"matched_note": matched_note, "per_method_knob": summary,
           "tests": {}}
    for label, mk_a, mk_b in comparisons:
        if mk_a not in cells or mk_b not in cells:
            print(f"  [skip] {label}: missing cell")
            continue
        res = {}
        for lvl, tbl in (("question_level", q_level), ("pool_level", p_level)):
            keys = sorted(set(tbl[mk_a]) & set(tbl[mk_b]))
            if not keys:
                continue
            for mi, mname in ((0, "set_metric"), (1, "item_metric")):
                a = [tbl[mk_a][k][mi] for k in keys]
                b = [tbl[mk_b][k][mi] for k in keys]
                res[f"{lvl}|{mname}"] = paired_test(a, b)
        out["tests"][label] = res
        q = res.get("question_level|set_metric", {})
        pl = res.get("pool_level|set_metric", {})
        qi = res.get("question_level|item_metric", {})
        print(f"  {label}")
        print(f"      set  metric: question-level d={q.get('mean_diff', float('nan')):+.4f} "
              f"[{q.get('ci_lo', float('nan')):+.4f},{q.get('ci_hi', float('nan')):+.4f}] "
              f"p={q.get('p', float('nan')):.4g} (n={q.get('n', 0)}) | "
              f"pool-level d={pl.get('mean_diff', float('nan')):+.4f} "
              f"p={pl.get('p', float('nan')):.4g} (n={pl.get('n', 0)})")
        print(f"      item metric: question-level d={qi.get('mean_diff', float('nan')):+.4f} "
              f"[{qi.get('ci_lo', float('nan')):+.4f},{qi.get('ci_hi', float('nan')):+.4f}] "
              f"p={qi.get('p', float('nan')):.4g}")
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
            n = a.n_pools or (30 if task == "ambrosia" else 20)
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
                # tau anchors: Samp tau=0.003 ~ old T=500 (the headline point);
                # MAP tau=3 ~ old tT=10.
                ("MAP(tau=3,sT=1.5) vs k-Means(sT=1.3)",
                 ("PoolDPP-MAP", "3"), ("k-Means", "1.3")),
                ("MAP(tau=3,sT=1.5) vs Ancestral(sT=1.3)",
                 ("PoolDPP-MAP", "3"), ("Ancestral", "1.3")),
                ("Samp(tau=0.003,sT=1.0) vs k-Means(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "0.003"), ("k-Means", "1.0")),
                ("Samp(tau=0.003,sT=1.0) vs Ancestral(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "0.003"), ("Ancestral", "1.0")),
                ("Samp(tau=0.003,sT=1.0) vs Naive(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "0.003"), ("Naive DPP", "1.0")),
                ("MAP(tau=3) vs Samp(tau=0.003)",
                 ("PoolDPP-MAP", "3"), ("PoolDPP-Samp", "0.003")),
                # frontier tilts of the unified kernel, and each
                # baseline at its own best-recall local temperature (1.0)
                ("Samp(tau=1,sT=1.0) vs Ancestral(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "1"), ("Ancestral", "1.0")),
                ("Samp(tau=1,sT=1.0) vs k-Means(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "1"), ("k-Means", "1.0")),
                ("Samp(tau=1,sT=1.0) vs Naive(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "1"), ("Naive DPP", "1.0")),
                ("Samp(tau=0.3,sT=1.0) vs Ancestral(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "0.3"), ("Ancestral", "1.0")),
                ("Samp(tau=0.3,sT=1.0) vs k-Means(sT=1.0) [matched sT]",
                 ("PoolDPP-Samp", "0.3"), ("k-Means", "1.0")),
                ("MAP(tau=3,sT=1.5) vs k-Means(sT=1.0) [k-means best recall]",
                 ("PoolDPP-MAP", "3"), ("k-Means", "1.0")),
                ("MAP(tau=3,sT=1.5) vs Ancestral(sT=1.0) [ancestral best recall]",
                 ("PoolDPP-MAP", "3"), ("Ancestral", "1.0")),
                ("MAP(tau=3,sT=1.5) vs Samp(tau=1,sT=1.0)",
                 ("PoolDPP-MAP", "3"), ("PoolDPP-Samp", "1")),
            ]
            note = ("PoolDPP-MAP is anchored at sT=1.5 and PoolDPP-Samp at "
                    "sT=1.0 in the headline figure, so 'matched T' is only "
                    "exact for the Samp comparisons; MAP is compared against "
                    "each baseline at that baseline's best sT. tau anchors "
                    "correspond to the old T anchors via tau = sigma_hat/(2T).")
        else:
            comparisons = [
                # tau anchors: 0.03 ~ old T=500, 1 ~ old T=10, 3 ~ old T=2.5.
                ("MAP(tau=0.03) vs k-Means(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-MAP", "0.03"), ("k-Means", "1.2")),
                ("MAP(tau=0.03) vs Ancestral(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-MAP", "0.03"), ("Ancestral", "1.2")),
                ("MAP(tau=1) vs k-Means(sT=1.2) [moderate tilt, matched sT]",
                 ("PoolDPP-MAP", "1"), ("k-Means", "1.2")),
                ("MAP(tau=1) vs Ancestral(sT=1.2) [moderate tilt, matched sT]",
                 ("PoolDPP-MAP", "1"), ("Ancestral", "1.2")),
                ("MAP(tau=3) vs Ancestral(sT=1.2) [sharp tilt, matched sT]",
                 ("PoolDPP-MAP", "3"), ("Ancestral", "1.2")),
                ("Samp(tau=0.03) vs Ancestral(sT=1.2) [matched sT=1.2]",
                 ("PoolDPP-Samp", "0.03"), ("Ancestral", "1.2")),
                ("Samp(tau=1) vs Ancestral(sT=1.2) [moderate tilt]",
                 ("PoolDPP-Samp", "1"), ("Ancestral", "1.2")),
                ("MAP(tau=0.03) vs MAP(tau=1) [diversity vs moderate]",
                 ("PoolDPP-MAP", "0.03"), ("PoolDPP-MAP", "1")),
            ]
            note = ("All PoolDPP variants are anchored at sT=1.2, so every "
                    "comparison against a baseline at sT=1.2 is exactly "
                    "matched in sampling temperature. tau anchors correspond "
                    "to the old T anchors via tau = sigma_hat/(2T).")
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
