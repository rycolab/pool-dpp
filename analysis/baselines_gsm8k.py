"""
GSM8K baselines on every axis of the paper (tab:baselines, tab:gsm8k-kernels, tab:gsm8k_meank_passk)
===================================================================================================
This script scores every method, on the same 100 pools at matched
sT=1.2, on four axes:

  pass@K            fraction of questions with >=1 correct chain among K=10
  per-chain acc     mean correctness of the K chains
  unique answers    distinct final '#### N' values among the K chains
                    (all methods select K distinct STRINGS; this counts
                    distinct final answers among them)
  log-det           the diversity axis of the headline figure

MBR/MMR/quality-topK are computed from the pool embeddings; PoolDPP-MAP,
PoolDPP-Samp, k-Means, Ancestral, Naive are read from the unified est layout
(est10{emb}/{run}/{model}-{method}/T{T}_s1.2.pkl or s1.2.pkl), every T of the
figure's grids.

Uncertainty: each metric is averaged within a pool (over questions and run
dirs), then reported as the mean over the 100 independent pools with a 95%
t-interval half-width across pools ({metric}_ci95). The Pareto frontier is
computed on the means. unique answers counts only chains with a final
'#### N'; n_chains_no_final_answer reports, per method, how many chains have none.

Failures stop the run with an error; nothing is skipped. A missing pool
embeddings file, est dir or selection file raises FileNotFoundError listing
every missing one (a stochastic method must exist in every run dir 0..N-1,
PoolDPP-MAP in run dir 0). A pool or selection file without exactly the 50
questions, a record that is not (prompt, sents, logps_t1, embs), a set of other
than K strings, a pool with fewer than K distinct strings, a non-finite metric
(e.g. a log-det whose Gram matrix plus 1e-6 I is not positive definite) or
fewer than 2 pools raises ValueError (see the checks in baselines.py).

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/baselines_gsm8k.py \
        [--variant qwen|qwenins] [--out analysis/baselines_gsm8k.json]
"""

import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from baselines import (check_pool_row, check_prompts, check_selections, finite_mean, fmt,  # noqa: E402
                       logdet, require_pools, selection_files, selections, t975)

M = "google__gemma-2-9b-it_qwen"
EST = "est10qwen"
ST = 1.2
K = 10
N_POOLS = 100
SAMP_TS = [0.428571, 0.666667, 0.818182, 1, 1.03, 1.05, 1.07, 1.1, 1.22222]
MAP_TS = [2.5, 5, 10, 15, 20, 25, 50, 100, 500]
ANS_RE = re.compile(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")


def canon_num(s):
    """'243.00' == '243', '72,500' == '72500'; see e2_gsm8k_decomposition."""
    if s is None:
        return None
    t = s.replace(",", "").rstrip(".")
    try:
        f = float(t)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return t

def est_methods(model):
    return (
        [("Ancestral", f"{model}-ancestral", f"s{fmt(ST)}.pkl")]
        + [("k-Means", f"{model}-kmeans", f"s{fmt(ST)}.pkl")]
        + [("Naive-DPP", f"{model}-naive", f"s{fmt(ST)}.pkl")]
        + [(f"PoolDPP-Samp T={fmt(t)}", f"{model}-samp", f"T{fmt(t)}_s{fmt(ST)}.pkl")
           for t in SAMP_TS]
        + [(f"PoolDPP-MAP T={fmt(t)}", f"{model}-map", f"T{fmt(t)}_s{fmt(ST)}.pkl")
           for t in MAP_TS]
    )


def ans_of(t):
    m = ANS_RE.search(t)
    return canon_num(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="qwen",
                    help="embedding variant: qwen (baseline kernel) or qwenins")
    ap.add_argument("--cache", default="cache-gsm8k")
    ap.add_argument("--n-pools", type=int, default=N_POOLS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model = f"google__gemma-2-9b-it_{a.variant}"
    est_sub = f"est10{a.variant}"

    from lm_dpp.utils.math_utils import build_math_prompt, is_correct, \
        load_gsm8k_dataset
    df = load_gsm8k_dataset().head(50)
    if len(df) < 50:  # offline load returns an empty frame
        raise SystemExit("GSM8K dataset did not load (HF offline?); refusing to score")
    gold_of = {build_math_prompt(r["question"]): r["answer"]
               for _, r in df.iterrows()}

    # acc[name][pool][metric] -> per-set values
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    corr_cache = {}

    no_answer = defaultdict(int)   # chains with no final '#### N', per method

    def add(name, pi, pt, sents, E, gold):
        c = np.array([corr_cache.setdefault((pt, s), 1.0 if is_correct(s, gold) else 0.0)
                      for s in sents])
        v = acc[name][pi]
        v["pass"].append(1.0 if c.sum() > 0 else 0.0)
        v["chain"].append(float(c.mean()))
        answers = [ans_of(s) for s in sents]
        no_answer[name] += sum(x is None for x in answers)
        v["uniq"].append(len(set(answers) - {None}))
        v["ld"].append(logdet(E))

    pool_dir = {pi: os.path.join(ROOT_DIR, a.cache, f"pool{pi}") for pi in range(a.n_pools)}
    require_pools(f"gsm8k ({a.variant})",
                  {pi: [os.path.join(d, f"{model}_embeddings.pkl"), os.path.join(d, est_sub)]
                   for pi, d in pool_dir.items()})
    for pi in range(a.n_pools):
        # pool-embedding methods
        p = os.path.join(pool_dir[pi], f"{model}_embeddings.pkl")
        emb = pickle.load(open(p, "rb"))
        if ST not in emb:
            raise ValueError(f"gsm8k: {p} has no sT={fmt(ST)} (has {sorted(emb)})")
        check_prompts(emb[ST], gold_of, f"gsm8k: {p} sT={fmt(ST)}")
        for pt, d in emb[ST].items():
            gold = gold_of[pt]
            u = list(d["u_sents"])
            lp1 = np.asarray(d["logps_t1"], float)
            E = np.asarray(d["embeddings"], float)
            check_pool_row(u, lp1, E, K, f"gsm8k: {p} sT={fmt(ST)} prompt {pt[:60]!r}")
            for m, idx in selections(E, lp1, K).items():
                add(m, pi, pt, [u[i] for i in idx], E[idx], gold)
        del emb
        # est-dir methods; PoolDPP-MAP is deterministic (run dir 0 only)
        est = os.path.join(pool_dir[pi], est_sub)
        for name, sub, fname in est_methods(model):
            for fp in selection_files(est, sub, fname, deterministic=sub.endswith("-map")):
                items = pickle.load(open(fp, "rb"))
                check_selections(items, fp, gold_of, K)
                for pt, sents, _, E in items:
                    add(name, pi, pt, sents, np.asarray(E), gold_of[pt])
        print(f"pool{pi} done", flush=True)

    names = {"pass": "pass_at_K", "chain": "per_chain_accuracy",
             "uniq": "unique_answers_per_set", "ld": "logdet"}
    out = {}
    print(f"\n{'method':<24s} {'pass@K':>16s} {'per-chain':>16s} "
          f"{'uniq ans':>13s} {'log-det':>15s} {'pools':>5s} {'n':>6s}")
    for name in sorted(acc):
        by_pool = acc[name]
        if sorted(by_pool) != list(range(a.n_pools)):
            raise ValueError(f"{name}: pools {sorted(by_pool)}, expected 0..{a.n_pools - 1}")
        o = {"n_pools": len(by_pool),
             "n": int(sum(len(v["pass"]) for v in by_pool.values())),
             "n_chains_no_final_answer": no_answer[name]}
        for k, lab in names.items():
            pm = np.array([finite_mean(by_pool[pi][k], f"{name} pool {pi} {lab}")
                           for pi in sorted(by_pool)])
            n = len(pm)
            o[lab] = float(pm.mean())
            sd = float(pm.std(ddof=1)) if n >= 2 else float("nan")
            o[f"{lab}_pool_sd"] = sd
            o[f"{lab}_ci95"] = t975(n) * sd / np.sqrt(n)
        out[name] = o
        print(f"{name:<24s} {o['pass_at_K']:>7.4f}+-{o['pass_at_K_ci95']:.4f} "
              f"{o['per_chain_accuracy']:>7.4f}+-{o['per_chain_accuracy_ci95']:.4f} "
              f"{o['unique_answers_per_set']:>6.2f}+-{o['unique_answers_per_set_ci95']:.2f} "
              f"{o['logdet']:>7.2f}+-{o['logdet_ci95']:.2f} "
              f"{o['n_pools']:>5d} {o['n']:>6d}")

    # Pareto on the headline axes (log-det, pass@K), higher = better on both
    pts = [(k, out[k]["logdet"], out[k]["pass_at_K"]) for k in out
           if not k.startswith("_")]
    front = [k for k, d, p in pts
             if not any((d2 >= d and p2 >= p) and (d2 > d or p2 > p)
                        for k2, d2, p2 in pts if k2 != k)]
    out["_pareto_frontier_logdet_passk"] = sorted(front)
    print(f"\nPareto frontier on (log-det, pass@K): {sorted(front)}")

    out["_meta"] = {"variant": a.variant, "sT": ST, "K": K,
                    "ci": "95% t-interval half-width across pools of per-pool means"}
    dest = a.out or os.path.join(
        SCRIPT_DIR, "baselines_gsm8k.json" if a.variant == "qwen"
        else f"baselines_gsm8k_{a.variant}.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
