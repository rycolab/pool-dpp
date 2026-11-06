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
computed on the means.

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
from baselines import fmt, logdet, selections, t975  # noqa: E402

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

    def add(name, pi, pt, sents, E, gold):
        c = np.array([corr_cache.setdefault((pt, s), 1.0 if is_correct(s, gold) else 0.0)
                      for s in sents])
        v = acc[name][pi]
        v["pass"].append(1.0 if c.sum() > 0 else 0.0)
        v["chain"].append(float(c.mean()))
        v["uniq"].append(len({ans_of(s) for s in sents} - {None}))
        v["ld"].append(logdet(E))

    for pi in range(a.n_pools):
        # pool-embedding methods
        p = os.path.join(ROOT_DIR, a.cache, f"pool{pi}",
                         f"{model}_embeddings.pkl")
        if os.path.exists(p):
            emb = pickle.load(open(p, "rb"))
            if ST in emb:
                for pt, d in emb[ST].items():
                    gold = gold_of.get(pt)
                    if gold is None:
                        continue
                    u = list(d["u_sents"])
                    lp1 = np.asarray(d["logps_t1"], float)
                    E = np.asarray(d["embeddings"], float)
                    if len(u) < K or len(u) != len(lp1):
                        continue
                    for m, idx in selections(E, lp1, K).items():
                        add(m, pi, pt, [u[i] for i in idx], E[idx], gold)
            del emb
        # est-dir methods
        est = os.path.join(ROOT_DIR, a.cache, f"pool{pi}", est_sub)
        if os.path.isdir(est):
            for name, sub, fname in est_methods(model):
                for r in sorted([x for x in os.listdir(est) if x.isdigit()],
                                key=int):
                    fp = os.path.join(est, r, sub, fname)
                    if not os.path.exists(fp):
                        continue
                    for item in pickle.load(open(fp, "rb")):
                        if len(item) != 4:
                            continue
                        pt, sents, _, E = item
                        gold = gold_of.get(pt)
                        if gold is None or not sents:
                            continue
                        add(name, pi, pt, sents, np.asarray(E), gold)
        print(f"pool{pi} done", flush=True)

    names = {"pass": "pass_at_K", "chain": "per_chain_accuracy",
             "uniq": "unique_answers_per_set", "ld": "logdet"}
    out = {}
    print(f"\n{'method':<24s} {'pass@K':>16s} {'per-chain':>16s} "
          f"{'uniq ans':>13s} {'log-det':>15s} {'pools':>5s} {'n':>6s}")
    for name in sorted(acc):
        by_pool = acc[name]
        o = {"n_pools": len(by_pool),
             "n": int(sum(len(v["pass"]) for v in by_pool.values()))}
        for k, lab in names.items():
            pm = np.array([np.nanmean(by_pool[pi][k]) for pi in sorted(by_pool)])
            pm = pm[np.isfinite(pm)]
            n = len(pm)
            o[lab] = float(pm.mean()) if n else float("nan")
            sd = float(pm.std(ddof=1)) if n >= 2 else float("nan")
            o[f"{lab}_pool_sd"] = sd
            o[f"{lab}_ci95"] = t975(n) * sd / np.sqrt(n) if n >= 2 else float("nan")
        out[name] = o
        print(f"{name:<24s} {o['pass_at_K']:>7.4f}+-{o['pass_at_K_ci95']:.4f} "
              f"{o['per_chain_accuracy']:>7.4f}+-{o['per_chain_accuracy_ci95']:.4f} "
              f"{o['unique_answers_per_set']:>6.2f}+-{o['unique_answers_per_set_ci95']:.2f} "
              f"{o['logdet']:>7.2f}+-{o['logdet_ci95']:.2f} "
              f"{o['n_pools']:>5d} {o['n']:>6d}")

    # Pareto on the headline axes (log-det, pass@K), higher = better on both
    pts = [(k, out[k]["logdet"], out[k]["pass_at_K"]) for k in out
           if not k.startswith("_") and np.isfinite(out[k]["logdet"])]
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
