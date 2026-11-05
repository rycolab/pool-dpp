"""
Set-diversity baselines on the same pools (tab:baselines)
=============================================================================
This script adds the two pool-reusable comparators, run on exactly the same
pools, the same embeddings and the same K as every method in the headline
figures, so the numbers are directly comparable:

  MMR   Maximal Marginal Relevance. Greedy; at each step picks
        argmax_i  lambda * quality_i - (1 - lambda) * max_{j in S} cos(i, j).
        lambda is swept, so MMR traces its own quality/diversity frontier.
        Quality is the min-max normalized log p_t1, matching what the DPP
        kernel uses as its quality signal.

  MBR   Minimum Bayes Risk reranking with a semantic utility. Each candidate
        scores as its mean cosine similarity to the rest of the pool (its
        consensus / centrality), and the top-K are returned. MBR optimizes
        agreement rather than coverage, so it is the natural anti-diversity
        comparator: it says what a strong non-diversity reranker gets.

Also reported for reference: quality-topK (no diversity term at all), and the
headline methods read from the unified estimation layout
(est{K}{emb}/{run}/{model}-{method}/...) on the same pools, so every row of
the comparison table comes from one script with one CI convention.

Stochastic Beam Search and Conditional Poisson SBS are NOT included: they
require new decoding runs rather than pool reranking (see beam_baselines.py).

Uncertainty: every metric is averaged within a pool (over prompts, and over
run dirs for the stochastic est-dir methods), then reported as the mean over
pools with a 95% t-interval half-width across pools (`{metric}_ci95`,
t_{0.975, n_pools-1} * sd / sqrt(n_pools)). Pools are independent LM samples, so this is the CI the figures use.

Scoring uses each task's own headline axes:
  fineweb   quality = mean log p_t1        diversity = log-det of MiniLM Gram
  ambrosia  precision = golden accuracy    recall = golden query coverage
  gsm8k     pass@K=10                      mean per-chain accuracy

Pools: fineweb = cache-fineweb (pool 0) + cache-fineweb-pools/pool*;
ambrosia = cache-ambrosia/pool{0..29}; gsm8k = cache-gsm8k/pool{0..19}.

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/baselines.py --task fineweb
    OPENBLAS_NUM_THREADS=1 python analysis/baselines.py --task ambrosia
    OPENBLAS_NUM_THREADS=1 python analysis/baselines.py --task gsm8k
"""

import argparse
import glob
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]


def fmt(x):
    return f"{float(x):g}"


def unit(E):
    E = np.asarray(E, dtype=np.float64)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def mmr_select(E, q, k, lam):
    """Greedy MMR. E must be unit-norm, q in [0,1]."""
    S = E @ E.T
    n = len(E)
    k = min(k, n)
    sel = [int(np.argmax(q))]
    maxsim = S[sel[0]].copy()
    for _ in range(k - 1):
        score = lam * q - (1.0 - lam) * maxsim
        score[sel] = -np.inf
        i = int(np.argmax(score))
        sel.append(i)
        maxsim = np.maximum(maxsim, S[i])
    return sel


def mbr_select(E, k):
    """Top-K by mean cosine similarity to the pool (consensus)."""
    S = E @ E.T
    return list(np.argsort(-S.mean(axis=1))[:min(k, len(E))])


def logdet(F):
    F = np.asarray(F, dtype=np.float64)
    if len(F) < 2:
        return np.nan
    G = F @ F.T + 1e-6 * np.eye(len(F))
    s, ld = np.linalg.slogdet(G)
    return ld if s > 0 else np.nan


def selections(E_raw, lp1, k):
    """-> {method_label: [indices]}"""
    E = unit(E_raw)
    q = minmax(lp1)
    out = {f"MMR|lam={lam}": mmr_select(E, q, k, lam) for lam in LAMBDAS}
    out["MBR|consensus"] = mbr_select(E, k)
    out["quality-topK"] = list(np.argsort(-np.asarray(lp1))[:k])
    return out


# ---------------------------------------------------------------------------
# pool-level aggregation
# ---------------------------------------------------------------------------
def t975(n):
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1)) if n >= 2 else float("nan")


def summarize_pools(acc, metrics):
    """acc[method][pool][metric] -> list of per-set values.
    -> {method: {metric: mean over pools, metric_ci95, metric_pool_sd,
                 n_pools, n}}"""
    res = {}
    for m, by_pool in acc.items():
        pools = sorted(by_pool)
        row = {"n_pools": len(pools),
               "n": int(sum(len(by_pool[p][metrics[0]]) for p in pools))}
        for met in metrics:
            pm = np.array([np.nanmean(by_pool[p][met]) for p in pools
                           if len(by_pool[p][met])], dtype=np.float64)
            pm = pm[np.isfinite(pm)]
            n = len(pm)
            row[met] = float(pm.mean()) if n else float("nan")
            sd = float(pm.std(ddof=1)) if n >= 2 else float("nan")
            row[f"{met}_pool_sd"] = sd
            row[f"{met}_ci95"] = t975(n) * sd / np.sqrt(n) if n >= 2 else float("nan")
        res[m] = row
    return res


def runs_of(est):
    return sorted([r for r in os.listdir(est)
                   if r.isdigit() and os.path.isdir(os.path.join(est, r))], key=int)


def est_items(est, model, spec):
    """Yield the saved selections of one est-dir method across its run dirs.
    spec = (method, fname)."""
    method, fname = spec
    for r in runs_of(est):
        p = os.path.join(est, r, f"{model}-{method}", fname)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            for item in pickle.load(f):
                if len(item) == 4:
                    yield item


def est_specs(st_base, samp_st, map_st, taus=TAUS):
    """Headline est-dir rows: label -> (method, fname)."""
    out = {}
    for sT in st_base:
        out[f"est:Ancestral|sT={fmt(sT)}"] = ("ancestral", f"s{fmt(sT)}.pkl")
        out[f"est:k-Means|sT={fmt(sT)}"] = ("kmeans", f"s{fmt(sT)}.pkl")
    for tau in taus:
        out[f"est:PoolDPP-Samp|tau={fmt(tau)}|sT={fmt(samp_st)}"] = (
            "samp", f"tau{fmt(tau)}_s{fmt(samp_st)}.pkl")
        out[f"est:PoolDPP-MAP|tau={fmt(tau)}|sT={fmt(map_st)}"] = (
            "map", f"tau{fmt(tau)}_s{fmt(map_st)}.pkl")
    return out


def _poolnum(d):
    return int(re.sub(r"\D", "", os.path.basename(d.rstrip("/"))) or 0)


# ---------------------------------------------------------------------------
def run_fineweb(k=10, sT=1.0, pool_dirs=None, est_rows=True):
    model = "google__gemma-2-9b-it_minilm"
    if pool_dirs is None:
        pool_dirs = [os.path.join(ROOT_DIR, "cache-fineweb")] + sorted(
            glob.glob(os.path.join(ROOT_DIR, "cache-fineweb-pools", "pool*")), key=_poolnum)
    Q, D = "quality_mean_logp", "kernel_logdet_minilm"
    acc = defaultdict(lambda: defaultdict(lambda: {Q: [], D: []}))
    for pi, pd in enumerate(pool_dirs):
        path = os.path.join(pd, f"{model}_embeddings.pkl")
        if not os.path.exists(path):
            print(f"  [skip] {path} missing", flush=True)
            continue
        with open(path, "rb") as f:
            emb = pickle.load(f)
        for _, d in emb[sT].items():
            u, lp1 = list(d["u_sents"]), np.asarray(d["logps_t1"], dtype=np.float64)
            E = np.asarray(d["embeddings"], dtype=np.float64)
            if len(u) < k or len(u) != len(lp1):
                continue
            for m, idx in selections(E, lp1, k).items():
                acc[m][pi][Q].append(float(lp1[idx].mean()))
                acc[m][pi][D].append(logdet(E[idx]))
        del emb
        est = os.path.join(pd, "est10minilm")
        if est_rows and os.path.isdir(est):
            for lab, spec in est_specs([0.6, 0.8, 1.0, 1.2, 1.5], sT, sT).items():
                for _, sents, lp, E in est_items(est, model, spec):
                    if len(sents) < 2:
                        continue
                    acc[lab][pi][Q].append(float(np.mean(lp)))
                    acc[lab][pi][D].append(logdet(E))
        print(f"  pool {pd} done", flush=True)
    return summarize_pools(acc, [Q, D])


def run_ambrosia(n_pools=30, k=3, sTs=(1.0, 1.2, 1.5), cache="cache-ambrosia",
                 est_rows=True):
    with open(os.path.join(SCRIPT_DIR, "pool_ceiling_sigtable.pkl"), "rb") as f:
        st = pickle.load(f)
    golden, sig_tables = st["golden"], st["sig_tables"]
    model = "google__gemma-2-9b-it_minilm"

    def parse_q(pt):
        s = pt.find("and the following question:\n")
        if s < 0:
            return None
        s += len("and the following question:\n")
        e = pt.find("\ngenerate the SQL query", s)
        return pt[s:e if e != -1 else len(pt)].strip()

    def norm(x):
        return x.strip().replace("```sql\n", "").replace("\n```", "")

    def score(q, sents):
        gs, tbl = golden[q], sig_tables[q]
        ded = list(set(sents))
        sigs = [tbl.get(norm(s)) for s in ded]
        return (len(set(sigs) & gs) / len(gs),
                sum(1 for s in sigs if s in gs) / len(ded))

    C, P = "recall_coverage", "precision_golden_accuracy"
    acc = defaultdict(lambda: defaultdict(lambda: {C: [], P: []}))
    for pi in range(n_pools):
        p = os.path.join(ROOT_DIR, cache, f"pool{pi}",
                         "google__gemma-2-9b-it_minilm_embeddings.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            emb = pickle.load(f)
        for sT in sTs:
            if sT not in emb:
                continue
            for pt, d in emb[sT].items():
                q = parse_q(pt)
                if q is None or q not in golden:
                    continue
                u, lp1 = list(d["u_sents"]), np.asarray(d["logps_t1"], float)
                E = np.asarray(d["embeddings"], dtype=np.float64)
                if len(u) < k or len(u) != len(lp1):
                    continue
                for m, idx in selections(E, lp1, k).items():
                    cov, prec = score(q, [u[i] for i in idx])
                    acc[f"{m}|sT={sT}"][pi][C].append(cov)
                    acc[f"{m}|sT={sT}"][pi][P].append(prec)
        del emb
        est = os.path.join(ROOT_DIR, cache, f"pool{pi}", "est3minilm")
        if est_rows and os.path.isdir(est):
            for lab, spec in est_specs([0.6, 0.7, 1.0, 1.2, 1.25, 1.3, 1.5],
                                       1.0, 1.5).items():
                for pt, sents, _, _ in est_items(est, model, spec):
                    q = parse_q(pt)
                    if q is None or q not in golden or not sents:
                        continue
                    cov, prec = score(q, sents)
                    acc[lab][pi][C].append(cov)
                    acc[lab][pi][P].append(prec)
        print(f"  pool{pi} done", flush=True)
    return summarize_pools(acc, [C, P])


def run_gsm8k(n_pools=20, k=10, sT=1.2, cache="cache-gsm8k", est_rows=True):
    from lm_dpp.utils.math_utils import build_math_prompt, is_correct, load_gsm8k_dataset
    df = load_gsm8k_dataset().head(50)
    if len(df) < 50:  # offline load returns an empty frame
        raise SystemExit("GSM8K dataset did not load (HF offline?); refusing to score")
    gold_of = {build_math_prompt(r["question"]): r["answer"] for _, r in df.iterrows()}
    model = "google__gemma-2-9b-it_qwen"

    PS, CH = "pass_at_K", "mean_per_chain_accuracy"
    acc = defaultdict(lambda: defaultdict(lambda: {PS: [], CH: []}))
    for pi in range(n_pools):
        p = os.path.join(ROOT_DIR, cache, f"pool{pi}", f"{model}_embeddings.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            emb = pickle.load(f)
        if sT not in emb:
            continue
        corr_of = {}
        for pt, d in emb[sT].items():
            gold = gold_of.get(pt)
            if gold is None:
                continue
            u, lp1 = list(d["u_sents"]), np.asarray(d["logps_t1"], float)
            E = np.asarray(d["embeddings"], dtype=np.float64)
            if len(u) < k or len(u) != len(lp1):
                continue
            corr = np.array([1.0 if is_correct(s, gold) else 0.0 for s in u])
            corr_of.update({(pt, s): c for s, c in zip(u, corr)})
            for m, idx in selections(E, lp1, k).items():
                c = corr[idx]
                acc[m][pi][PS].append(1.0 if c.sum() > 0 else 0.0)
                acc[m][pi][CH].append(float(c.mean()))
        del emb
        est = os.path.join(ROOT_DIR, cache, f"pool{pi}", "est10qwen")
        if est_rows and os.path.isdir(est):
            for lab, spec in est_specs([sT], sT, sT).items():
                for pt, sents, _, _ in est_items(est, model, spec):
                    gold = gold_of.get(pt)
                    if gold is None or not sents:
                        continue
                    c = np.array([corr_of[(pt, s)] if (pt, s) in corr_of
                                  else float(is_correct(s, gold)) for s in sents])
                    acc[lab][pi][PS].append(1.0 if c.sum() > 0 else 0.0)
                    acc[lab][pi][CH].append(float(c.mean()))
        print(f"  pool{pi} done", flush=True)
    return summarize_pools(acc, [PS, CH])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fineweb", "ambrosia", "gsm8k"], required=True)
    ap.add_argument("--n-pools", type=int, default=None)
    ap.add_argument("--cache", default=None, help="ambrosia/gsm8k cache dir override")
    ap.add_argument("--fineweb-pools", nargs="+", default=None,
                    help="fineweb pool dirs (default cache-fineweb + cache-fineweb-pools/pool*)")
    ap.add_argument("--no-est-rows", action="store_true",
                    help="skip the est-dir headline method rows")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    est_rows = not a.no_est_rows

    if a.task == "fineweb":
        pools = ([os.path.join(ROOT_DIR, p) for p in a.fineweb_pools]
                 if a.fineweb_pools else None)
        res = run_fineweb(pool_dirs=pools, est_rows=est_rows)
        hdr = ("quality_mean_logp", "kernel_logdet_minilm")
    elif a.task == "ambrosia":
        res = run_ambrosia(a.n_pools or 30, cache=a.cache or "cache-ambrosia",
                           est_rows=est_rows)
        hdr = ("recall_coverage", "precision_golden_accuracy")
    else:
        res = run_gsm8k(a.n_pools or 20, cache=a.cache or "cache-gsm8k",
                        est_rows=est_rows)
        hdr = ("pass_at_K", "mean_per_chain_accuracy")

    print(f"\n=== baselines: {a.task} (mean over pools +- 95% t-CI half-width) ===")
    for m in sorted(res):
        v = res[m]
        print(f"  {m:<44s} {hdr[0]}={v[hdr[0]]:+9.4f} +- {v[hdr[0] + '_ci95']:.4f}  "
              f"{hdr[1]}={v[hdr[1]]:+9.4f} +- {v[hdr[1] + '_ci95']:.4f}  "
              f"(pools={v['n_pools']}, sets={v['n']})")

    res["_meta"] = {"ci": "95% t-interval half-width across pools of per-pool means",
                    "task": a.task}
    dest = a.out or os.path.join(SCRIPT_DIR, f"baselines_{a.task}.json")
    with open(dest, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
