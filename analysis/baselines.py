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

Pools: fineweb = cache-fineweb (pool 0) + cache-fineweb-pools/pool{1..9};
ambrosia = cache-ambrosia/pool{0..99}; gsm8k = cache-gsm8k/pool{0..99}.

Failures stop the run with an error; nothing is skipped. A missing pool, est
dir or selection file raises FileNotFoundError listing every missing one. A
pool or selection file that lacks one of the task's prompts (or has an extra
or repeated one), a record that is not (prompt, sents, logps_t1, embs), a set
of other than K strings, a pool with fewer than K distinct strings, a
non-finite metric (e.g. a log-det whose Gram matrix is not positive definite),
fewer than 2 pools for a CI, or an Ambrosia string missing from the execution
table of pool_ceiling.py raises ValueError. Stochastic est-dir methods are read
from every run dir 0..N-1 (the run dirs must be exactly these); PoolDPP-MAP is
deterministic and read from run dir 0 only.

The log-det adds 1e-6 * I to the Gram matrix (an existing ridge, kept as is);
unit() adds 1e-12 to row norms.

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/baselines.py --task fineweb
    OPENBLAS_NUM_THREADS=1 python analysis/baselines.py --task ambrosia
    OPENBLAS_NUM_THREADS=1 python analysis/baselines.py --task gsm8k
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
# T grids of the frontier figures
SAMP_TS = {"fineweb": [0.428571, 0.666667, 0.818182, 1, 1.22222],
           "ambrosia": [0.428571, 0.538462, 0.666667, 0.818182, 1],
           "gsm8k": [0.428571, 0.666667, 0.818182, 1, 1.03, 1.05, 1.07, 1.1, 1.22222]}
MAP_TS = {"fineweb": [500, 100, 50, 25, 10], "ambrosia": [10, 7.5, 3, 1],
          "gsm8k": [2.5, 5, 10, 15, 20, 25, 50, 100, 500]}
# extra FineWeb pools cache-fineweb-pools/pool{1..N}: N_EXTRA in scripts/run_fineweb.sh
FINEWEB_N_EXTRA = 9
# est-dir methods computed once, into run dir 0 (dpp-estimation --N 1)
DETERMINISTIC = {"map"}


def fmt(x):
    return f"{float(x):g}"


# ---------------------------------------------------------------------------
# checks: a missing or incomplete input stops the run
# ---------------------------------------------------------------------------
def _ranges(idx):
    """[0, 1, 2, 5, 7, 8] -> '0..2, 5, 7..8'"""
    idx, out, i = sorted(idx), [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(f"{idx[i]}" if i == j else f"{idx[i]}..{idx[j]}")
        i = j + 1
    return ", ".join(out)


def require_pools(task, paths_of_pool):
    """paths_of_pool: {pool index: [paths that must exist]}. Raises FileNotFoundError
    naming every pool with a missing path."""
    missing = {pi: [p for p in ps if not os.path.exists(p)] for pi, ps in paths_of_pool.items()}
    missing = {pi: ps for pi, ps in missing.items() if ps}
    if missing:
        first = [p for pi in sorted(missing)[:3] for p in missing[pi]]
        raise FileNotFoundError(
            f"{task}: {len(missing)} of {len(paths_of_pool)} pools missing: pool "
            f"{_ranges(missing)} (first missing paths: {first})")


def require_files(paths, what):
    """Raises FileNotFoundError naming how many of `paths` are missing and the first few."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{what}: {len(missing)} of {len(paths)} files missing, "
                                f"first: {missing[:5]}")


def run_dirs(est):
    """Run dirs of an est dir, which must be exactly 0..N-1."""
    runs = sorted(int(r) for r in os.listdir(est)
                  if r.isdigit() and os.path.isdir(os.path.join(est, r)))
    if not runs or runs != list(range(len(runs))):
        raise ValueError(f"{est}: run dirs are {runs}, expected 0..N-1")
    return [str(r) for r in runs]


def selection_files(est, sub, fname, deterministic):
    """Paths of one method's selection file: run dir 0 for a deterministic method,
    every run dir otherwise. Raises FileNotFoundError if any is missing."""
    runs = ["0"] if deterministic else run_dirs(est)
    paths = [os.path.join(est, r, sub, fname) for r in runs]
    require_files(paths, f"{est}: {sub}/{fname} "
                         f"({'deterministic, run 0' if deterministic else 'every run dir'})")
    return paths


def check_selections(items, path, expected_prompts, k):
    """Raises ValueError unless the selection file `items` has exactly one record
    (prompt, sents, logps_t1, embs) per prompt of `expected_prompts`, each with k strings."""
    seen = set()
    for item in items:
        if len(item) != 4:
            raise ValueError(f"{path}: a record has {len(item)} fields, expected "
                             f"(prompt, sents, logps_t1, embs)")
        pt, sents, lp, E = item
        if pt in seen:
            raise ValueError(f"{path}: prompt {pt[:60]!r} appears twice")
        if pt not in expected_prompts:
            raise ValueError(f"{path}: prompt {pt[:60]!r} is not one of the task's prompts")
        seen.add(pt)
        if not (len(sents) == len(lp) == len(E) == k):
            raise ValueError(f"{path}: prompt {pt[:60]!r} has {len(sents)} strings, "
                             f"{len(lp)} log-probs and {len(E)} embeddings; expected K={k}")
    missing = set(expected_prompts) - seen
    if missing:
        raise ValueError(f"{path}: {len(missing)} of {len(expected_prompts)} prompts missing, "
                         f"first: {sorted(missing)[0][:60]!r}")


def check_prompts(found, expected, where):
    """Raises ValueError unless the prompt set `found` equals `expected`."""
    found, expected = set(found), set(expected)
    if found != expected:
        miss, extra = expected - found, found - expected
        raise ValueError(f"{where}: {len(miss)} of the task's {len(expected)} prompts missing"
                         f"{' (first: ' + repr(sorted(miss)[0][:60]) + ')' if miss else ''}, "
                         f"{len(extra)} unexpected"
                         f"{' (first: ' + repr(sorted(extra)[0][:60]) + ')' if extra else ''}")


def check_pool_row(u, lp1, E, k, where):
    """Raises ValueError unless a pool row has matching lengths and at least k distinct strings."""
    if not (len(u) == len(lp1) == len(E)):
        raise ValueError(f"{where}: {len(u)} strings, {len(lp1)} log-probs, {len(E)} embeddings")
    if len(u) < k:
        raise ValueError(f"{where}: {len(u)} distinct strings, fewer than K={k}")


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
    """log det(F F^T + 1e-6 I). Raises ValueError for fewer than 2 rows or a Gram
    matrix that is not positive definite (a numerical failure)."""
    F = np.asarray(F, dtype=np.float64)
    if len(F) < 2:
        raise ValueError(f"log-det of a set of {len(F)} strings: the diversity axis needs at least 2")
    G = F @ F.T + 1e-6 * np.eye(len(F))
    s, ld = np.linalg.slogdet(G)
    if not (s > 0 and np.isfinite(ld)):
        raise ValueError(f"log-det failed: slogdet sign {s}, value {ld} for a set of {len(F)} "
                         f"strings (Gram matrix plus 1e-6 I is not positive definite)")
    return ld


def selections(E_raw, lp1, k):
    """-> {method_label: [indices]}"""
    lp1 = np.asarray(lp1, dtype=np.float64)
    if len(E_raw) < k:
        raise ValueError(f"pool has {len(E_raw)} distinct strings, fewer than K={k}")
    if not np.all(np.isfinite(lp1)):
        raise ValueError(f"{int((~np.isfinite(lp1)).sum())} of {len(lp1)} log-probs are not finite")
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
    """t_{0.975, n-1}. Raises ValueError for n < 2 (no interval from one pool)."""
    from scipy import stats
    if n < 2:
        raise ValueError(f"a 95% t-interval needs at least 2 pools, got {n}")
    return float(stats.t.ppf(0.975, n - 1))


def finite_mean(vals, where):
    """Mean of a non-empty list of finite values; raises ValueError otherwise."""
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        raise ValueError(f"{where}: no values")
    bad = np.flatnonzero(~np.isfinite(v))
    if bad.size:
        raise ValueError(f"{where}: {bad.size} of {v.size} values are not finite "
                         f"(first: index {int(bad[0])}, {v[bad[0]]})")
    return float(v.mean())


def summarize_pools(acc, metrics):
    """acc[method][pool][metric] -> list of per-set values.
    -> {method: {metric: mean over pools, metric_ci95, metric_pool_sd,
                 n_pools, n}}
    Every method must cover the same pools, with at least one finite value per
    pool and metric, and at least 2 pools; otherwise ValueError."""
    res = {}
    pool_sets = {m: sorted(by_pool) for m, by_pool in acc.items()}
    if len({tuple(p) for p in pool_sets.values()}) > 1:
        raise ValueError(f"methods cover different pools: "
                         f"{ {m: _ranges(p) for m, p in pool_sets.items()} }")
    for m, by_pool in acc.items():
        pools = sorted(by_pool)
        row = {"n_pools": len(pools),
               "n": int(sum(len(by_pool[p][metrics[0]]) for p in pools))}
        for met in metrics:
            pm = np.array([finite_mean(by_pool[p][met], f"{m} pool {p} {met}") for p in pools],
                          dtype=np.float64)
            n = len(pm)
            row[met] = float(pm.mean())
            sd = float(pm.std(ddof=1)) if n >= 2 else float("nan")
            row[f"{met}_pool_sd"] = sd
            row[f"{met}_ci95"] = t975(n) * sd / np.sqrt(n)
        res[m] = row
    return res


def est_items(est, model, spec, expected_prompts, k):
    """Yield the saved selections of one est-dir method across its run dirs
    (run dir 0 only for a deterministic method). spec = (method, fname). Each file
    must hold one K-string selection per prompt of `expected_prompts`."""
    method, fname = spec
    for p in selection_files(est, f"{model}-{method}", fname, method in DETERMINISTIC):
        with open(p, "rb") as f:
            items = pickle.load(f)
        check_selections(items, p, expected_prompts, k)
        yield from items


def est_specs(task, st_base, samp_st, map_st):
    """Headline est-dir rows: label -> (method, fname)."""
    out = {}
    for sT in st_base:
        out[f"est:Ancestral|sT={fmt(sT)}"] = ("ancestral", f"s{fmt(sT)}.pkl")
        out[f"est:k-Means|sT={fmt(sT)}"] = ("kmeans", f"s{fmt(sT)}.pkl")
    for T in SAMP_TS[task]:
        out[f"est:PoolDPP-Samp|T={fmt(T)}|sT={fmt(samp_st)}"] = ("samp", f"T{fmt(T)}_s{fmt(samp_st)}.pkl")
    for T in MAP_TS[task]:
        out[f"est:PoolDPP-MAP|T={fmt(T)}|sT={fmt(map_st)}"] = ("map", f"T{fmt(T)}_s{fmt(map_st)}.pkl")
    return out


# ---------------------------------------------------------------------------
def run_fineweb(k=10, sT=1.0, pool_dirs=None, est_rows=True):
    from lm_dpp.utils.fineweb_prompts import prompts as fineweb_prompts
    model = "google__gemma-2-9b-it_minilm"
    if pool_dirs is None:
        pool_dirs = [os.path.join(ROOT_DIR, "cache-fineweb")] + [
            os.path.join(ROOT_DIR, "cache-fineweb-pools", f"pool{i}")
            for i in range(1, FINEWEB_N_EXTRA + 1)]
    expected = set(fineweb_prompts)
    require_pools("fineweb", {pi: [os.path.join(pd, f"{model}_embeddings.pkl")]
                              + ([os.path.join(pd, "est10minilm")] if est_rows else [])
                              for pi, pd in enumerate(pool_dirs)})
    Q, D = "quality_mean_logp", "kernel_logdet_minilm"
    acc = defaultdict(lambda: defaultdict(lambda: {Q: [], D: []}))
    for pi, pd in enumerate(pool_dirs):
        path = os.path.join(pd, f"{model}_embeddings.pkl")
        with open(path, "rb") as f:
            emb = pickle.load(f)
        if sT not in emb:
            raise ValueError(f"fineweb: {path} has no sT={fmt(sT)} (has {sorted(emb)})")
        check_prompts(emb[sT], expected, f"fineweb: {path} sT={fmt(sT)}")
        for pt, d in emb[sT].items():
            u, lp1 = list(d["u_sents"]), np.asarray(d["logps_t1"], dtype=np.float64)
            E = np.asarray(d["embeddings"], dtype=np.float64)
            check_pool_row(u, lp1, E, k, f"fineweb: {path} sT={fmt(sT)} prompt {pt[:60]!r}")
            for m, idx in selections(E, lp1, k).items():
                acc[m][pi][Q].append(float(lp1[idx].mean()))
                acc[m][pi][D].append(logdet(E[idx]))
        del emb
        est = os.path.join(pd, "est10minilm")
        if est_rows:
            for lab, spec in est_specs("fineweb", [0.6, 0.8, 1.0, 1.2, 1.5], sT, sT).items():
                for _, sents, lp, E in est_items(est, model, spec, expected, k):
                    acc[lab][pi][Q].append(float(np.mean(lp)))
                    acc[lab][pi][D].append(logdet(E))
        print(f"  pool {pd} done", flush=True)
    return summarize_pools(acc, [Q, D])


def run_ambrosia(n_pools=100, k=3, sTs=(1.0, 1.2, 1.5), cache="cache-ambrosia",
                 est_rows=True):
    with open(os.path.join(SCRIPT_DIR, "pool_ceiling_sigtable.pkl"), "rb") as f:
        st = pickle.load(f)
    golden, sig_tables = st["golden"], st["sig_tables"]
    model = "google__gemma-2-9b-it_minilm"

    def parse_q(pt):
        s = pt.find("and the following question:\n")
        if s < 0:
            raise ValueError(f"ambrosia: prompt {pt[:60]!r} has no 'and the following question:'")
        s += len("and the following question:\n")
        e = pt.find("\ngenerate the SQL query", s)
        q = pt[s:e if e != -1 else len(pt)].strip()
        if q not in golden:
            raise ValueError(f"ambrosia: question {q[:60]!r} is not in the golden table of "
                             f"pool_ceiling_sigtable.pkl")
        return q

    def norm(x):
        return x.strip().replace("```sql\n", "").replace("\n```", "")

    def score(q, sents):
        gs, tbl = golden[q], sig_tables[q]
        ded = list(set(sents))
        # tbl[s] is None for a query that does not execute; a string absent from the
        # table was never executed, which is an error, not a non-runnable query
        absent = [s for s in ded if norm(s) not in tbl]
        if absent:
            raise ValueError(f"ambrosia: {len(absent)} selected strings for question {q[:60]!r} "
                             f"are not in the execution table (first: {absent[0][:60]!r}); "
                             f"rerun pool_ceiling.py --part A on the same pools")
        sigs = [tbl[norm(s)] for s in ded]
        return (len(set(sigs) & gs) / len(gs),
                sum(1 for s in sigs if s in gs) / len(ded))

    C, P = "recall_coverage", "precision_golden_accuracy"
    acc = defaultdict(lambda: defaultdict(lambda: {C: [], P: []}))
    pool_dir = {pi: os.path.join(ROOT_DIR, cache, f"pool{pi}") for pi in range(n_pools)}
    require_pools("ambrosia", {pi: [os.path.join(d, f"{model}_embeddings.pkl")]
                               + ([os.path.join(d, "est3minilm")] if est_rows else [])
                               for pi, d in pool_dir.items()})
    for pi in range(n_pools):
        p = os.path.join(pool_dir[pi], f"{model}_embeddings.pkl")
        with open(p, "rb") as f:
            emb = pickle.load(f)
        expected = None
        for sT in sTs:
            if sT not in emb:
                raise ValueError(f"ambrosia: {p} has no sT={fmt(sT)} (has {sorted(emb)})")
            qs = [parse_q(pt) for pt in emb[sT]]
            check_prompts(qs, golden, f"ambrosia: {p} sT={fmt(sT)} (questions)")
            expected = set(emb[sT]) if expected is None else expected
            check_prompts(emb[sT], expected, f"ambrosia: {p} sT={fmt(sT)}")
            for pt, d in emb[sT].items():
                q = parse_q(pt)
                u, lp1 = list(d["u_sents"]), np.asarray(d["logps_t1"], float)
                E = np.asarray(d["embeddings"], dtype=np.float64)
                check_pool_row(u, lp1, E, k, f"ambrosia: {p} sT={fmt(sT)} prompt {pt[:60]!r}")
                for m, idx in selections(E, lp1, k).items():
                    cov, prec = score(q, [u[i] for i in idx])
                    acc[f"{m}|sT={sT}"][pi][C].append(cov)
                    acc[f"{m}|sT={sT}"][pi][P].append(prec)
        del emb
        est = os.path.join(pool_dir[pi], "est3minilm")
        if est_rows:
            for lab, spec in est_specs("ambrosia", [0.6, 0.7, 1.0, 1.2, 1.25, 1.3, 1.5],
                                       1.0, 1.5).items():
                for pt, sents, _, _ in est_items(est, model, spec, expected, k):
                    cov, prec = score(parse_q(pt), sents)
                    acc[lab][pi][C].append(cov)
                    acc[lab][pi][P].append(prec)
        print(f"  pool{pi} done", flush=True)
    return summarize_pools(acc, [C, P])


def run_gsm8k(n_pools=100, k=10, sT=1.2, cache="cache-gsm8k", est_rows=True):
    from lm_dpp.utils.math_utils import build_math_prompt, is_correct, load_gsm8k_dataset
    df = load_gsm8k_dataset().head(50)
    if len(df) < 50:  # offline load returns an empty frame
        raise SystemExit("GSM8K dataset did not load (HF offline?); refusing to score")
    gold_of = {build_math_prompt(r["question"]): r["answer"] for _, r in df.iterrows()}
    model = "google__gemma-2-9b-it_qwen"

    PS, CH = "pass_at_K", "mean_per_chain_accuracy"
    acc = defaultdict(lambda: defaultdict(lambda: {PS: [], CH: []}))
    pool_dir = {pi: os.path.join(ROOT_DIR, cache, f"pool{pi}") for pi in range(n_pools)}
    require_pools("gsm8k", {pi: [os.path.join(d, f"{model}_embeddings.pkl")]
                            + ([os.path.join(d, "est10qwen")] if est_rows else [])
                            for pi, d in pool_dir.items()})
    for pi in range(n_pools):
        p = os.path.join(pool_dir[pi], f"{model}_embeddings.pkl")
        with open(p, "rb") as f:
            emb = pickle.load(f)
        if sT not in emb:
            raise ValueError(f"gsm8k: {p} has no sT={fmt(sT)} (has {sorted(emb)})")
        check_prompts(emb[sT], gold_of, f"gsm8k: {p} sT={fmt(sT)}")
        corr_of = {}
        for pt, d in emb[sT].items():
            gold = gold_of[pt]
            u, lp1 = list(d["u_sents"]), np.asarray(d["logps_t1"], float)
            E = np.asarray(d["embeddings"], dtype=np.float64)
            check_pool_row(u, lp1, E, k, f"gsm8k: {p} sT={fmt(sT)} prompt {pt[:60]!r}")
            corr = np.array([1.0 if is_correct(s, gold) else 0.0 for s in u])
            corr_of.update({(pt, s): c for s, c in zip(u, corr)})
            for m, idx in selections(E, lp1, k).items():
                c = corr[idx]
                acc[m][pi][PS].append(1.0 if c.sum() > 0 else 0.0)
                acc[m][pi][CH].append(float(c.mean()))
        del emb
        est = os.path.join(pool_dir[pi], "est10qwen")
        if est_rows:
            for lab, spec in est_specs("gsm8k", [sT], sT, sT).items():
                for pt, sents, _, _ in est_items(est, model, spec, gold_of, k):
                    gold = gold_of[pt]
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
                    help="fineweb pool dirs (default cache-fineweb + cache-fineweb-pools/pool1..9)")
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
        res = run_ambrosia(a.n_pools or 100, cache=a.cache or "cache-ambrosia",
                           est_rows=est_rows)
        hdr = ("recall_coverage", "precision_golden_accuracy")
    else:
        res = run_gsm8k(a.n_pools or 100, cache=a.cache or "cache-gsm8k",
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
