"""
Non-kernel diversity metrics on FineWeb (tab:intrinsic-heldout)
===============================================================
The FineWeb figure plots mean log-prob
against log-det of the MiniLM Gram matrix, and PoolDPP-MAP selects by
maximizing sum(log p)/T + logdet of exactly that matrix. It is graded on its
own objective.

This script re-scores the same selected sets with diversity metrics that are
NOT the kernel:

  distinct-1/2/3   fraction of distinct n-grams among all n-grams in the set.
                   Purely lexical, no embedder at all.
  self-BLEU-ish    mean pairwise token-level Jaccard overlap (lower = more
                   diverse). No embedder.
  held-out cosine  mean pairwise cosine distance under Qwen3-Embedding-0.6B,
                   which is NOT the kernel used for these runs (MiniLM is).
  held-out logdet  log-det of the Qwen Gram matrix of the same selected set.

Cosine diversity under the SAME embedder would not fix the circularity, so it
is deliberately not the headline here; the MiniLM log-det is reported only as
the reference column that the circularity charge is about.

Inputs are pool 0 (cache-fineweb/est10minilm). Failures stop the run with an
error; nothing is skipped. A missing selection file raises FileNotFoundError (a
stochastic method must exist in every run dir 0..N-1, PoolDPP-MAP in run dir 0).
A selection file without exactly the 50 FineWeb prompts, a record that is not
(prompt, sents, logps_t1, embs), a set of other than K=10 strings, a selected
string with no held-out Qwen embedding (n_sets_missing_heldout is therefore
always 0), a set with no n-grams for distinct-n, a pair of strings that are
both empty after whitespace tokenization (Jaccard undefined), or a non-finite
metric (e.g. a log-det whose Gram matrix plus 1e-6 I is not positive definite)
raises ValueError. The log-det adds 1e-6 * I and the cosine adds 1e-12 to row
norms (existing approximations, kept as is).

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/heldout_diversity.py
"""

import json
import os
import pickle
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from baselines import check_selections, finite_mean, logdet, selection_files  # noqa: E402

K = 10
EST_DIR = os.path.join(ROOT_DIR, "cache-fineweb", "est10minilm")
MODEL = "google__gemma-2-9b-it_minilm"
QWEN_EMB = os.path.join(ROOT_DIR, "cache-fineweb",
                        "google__gemma-2-9b-it_qwen_embeddings.pkl")

ST_SWEEP = [0.6, 0.8, 1.0, 1.2, 1.5]
# PoolDPP-Samp / -MAP hold sT=1.0 and sweep T on the grids of the figure
SAMP_TS = [0.428571, 0.666667, 0.818182, 1, 1.22222]
MAP_TS = [500, 100, 50, 25, 10]


def fmt(x):
    return f"{float(x):g}"


METHODS = (
    [("Ancestral", sT, f"{MODEL}-ancestral", f"s{fmt(sT)}.pkl") for sT in ST_SWEEP]
    + [("k-Means", sT, f"{MODEL}-kmeans", f"s{fmt(sT)}.pkl") for sT in ST_SWEEP]
    + [("Naive DPP", sT, f"{MODEL}-naive", f"s{fmt(sT)}.pkl") for sT in ST_SWEEP]
    + [("PoolDPP-Samp", f"T={fmt(t)}", f"{MODEL}-samp", f"T{fmt(t)}_s1.pkl")
       for t in SAMP_TS]
    + [("PoolDPP-MAP", f"T={fmt(t)}", f"{MODEL}-map", f"T{fmt(t)}_s1.pkl")
       for t in MAP_TS]
)


def toks(s):
    return s.split()


def distinct_n(sents, n):
    """Fraction of distinct n-grams; raises ValueError if the set has no n-gram."""
    grams = []
    for s in sents:
        t = toks(s)
        grams.extend(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    if not grams:
        raise ValueError(f"distinct-{n} is undefined: none of the {len(sents)} strings has "
                         f"{n} whitespace tokens (first: {sents[0][:60]!r})" if sents
                         else f"distinct-{n} of an empty set")
    return len(set(grams)) / len(grams)


def mean_pairwise_jaccard(sents):
    sets = [set(toks(s)) for s in sents]
    if len(sets) < 2:
        raise ValueError(f"pairwise Jaccard of a set of {len(sets)} strings")
    vals = []
    for a, b in combinations(sets, 2):
        u = len(a | b)
        if not u:
            raise ValueError("pairwise Jaccard is undefined for two strings that are both "
                             "empty after whitespace tokenization")
        vals.append(len(a & b) / u)
    return float(np.mean(vals))


def mean_pairwise_cosdist(F):
    F = np.asarray(F, dtype=np.float64)
    if len(F) < 2:
        raise ValueError(f"pairwise cosine distance of a set of {len(F)} strings")
    nrm = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    S = nrm @ nrm.T
    iu = np.triu_indices(len(F), 1)
    return float(1.0 - S[iu].mean())


def main():
    print("Loading held-out Qwen embeddings ...", flush=True)
    with open(QWEN_EMB, "rb") as f:
        qemb = pickle.load(f)
    # sentence -> qwen vector, per sampling temperature, merged
    lookup = {}
    for sT, per_prompt in qemb.items():
        for _, d in per_prompt.items():
            for s, v in zip(d["u_sents"], d["embeddings"]):
                lookup.setdefault(s, v)
    print(f"  {len(lookup)} unique sentences with held-out embeddings", flush=True)

    from lm_dpp.utils.fineweb_prompts import prompts as fineweb_prompts
    expected = set(fineweb_prompts)
    out = {}
    for name, knob, sub, fname in METHODS:
        key = f"{name}|{knob}"
        acc = defaultdict(list)
        n_sets, n_missing = 0, 0
        for p in selection_files(EST_DIR, sub, fname, deterministic=(name == "PoolDPP-MAP")):
            with open(p, "rb") as f:
                items = pickle.load(f)
            check_selections(items, p, expected, K)
            for pt, sents, lp, emb_minilm in items:
                n_sets += 1
                # quality axis, identical to analysis/plot_intrinsic.py
                acc["quality_mean_logp"].append(float(np.mean(lp)))
                acc["distinct_1"].append(distinct_n(sents, 1))
                acc["distinct_2"].append(distinct_n(sents, 2))
                acc["distinct_3"].append(distinct_n(sents, 3))
                acc["pairwise_jaccard"].append(mean_pairwise_jaccard(sents))
                acc["kernel_logdet_minilm"].append(logdet(emb_minilm))
                absent = [s for s in sents if s not in lookup]
                if absent:
                    raise ValueError(f"{p}: prompt {pt[:60]!r}: {len(absent)} selected strings "
                                     f"have no held-out embedding in {QWEN_EMB} "
                                     f"(first: {absent[0][:60]!r})")
                qv = [lookup[s] for s in sents]
                acc["heldout_logdet_qwen"].append(logdet(qv))
                acc["heldout_cosdist_qwen"].append(mean_pairwise_cosdist(qv))
        out[key] = {k: finite_mean(v, f"{key} {k}") for k, v in acc.items()}
        out[key]["n_sets"] = n_sets
        out[key]["n_sets_missing_heldout"] = n_missing
        print(f"  {key:<24s} n={n_sets:<5d} d1={out[key]['distinct_1']:.4f} "
              f"d2={out[key]['distinct_2']:.4f} "
              f"jac={out[key]['pairwise_jaccard']:.4f} "
              f"kernel_ld={out[key]['kernel_logdet_minilm']:+.2f} "
              f"heldout_ld={out[key]['heldout_logdet_qwen']:+.2f}",
              flush=True)

    dest = os.path.join(SCRIPT_DIR, "heldout_diversity.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)

    # rank agreement: does the kernel metric order methods the same way the
    # non-kernel metrics do?
    keys = list(out)
    if len(keys) >= 3:
        from scipy.stats import spearmanr
        kern = [out[k]["kernel_logdet_minilm"] for k in keys]
        print("\nSpearman rank correlation with the kernel metric "
              "(MiniLM log-det), across all method/knob points:")
        for m in ["heldout_logdet_qwen", "heldout_cosdist_qwen",
                  "distinct_1", "distinct_2", "distinct_3", "pairwise_jaccard"]:
            v = [out[k][m] for k in keys]
            rho, p = spearmanr(kern, v)
            print(f"  {m:<24s} rho={rho:+.3f}  p={p:.3g}")
            out.setdefault("_rank_agreement", {})[m] = {"spearman_rho": float(rho),
                                                        "p": float(p)}
        with open(dest, "w") as f:
            json.dump(out, f, indent=2)

    # ------------------------------------------------------------------
    # The decisive test for C3: does PoolDPP-MAP still sit on the Pareto
    # frontier when diversity is measured by something that is NOT the kernel?
    # A point is dominated if another point is >= on quality AND >= on
    # diversity, with at least one strict.
    # ------------------------------------------------------------------
    print("\nPareto frontier under each diversity metric "
          "(quality = mean log-prob, both axes higher = better):")
    pareto_out = {}
    for dm, sign in [("kernel_logdet_minilm", +1.0),
                     ("heldout_logdet_qwen", +1.0),
                     ("heldout_cosdist_qwen", +1.0),
                     ("distinct_1", +1.0), ("distinct_2", +1.0),
                     ("pairwise_jaccard", -1.0)]:
        pts = [(k, out[k]["quality_mean_logp"], sign * out[k][dm])
               for k in out if not k.startswith("_")]
        front = []
        for k, q, d in pts:
            dominated = any((q2 >= q and d2 >= d) and (q2 > q or d2 > d)
                            for k2, q2, d2 in pts if k2 != k)
            if not dominated:
                front.append(k)
        front_sorted = sorted(front)
        pareto_out[dm] = front_sorted
        n_map = sum(1 for k in front_sorted if k.startswith("PoolDPP-MAP"))
        n_samp = sum(1 for k in front_sorted if k.startswith("PoolDPP-Samp"))
        print(f"  {dm:<24s} frontier size={len(front_sorted):<3d} "
              f"PoolDPP-MAP points={n_map} PoolDPP-Samp points={n_samp}")
        print(f"      {front_sorted}")
    out["_pareto_frontier_by_metric"] = pareto_out
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
