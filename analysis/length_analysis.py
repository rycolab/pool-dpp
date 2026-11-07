"""Length and log-probability (tab:length)
=======================================
Every reported number uses the raw sequence log-probability. Since log p is a
sum over tokens, it falls roughly linearly with length, so any method that
selects on log p also selects on length.

Part 1  pool-level correlation between length and log p (Pearson and Spearman),
        per task and sampling temperature
Part 2  a length-only selection baseline on FineWeb, scored on the axes of
        fig:fineweb_pareto (quality = mean log p, diversity = MiniLM log-det):
          length-topk    the K longest pool items
          length-kmeans  1-D K-means on length, highest-log-p item per bin

Inputs are pool 0 of each task. A missing embeddings file, a missing sT=1.0
FineWeb sample, mismatched string/log-prob/embedding lengths, a FineWeb pool
without exactly the 50 prompts or with fewer than K distinct strings for a
prompt, a sampling temperature with fewer than 10 strings, or a non-finite
metric (e.g. a log-det whose Gram matrix plus 1e-6 I is not positive definite;
the 1e-6 ridge is an existing approximation) raises an error; nothing is
skipped.

length-kmeans uses K equal-width length bins, so it picks fewer than K strings
when a bin is empty, and none when every string has the same length; a prompt
is left out when fewer than 2 strings are picked. The output records these
counts per method: n_prompts (prompts scored), n_prompts_excluded,
n_prompts_fewer_than_K, min_set_size and mean_set_size.

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/length_analysis.py
"""

import json
import os
import pickle
from collections import defaultdict

import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from baselines import check_pool_row, check_prompts, finite_mean, logdet, require_files  # noqa: E402

TASKS = {
    "fineweb": {"emb": "cache-fineweb/google__gemma-2-9b-it_minilm_embeddings.pkl"},
    "ambrosia": {"emb": "cache-ambrosia/pool0/google__gemma-2-9b-it_minilm_embeddings.pkl"},
    "gsm8k": {"emb": "cache-gsm8k/pool0/google__gemma-2-9b-it_qwen_embeddings.pkl"},
}


def pool_length_vs_logp():
    corr_out = {}
    require_files([os.path.join(ROOT_DIR, cfg["emb"]) for cfg in TASKS.values()],
                  "length_analysis pool-0 embeddings")
    for task, cfg in TASKS.items():
        emb_path = os.path.join(ROOT_DIR, cfg["emb"])
        with open(emb_path, "rb") as f:
            emb = pickle.load(f)
        print(f"--- {task}: pool length vs log p ---")
        for sT in sorted(emb.keys()):
            L, P = [], []
            for pt, d in emb[sT].items():
                if len(d["u_sents"]) != len(d["logps_t1"]):
                    raise ValueError(f"{task}: {emb_path} sT={sT} prompt {pt[:60]!r}: "
                                     f"{len(d['u_sents'])} strings, {len(d['logps_t1'])} log-probs")
                for s, lp1 in zip(d["u_sents"], d["logps_t1"]):
                    L.append(len(s))
                    P.append(float(lp1))
            if len(L) < 10:
                raise ValueError(f"{task}: {emb_path} sT={sT} has {len(L)} strings, fewer than 10")
            if not np.all(np.isfinite(P)):
                raise ValueError(f"{task}: {emb_path} sT={sT}: non-finite log-probs")
            pr = pearsonr(L, P)
            sr = spearmanr(L, P)
            corr_out[f"{task}|sT={sT}"] = {
                "n": len(L), "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
                "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]),
                "mean_chars": float(np.mean(L)),
            }
            print(f"  sT={sT:<5} n={len(L):<7d} pearson r={pr[0]:+.4f} "
                  f"spearman rho={sr[0]:+.4f}")
    return corr_out


def length_baseline(k=10, sT=1.0):
    """Length-only selection on FineWeb, on the headline axes."""
    from lm_dpp.utils.fineweb_prompts import prompts as fineweb_prompts
    emb_path = os.path.join(ROOT_DIR, TASKS["fineweb"]["emb"])
    with open(emb_path, "rb") as f:
        emb = pickle.load(f)
    if sT not in emb:
        raise ValueError(f"fineweb: {emb_path} has no sT={sT} (has {sorted(emb)})")
    check_prompts(emb[sT], fineweb_prompts, f"fineweb: {emb_path} sT={sT}")
    res = defaultdict(lambda: {"q": [], "d": [], "size": []})
    n_excluded = defaultdict(int)
    for pt, d in emb[sT].items():
        u = list(d["u_sents"])
        lp1 = np.asarray(d["logps_t1"], dtype=np.float64)
        E = np.asarray(d["embeddings"], dtype=np.float64)
        check_pool_row(u, lp1, E, k, f"fineweb: {emb_path} sT={sT} prompt {pt[:60]!r}")
        L = np.array([len(s) for s in u], dtype=np.float64)

        # (a) the K longest
        idx = np.argsort(-L)[:k]
        res["length-topk"]["q"].append(float(lp1[idx].mean()))
        res["length-topk"]["d"].append(logdet(E[idx]))
        res["length-topk"]["size"].append(len(idx))

        # (b) 1-D K-means on length, best log p per bin
        lo, hi = L.min(), L.max()
        if hi > lo:
            edges = np.linspace(lo, hi, k + 1)
            bins = np.clip(np.digitize(L, edges[1:-1]), 0, k - 1)
            pick = []
            for b in range(k):
                m = np.where(bins == b)[0]
                if len(m):
                    pick.append(m[np.argmax(lp1[m])])
            if len(pick) >= 2:
                pick = np.array(pick)
                res["length-kmeans"]["q"].append(float(lp1[pick].mean()))
                res["length-kmeans"]["d"].append(logdet(E[pick]))
                res["length-kmeans"]["size"].append(len(pick))
            else:
                n_excluded["length-kmeans"] += 1
        else:
            n_excluded["length-kmeans"] += 1

        # (c) reference: pure top-K by log p (no diversity term at all)
        idx = np.argsort(-lp1)[:k]
        res["quality-topk"]["q"].append(float(lp1[idx].mean()))
        res["quality-topk"]["d"].append(logdet(E[idx]))
        res["quality-topk"]["size"].append(len(idx))

    out = {m: {"quality_mean_logp": finite_mean(v["q"], f"{m} quality"),
               "kernel_logdet_minilm": finite_mean(v["d"], f"{m} log-det"),
               "n_prompts": len(v["q"]),
               "n_prompts_excluded": n_excluded[m],
               "n_prompts_fewer_than_K": int(sum(n < k for n in v["size"])),
               "min_set_size": int(min(v["size"])),
               "mean_set_size": float(np.mean(v["size"]))}
           for m, v in res.items()}
    return out


def main():
    corr_out = pool_length_vs_logp()

    print("\n=== Length-only selection on FineWeb (sT=1.0, K=10) ===")
    baseline = length_baseline()
    for m, v in baseline.items():
        print(f"  {m:<16s} quality={v['quality_mean_logp']:+9.3f}  "
              f"logdet={v['kernel_logdet_minilm']:+7.3f}  (n={v['n_prompts']})")

    dest = os.path.join(SCRIPT_DIR, "length_analysis.json")
    with open(dest, "w") as f:
        json.dump({"pool_length_vs_logp": corr_out,
                   "length_only_baseline_fineweb": baseline}, f, indent=2)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
