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

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/length_analysis.py
"""

import json
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

TASKS = {
    "fineweb": {"emb": "cache-fineweb/google__gemma-2-9b-it_minilm_embeddings.pkl"},
    "ambrosia": {"emb": "cache-ambrosia/pool0/google__gemma-2-9b-it_minilm_embeddings.pkl"},
    "gsm8k": {"emb": "cache-gsm8k/pool0/google__gemma-2-9b-it_qwen_embeddings.pkl"},
}


def logdet(F):
    F = np.asarray(F, dtype=np.float64)
    if len(F) < 2:
        return np.nan
    G = F @ F.T + 1e-6 * np.eye(len(F))
    s, ld = np.linalg.slogdet(G)
    return ld if s > 0 else np.nan


def pool_length_vs_logp():
    corr_out = {}
    for task, cfg in TASKS.items():
        emb_path = os.path.join(ROOT_DIR, cfg["emb"])
        if not os.path.exists(emb_path):
            continue
        with open(emb_path, "rb") as f:
            emb = pickle.load(f)
        print(f"--- {task}: pool length vs log p ---")
        for sT in sorted(emb.keys()):
            L, P = [], []
            for _, d in emb[sT].items():
                for s, lp1 in zip(d["u_sents"], d["logps_t1"]):
                    L.append(len(s))
                    P.append(float(lp1))
            if len(L) < 10:
                continue
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
    emb_path = os.path.join(ROOT_DIR, TASKS["fineweb"]["emb"])
    with open(emb_path, "rb") as f:
        emb = pickle.load(f)
    if sT not in emb:
        return {}
    res = defaultdict(lambda: {"q": [], "d": []})
    for _, d in emb[sT].items():
        u = list(d["u_sents"])
        lp1 = np.asarray(d["logps_t1"], dtype=np.float64)
        E = np.asarray(d["embeddings"], dtype=np.float64)
        if len(u) < k or len(u) != len(lp1):
            continue
        L = np.array([len(s) for s in u], dtype=np.float64)

        # (a) the K longest
        idx = np.argsort(-L)[:k]
        res["length-topk"]["q"].append(float(lp1[idx].mean()))
        res["length-topk"]["d"].append(logdet(E[idx]))

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

        # (c) reference: pure top-K by log p (no diversity term at all)
        idx = np.argsort(-lp1)[:k]
        res["quality-topk"]["q"].append(float(lp1[idx].mean()))
        res["quality-topk"]["d"].append(logdet(E[idx]))

    out = {m: {"quality_mean_logp": float(np.nanmean(v["q"])),
               "kernel_logdet_minilm": float(np.nanmean(v["d"])),
               "n_prompts": len(v["q"])}
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
