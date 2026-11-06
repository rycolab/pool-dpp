"""
Weight diagnostics of the frontier kernels (tab:weights)
========================================================
Per task and T on one pool, the weights of the figure's kernel at that task's local
temperature sT:

    PoolDPP-Samp  w(y) = plm(y)^(1/T) / q(y)
    PoolDPP-MAP   w(y) = plm(y)^(1/T)         (Ambrosia at sT=1.5)

With wbar the weights self-normalized over the N draws of a prompt, ESS = 1 / sum_n wbar_n^2
and the sup ratio is N max_n wbar_n. Per (task, sT, T) we report over prompts:
    ESS mean/N, median, q25/q75, min; #prompts with ESS < K;
    median of max wbar; mean of Var(wbar); mean sup ratio; median log-weight range.

Output: analysis/weight_diagnostics.json

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/weight_diagnostics.py
"""

import json
import os
import pickle

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

FINEWEB = "cache-fineweb/google__gemma-2-9b-it_minilm_embeddings.pkl"
AMBROSIA = "cache-ambrosia/pool0/google__gemma-2-9b-it_minilm_embeddings.pkl"
GSM8K = "cache-gsm8k/pool0/google__gemma-2-9b-it_qwen_embeddings.pkl"
# task, embeddings pkl, sT, K, weight divides by q, T grid of the figure
CONFIGS = [
    ("gsm8k", GSM8K, 1.2, 10, True, [0.428571, 0.666667, 0.818182, 1, 1.03, 1.05, 1.07, 1.1, 1.22222]),
    ("fineweb", FINEWEB, 1.0, 10, True, [0.428571, 0.666667, 0.818182, 1, 1.22222]),
    ("ambrosia", AMBROSIA, 1.0, 3, True, [0.428571, 0.538462, 0.666667, 0.818182, 1]),
    ("ambrosia", AMBROSIA, 1.5, 3, False, [10, 7.5, 3, 1]),
]


def fmt(x):
    return f"{float(x):g}"


def per_prompt(d, T, uses_q):
    lp1 = np.asarray(d["logps_t1"], dtype=np.float64)
    c = np.asarray(d["counts"], dtype=np.float64)
    lw = lp1 / T
    if uses_q:
        lw = lw - np.asarray(d["logps"], dtype=np.float64)
    w = np.exp(lw - lw.max())
    wbar = w / (w * c).sum()
    N = c.sum()
    return dict(ess=1.0 / (c * wbar ** 2).sum(), max_wbar=float(wbar.max()),
                var_wbar=float((c * wbar ** 2).sum() / N - 1.0 / N ** 2),
                sup_ratio=float(N * wbar.max()), range=float(lw.max() - lw.min()), N=float(N))


def main():
    out = {}
    for task, rel, sT, K, uses_q, Ts in CONFIGS:
        path = os.path.join(ROOT_DIR, rel)
        if not os.path.exists(path):
            print(f"[skip] {task}: missing {rel}")
            continue
        with open(path, "rb") as f:
            emb = pickle.load(f)[sT]
        for T in Ts:
            rows = [per_prompt(d, T, uses_q) for d in emb.values() if len(d["logps_t1"]) >= 2]
            ess = np.array([r["ess"] for r in rows])
            N = rows[0]["N"]
            key = f"{task}|sT={fmt(sT)}|T={fmt(T)}"
            o = out[key] = {
                "n_prompts": len(ess), "N": N, "K": K, "mode": "samp" if uses_q else "map",
                "ess_mean_over_N": float(ess.mean() / N),
                "ess_median": float(np.median(ess)),
                "ess_q25": float(np.quantile(ess, 0.25)),
                "ess_q75": float(np.quantile(ess, 0.75)),
                "ess_min": float(ess.min()),
                "n_collapsed_lt_K": int((ess < K).sum()),
                "max_normalized_weight_median": float(np.median([r["max_wbar"] for r in rows])),
                "var_normalized_weight_mean": float(np.mean([r["var_wbar"] for r in rows])),
                "sup_ratio_mean": float(np.mean([r["sup_ratio"] for r in rows])),
                "logw_range_median": float(np.median([r["range"] for r in rows])),
            }
            print(f"{key:30s} ESS/N={o['ess_mean_over_N']:.3f} "
                  f"med={o['ess_median']:7.1f} [{o['ess_q25']:7.1f},{o['ess_q75']:7.1f}] "
                  f"min={o['ess_min']:6.1f} coll<K={o['n_collapsed_lt_K']:2d}/{o['n_prompts']} "
                  f"maxw={o['max_normalized_weight_median']:.3f} "
                  f"1e4var={1e4 * o['var_normalized_weight_mean']:.3f} "
                  f"sup={o['sup_ratio_mean']:.1f} range={o['logw_range_median']:5.1f} nats")

    dest = os.path.join(SCRIPT_DIR, "weight_diagnostics.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
