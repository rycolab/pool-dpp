"""
Weight diagnostics for the unified quality kernel (tab:weights)
=====================================================================
Per-task, per-anchor, per-tau distributions of the importance-weight ESS for the
unified tilted target,

    w(y) = plm(y)^(2*beta),   beta = tau / sigma_hat(task, LM, sT),

with sigma_hat read from the calibration JSON written by
`dpp-estimation --calibrate`. Weights are bounded above by construction (w <= 1 before normalization).

Per (task, sT, tau) we report over prompts:
    ESS mean/N, median, q25/q75, min; #prompts with ESS < K;
    max normalized weight; log-weight range 2*beta*(max lp1 - min lp1).

Output: analysis/weight_diagnostics.json (+ a LaTeX tabular on stdout).

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/weight_diagnostics.py
"""

import json
import os
import pickle

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

# task, embeddings pkl, calibration json, anchor sTs, K
CONFIGS = [
    ("fineweb", "cache-fineweb/google__gemma-2-9b-it_minilm_embeddings.pkl",
     "cache-fineweb/calibration_google__gemma-2-9b-it_minilm.json", [1.0], 10),
    ("ambrosia", "cache-ambrosia/pool0/google__gemma-2-9b-it_minilm_embeddings.pkl",
     "cache-ambrosia/calibration_google__gemma-2-9b-it_minilm.json", [1.0, 1.5], 3),
    ("gsm8k", "cache-gsm8k/pool0/google__gemma-2-9b-it_qwen_embeddings.pkl",
     "cache-gsm8k/calibration_google__gemma-2-9b-it_qwen.json", [1.2], 10),
]


def fmt(x):
    return f"{float(x):g}"


def per_prompt_ess(lp1, counts, beta):
    lp1 = np.asarray(lp1, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)
    lw = 2.0 * beta * lp1
    w = np.exp(lw - lw.max())
    tot = (w * c).sum()
    wbar = w / tot
    ess = 1.0 / (c * wbar ** 2).sum()
    return ess, float(wbar.max()), float(2.0 * beta * (lp1.max() - lp1.min()))


def main():
    out = {}
    for task, rel, cal_rel, sTs, K in CONFIGS:
        emb_path = os.path.join(ROOT_DIR, rel)
        cal_path = os.path.join(ROOT_DIR, cal_rel)
        if not (os.path.exists(emb_path) and os.path.exists(cal_path)):
            print(f"[skip] {task}: missing {rel} or calibration")
            continue
        with open(emb_path, "rb") as f:
            emb_all = pickle.load(f)
        with open(cal_path) as f:
            sigma = {float(k): float(v) for k, v in json.load(f)["sigma_hat"].items()}
        for sT in sTs:
            emb = emb_all[sT]
            for tau in TAUS:
                beta = tau / sigma[sT]
                ess_l, maxw_l, rng_l, N = [], [], [], None
                for d in emb.values():
                    lp1 = np.asarray(d["logps_t1"])
                    if len(lp1) < 2:
                        continue
                    ess, maxw, rng = per_prompt_ess(lp1, d["counts"], beta)
                    ess_l.append(ess)
                    maxw_l.append(maxw)
                    rng_l.append(rng)
                    N = float(np.sum(d["counts"]))
                ess = np.array(ess_l)
                key = f"{task}|sT={fmt(sT)}|tau={fmt(tau)}"
                out[key] = {
                    "n_prompts": len(ess), "N": N, "K": K,
                    "beta": beta,
                    "ess_mean_over_N": float(ess.mean() / N),
                    "ess_median": float(np.median(ess)),
                    "ess_q25": float(np.quantile(ess, 0.25)),
                    "ess_q75": float(np.quantile(ess, 0.75)),
                    "ess_min": float(ess.min()),
                    "n_collapsed_lt_K": int((ess < K).sum()),
                    "max_normalized_weight_median": float(np.median(maxw_l)),
                    "logw_range_median": float(np.median(rng_l)),
                }
                o = out[key]
                print(f"{key:30s} ESS/N={o['ess_mean_over_N']:.3f} "
                      f"med={o['ess_median']:7.1f} [{o['ess_q25']:7.1f},{o['ess_q75']:7.1f}] "
                      f"min={o['ess_min']:6.1f} coll<K={o['n_collapsed_lt_K']:2d}/{o['n_prompts']} "
                      f"maxw={o['max_normalized_weight_median']:.3f} "
                      f"range={o['logw_range_median']:5.1f} nats")

    dest = os.path.join(SCRIPT_DIR, "weight_diagnostics.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {dest}\n")

    # LaTeX tabular for the appendix (median ESS with collapse counts)
    print("% ---- appendix table: per-anchor ESS (unified scheme) ----")
    print(r"\begin{tabular}{llrrrrr}")
    print(r"\toprule")
    print(r"task & anchor & $\tautilt$ & ESS med & [q25, q75] & min & \#ESS$<K$ \\")
    print(r"\midrule")
    for key, o in out.items():
        task, sT, tau = key.split("|")
        print(f"{task} & {sT.replace('sT=', '$s_T$=')} & {tau.replace('tau=', '')} & "
              f"{o['ess_median']:.1f} & [{o['ess_q25']:.1f}, {o['ess_q75']:.1f}] & "
              f"{o['ess_min']:.1f} & {o['n_collapsed_lt_K']}/{o['n_prompts']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
