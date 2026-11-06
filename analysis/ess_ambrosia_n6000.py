"""ESS of the figure kernels on Ambrosia at N'=6000 (tab:weights)
===============================================================================
Merges the first 30 pools of N=200 into one candidate set per question at sT=1.0 and
sT=1.5 (unique strings, counts summed). Reports the ESS at N=200 (pool 0) and
N=6000 of the PoolDPP-Samp weights plm^(1/T)/q at sT=1.0 and of the PoolDPP-MAP
weights plm^(1/T) at sT=1.5, and the bound on the KL at
N=200 and N=6000 from the per-question bound rows (bound_raw_ambrosia.json),
median over questions:
  "bound_m1|{setting}": median_bound_m1_N200, median_bound_m1_N6000 (new,
      primary): second-moment hybrid
      log(1 + a_1 chi2_1 + sum_{c>=2} a_c ((K^c/(K)_c)^2 m_1^c - 1)), m_1 = 1 + chi2_1;
  "cert|{setting}": median_cert_N200, median_cert_N6000 (secondary, meaning
      unchanged): r_1 hybrid log(1 + a_1 chi2_1 + sum_{c>=2} a_c (K^c/(K)_c r_1^c - 1)).
Formulas in bound_formulas.py.

Output: analysis/ess_ambrosia_n6000.json
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import bound_formulas as bf  # noqa: E402
MODEL = "google__gemma-2-9b-it_minilm"
FNAME = f"{MODEL}_embeddings.pkl"
STS = [1.0, 1.5]
TS = {1.0: [0.428571, 0.538462, 0.666667, 0.818182, 1], 1.5: [10, 7.5, 3, 1]}
USES_Q = {1.0: True, 1.5: False}  # PoolDPP-Samp at sT=1.0 divides by q, PoolDPP-MAP at sT=1.5 does not
K = 3
N_POOLS = 30


def fmt(x):
    return f"{float(x):g}"


def merge(cache, sT, n_pools):
    store = {}
    for pi in range(n_pools):
        p = os.path.join(ROOT_DIR, cache, f"pool{pi}", FNAME)
        with open(p, "rb") as f:
            emb = pickle.load(f)[sT]
        for pt, d in emb.items():
            st = store.setdefault(pt, dict(idx={}, sents=[], phi=[], lp=[],
                                           lp1=[], cnt=[]))
            c_arr = np.asarray(d["counts"], dtype=np.int64)
            p_arr = np.asarray(d["embeddings"])
            l_arr = np.asarray(d["logps"], dtype=np.float64)
            l1_arr = np.asarray(d["logps_t1"], dtype=np.float64)
            for j, sj in enumerate(d["u_sents"]):
                k = st["idx"].get(sj)
                if k is None:
                    k = len(st["sents"])
                    st["idx"][sj] = k
                    st["sents"].append(sj)
                    st["phi"].append(p_arr[j])
                    st["lp"].append(l_arr[j])
                    st["lp1"].append(l1_arr[j])
                    st["cnt"].append(0)
                st["cnt"][k] += int(c_arr[j])
    out = {}
    for pt, st in store.items():
        out[pt] = dict(
            u_sents=np.asarray(st["sents"]),
            counts=np.asarray(st["cnt"], dtype=np.int64),
            logps=np.asarray(st["lp"]), logps_t1=np.asarray(st["lp1"]),
            embeddings=np.asarray(st["phi"]))
    return out


def ess_summary(per_prompt, T, uses_q):
    ess, N = [], None
    for _, d in per_prompt.items():
        lp1 = np.asarray(d["logps_t1"], dtype=np.float64)
        c = np.asarray(d["counts"], dtype=np.float64)
        if len(lp1) < 2:
            continue
        lw = lp1 / T - (np.asarray(d["logps"], dtype=np.float64) if uses_q else 0.0)
        w = np.exp(lw - lw.max())
        wbar = w / (w * c).sum()
        ess.append(1.0 / (c * wbar ** 2).sum())
        N = float(c.sum())
    ess = np.asarray(ess)
    return {"N": N, "n_prompts": len(ess),
            "ess_median": float(np.median(ess)),
            "ess_q25": float(np.quantile(ess, 0.25)),
            "ess_q75": float(np.quantile(ess, 0.75)),
            "ess_min": float(ess.min()),
            "n_collapsed_lt_K": int((ess < K).sum())}


def bound_m1_at(N, K_, chi2_1):
    """Second-moment hybrid bound on the KL at pool size N (primary)."""
    return bf.kl_hybrid_m1(N, K_, chi2_1)


def cert_at(N, K_, chi2_1, r1):
    """r_1 hybrid bound on the KL at pool size N (secondary)."""
    return bf.kl_hybrid_r1(N, K_, chi2_1, r1)


def run(a):
    merged = {sT: merge(a.cache, sT, a.n_pools) for sT in STS}
    n_uniq = {fmt(sT): float(np.mean([len(d["u_sents"]) for d in merged[sT].values()]))
              for sT in STS}
    n_draws = {fmt(sT): float(np.mean([d["counts"].sum() for d in merged[sT].values()]))
               for sT in STS}
    print(f"merged {len(merged[1.0])} prompts; mean uniques {n_uniq}; draws {n_draws}")

    with open(os.path.join(ROOT_DIR, a.cache, "pool0", FNAME), "rb") as f:
        pool0 = pickle.load(f)
    out = {"_meta": dict(cache=a.cache, n_pools=a.n_pools, mean_unique=n_uniq, mean_draws=n_draws,
                         weight="sT=1: plm^(1/T)/q (Samp); sT=1.5: plm^(1/T) (MAP)")}
    for sT in STS:
        for T in TS[sT]:
            o = {"N200_pool0": ess_summary(pool0[sT], T, USES_Q[sT]),
                 "N6000_merged": ess_summary(merged[sT], T, USES_Q[sT])}
            out[f"ess|sT={fmt(sT)}|T={fmt(T)}"] = o
            m = o["N6000_merged"]
            print(f"ESS sT={fmt(sT)} T={fmt(T)}: N=6000 med={m['ess_median']:8.1f} "
                  f"[{m['ess_q25']:8.1f},{m['ess_q75']:8.1f}] min={m['ess_min']:7.1f} "
                  f"collapsed<K={m['n_collapsed_lt_K']}/{m['n_prompts']}  "
                  f"(N=200 med={o['N200_pool0']['ess_median']:.1f})")

    xr = os.path.join(SCRIPT_DIR, a.bound_raw)
    if os.path.exists(xr):
        for key, rows in json.load(open(xr)).items():
            if not isinstance(rows, list) or not rows or "chi2_1" not in rows[0]:
                continue
            for r in rows:
                bf.m1_of_row(r)  # rows that store m1 must have m1 = 1 + chi2_1
            o_m1 = {"median_bound_m1_N200": float(np.median(
                        [bound_m1_at(200, K, r["chi2_1"]) for r in rows])),
                    "median_bound_m1_N6000": float(np.median(
                        [bound_m1_at(6000, K, r["chi2_1"]) for r in rows]))}
            out[f"bound_m1|{key}"] = o_m1
            print(f"bound on the KL (m1) {key}: N=200 {o_m1['median_bound_m1_N200']:.4f} -> "
                  f"N=6000 {o_m1['median_bound_m1_N6000']:.5f}")
            o = {"median_cert_N200": float(np.median([cert_at(200, K, r["chi2_1"], r["r1"])
                                                      for r in rows])),
                 "median_cert_N6000": float(np.median([cert_at(6000, K, r["chi2_1"], r["r1"])
                                                       for r in rows]))}
            out[f"cert|{key}"] = o
            print(f"  r1 version (secondary) {key}: N=200 {o['median_cert_N200']:.4f} -> "
                  f"N=6000 {o['median_cert_N6000']:.5f}")
    else:
        print(f"[warn] {xr} missing; bounds on the KL skipped")
    with open(os.path.join(SCRIPT_DIR, "ess_ambrosia_n6000.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("Saved analysis/ess_ambrosia_n6000.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache-ambrosia")
    ap.add_argument("--n-pools", type=int, default=N_POOLS)
    ap.add_argument("--bound-raw", default="bound_raw_ambrosia.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
