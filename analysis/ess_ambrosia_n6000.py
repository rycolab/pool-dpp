"""ESS of the unified-kernel weights on Ambrosia at N'=6000 (tab:weights, panel b)
===============================================================================
Merges the 30 pools of N=200 into one candidate set per question at sT=1.0 and
sT=1.5 (unique strings, counts summed). Reports the ESS of the L weights
plm^(2 tau/sigma_hat) at N=200 (pool 0) and N=6000, and the certified KL at
N=200 and N=6000 from the per-question bound rows.

sigma_hat is the cache-ambrosia calibration (pool 0), the constant the N=200
runs use, so a tau means the same kernel at both pool sizes.

Output: analysis/ess_ambrosia_n6000.json
"""

import argparse
import json
import math
import os
import pickle

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL = "google__gemma-2-9b-it_minilm"
FNAME = f"{MODEL}_embeddings.pkl"
STS = [1.0, 1.5]
TAUS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
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


def ess_summary(per_prompt, beta):
    ess, N = [], None
    for _, d in per_prompt.items():
        lp1 = np.asarray(d["logps_t1"], dtype=np.float64)
        c = np.asarray(d["counts"], dtype=np.float64)
        if len(lp1) < 2:
            continue
        lw = 2.0 * beta * lp1
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


def cert_at(N, K_, chi2_1, r1):
    denom = math.comb(N, K_)
    ac = [math.comb(K_, c) * math.comb(N - K_, K_ - c) / denom
          for c in range(1, K_ + 1)]
    v = ac[0] * chi2_1
    for c in range(2, K_ + 1):
        if ac[c - 1] == 0.0:
            continue
        v += ac[c - 1] * max(K_ ** c / math.perm(K_, c) * r1 ** c - 1.0, 0.0)
    return math.log1p(v)


def run(a):
    merged = {sT: merge(a.cache, sT, a.n_pools) for sT in STS}
    n_uniq = {fmt(sT): float(np.mean([len(d["u_sents"]) for d in merged[sT].values()]))
              for sT in STS}
    n_draws = {fmt(sT): float(np.mean([d["counts"].sum() for d in merged[sT].values()]))
               for sT in STS}
    print(f"merged {len(merged[1.0])} prompts; mean uniques {n_uniq}; draws {n_draws}")

    with open(os.path.join(ROOT_DIR, a.calibration)) as f:
        sig = {float(k): float(v) for k, v in json.load(f)["sigma_hat"].items()}
    with open(os.path.join(ROOT_DIR, a.cache, "pool0", FNAME), "rb") as f:
        pool0 = pickle.load(f)
    out = {"_meta": dict(cache=a.cache, n_pools=a.n_pools,
                         calibration=a.calibration, sigma_hat={fmt(s): sig[s] for s in STS},
                         mean_unique=n_uniq, mean_draws=n_draws,
                         weight="L weight plm^(2 tau/sigma_hat)")}
    for sT in STS:
        for tau in TAUS:
            beta = tau / sig[sT]
            o = {"N200_pool0": ess_summary(pool0[sT], beta),
                 "N6000_merged": ess_summary(merged[sT], beta)}
            out[f"ess|sT={fmt(sT)}|tau={fmt(tau)}"] = o
            m = o["N6000_merged"]
            print(f"ESS sT={fmt(sT)} tau={fmt(tau)}: N=6000 med={m['ess_median']:8.1f} "
                  f"[{m['ess_q25']:8.1f},{m['ess_q75']:8.1f}] min={m['ess_min']:7.1f} "
                  f"collapsed<K={m['n_collapsed_lt_K']}/{m['n_prompts']}  "
                  f"(N=200 med={o['N200_pool0']['ess_median']:.1f})")

    xr = os.path.join(SCRIPT_DIR, a.bound_raw)
    if os.path.exists(xr):
        for key, rows in json.load(open(xr)).items():
            if not isinstance(rows, list) or not rows or "chi2_1" not in rows[0]:
                continue
            o = {"median_cert_N200": float(np.median([cert_at(200, K, r["chi2_1"], r["r1"])
                                                      for r in rows])),
                 "median_cert_N6000": float(np.median([cert_at(6000, K, r["chi2_1"], r["r1"])
                                                       for r in rows]))}
            out[f"cert|{key}"] = o
            print(f"cert {key}: N=200 {o['median_cert_N200']:.4f} -> "
                  f"N=6000 {o['median_cert_N6000']:.5f}")
    else:
        print(f"[warn] {xr} missing; certificates skipped")
    with open(os.path.join(SCRIPT_DIR, "ess_ambrosia_n6000.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("Saved analysis/ess_ambrosia_n6000.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache-ambrosia")
    ap.add_argument("--n-pools", type=int, default=N_POOLS)
    ap.add_argument("--calibration",
                    default="cache-ambrosia/calibration_google__gemma-2-9b-it_minilm.json")
    ap.add_argument("--bound-raw", default="bound_raw_ambrosia.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
