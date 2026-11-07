"""ESS of the PoolDPP-Samp weights on FineWeb at N'=8000 (tab:weights)
==============================================================================
Two pools of 4000 draws per prompt (seeds 101 and 202, sT=1.0), embedded with
MiniLM and merged per prompt (unique strings, counts summed). Reports the ESS of
the weights plm^(1/T)/q for N=4000 (each half) and N=8000 (merged)
next to N=400 (pool 0), the cross/within collision ratio between the halves, and
the bound on the KL at N in {400, 4000, 8000} from the per-question bound rows
(bound_raw_fineweb.json), median over questions:
  "bound_m1|{setting}" (primary): second-moment hybrid
      log(1 + a_1 chi2_1 + sum_{c>=2} a_c ((K^c/(K)_c)^2 m_1^c - 1)), m_1 = 1 + chi2_1;
  "cert|{setting}" (secondary): r_1 hybrid
      log(1 + a_1 chi2_1 + sum_{c>=2} a_c (K^c/(K)_c r_1^c - 1)).
Formulas in bound_formulas.py.

Failures stop the run with an error: a missing pool file or bound_raw file, pools with different
prompt sets, a prompt whose number of draws is not N, logps and logps_t1 of different lengths, a
non-finite log-weight, or a bound_raw entry that is not a non-empty list of rows (keys starting
with "_" are metadata and are not read). Prompts with fewer than 2 distinct strings are left out of
the ESS (their ESS is N by construction); their number is written as n_prompts_excluded_lt2_unique.

Output: analysis/ess_fineweb_n8000.json
"""

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import bound_formulas as bf  # noqa: E402
TS = [0.428571, 0.666667, 0.818182, 1, 1.22222]
K = 10
MODEL = "google__gemma-2-9b-it_minilm"
FNAME = f"{MODEL}_embeddings.pkl"


def fmt(x):
    return f"{float(x):g}"


def load(cache):
    path = os.path.join(ROOT_DIR, cache, FNAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"fineweb: missing pool file {path}")
    with open(path, "rb") as f:
        return pickle.load(f)[1.0]


def same_prompts(named):
    """Raise unless every {name: per-prompt dict} has the same prompts."""
    (n0, d0), *rest = named.items()
    for n, d in rest:
        if set(d) != set(d0):
            raise ValueError(f"fineweb: {n} and {n0} have different prompts "
                             f"({len(set(d0) - set(d))} only in {n0}, {len(set(d) - set(d0))} only in {n})")


def merge_two(da, db):
    """Merge two per-prompt dicts, summing counts on duplicate strings."""
    same_prompts({"half a": da, "half b": db})
    out = {}
    for pt in da:
        idx, sents, phi, lp, lp1, cnt = {}, [], [], [], [], []
        for d in (da[pt], db[pt]):
            s_arr = list(d["u_sents"])
            c_arr = np.asarray(d["counts"], dtype=np.int64)
            p_arr = np.asarray(d["embeddings"])
            l_arr = np.asarray(d["logps"], dtype=np.float64)
            l1_arr = np.asarray(d["logps_t1"], dtype=np.float64)
            for j, sj in enumerate(s_arr):
                k = idx.get(sj)
                if k is None:
                    k = len(sents)
                    idx[sj] = k
                    sents.append(sj)
                    phi.append(p_arr[j])
                    lp.append(l_arr[j])
                    lp1.append(l1_arr[j])
                    cnt.append(0)
                cnt[k] += int(c_arr[j])
        out[pt] = dict(u_sents=np.asarray(sents), counts=np.asarray(cnt, dtype=np.int64),
                       logps=np.asarray(lp), logps_t1=np.asarray(lp1),
                       embeddings=np.asarray(phi))
    return out


def ess_rows(per_prompt, T, N, where):
    """-> (ESS per prompt, #prompts left out for having fewer than 2 distinct strings)."""
    rows, n_lt2 = [], 0
    for pt, d in per_prompt.items():
        lp = np.asarray(d["logps"], dtype=np.float64)
        lp1 = np.asarray(d["logps_t1"], dtype=np.float64)
        c = np.asarray(d["counts"], dtype=np.float64)
        if lp.shape != lp1.shape or lp.shape != c.shape:
            raise ValueError(f"{where} prompt {pt[:60]!r}: logps {lp.shape}, logps_t1 {lp1.shape}, "
                             f"counts {c.shape} differ")
        if c.sum() != N:
            raise ValueError(f"{where} prompt {pt[:60]!r}: {c.sum():g} draws, expected N={N}")
        if len(lp) < 2:
            n_lt2 += 1
            continue
        lw = lp1 / T - lp
        if not np.all(np.isfinite(lw)):
            raise ValueError(f"{where} prompt {pt[:60]!r}: non-finite log-weight at T={fmt(T)}")
        w = np.exp(lw - lw.max())
        wbar = w / (w * c).sum()
        rows.append(1.0 / (c * wbar ** 2).sum())
    return np.asarray(rows), n_lt2


def summarize(ess_and_excluded, N):
    ess, n_lt2 = ess_and_excluded
    return {
        "N": N, "n_prompts": len(ess), "n_prompts_excluded_lt2_unique": n_lt2,
        "ess_mean_over_N": float(ess.mean() / N),
        "ess_median": float(np.median(ess)),
        "ess_q25": float(np.quantile(ess, 0.25)),
        "ess_q75": float(np.quantile(ess, 0.75)),
        "ess_min": float(ess.min()),
        "n_collapsed_lt_K": int((ess < K).sum()),
    }


def bound_rows(path):
    """(setting, rows) of a bound_raw file; keys starting with "_" are metadata. Raises on an
    entry that is not a non-empty list of rows with a finite chi2_1 and r1."""
    out = []
    for key, rows in json.load(open(path)).items():
        if key.startswith("_"):
            continue
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: entry {key!r} is not a non-empty list of rows")
        for i, r in enumerate(rows):
            bad = [f for f in ("chi2_1", "r1") if not np.isfinite(r.get(f, np.nan))]
            if bad:
                raise ValueError(f"{path}: {key!r} row {i} has no finite {bad}")
            bf.m1_of_row(r)  # rows that store m1 must have m1 = 1 + chi2_1
        out.append((key, rows))
    return out


def bound_m1_at(N, chi2_1):
    """Second-moment hybrid bound on the KL at pool size N (primary)."""
    return bf.kl_hybrid_m1(N, K, chi2_1)


def cert_at(N, chi2_1, r1):
    """r_1 hybrid bound on the KL at pool size N (secondary)."""
    return bf.kl_hybrid_r1(N, K, chi2_1, r1)


def cross_within_ratio(da, db):
    same_prompts({"half a": da, "half b": db})
    X = E = 0.0
    for pt in da:
        ca = dict(zip(da[pt]["u_sents"], np.asarray(da[pt]["counts"], float)))
        cb = dict(zip(db[pt]["u_sents"], np.asarray(db[pt]["counts"], float)))
        X += sum(c * cb[s] for s, c in ca.items() if s in cb)
        Na, Nb = sum(ca.values()), sum(cb.values())
        Qa = sum(c * (c - 1) / 2 for c in ca.values()) / (Na * (Na - 1) / 2)
        Qb = sum(c * (c - 1) / 2 for c in cb.values()) / (Nb * (Nb - 1) / 2)
        E += Na * Nb * (Qa + Qb) / 2
    return dict(cross_matched_pairs=float(X), expected_if_independent=float(E),
                ratio=(float(X / E) if E > 0 else None))


def run(a):
    da, db = load(a.halves[0]), load(a.halves[1])
    merged = merge_two(da, db)
    print(f"merged {len(merged)} prompts; uniques per prompt: "
          f"{np.mean([len(d['u_sents']) for d in merged.values()]):.0f} avg")

    pool0 = load(a.pool0_cache)
    same_prompts({a.pool0_cache: pool0, a.halves[0]: da, a.halves[1]: db})
    out = {"_meta": dict(halves=a.halves, weight="plm^(1/T)/q"),
           "independence_between_halves": cross_within_ratio(da, db)}
    print("independence:", out["independence_between_halves"])
    for T in TS:
        row = {"pool0_N400": summarize(ess_rows(pool0, T, 400, a.pool0_cache), 400),
               "half_a_N4000": summarize(ess_rows(da, T, 4000, a.halves[0]), 4000),
               "half_b_N4000": summarize(ess_rows(db, T, 4000, a.halves[1]), 4000),
               "merged_N8000": summarize(ess_rows(merged, T, 8000, "merged halves"), 8000)}
        out[f"T={fmt(T)}"] = row
        m = row["merged_N8000"]
        print(f"T={fmt(T)}: N=8000 ESS med={m['ess_median']:7.1f} "
              f"[{m['ess_q25']:7.1f},{m['ess_q75']:7.1f}] min={m['ess_min']:6.1f} "
              f"collapsed<K={m['n_collapsed_lt_K']}/{m['n_prompts']}")
    xr = os.path.join(SCRIPT_DIR, a.bound_raw)
    if not os.path.exists(xr):
        raise FileNotFoundError(f"{xr} missing; run bound_empirical.py --task fineweb first")
    for key, rows in bound_rows(xr):
        out[f"bound_m1|{key}"] = {str(N): float(np.median([bound_m1_at(N, r["chi2_1"])
                                                          for r in rows]))
                                  for N in (400, 4000, 8000)}
        print(f"bound on the KL (m1) {key}: {out[f'bound_m1|{key}']}")
        out[f"cert|{key}"] = {str(N): float(np.median([cert_at(N, r["chi2_1"], r["r1"])
                                                      for r in rows]))
                              for N in (400, 4000, 8000)}
        print(f"  r1 version (secondary) {key}: {out[f'cert|{key}']}")
    ind = out["independence_between_halves"]
    X, E = ind["cross_matched_pairs"], ind["expected_if_independent"]
    if X > 1.5 * E + 3 * math.sqrt(E) + 5:   # checked before saving, so no output is written
        print(f"ERROR: halves are not independent (X={X:.0f}, E={E:.1f})")
        raise SystemExit(2)
    with open(os.path.join(SCRIPT_DIR, "ess_fineweb_n8000.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("Saved analysis/ess_fineweb_n8000.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halves", nargs=2,
                    default=["cache-fineweb-n8000-a", "cache-fineweb-n8000-b"])
    ap.add_argument("--pool0-cache", default="cache-fineweb")
    ap.add_argument("--bound-raw", default="bound_raw_fineweb.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
