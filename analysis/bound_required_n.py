"""
Pool size required to certify a target KL, from the Z-free identity (tab:zfree-required-N)
=========================================================================
For a FIXED instance (kappa, phi, q) the identity makes the bound an explicit
function of the number of proposal draws M:

    Var(Zhat)/Z^2 = sum_c a_c(M) chi2_c,   a_c(M) = C(K,c)C(M-K,K-c)/C(M,K)

so chi2_1 and r_1 (properties of the target and the proposal, not of M) fix the
whole curve. We solve for the smallest M with

    lower  : log(1 + a_1(M) chi2_1)                       <= target   (necessary)
    hybrid : log(1 + a_1(M) chi2_1 + NA tail(M, r_1))     <= target   (sufficient)

Inputs are the per-prompt rows analysis/bound_raw_{task}.json.

Usage: python analysis/bound_required_n.py
"""

import argparse
import glob
import json
import math
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# task -> (K, experimental pool size N)
TASK_PARAMS = {"ambrosia": (3, 200), "fineweb": (10, 400), "gsm8k": (10, 100)}
TARGETS = [1.0, 0.1]


def a_coeffs(M, K):
    denom = math.comb(M, K)
    return [math.comb(K, c) * math.comb(M - K, K - c) / denom
            if (M - K) >= (K - c) else 0.0 for c in range(1, K + 1)]


def bound_at(M, K, chi2_1, r1, hybrid):
    ac = a_coeffs(M, K)
    v = ac[0] * chi2_1
    if hybrid:
        for c in range(2, K + 1):
            if ac[c - 1] == 0.0:
                continue
            v += ac[c - 1] * max(K ** c / math.perm(K, c) * r1 ** c - 1.0, 0.0)
    return math.log1p(v)


def required_M(K, chi2_1, r1, target, hybrid, M_hi=10 ** 9):
    lo, hi = 2 * K, M_hi
    if bound_at(hi, K, chi2_1, r1, hybrid) > target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if mid <= 2 * K or bound_at(mid, K, chi2_1, r1, hybrid) > target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=None, help="bound_raw json files (globs ok)")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "bound_required_n.json"))
    args = ap.parse_args()
    pats = args.inputs or [os.path.join(SCRIPT_DIR, "bound_raw_*.json")]
    paths = sorted({p for pat in pats for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no inputs match {pats}")
    rows_out = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for key, rows in data.items():
            if key.endswith("|summary") or not isinstance(rows, list) or not rows:
                continue
            if not isinstance(rows[0], dict) or "chi2_1" not in rows[0]:
                continue  # T1 enumeration rows, meta
            K = TASK_PARAMS[key.split("|")[0]][0]
            out = {}
            for target in TARGETS:
                for hybrid, name in ((False, "necessary_leading"),
                                     (True, "sufficient_hybrid")):
                    Ms = []
                    for r in rows:
                        M = required_M(K, r["chi2_1"], r["r1"], target, hybrid)
                        if M is not None:
                            Ms.append(M)
                    if Ms:
                        out[f"KL<={target}|{name}"] = dict(
                            median_M=int(np.median(Ms)),
                            q25=int(np.quantile(Ms, .25)),
                            q75=int(np.quantile(Ms, .75)),
                            log10_median=float(np.log10(np.median(Ms))),
                            n=len(Ms),
                        )
            rows_out[key] = out
            print(f"\n=== {key} (K={K}, {len(rows)} prompts) ===")
            for k, v in out.items():
                print(f"  {k:34s} median M = {v['median_M']:>12,d} "
                      f"(10^{v['log10_median']:.2f})  IQR [{v['q25']:,d}, {v['q75']:,d}]")
    rows_out["_inputs"] = [os.path.relpath(p, os.path.dirname(SCRIPT_DIR)) for p in paths]
    dest = args.out
    with open(dest, "w") as f:
        json.dump(rows_out, f, indent=2)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
