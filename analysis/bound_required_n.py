"""
Pool size required to bound the KL by a target, from the Z-free identity (tab:zfree-required-N)
=========================================================================
For a FIXED instance (kappa, phi, q) the identity makes the bound an explicit
function of the number of proposal draws N (spelled M in the code):

    Var(Zhat)/Z^2 = sum_c a_c(M) chi2_c,   a_c(M) = C(K,c)C(M-K,K-c)/C(M,K)

so Var_q(u_1) = chi2_1 (and, for the secondary bound, r_1), which do not depend
on M, fix the whole curve. For each prompt we find the smallest M0 such that,
for every M >= M0,

    lower H   : log(1 + (K^2/M) chi2_1)                        <= target  (necessary, primary)
    lower a1  : log(1 + a_1(M) chi2_1)                         <= target  (necessary, secondary)
    hybrid m1 : log(1 + a_1(M) chi2_1 + tail_m1(M, m_1))       <= target  (sufficient, primary)
    hybrid r1 : log(1 + a_1(M) chi2_1 + tail_r1(M, r_1))       <= target  (sufficient, secondary)

with m_1 = E_q[u_1^2] = 1 + chi2_1, tail_m1 = sum_{c>=2} a_c ((K^c/(K)_c)^2 m_1^c - 1)
and tail_r1 = sum_{c>=2} a_c (K^c/(K)_c r_1^c - 1) (formulas in bound_formulas.py).
The m1 bound depends on the prompt only through chi2_1, so bound_raw files written
before the key ``m1`` existed give the same answer.

"Necessary" is about the Var(Zhat)/Z^2 route: below that M, log(1 + Var/Z^2)
exceeds the target, so this route cannot bound the KL by the target there (the
KL itself may still be smaller). The primary lower bound is Hoeffding's,
Var(Zhat)/Z^2 >= (K^2/M) chi2_1 (Hoeffding 1948, Thm 5.2: Var(u_c)/c is
nondecreasing in c, and sum_c c a_c = K^2/M). It is at least a_1(M) chi2_1 and
decreases in M at every M, so its required M is the closed form
M >= K^2 chi2_1 / (e^target - 1) (bound_formulas.required_N_lower_hoeffding),
with no floor effects.

Search for the other three (bound_formulas.required_N). They are non-increasing in M only for
M >= K^2 - 1; below that the a_c are not monotone (a_1 = 0 for M < 2K - 1). So
we bisect on [K^2 - 1, --M-hi] and walk down below K^2 - 1 when the bound is
already under the target there (a result of K means the bound rules out no pool
size; n_at_K counts these prompts). A prompt that does not reach the target by
--M-hi raises an error.

Output keys (analysis/bound_required_n.json), per "{task}|T=..." setting:
  "KL<={target}|necessary_hoeffding"   lower bound on the required M from (K^2/M) chi2_1
                                       (primary)
  "KL<={target}|necessary_leading"     lower bound on the required M from a_1(M) chi2_1
                                       (secondary)
  "KL<={target}|sufficient_hybrid_m1"  M from the second-moment hybrid (primary)
  "KL<={target}|sufficient_hybrid"     M from the r_1 hybrid (secondary)
each with median_M, q25, q75, log10_median, n (= number of prompts) and
n_at_K (prompts whose answer is M = K).

Inputs are the per-prompt rows analysis/bound_raw_{task}.json.

Usage: python analysis/bound_required_n.py
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import bound_formulas as bf  # noqa: E402

# task -> (K, experimental pool size N)
TASK_PARAMS = {"ambrosia": (3, 200), "fineweb": (10, 400), "gsm8k": (10, 100)}
TARGETS = [1.0, 0.1]
M_HI = 10 ** 9

# name -> log(1 + bound on Var(Zhat)/Z^2) at pool size M for one row (K fixed per task)
BOUNDS = {
    "necessary_hoeffding": lambda M, K, r: math.log1p(bf.lower_hoeffding(M, K, r["chi2_1"])),
    "necessary_leading": lambda M, K, r: math.log1p(bf.leading(M, K, r["chi2_1"])),
    "sufficient_hybrid_m1": lambda M, K, r: bf.kl_hybrid_m1(M, K, r["chi2_1"]),
    "sufficient_hybrid": lambda M, K, r: bf.kl_hybrid_r1(M, K, r["chi2_1"], r["r1"]),
}


def required_M(K, row, target, name, M_hi=M_HI):
    """Smallest pool size M0 with BOUNDS[name](M) <= target for every M >= M0."""
    if name == "necessary_hoeffding":   # monotone in M: closed form
        return bf.required_N_lower_hoeffding(K, row["chi2_1"], target)
    return bf.required_N(lambda M: BOUNDS[name](M, K, row), K, target, M_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=None, help="bound_raw json files (globs ok)")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "bound_required_n.json"))
    ap.add_argument("--M-hi", type=int, default=M_HI,
                    help="largest pool size searched; a prompt above target there is an error")
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
            for r in rows:
                bf.m1_of_row(r)  # rows that store m1 must have m1 = 1 + chi2_1
            out = {}
            for target in TARGETS:
                for name in BOUNDS:
                    Ms = [required_M(K, r, target, name, args.M_hi) for r in rows]
                    out[f"KL<={target}|{name}"] = dict(
                        median_M=int(np.median(Ms)),
                        q25=int(np.quantile(Ms, .25)),
                        q75=int(np.quantile(Ms, .75)),
                        log10_median=float(np.log10(np.median(Ms))),
                        n=len(Ms),
                        n_at_K=int(sum(M == K for M in Ms)),
                    )
            rows_out[key] = out
            print(f"\n=== {key} (K={K}, {len(rows)} prompts) ===")
            for k, v in out.items():
                tag = ("  [r1, secondary]" if k.endswith("|sufficient_hybrid")
                       else "  [a1, secondary]" if k.endswith("|necessary_leading") else "")
                print(f"  {k:36s} median M = {v['median_M']:>12,d} "
                      f"(10^{v['log10_median']:.2f})  IQR [{v['q25']:,d}, {v['q75']:,d}]"
                      f"  at M=K: {v['n_at_K']}{tag}")
    rows_out["_inputs"] = [os.path.relpath(p, os.path.dirname(SCRIPT_DIR)) for p in paths]
    dest = args.out
    with open(dest, "w") as f:
        json.dump(rows_out, f, indent=2)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
