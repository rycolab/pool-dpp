"""
Tables for the Z-free bound test (tab:zfree-certificates, reads bound_raw_*.json)
=================================================================================
  Tab A  bound at the experimental setting: exact leading term, honest MC value
         of Var(Zhat)/Z^2, the bound on the KL, and tightness vs the exact
         reduced quantity
  Tab B  M-sweep on the true instance (disjoint within-pool blocks) against the
         bracket [(K^2/m) chi2_1, m_1 upper bound] (primary) and the earlier
         [a_1(m) chi2_1, r_1 upper bound] (secondary)
  Tab C  independence diagnostics for the pools

Two bounds on the KL (formulas in bound_formulas.py):
  primary, second moment, m_1 = E_q[u_1^2] = 1 + chi2_1:
      hybrid log(1 + a_1 chi2_1 + sum_{c>=2} a_c ((K^c/(K)_c)^2 m_1^c - 1))
          (= log sum_{c=0}^K a_c (K^c/(K)_c)^2 m_1^c, cor:kl-sum),
      closed form log(e^theta - 1 + a_0) <= theta = K^2 m_1/(N-K+1)
          (thm:zfree; a_0 = C(N-K,K)/C(N,K));
  secondary (the paper's footnote), r_1 = sup u_1:
      hybrid log(1 + a_1 chi2_1 + sum_{c>=2} a_c (K^c/(K)_c r_1^c - 1)),
      closed form K^2 r_1/(N-K+1).
Two lower bounds on Var(Zhat)/Z^2: (K^2/N) chi2_1 (Hoeffding 1948, Thm 5.2;
primary) and the exact leading term a_1 chi2_1 (secondary).
The m_1 and Hoeffding quantities are recomputed here from chi2_1 (m_1 = 1 + chi2_1 by the
definition of chi2_1), so bound_raw files written before the m1 keys existed
give the same tables; when a row stores m1 it is checked against 1 + chi2_1.

Output keys (analysis/bound_tables.json), per setting:
  missing_mass                                 median Good-Turing ESTIMATE of the
                                               proposal mass no pool drew (unchanged)
  log10_covered_mass, covered_mass             (new) median exact proposal mass
                                               sum_y q(y) of the distinct observed
                                               strings, as log10 and as a value
                                               (bound_empirical.py, "Covered proposal
                                               mass"); None, and printed as missing,
                                               for bound_raw files written before
                                               these keys existed
  m1 [median, q25, q75]                        (new)
  lower_hoeffding, lower_hoeffding_over_mc     (new, primary) median (K^2/N) chi2_1 and
                                               its median ratio to the MC relvar
  leading, leading_over_mc                     (secondary lower bound; meaning unchanged)
  bound_hybrid_m1, bound_simple_m1             (new, primary) medians; bound_simple_m1 is
                                               log(e^theta - 1 + a_0) (it was theta in the
                                               first commit of this change)
  bound_theta_m1                               (new) median theta = K^2 m_1/(N-K+1)
  tightness_m1_over_actual                     (new) median bound_hybrid_m1 / log(1 + MC relvar)
  frac_hybrid_m1_above_r1                      (new) fraction of prompts where the
                                               m_1 hybrid exceeds the r_1 hybrid
  r1, cert_hybrid, cert_simple,
  tightness_cert_over_actual                   (secondary, r_1; meaning unchanged)
  sweep[m].lower_hoeffding, upper_m1           (new) primary bracket, medians
  sweep[m].frac_in_bracket_m1                  (new) fraction in [lower_hoeffding, upper_m1]
                                               (in the first commit of this change the
                                               lower end was a_1 chi2_1)
  sweep[m].frac_below_lower_hoeffding          (new) fraction below (K^2/m) chi2_1
  sweep[m].lower, upper, frac_in_bracket,
  frac_below_lower                             (a_1 lower, r_1 upper; meaning unchanged)
  all other keys unchanged.

Usage: python analysis/bound_tables.py
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


def q(vals, p):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.quantile(v, p)) if v else float("nan")


def med(vals):
    return q(vals, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=None, help="bound_raw json files (globs ok)")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "bound_tables.json"))
    args = ap.parse_args()
    pats = args.inputs or [os.path.join(SCRIPT_DIR, "bound_raw_*.json")]
    paths = sorted({p for pat in pats for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no inputs match {pats}")
    out = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for key, rows in data.items():
            if not rows or not isinstance(rows, list) or "chi2_1" not in rows[0]:
                continue
            K, M_exp = TASK_PARAMS[key.split("|")[0]]

            lead = [r["leading"] for r in rows]
            mc = [r["mc_relvar"] for r in rows]
            meas = [r["measured_relvar"] for r in rows]
            # primary: second-moment bound, from chi2_1 (m_1 = 1 + chi2_1)
            m1 = [bf.m1_of_row(r) for r in rows]
            hyb_m1 = [bf.kl_hybrid_m1(M_exp, K, r["chi2_1"]) for r in rows]
            simple_m1 = [bf.kl_closed_m1(M_exp, K, x) for x in m1]
            theta_m1 = [bf.kl_theta_m1(M_exp, K, x) for x in m1]
            lower_h = [bf.lower_hoeffding(M_exp, K, r["chi2_1"]) for r in rows]
            # secondary: r_1 bound, as stored by bound_empirical.py
            cert = [r["cert_hybrid_KL"] for r in rows]
            simple = [r["cert_simple_KL"] for r in rows]
            # stored values must match the shared formulas at this (K, M_exp)
            for r, h in zip(rows, hyb_m1):
                chk = [(h, r.get("bound_hybrid_m1_KL"), "bound_hybrid_m1_KL"),
                       (bf.kl_hybrid_r1(M_exp, K, r["chi2_1"], r["r1"]),
                        r["cert_hybrid_KL"], "cert_hybrid_KL")]
                for new_v, stored, name in chk:
                    if stored is not None and not math.isclose(new_v, stored, rel_tol=1e-9):
                        raise ValueError(f"{key}: stored {name}={stored!r} but the "
                                         f"formula at K={K}, N={M_exp} gives {new_v!r}")
            frac_m1_above = float(np.mean([h > c for h, c in zip(hyb_m1, cert)]))
            # exact KL implied by the honest MC value of the reduced quantity
            kl_mc = [math.log1p(x) for x in mc if np.isfinite(x)]
            tight_m1 = [h / math.log1p(r["mc_relvar"]) for h, r in zip(hyb_m1, rows)
                        if r["mc_relvar"] and np.isfinite(r["mc_relvar"])]
            tight = [r["cert_hybrid_KL"] / math.log1p(r["mc_relvar"])
                     for r in rows if r["mc_relvar"] and np.isfinite(r["mc_relvar"])]
            lead_frac = [r["leading"] / r["mc_relvar"] for r in rows
                         if r["mc_relvar"] and np.isfinite(r["mc_relvar"]) and r["mc_relvar"] > 0]
            lower_h_frac = [lh / r["mc_relvar"] for lh, r in zip(lower_h, rows)
                            if r["mc_relvar"] and np.isfinite(r["mc_relvar"])
                            and r["mc_relvar"] > 0]
            # ratio of the honest value to the across-pool measurement
            infl = [r["mc_relvar"] / r["measured_relvar"] for r in rows
                    if r.get("measured_relvar") and r["measured_relvar"] > 0
                    and np.isfinite(r["mc_relvar"])]

            Ms = sorted({int(m) for r in rows for m in r["sub_relvar"]})
            sweep = {}
            for m in Ms:
                o = [r["sub_relvar"][str(m)] if str(m) in r["sub_relvar"]
                     else r["sub_relvar"].get(m) for r in rows]
                o = [x for x in o if x is not None]
                lo = [r["pred_leading_by_m"].get(str(m), r["pred_leading_by_m"].get(m))
                      for r in rows]
                hi = [r["pred_upper_by_m"].get(str(m), r["pred_upper_by_m"].get(m))
                      for r in rows]
                lo = [x for x in lo if x is not None]
                hi = [x for x in hi if x is not None]
                # primary bracket at m, recomputed from chi2_1:
                # Hoeffding lower (K^2/m) chi2_1 and second-moment upper bound
                lo_h = [bf.lower_hoeffding(m, K, r["chi2_1"]) for r in rows]
                hi_m1 = [bf.relvar_bound_m1(m, K, r["chi2_1"]) for r in rows]
                inb, inb_m1, below_h = [], [], []
                for r, lv_h, hv_m1 in zip(rows, lo_h, hi_m1):
                    ov = r["sub_relvar"].get(str(m), r["sub_relvar"].get(m))
                    lv = r["pred_leading_by_m"].get(str(m), r["pred_leading_by_m"].get(m))
                    hv = r["pred_upper_by_m"].get(str(m), r["pred_upper_by_m"].get(m))
                    if ov is not None:
                        inb_m1.append(lv_h <= ov <= hv_m1)
                        below_h.append(ov < lv_h)
                    if ov is not None and lv is not None and hv is not None:
                        inb.append(lv <= ov <= hv)
                sweep[m] = dict(obs=med(o), lower_hoeffding=med(lo_h), lower=med(lo),
                                upper_m1=med(hi_m1), upper=med(hi),
                                obs_times_m=med(o) * m,
                                frac_in_bracket_m1=(float(np.mean(inb_m1))
                                                    if inb_m1 else None),
                                frac_below_lower_hoeffding=(float(np.mean(below_h))
                                                            if below_h else None),
                                frac_in_bracket=float(np.mean(inb)) if inb else None,
                                frac_below_lower=float(np.mean(
                                    [not b for b in inb])) if inb else None,
                                n_prompts=len(o))

            iid = {k: med([r["iid"][k] for r in rows]) for k in rows[0]["iid"]}

            # exact covered proposal mass (absent from bound_raw files written
            # before bound_empirical.py stored it; a file with it on some rows
            # only is corrupt)
            has_cov = ["log10_covered_mass" in r for r in rows]
            if all(has_cov):
                log10_cov = med([r["log10_covered_mass"] for r in rows])
                cov = med([r["covered_mass"] for r in rows])
            elif not any(has_cov):
                log10_cov = cov = None
            else:
                raise ValueError(f"{key}: log10_covered_mass is on {sum(has_cov)} of "
                                 f"{len(rows)} rows of {path}")

            out[key] = dict(
                n_prompts=len(rows), K=K, M_exp=M_exp,
                n_uniq=med([r["n_uniq"] for r in rows]),
                missing_mass=med([r["missing_mass"] for r in rows]),   # Good-Turing estimate
                log10_covered_mass=log10_cov, covered_mass=cov,          # exact
                chi2_1=[med([r["chi2_1"] for r in rows]),
                        q([r["chi2_1"] for r in rows], .25),
                        q([r["chi2_1"] for r in rows], .75)],
                m1=[med(m1), q(m1, .25), q(m1, .75)],
                r1=med([r["r1"] for r in rows]),
                unbiasedness_EZ_over_Z=med([r["mc_mean_ratio"] for r in rows]),
                lower_hoeffding=med(lower_h), lower_hoeffding_over_mc=med(lower_h_frac),
                leading=med(lead), mc_relvar=med(mc),
                mc_relvar_iqr=[q(mc, .25), q(mc, .75)],
                leading_over_mc=med(lead_frac),
                kl_from_mc=med(kl_mc),
                bound_hybrid_m1=med(hyb_m1), bound_simple_m1=med(simple_m1),
                bound_theta_m1=med(theta_m1),
                tightness_m1_over_actual=med(tight_m1),
                frac_hybrid_m1_above_r1=frac_m1_above,
                cert_hybrid=med(cert),
                cert_simple=med(simple),
                tightness_cert_over_actual=med(tight),
                measured_across_pool=med(meas),
                inflation_mc_over_measured=med(infl),
                sweep=sweep, iid=iid,
            )

            s = out[key]
            print(f"\n{'='*78}\n{key}   (K={K}, M_exp={M_exp}, "
                  f"{s['n_prompts']} prompts)\n{'='*78}")
            print(f"  ground set |Y| (median)              {s['n_uniq']:.0f}")
            print(f"  Good-Turing missing mass (estimate)  {s['missing_mass']:.4f}")
            if s["log10_covered_mass"] is None:
                print("  exact covered proposal mass          not in this bound_raw file "
                      "(rerun bound_empirical.py)")
            else:
                print(f"  exact covered proposal mass (log10, median)  "
                      f"{s['log10_covered_mass']:.4g}")
                print(f"  exact covered proposal mass (median)         "
                      f"{s['covered_mass']:.4g}")
            print(f"  E[Zhat]/Z from MC (must be 1)        {s['unbiasedness_EZ_over_Z']:.4f}")
            print(f"  chi2(pi_1||q)  median [IQR]          {s['chi2_1'][0]:.3f} "
                  f"[{s['chi2_1'][1]:.3f}, {s['chi2_1'][2]:.3f}]")
            print(f"  m_1 = E_q[u_1^2] median [IQR]        {s['m1'][0]:.3f} "
                  f"[{s['m1'][1]:.3f}, {s['m1'][2]:.3f}]")
            print(f"  r_1 = sup pi_1/q (secondary)         {s['r1']:.3f}")
            print(f"  lower bound (K^2/N) chi2_1           {s['lower_hoeffding']:.4f}")
            print(f"  leading term a_1 chi2_1 (secondary)  {s['leading']:.4f}")
            print(f"  honest Var(Zhat)/Z^2 (MC)            {s['mc_relvar']:.4f} "
                  f"[{s['mc_relvar_iqr'][0]:.4f}, {s['mc_relvar_iqr'][1]:.4f}]")
            print(f"  (K^2/N) chi2_1 / total               {s['lower_hoeffding_over_mc']:.3f}")
            print(f"  leading / total (secondary)          {s['leading_over_mc']:.3f}")
            print(f"  KL from the honest value             {s['kl_from_mc']:.4f}")
            print(f"  bound on the KL, m_1 hybrid          {s['bound_hybrid_m1']:.4f}")
            print(f"  bound on the KL, log(e^th-1+a_0)     {s['bound_simple_m1']:.4f}")
            print(f"    th = K^2 m_1/(M-K+1)               {s['bound_theta_m1']:.4f}")
            print(f"  bound (m_1 hybrid) / actual          {s['tightness_m1_over_actual']:.2f}x")
            print(f"  r_1 versions (secondary): hybrid     {s['cert_hybrid']:.4f}")
            print(f"                     K^2 r_1/(M-K+1)   {s['cert_simple']:.4f}")
            print(f"                     hybrid / actual   {s['tightness_cert_over_actual']:.2f}x")
            print(f"  prompts with m_1 hybrid > r_1 hybrid {s['frac_hybrid_m1_above_r1']:.3f}")
            print(f"  across-pool measurement        {s['measured_across_pool']:.5f}")
            print(f"  honest / across-pool measurement     {s['inflation_mc_over_measured']:.1f}x")
            print("  M-sweep on the true instance (disjoint within-pool blocks):")
            print(f"    {'m':>5} {'observed':>10} {'lower (K^2/m)':>14} {'lower (a_1)':>12} "
                  f"{'upper (m_1)':>12} {'upper (r_1)':>12} {'obs*m':>8} "
                  f"{'in primary bracket':>19} {'(a_1, r_1)':>11}")
            for m, v in s["sweep"].items():
                print(f"    {m:>5} {v['obs']:>10.4f} {v['lower_hoeffding']:>14.4f} "
                      f"{v['lower']:>12.4f} "
                      f"{v['upper_m1']:>12.4f} {v['upper']:>12.4f} "
                      f"{v['obs_times_m']:>8.2f} "
                      f"{v['frac_in_bracket_m1']:>19.2f} {v['frac_in_bracket']:>11.2f}")
            print("  pool independence diagnostics (per pair of pools):")
            print(f"    within-pool collisions   obs {s['iid']['obs_collisions']:.3g}  "
                  f"pred(LM) {s['iid']['pred_collisions']:.3g}")
            print(f"    shared strings           obs {s['iid']['obs_shared_pairwise']:.3g}  "
                  f"pred(LM) {s['iid']['pred_shared_pairwise']:.3g}  "
                  f"pred(empirical q) {s['iid'].get('pred_shared_pairwise_emp', float('nan')):.3g}")
            print(f"    max_y q(y) from LM logps  {s['iid']['max_q']:.3g}")

    out["_inputs"] = [os.path.relpath(p, os.path.dirname(SCRIPT_DIR)) for p in paths]
    dest = args.out
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
