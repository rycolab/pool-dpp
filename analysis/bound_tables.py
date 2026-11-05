"""
Tables for the Z-free bound test (tab:zfree-certificates, reads bound_raw_*.json)
=================================================================================
  Tab A  bound at the experimental setting: exact leading term, honest MC value
         of Var(Zhat)/Z^2, certified KL, and tightness vs the exact reduced
         quantity
  Tab B  M-sweep on the true instance (disjoint within-pool blocks) against the
         identity's bracket [a_1(m) chi2_1, a_1 chi2_1 + NA tail]
  Tab C  independence diagnostics for the pools

Usage: python analysis/bound_tables.py
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
            cert = [r["cert_hybrid_KL"] for r in rows]
            simple = [r["cert_simple_KL"] for r in rows]
            # exact KL implied by the honest MC value of the reduced quantity
            kl_mc = [math.log1p(x) for x in mc if np.isfinite(x)]
            tight = [r["cert_hybrid_KL"] / math.log1p(r["mc_relvar"])
                     for r in rows if r["mc_relvar"] and np.isfinite(r["mc_relvar"])]
            lead_frac = [r["leading"] / r["mc_relvar"] for r in rows
                         if r["mc_relvar"] and np.isfinite(r["mc_relvar"]) and r["mc_relvar"] > 0]
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
                inb = []
                for r in rows:
                    ov = r["sub_relvar"].get(str(m), r["sub_relvar"].get(m))
                    lv = r["pred_leading_by_m"].get(str(m), r["pred_leading_by_m"].get(m))
                    hv = r["pred_upper_by_m"].get(str(m), r["pred_upper_by_m"].get(m))
                    if ov is not None and lv is not None and hv is not None:
                        inb.append(lv <= ov <= hv)
                sweep[m] = dict(obs=med(o), lower=med(lo), upper=med(hi),
                                obs_times_m=med(o) * m,
                                frac_in_bracket=float(np.mean(inb)) if inb else None,
                                frac_below_lower=float(np.mean(
                                    [not b for b in inb])) if inb else None,
                                n_prompts=len(o))

            iid = {k: med([r["iid"][k] for r in rows]) for k in rows[0]["iid"]}

            out[key] = dict(
                n_prompts=len(rows), K=K, M_exp=M_exp,
                n_uniq=med([r["n_uniq"] for r in rows]),
                missing_mass=med([r["missing_mass"] for r in rows]),
                chi2_1=[med([r["chi2_1"] for r in rows]),
                        q([r["chi2_1"] for r in rows], .25),
                        q([r["chi2_1"] for r in rows], .75)],
                r1=med([r["r1"] for r in rows]),
                unbiasedness_EZ_over_Z=med([r["mc_mean_ratio"] for r in rows]),
                leading=med(lead), mc_relvar=med(mc),
                mc_relvar_iqr=[q(mc, .25), q(mc, .75)],
                leading_over_mc=med(lead_frac),
                kl_from_mc=med(kl_mc), cert_hybrid=med(cert),
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
            print(f"  Good-Turing missing mass             {s['missing_mass']:.4f}")
            print(f"  E[Zhat]/Z from MC (must be 1)        {s['unbiasedness_EZ_over_Z']:.4f}")
            print(f"  chi2(pi_1||q)  median [IQR]          {s['chi2_1'][0]:.3f} "
                  f"[{s['chi2_1'][1]:.3f}, {s['chi2_1'][2]:.3f}]")
            print(f"  r_1 = sup pi_1/q                     {s['r1']:.3f}")
            print(f"  identity leading term a_1 chi2_1     {s['leading']:.4f}")
            print(f"  honest Var(Zhat)/Z^2 (MC)            {s['mc_relvar']:.4f} "
                  f"[{s['mc_relvar_iqr'][0]:.4f}, {s['mc_relvar_iqr'][1]:.4f}]")
            print(f"  leading / total                      {s['leading_over_mc']:.3f}")
            print(f"  KL from the honest value             {s['kl_from_mc']:.4f}")
            print(f"  certified KL (exact lead + NA tail)  {s['cert_hybrid']:.4f}")
            print(f"  certified KL (K^2 r_1/(M-K+1))       {s['cert_simple']:.4f}")
            print(f"  certificate / actual                 {s['tightness_cert_over_actual']:.2f}x")
            print(f"  across-pool measurement        {s['measured_across_pool']:.5f}")
            print(f"  honest / across-pool measurement     {s['inflation_mc_over_measured']:.1f}x")
            print("  M-sweep on the true instance (disjoint within-pool blocks):")
            print(f"    {'m':>5} {'observed':>10} {'identity lower':>15} "
                  f"{'identity upper':>15} {'obs*m':>8} {'in bracket':>11}")
            for m, v in s["sweep"].items():
                print(f"    {m:>5} {v['obs']:>10.4f} {v['lower']:>15.4f} "
                      f"{v['upper']:>15.4f} {v['obs_times_m']:>8.2f} "
                      f"{v['frac_in_bracket']:>11.2f}")
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
