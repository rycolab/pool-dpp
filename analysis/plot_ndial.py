"""
N-dial plot (fig:ndial, tab:ndial): bound on the KL vs pool size N on GSM8K.

Per prompt, the plotted bound is the second-moment hybrid
    KL <= log(1 + a_1(N) chi2_1 + sum_{c>=2} a_c(N) ((K^c/(K)_c)^2 m_1^c - 1)),
    m_1 = E_q[u_1^2] = 1 + chi2_1,
with chi2_1 = Var_q(u_1) N-independent (bound_raw per-question rows, built from
the N=100 pools); we sweep N and take the median over prompts. Holding chi2_1
fixed while N grows is the extrapolation: it treats the instance (q and the
target) as fixed and only the number of draws as changing.

The r_1 hybrid,
    KL <= log(1 + a_1(N) chi2_1 + sum_{c>=2} a_c(N) (K^c/(K)_c r_1^c - 1)),
is not plotted; its medians are written to the JSON as a
secondary output (the paper keeps r_1 in a footnote). Formulas in
bound_formulas.py.

The dot at N=2,500 is the value gsm8k_n2500.py wrote under "bound_m1|{setting}".
That script computes it from the same bound_raw rows with the same formula, so
the dot checks that the two scripts agree; it is not an independent
measurement on the N=2,500 pools.

Inputs: analysis/bound_raw_gsm8k.json at T=1 and T=1.05, and analysis/gsm8k_n2500.json for the
dot. The script stops if either file or setting is missing (it used to drop the curve or the dot
with a printed note), or if a row's chi2_1 is not finite.

Output: --out (default analysis/ndial_gsm8k.pdf) + the table
values on stdout and in analysis/ndial_gsm8k.json, with keys
  median_bound_KL_m1   {setting: {N: median second-moment hybrid}}  (primary)
  median_certified_KL  {setting: {N: median r_1 hybrid}}  (secondary)
"""
import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import bound_formulas as bf  # noqa: E402

K = 10
N_EXPERIMENTS = 100   # pool size of the GSM8K experiments


def bound_m1(N, chi2_1, K=K):
    """Second-moment hybrid bound on the KL at pool size N (plotted)."""
    return bf.kl_hybrid_m1(N, K, chi2_1)


def cert(N, chi2_1, r1, K=K):
    """r_1 hybrid bound on the KL at pool size N (secondary, JSON only)."""
    return bf.kl_hybrid_r1(N, K, chi2_1, r1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "ndial_gsm8k.pdf"))
    a = ap.parse_args()
    raw_name = "bound_raw_gsm8k.json"
    curves = [("gsm8k|T=1", "#d62728", r"$T=1$"), ("gsm8k|T=1.05", "#1f77b4", r"$T=1.05$")]
    with open(os.path.join(SCRIPT_DIR, raw_name)) as f:
        raw = json.load(f)
    n2500_path = os.path.join(SCRIPT_DIR, "gsm8k_n2500.json")
    with open(n2500_path) as f:   # run gsm8k_n2500.py first (make_figures.sh: ndial)
        n2500 = json.load(f)

    Ns = np.unique(np.round(np.logspace(2, math.log10(4000), 25)).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300})

    table, table_r1 = {}, {}
    for key, color, label in curves:
        if key not in raw:
            raise KeyError(f"{key} not in {raw_name}")
        rows = raw[key]
        if not rows:
            raise ValueError(f"{key}: no rows in {raw_name}")
        for r in rows:
            bf.m1_of_row(r)  # rows that store m1 must have m1 = 1 + chi2_1
            if not math.isfinite(r["chi2_1"]):
                raise ValueError(f"{key}: chi2_1 = {r['chi2_1']} in {raw_name}")
        med = [float(np.median([bound_m1(int(N), r["chi2_1"]) for r in rows]))
               for N in Ns]
        ax.plot(Ns, med, color=color, label=label)
        table[key] = {int(N): float(np.median([bound_m1(int(N), r["chi2_1"])
                                               for r in rows]))
                      for N in [100, 500, 1000, 2500]}
        table_r1[key] = {int(N): float(np.median([cert(int(N), r["chi2_1"], r["r1"])
                                                  for r in rows]))
                         for N in [100, 500, 1000, 2500]}
        # consistency dot: gsm8k_n2500.py's value from the same rows and formula
        v = n2500.get(f"bound_m1|{key}", {}).get("2500")
        if v is None:
            raise KeyError(f"no bound_m1|{key} at N=2500 in {n2500_path}")
        ax.plot([2500], [v], "o", color=color, markersize=5)

    ax.axhline(0.1, color="gray", lw=0.8, ls="--")
    ax.text(N_EXPERIMENTS * 1.25, 0.115, r"KL $\leq$ 0.1", fontsize=8, color="gray")
    ax.axvline(N_EXPERIMENTS, color="gray", lw=0.8, ls=":")
    # x in data units, y in axes fraction, so the label stays inside the axes
    ax.text(N_EXPERIMENTS * 1.05, 0.04, "pool size of the experiments", fontsize=8,
            color="gray", rotation=90, va="bottom",
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pool size $N$")
    ax.set_ylabel("bound on the KL (nats)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(pad=0.4)
    dest = a.out
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    fig.savefig(dest, bbox_inches="tight")
    print(f"Saved {dest}")
    with open(os.path.join(SCRIPT_DIR, "ndial_gsm8k.json"), "w") as f:
        json.dump({"inputs": raw_name, "median_bound_KL_m1": table,
                   "median_certified_KL": table_r1}, f, indent=1)

    print("\nbound on the KL, second-moment hybrid (medians):")
    for key, vals in table.items():
        print(key, {n: round(v, 3) for n, v in vals.items()})
    print("\nr_1 hybrid, secondary (medians):")
    for key, vals in table_r1.items():
        print(key, {n: round(v, 3) for n, v in vals.items()})
    print("\ngsm8k_n2500.json:", {k: v for k, v in n2500.items()
                                  if k.startswith(("bound_m1|", "cert|"))})
    print("samp pass@10 N=2500:", n2500.get("samp_N2500"),
          "ceiling:", n2500.get("ceiling_pass_at_2500"))


if __name__ == "__main__":
    main()
