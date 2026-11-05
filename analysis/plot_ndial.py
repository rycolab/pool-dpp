"""
N-dial plot (fig:ndial, tab:ndial): certified KL vs pool size N on GSM8K.

Per prompt, the certificate of cor:certificates(a) is
    KL <= log(1 + a_1(N) chi2_1 + sum_{c>=2} a_c(N) (K^c/(K)_c r_1^c - 1))
with chi2_1 = Var_q(u_1) and r_1 N-independent (bound_raw per-question rows); we
sweep N and take the median over prompts. The N=2,500 points are cross-checked
against the fresh N=2,500 run (gsm8k_n2500.json).

Inputs: analysis/bound_raw_gsm8k.json at the anchors tau=1 (moderate tilt) and
tau=0.03 (diversity tilt).

Output: --out (default analysis/ndial_gsm8k.pdf) + the table
values on stdout and in analysis/ndial_gsm8k.json.
"""
import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
K = 10


def a_coeffs(N, K):
    denom = math.comb(N, K)
    return np.array([
        math.comb(K, c) * math.comb(N - K, K - c) / denom
        if (N - K) >= (K - c) else 0.0
        for c in range(1, K + 1)
    ])


def cert(N, chi2_1, r1, K=K):
    a = a_coeffs(N, K)
    ks = np.array([K ** c for c in range(1, K + 1)], dtype=float)
    fall = np.array([math.perm(K, c) for c in range(1, K + 1)], dtype=float)  # (K)_c
    tail_terms = ks / fall * np.array([r1 ** c for c in range(1, K + 1)]) - 1.0
    total = a[0] * chi2_1 + float((a[1:] * tail_terms[1:]).sum())
    return math.log1p(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "ndial_gsm8k.pdf"))
    a = ap.parse_args()
    raw_name = "bound_raw_gsm8k.json"
    curves = [("gsm8k|tau|tau=1", "#d62728", r"moderate tilt ($\tau=1$)"),
              ("gsm8k|tau|tau=0.03", "#1f77b4", r"diversity tilt ($\tau=0.03$)")]
    with open(os.path.join(SCRIPT_DIR, raw_name)) as f:
        raw = json.load(f)
    n2500_path = os.path.join(SCRIPT_DIR, "gsm8k_n2500.json")
    n2500 = json.load(open(n2500_path)) if os.path.exists(n2500_path) else {}

    Ns = np.unique(np.round(np.logspace(2, math.log10(4000), 25)).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300})

    table = {}
    for key, color, label in curves:
        if key not in raw:
            print(f"[skip] {key} not in {raw_name}")
            continue
        rows = raw[key]
        med = [float(np.median([cert(int(N), r["chi2_1"], r["r1"]) for r in rows]))
               for N in Ns]
        ax.plot(Ns, med, color=color, label=f"certified KL, {label}")
        table[key] = {int(N): float(np.median([cert(int(N), r["chi2_1"], r["r1"])
                                               for r in rows]))
                      for N in [100, 500, 1000, 2500]}
        # cross-check dot from the N=2500 run
        v = n2500.get(f"cert|{key}", {}).get("2500")
        if v is not None:
            ax.plot([2500], [v], "o", color=color, markersize=5)

    ax.axhline(0.1, color="gray", lw=0.8, ls="--")
    ax.text(110, 0.115, r"KL $\leq$ 0.1", fontsize=8, color="gray")
    ax.axvline(100, color="gray", lw=0.8, ls=":")
    ax.text(105, 8, "run size", fontsize=8, color="gray", rotation=90)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pool size $N$")
    ax.set_ylabel("certified KL (nats)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(pad=0.4)
    dest = a.out
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    fig.savefig(dest, bbox_inches="tight")
    print(f"Saved {dest}")
    with open(os.path.join(SCRIPT_DIR, "ndial_gsm8k.json"), "w") as f:
        json.dump({"inputs": raw_name, "median_certified_KL": table}, f, indent=1)

    print("\ncertified KL (medians):")
    for key, vals in table.items():
        print(key, {n: round(v, 3) for n, v in vals.items()})
    print("\nN=2500 run:", {k: v for k, v in n2500.items() if k.startswith("cert")})
    print("samp pass@10 N=2500:", n2500.get("samp_N2500"),
          "ceiling:", n2500.get("ceiling_pass_at_2500"))


if __name__ == "__main__":
    main()
