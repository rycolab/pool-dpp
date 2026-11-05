"""
FineWeb ablation plots (fig:abl_pool_size, fig:abl_k_size, fig:abl_lm_emb).

Reads the unified estimation layout produced by scripts/run_fineweb.sh and
scripts/run_fineweb_ablations.sh on cache-fineweb/:
  est10minilm_pool{N}/          pool-size ablation (K=10)
  est{K}minilm/                 subset-size ablation (N=400)
  est10{emb}/                   LM x embedding grid
The PoolDPP-Samp arm holds sT=1.0 and tau=3; PoolDPP-MAP appears in the grid
panel via the same tau.

Also fig:abl_multimodel (intrinsic_multimodel.pdf): Gemma-2-9B, Qwen3-8B and
Llama-3.1-8B with MiniLM at K=10, all four headline methods (Ancestral and
k-Means sweep sT; PoolDPP-Samp and -MAP hold sT=1.0 and sweep tau).

Outputs: {out-dir}/ablation_{pool_size,k_size,lm_emb}.pdf, intrinsic_multimodel.pdf
(default out-dir analysis)

Usage: OPENBLAS_NUM_THREADS=1 python analysis/plot_ablations.py \
           [--out-dir analysis] [--base cache-fineweb]
"""
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = Path(ROOT_DIR) / "cache-fineweb"
OUT_DIR = Path(SCRIPT_DIR)

MODEL_DEFAULT = "google__gemma-2-9b-it_minilm"
ST = 1.0
TAU_HELD = 3
TAU_MAP = 0.3   # MAP arm of the pool/K ablations: old T=50 anchor, tau = sigma_hat/(2*50) ~ 0.24 -> grid 0.3
N_RUNS = 10

COLORS = {"Ancestral": "#1f77b4", "K-Means": "#9467bd", "PoolDPP-Samp": "#ff7f0e",
          "PoolDPP-MAP": "#d62728"}
MARKERS = {"Ancestral": "^", "K-Means": "s", "PoolDPP-Samp": "o", "PoolDPP-MAP": "D"}

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300,
})


def fmt(x):
    return f"{float(x):g}"


def ci95(std, n):
    if n <= 1:
        return 0.0
    return scipy_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def per_run_metrics(data):
    qs, ds = [], []
    for item in data:
        if len(item) != 4:
            continue
        _, sents, lp, emb = item
        if len(sents) < 2:
            continue
        qs.append(np.mean(lp))
        F = np.asarray(emb)
        G = F @ F.T + 1e-6 * np.eye(len(sents))
        s, ld = np.linalg.slogdet(G)
        ds.append(ld if s > 0 else np.nan)
    return np.nanmean(qs) if qs else np.nan, np.nanmean(ds) if ds else np.nan


def method_path(rd, model, mt, st=ST, tau=TAU_HELD):
    if mt == "ancestral":
        return rd / f"{model}-ancestral" / f"s{fmt(st)}.pkl"
    if mt == "kmeans":
        return rd / f"{model}-kmeans" / f"s{fmt(st)}.pkl"
    if mt == "samp":
        return rd / f"{model}-samp" / f"tau{fmt(tau)}_s{fmt(st)}.pkl"
    if mt == "map":
        return rd / f"{model}-map" / f"tau{fmt(tau)}_s{fmt(st)}.pkl"
    raise ValueError(mt)


def collect(est_dir, mt, model=MODEL_DEFAULT, n_runs=N_RUNS, **kw):
    qs, ds = [], []
    for r in range(n_runs):
        p = method_path(est_dir / str(r), model, mt, **kw)
        if not p.exists():
            continue
        q, d = per_run_metrics(load_pkl(p))
        if not (np.isnan(q) or np.isnan(d)):
            qs.append(q)
            ds.append(d)
    n = len(qs)
    return (np.mean(qs) if n else np.nan, np.std(qs) if n else 0.0,
            np.mean(ds) if n else np.nan, np.std(ds) if n else 0.0, n)


def plot_xy(ax, xs, results, ylabel, xlabel):
    for name, m in results.items():
        means, stds, ns = (np.array(m["mean"]), np.array(m["std"]), np.array(m["n"]))
        errs = np.array([ci95(s, n) for s, n in zip(stds, ns)])
        ax.errorbar(xs, means, yerr=errs, label=name,
                    color=COLORS[name], marker=MARKERS[name], markersize=4,
                    capsize=2, capthick=0.5, elinewidth=0.6, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


ARMS = [("Ancestral", "ancestral", {}), ("K-Means", "kmeans", {}), ("PoolDPP-Samp", "samp", {}),
        ("PoolDPP-MAP", "map", {"tau": TAU_MAP})]


def sweep_ablation(values, est_for, out_name, xlabel, second_panel):
    cols = {k: {"q_mean": [], "q_std": [], "d_mean": [], "d_std": [], "n": []}
            for k, _, _ in ARMS}
    for v in values:
        for name, mt, kw in ARMS:
            qm, qs, dm, ds, n = collect(est_for(v), mt, **kw)
            c = cols[name]
            c["q_mean"].append(qm); c["q_std"].append(qs)
            c["d_mean"].append(dm); c["d_std"].append(ds); c["n"].append(n)

    fig, (axQ, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.0))
    res_q = {k: {"mean": c["q_mean"], "std": c["q_std"], "n": c["n"]}
             for k, c in cols.items()}
    plot_xy(axQ, values, res_q, "Quality (mean log-prob)", xlabel)
    if second_panel == "combined":
        res_2 = {k: {"mean": [a + b for a, b in zip(c["q_mean"], c["d_mean"])],
                     "std": [np.hypot(a, b) for a, b in zip(c["q_std"], c["d_std"])],
                     "n": c["n"]} for k, c in cols.items()}
        plot_xy(ax2, values, res_2, "Quality + Diversity", xlabel)
        ax2.legend(loc="lower right", frameon=True, framealpha=0.9)
    else:
        res_2 = {k: {"mean": c["d_mean"], "std": c["d_std"], "n": c["n"]}
                 for k, c in cols.items()}
        plot_xy(ax2, values, res_2, "Diversity (log-det)", xlabel)
        ax2.legend(loc="lower left", frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.4)
    out = OUT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def lm_emb_grid():
    lms = [("Gemma-2-9B", "google__gemma-2-9b-it"),
           ("Qwen3-8B", "Qwen__Qwen3-8B")]
    embs = [("MiniLM-L6", "minilm"), ("Qwen3-Emb", "qwen")]
    st_sweep = [0.6, 0.8, 1.0, 1.2, 1.5]
    taus = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0))
    for ri, (lm_name, lm_pre) in enumerate(lms):
        for ci, (emb_name, emb_key) in enumerate(embs):
            ax = axes[ri, ci]
            est = BASE / f"est10{emb_key}"
            model = f"{lm_pre}_{emb_key}"
            for name, mt in [("Ancestral", "ancestral"), ("K-Means", "kmeans")]:
                pts = [collect(est, mt, model=model, st=sT) for sT in st_sweep]
                xs = [p[2] for p in pts if p[4]]
                ys = [p[0] for p in pts if p[4]]
                xe = [ci95(p[3], p[4]) for p in pts if p[4]]
                ye = [ci95(p[1], p[4]) for p in pts if p[4]]
                if xs:
                    ax.errorbar(xs, ys, xerr=xe, yerr=ye, color=COLORS[name],
                                marker=MARKERS[name], markersize=3, capsize=1.5,
                                capthick=0.4, elinewidth=0.4, linewidth=1,
                                label=name, alpha=0.85)
            pts = [collect(est, "samp", model=model, tau=tau) for tau in taus]
            good = [p for p in pts if p[4]]
            if good:
                xs = [p[2] for p in good]
                ys = [p[0] for p in good]
                xe = [ci95(p[3], p[4]) for p in good]
                ye = [ci95(p[1], p[4]) for p in good]
                ax.errorbar(xs, ys, xerr=xe, yerr=ye, color=COLORS["PoolDPP-Samp"],
                            marker="o", markersize=3, capsize=1.5, capthick=0.4,
                            elinewidth=0.4, linewidth=1, label="PoolDPP-Samp",
                            alpha=0.85)
            pts = [collect(est, "map", model=model, tau=tau) for tau in taus]
            good = [p for p in pts if p[4]]
            if good:
                ax.errorbar([p[2] for p in good], [p[0] for p in good],
                            color=COLORS["PoolDPP-MAP"], marker=MARKERS["PoolDPP-MAP"],
                            markersize=3, linewidth=1, label="PoolDPP-MAP", alpha=0.85)
            ax.set_title(f"{lm_name} / {emb_name}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if ri == 1:
                ax.set_xlabel("Diversity (log-det)")
            if ci == 0:
                ax.set_ylabel("Quality (log-prob)")
    axes[0, 0].legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.5)
    out = OUT_DIR / "ablation_lm_emb.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def multimodel():
    lms = [("Gemma-2-9B", "google__gemma-2-9b-it"),
           ("Qwen3-8B", "Qwen__Qwen3-8B"),
           ("Llama-3.1-8B", "meta-llama__Llama-3.1-8B-Instruct")]
    st_sweep = [0.6, 0.8, 1.0, 1.2, 1.5]
    taus = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    est = BASE / "est10minilm"
    present = [(n, m) for n, m in lms
               if (est / "0" / f"{m}_minilm-samp").is_dir()]
    if not present:
        print("multimodel: no LM has samp results, skipped")
        return
    fig, axes = plt.subplots(1, len(present), figsize=(5.5, 2.2), squeeze=False)
    for ax, (lm_name, lm_pre) in zip(axes[0], present):
        model = f"{lm_pre}_minilm"
        for name, mt, sweep, key in [("Ancestral", "ancestral", st_sweep, "st"),
                                     ("K-Means", "kmeans", st_sweep, "st"),
                                     ("PoolDPP-Samp", "samp", taus, "tau"),
                                     ("PoolDPP-MAP", "map", taus, "tau")]:
            pts = [collect(est, mt, model=model, **{key: v}) for v in sweep]
            good = [p for p in pts if p[4]]
            if not good:
                continue
            ax.errorbar([p[2] for p in good], [p[0] for p in good],
                        xerr=[ci95(p[3], p[4]) for p in good],
                        yerr=[ci95(p[1], p[4]) for p in good],
                        color=COLORS[name], marker=MARKERS[name], markersize=3,
                        capsize=1.5, capthick=0.4, elinewidth=0.4, linewidth=1,
                        label=name, alpha=0.85)
        ax.set_title(lm_name, fontsize=9)
        ax.set_xlabel("Diversity (log-det)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Quality (log-prob)")
    axes[0, 0].legend(loc="lower left", fontsize=6, frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.5)
    out = OUT_DIR / "intrinsic_multimodel.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    global BASE, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="default analysis")
    ap.add_argument("--base", default=None, help="fineweb cache (default cache-fineweb)")
    a = ap.parse_args()
    if a.out_dir:
        OUT_DIR = Path(ROOT_DIR) / a.out_dir
    if a.base:
        BASE = Path(ROOT_DIR) / a.base
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_ablation(
        [25, 50, 100, 150, 200, 400],
        lambda N: BASE / (f"est10minilm_pool{N}" if N != 400 else "est10minilm"),
        "ablation_pool_size.pdf", "Pool size $N$", second_panel="combined")
    sweep_ablation(
        [3, 5, 10, 15, 20],
        lambda K: BASE / f"est{K}minilm",
        "ablation_k_size.pdf", "Subset size $K$", second_panel="diversity")
    lm_emb_grid()
    multimodel()


if __name__ == "__main__":
    main()
