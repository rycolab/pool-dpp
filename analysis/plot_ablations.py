"""
FineWeb ablation plots (fig:abl_pool_size, fig:abl_k_size, fig:abl_lm_emb).

Reads the unified estimation layout produced by scripts/run_fineweb.sh and
scripts/run_fineweb_ablations.sh on cache-fineweb/:
  est10minilm_pool{N}/          pool-size ablation (K=10)
  est{K}minilm/                 subset-size ablation (N=400)
  est10{emb}/                   LM x embedding grid
The pool-size and K panels hold sT=1.0, PoolDPP-Samp at T=0.82 and PoolDPP-MAP at T=50. The grid sweeps T.

Also fig:abl_multimodel (intrinsic_multimodel.pdf): Gemma-2-9B, Qwen3-8B and
Llama-3.1-8B with MiniLM at K=10, all four headline methods (Ancestral and
k-Means sweep sT; PoolDPP-Samp and -MAP hold sT=1.0 and sweep T).

Outputs: {out-dir}/ablation_{pool_size,k_size,lm_emb}.pdf, intrinsic_multimodel.pdf
(default out-dir analysis)

Error bars are 95% t-intervals over the N_RUNS=10 selection runs on pool 0, with the sample
standard deviation (ddof=1; before, ddof=0 made them sqrt(9/10), about 5%, too narrow).
PoolDPP-MAP is deterministic and has one run, so it has no interval (drawn with zero width, as
before). The "Quality + Diversity" panel's interval combines the two standard deviations as
sqrt(sd_q^2 + sd_d^2), which ignores their covariance across runs (unchanged).

Missing data stops the script (no run, prompt, LM, or point is skipped): it raises if an est
dir's run dirs are not exactly 0..N_RUNS-1, if a stochastic method's selection file is missing
from any run dir, if PoolDPP-MAP's file is missing from run dir 0, if a selection file does not
cover exactly the 50 FineWeb prompts with K strings each (sets with fewer than 2 strings used to
be skipped), if a log-determinant is not finite, or if an LM of the multimodel figure has no
results (it used to be left out of the figure). The log-determinant adds a ridge of 1e-6 to the
Gram matrix (unchanged).

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
SAMP_T = 0.818182   # held in the pool-size and K panels
MAP_T = 50
SAMP_TS = [0.428571, 0.666667, 0.818182, 1, 1.22222]   # the grids of scripts/run_fineweb.sh
MAP_TS = [500, 100, 50, 25, 10]
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
    """Half-width of the 95% t-interval; std is the sample standard deviation (ddof=1)."""
    if n < 2:
        raise ValueError(f"a CI needs at least 2 runs, got {n}")
    return scipy_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)


def half_width(std, n):
    """ci95, or 0 for a deterministic arm (std None: one run, no interval)."""
    return 0.0 if std is None else ci95(std, n)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def expected_prompts():
    from lm_dpp.utils.fineweb_prompts import prompts
    return set(prompts)


def check_selection(data, expected, k, where):
    """Every record is (prompt, sents, logps_t1, embs) with k strings, and the prompts are
    exactly `expected`, each once. Raises ValueError otherwise."""
    seen = []
    for item in data:
        if len(item) != 4:
            raise ValueError(f"{where}: a record has {len(item)} fields, expected 4")
        prompt, sents, lps, emb = item
        if not (len(sents) == len(lps) == len(emb) == k):
            raise ValueError(f"{where}: prompt {prompt[:60]!r} has {len(sents)} strings, {len(lps)} "
                             f"log-probs and {len(emb)} embeddings; expected K={k}")
        seen.append(prompt)
    dup = len(seen) - len(set(seen))
    missing, extra = set(expected) - set(seen), set(seen) - set(expected)
    if dup or missing or extra:
        ex = next(iter(missing or extra), "")
        raise ValueError(f"{where}: {len(missing)} expected prompts missing, {len(extra)} unexpected, "
                         f"{dup} duplicated (e.g. {ex[:60]!r})")


def per_run_metrics(data, k, where="selection"):
    """(mean log p, mean log-det) over the 50 prompts; raises on a non-finite value."""
    check_selection(data, expected_prompts(), k, where)
    qs, ds = [], []
    for prompt, sents, lp, emb in data:
        qs.append(np.mean(lp))
        F = np.asarray(emb)
        G = F @ F.T + 1e-6 * np.eye(len(sents))
        s, ld = np.linalg.slogdet(G)
        if not (s > 0 and np.isfinite(ld) and np.isfinite(qs[-1])):
            raise ValueError(f"{where}, prompt {prompt[:60]!r}: log-det sign {s}, value {ld}; "
                             f"mean log p {qs[-1]}")
        ds.append(ld)
    return np.mean(qs), np.mean(ds)


def method_path(rd, model, mt, st=ST, T=SAMP_T):
    if mt == "ancestral":
        return rd / f"{model}-ancestral" / f"s{fmt(st)}.pkl"
    if mt == "kmeans":
        return rd / f"{model}-kmeans" / f"s{fmt(st)}.pkl"
    if mt == "samp":
        return rd / f"{model}-samp" / f"T{fmt(T)}_s{fmt(st)}.pkl"
    if mt == "map":
        return rd / f"{model}-map" / f"T{fmt(T)}_s{fmt(st)}.pkl"
    raise ValueError(mt)


def collect(est_dir, mt, model=MODEL_DEFAULT, n_runs=N_RUNS, k=10, **kw):
    """(q_mean, q_std, d_mean, d_std, n) over the runs of one method. Stochastic methods read run
    dirs 0..n_runs-1, which must be exactly the est dir's run dirs, and std is the sample
    standard deviation (ddof=1). PoolDPP-MAP is deterministic: run dir 0 only, std None."""
    runs = sorted(int(d.name) for d in Path(est_dir).iterdir() if d.name.isdigit() and d.is_dir()) \
        if Path(est_dir).is_dir() else []
    if runs != list(range(n_runs)):
        raise FileNotFoundError(f"{est_dir}: run dirs are {runs}, expected 0..{n_runs - 1}")
    deterministic = mt == "map"
    qs, ds = [], []
    for r in ([0] if deterministic else range(n_runs)):
        p = method_path(est_dir / str(r), model, mt, **kw)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing ({mt}, {model}, {kw})")
        q, d = per_run_metrics(load_pkl(p), k, str(p))
        qs.append(q)
        ds.append(d)
    n = len(qs)
    if deterministic:
        return qs[0], None, ds[0], None, n
    return np.mean(qs), np.std(qs, ddof=1), np.mean(ds), np.std(ds, ddof=1), n


def plot_xy(ax, xs, results, ylabel, xlabel):
    for name, m in results.items():
        means, stds, ns = np.array(m["mean"]), list(m["std"]), np.array(m["n"])
        errs = np.array([half_width(s, n) for s, n in zip(stds, ns)])
        ax.errorbar(xs, means, yerr=errs, label=name,
                    color=COLORS[name], marker=MARKERS[name], markersize=4,
                    capsize=2, capthick=0.5, elinewidth=0.6, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


ARMS = [("Ancestral", "ancestral", {}), ("K-Means", "kmeans", {}), ("PoolDPP-Samp", "samp", {}),
        ("PoolDPP-MAP", "map", {"T": MAP_T})]


def sweep_ablation(values, est_for, out_name, xlabel, second_panel, k_of=lambda v: 10):
    cols = {k: {"q_mean": [], "q_std": [], "d_mean": [], "d_std": [], "n": []}
            for k, _, _ in ARMS}
    for v in values:
        for name, mt, kw in ARMS:
            qm, qs, dm, ds, n = collect(est_for(v), mt, k=k_of(v), **kw)
            c = cols[name]
            c["q_mean"].append(qm); c["q_std"].append(qs)
            c["d_mean"].append(dm); c["d_std"].append(ds); c["n"].append(n)

    fig, (axQ, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.0))
    res_q = {k: {"mean": c["q_mean"], "std": c["q_std"], "n": c["n"]}
             for k, c in cols.items()}
    plot_xy(axQ, values, res_q, "Quality (mean log-prob)", xlabel)
    if second_panel == "combined":
        res_2 = {k: {"mean": [a + b for a, b in zip(c["q_mean"], c["d_mean"])],
                     "std": [None if a is None else np.hypot(a, b)
                             for a, b in zip(c["q_std"], c["d_std"])],
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

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0))
    for ri, (lm_name, lm_pre) in enumerate(lms):
        for ci, (emb_name, emb_key) in enumerate(embs):
            ax = axes[ri, ci]
            est = BASE / f"est10{emb_key}"
            model = f"{lm_pre}_{emb_key}"
            for name, mt in [("Ancestral", "ancestral"), ("K-Means", "kmeans")]:
                pts = [collect(est, mt, model=model, st=sT) for sT in st_sweep]
                xs = [p[2] for p in pts]
                ys = [p[0] for p in pts]
                xe = [ci95(p[3], p[4]) for p in pts]
                ye = [ci95(p[1], p[4]) for p in pts]
                ax.errorbar(xs, ys, xerr=xe, yerr=ye, color=COLORS[name],
                            marker=MARKERS[name], markersize=3, capsize=1.5,
                            capthick=0.4, elinewidth=0.4, linewidth=1,
                            label=name, alpha=0.85)
            pts = [collect(est, "samp", model=model, T=T) for T in SAMP_TS]
            xs = [p[2] for p in pts]
            ys = [p[0] for p in pts]
            xe = [ci95(p[3], p[4]) for p in pts]
            ye = [ci95(p[1], p[4]) for p in pts]
            ax.errorbar(xs, ys, xerr=xe, yerr=ye, color=COLORS["PoolDPP-Samp"],
                        marker="o", markersize=3, capsize=1.5, capthick=0.4,
                        elinewidth=0.4, linewidth=1, label="PoolDPP-Samp",
                        alpha=0.85)
            pts = [collect(est, "map", model=model, T=T) for T in MAP_TS]
            ax.errorbar([p[2] for p in pts], [p[0] for p in pts],
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
    est = BASE / "est10minilm"
    absent = [m for _, m in lms if not (est / "0" / f"{m}_minilm-samp").is_dir()]
    if absent:
        raise FileNotFoundError(f"multimodel: no PoolDPP-Samp results in {est}/0 for {absent}")
    present = lms
    fig, axes = plt.subplots(1, len(present), figsize=(5.5, 2.2), squeeze=False)
    for ax, (lm_name, lm_pre) in zip(axes[0], present):
        model = f"{lm_pre}_minilm"
        for name, mt, sweep, key in [("Ancestral", "ancestral", st_sweep, "st"),
                                     ("K-Means", "kmeans", st_sweep, "st"),
                                     ("PoolDPP-Samp", "samp", SAMP_TS, "T"),
                                     ("PoolDPP-MAP", "map", MAP_TS, "T")]:
            pts = [collect(est, mt, model=model, **{key: v}) for v in sweep]
            ax.errorbar([p[2] for p in pts], [p[0] for p in pts],
                        xerr=[half_width(p[3], p[4]) for p in pts],
                        yerr=[half_width(p[1], p[4]) for p in pts],
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
        "ablation_k_size.pdf", "Subset size $K$", second_panel="diversity", k_of=lambda K: K)
    lm_emb_grid()
    multimodel()


if __name__ == "__main__":
    main()
