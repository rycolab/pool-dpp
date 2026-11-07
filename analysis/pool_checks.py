"""
GSM8K pools from different runs
===============================
The 100 GSM8K pools were drawn in three runs (README, "Pools"), so this compares them.
  old vs new: pools 0-29 against pools 30-99, every plotted setting of both kernels, on log-det and pass@10
  cards:      within pools 30-99, RTX 3090 against RTX 4090
  runs:       MAP, k-means, and ancestral sampling per group of pools
  draws:      the draws themselves at each local temperature, including their tails at sT=1.2 and the spread of the plain embeddings
  seeds:      pools with different seeds must not repeat a string at the same draw index
z is the difference of the group means over its standard error, with pools as the replicates.

Usage: python analysis/pool_checks.py
Output: analysis/pool_checks.json
"""

import json
import os
import pickle
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import plot_gsm8k as P  # noqa: E402

MN = "google__gemma-2-9b-it"
OLD, NEW = list(range(30)), list(range(30, 100))
RTX3090 = [79, 81, 83, 88, 89, 90, 95, 96, 98, 99]
GROUPS = {"0-19": list(range(20)), "20-29, large cards": [20, 22, 23, 24, 26, 27, 28], "21, 25, 29, RTX 3090": [21, 25, 29], "30-99": NEW}
f = P.fmt


def z(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((b.mean() - a.mean()) / np.hypot(a.std(ddof=1) / np.sqrt(len(a)), b.std(ddof=1) / np.sqrt(len(b))))


def mean_ci(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), ci95=float(1.96 * a.std(ddof=1) / np.sqrt(len(a))))


def per_pool(kernel, metrics):
    sub, model, _ = P.KERNELS[kernel]
    curves = {"Ancestral": (f"{model}-ancestral", P.ST_SWEEP, lambda v: f"s{f(v)}.pkl"), "Naive-DPP": (f"{model}-naive", P.NAIVE_ST_SWEEP, lambda v: f"s{f(v)}.pkl"),
              "k-Means": (f"{model}-kmeans", P.ST_SWEEP, lambda v: f"s{f(v)}.pkl"), "PoolDPP-Samp": (f"{model}-samp", P.SAMP_TS, lambda t: f"T{f(t)}_s{f(P.SAMP_ST_HELD)}.pkl"),
              "PoolDPP-MAP": (f"{model}-map", P.MAP_TS, lambda t: f"T{f(t)}_s{f(P.MAP_ST_HELD)}.pkl")}
    out = {}
    for name, (s, sweep, key) in curves.items():
        vals = {i: P.collect_pool_level([os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{i}", sub)], s, sweep, key, metrics, "pool_t") for i in OLD + NEW}
        for v in sweep:
            out[f"{name}|{f(v)}"] = (np.array([vals[i][v]["x_mean"] for i in range(100)]), np.array([vals[i][v]["y_mean"] for i in range(100)]))
    return out


def compare(vals, a, b):
    res = {}
    for k, (x, y) in vals.items():
        res[k] = dict(logdet=[float(x[a].mean()), float(x[b].mean()), z(x[a], x[b])], pass_at_10=[float(y[a].mean()), float(y[b].mean()), z(y[a], y[b])])
    zs = np.array([q for v in res.values() for q in (v["logdet"][2], v["pass_at_10"][2])])
    return dict(settings=res, comparisons=len(zs), max_abs_z=float(np.abs(zs).max()), above_1_96=int((np.abs(zs) > 1.96).sum()), mean_z=float(zs.mean()),
                logdet_negative=int(sum(v["logdet"][2] < 0 for v in res.values())), logdet_settings=len(res))


def draws():
    raw_all, by_T, tail = {}, {}, {}
    for i in range(100):
        d = os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{i}")
        s = pickle.load(open(os.path.join(d, f"{MN}.pkl"), "rb"))
        e = pickle.load(open(os.path.join(d, f"{MN}_qwen_embeddings.pkl"), "rb"))[1.2]
        for T, b in s.items():
            top, uq, ln, t1 = [], [], [], []
            for raw, _, t1lp, lengths in b.values():
                raw = [x[0] if isinstance(x, tuple) else x for x in raw]
                top.append(np.unique(raw, return_counts=True)[1].max() / len(raw)); uq.append(len(set(raw))); ln.append(np.mean(lengths)); t1.append(np.mean(t1lp))
            by_T.setdefault(f"{T:g}", {}).setdefault(i, [np.mean(top), np.mean(uq), np.mean(ln), np.mean(t1)])
        a, cos, cos01, top10 = [], [], [], []
        raw_all[i] = {}
        for pt, (raw, _, t1lp, lengths) in s[1.2].items():
            raw = [x[0] if isinstance(x, tuple) else x for x in raw]
            raw_all[i][pt] = raw
            L, T1 = np.asarray(lengths, float), np.asarray(t1lp, float)
            a.append((np.mean([bool(re.search(r"[^\x00-\x7f]", x)) for x in raw]), np.mean(L < 40), np.quantile(L, .01), np.quantile(L, .99), np.quantile(T1, .01), np.mean(["####" not in x for x in raw])))
        for v in e.values():
            E = np.asarray(v["embeddings"], float)
            C = E @ E.T
            iu = np.triu_indices(len(E), 1)
            cos.append(C[iu].mean()); cos01.append(np.quantile(C[iu], .01))
            top10.append(np.sum(np.log(np.clip(np.linalg.eigvalsh(C)[::-1][:10], 1e-12, None))))
        tail[i] = list(np.mean(a, axis=0)) + [np.mean(cos), np.mean(cos01), np.mean(top10)]
        print("draws", i, flush=True)
    names = ["top string share", "distinct strings", "length", "t1 logp"]
    T_out = {T: {n: dict(old=float(np.mean([v[i][j] for i in OLD])), new=float(np.mean([v[i][j] for i in NEW])), z=z([v[i][j] for i in OLD], [v[i][j] for i in NEW])) for j, n in enumerate(names)} for T, v in by_T.items()}
    tn = ["non-ascii share", "share under 40 tokens", "length q01", "length q99", "t1 logp q01", "share without ####", "mean cosine", "cosine q01", "sum of top 10 log eigenvalues"]
    X = np.array([tail[i] for i in range(100)])
    tail_out = {n: dict(groups={g: mean_ci(X[idx, j]) for g, idx in GROUPS.items()}, old_vs_new_z=z(X[OLD, j], X[NEW, j])) for j, n in enumerate(tn)}
    pm = [np.mean([u == v for u, v in zip(raw_all[x][pt], raw_all[y][pt])]) for x in range(100) for y in range(x + 1, 100) for pt in raw_all[x]]
    return T_out, tail_out, dict(mean=float(np.mean(pm)), max_over_pair_and_prompt=float(np.max(pm)))


def main():
    df = P.load_gsm8k_dataset().head(P.N_QUESTIONS)
    gold = {P.build_math_prompt(r["question"]): r["answer"] for _, r in df.iterrows()}
    metrics = P.make_metrics(gold)
    out = dict(groups={k: v for k, v in GROUPS.items()}, rtx3090_in_30_99=RTX3090)
    a4090 = [i for i in NEW if i not in RTX3090]
    for kernel in ("baseline", "instructed"):
        vals = per_pool(kernel, metrics)
        out[kernel] = dict(old_vs_new=compare(vals, OLD, NEW), rtx3090_vs_rtx4090=compare(vals, RTX3090, a4090),
                           runs={k: {g: dict(logdet=mean_ci(vals[k][0][idx]), pass_at_10=mean_ci(vals[k][1][idx])) for g, idx in GROUPS.items()}
                                 for k in ("PoolDPP-MAP|10", "PoolDPP-MAP|50", "PoolDPP-MAP|500", "PoolDPP-Samp|1.05", "k-Means|1.2", "Ancestral|1.2")})
        o = out[kernel]["old_vs_new"]
        print(kernel, "old vs new", o["comparisons"], "max |z|", round(o["max_abs_z"], 2), "above 1.96", o["above_1_96"], "log-det negative", o["logdet_negative"], "of", o["logdet_settings"], flush=True)
    out["draws_by_T"], out["draws_tail_T1_2"], out["same_string_same_index"] = draws()
    dest = os.path.join(SCRIPT_DIR, "pool_checks.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Saved", dest)


if __name__ == "__main__":
    main()
