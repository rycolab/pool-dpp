"""
Prompts whose pool has Zhat = 0 (sec:experiments, the note on redraws)
======================================================================
A (prompt, pool) pair has Zhat = 0 exactly when its embeddings span fewer than K dimensions. For each task and
local sampling temperature this counts those pairs by exact equality of embedding rows and by numerical rank,
names the prompts, and gives the smallest number of distinct embeddings in any pool.

On Ambrosia, PoolDPP-Samp redraws those pairs from the next pool in index order whose embeddings span K dimensions.
At sT=1, the local temperature of PoolDPP-Samp, this also checks every selection record of those pairs against that rule.

Usage: python analysis/zhat_zero_counts.py [--workers N]
Output: analysis/zhat_zero_counts.json
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
from lm_dpp.estimation import RANK_TOL  # noqa: E402

MN = "google__gemma-2-9b-it"
TASKS = {
    "ambrosia": dict(K=3, emb="minilm", dirs=[f"cache-ambrosia/pool{i}" for i in range(100)]),
    "gsm8k": dict(K=10, emb="qwen", dirs=[f"cache-gsm8k/pool{i}" for i in range(100)]),
    "fineweb": dict(K=10, emb="minilm", dirs=["cache-fineweb"] + [f"cache-fineweb-pools/pool{i}" for i in range(1, 10)]),
}


def label(prompt):
    """The question of an Ambrosia prompt, or the end of any other prompt"""
    m = re.search(r"following question:\n(.*?)\ngenerate", prompt, re.S)
    return m.group(1) if m else prompt[-80:]


def rows(d):
    return os.path.join(ROOT_DIR, d)


def count_pool(job):
    d, K, emb = job
    b = pickle.load(open(os.path.join(rows(d), f"{MN}_{emb}_embeddings.pkl"), "rb"))
    out = {}
    for sT, cells in b.items():
        exact, by_rank, low = [], [], 10 ** 9
        for p, v in cells.items():
            E = np.asarray(v["embeddings"], dtype=float)
            n = len(np.unique(E, axis=0))
            low = min(low, n)
            if n < K:
                exact.append(p)
            if np.linalg.matrix_rank(E, tol=RANK_TOL) < K:
                by_rank.append(p)
        out[float(sT)] = dict(n=len(cells), exact=exact, rank=by_rank, low=low)
    return out


def span_pool(job):
    """For the flagged prompts at sT=1: does this pool span K dimensions, and which strings does it hold"""
    d, K, emb, prompts = job
    b = pickle.load(open(os.path.join(rows(d), f"{MN}_{emb}_embeddings.pkl"), "rb"))[1.0]
    return {p: (bool(np.linalg.matrix_rank(np.asarray(b[p]["embeddings"], float), tol=RANK_TOL) >= K), {str(s) for s in b[p]["u_sents"]}) for p in prompts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    res = {}
    with Pool(a.workers) as W:
        for task, c in TASKS.items():
            K, emb, dirs = c["K"], c["emb"], c["dirs"]
            per = W.map(count_pool, [(d, K, emb) for d in dirs], chunksize=1)
            by = {}
            for sT in sorted({t for p in per for t in p}):
                cells = [p[sT] for p in per if sT in p]
                who = Counter(label(x) for q in cells for x in q["exact"])
                by[f"{sT:g}"] = dict(pairs=sum(q["n"] for q in cells), exact=sum(len(q["exact"]) for q in cells), by_rank=sum(len(q["rank"]) for q in cells),
                                     smallest_distinct=min(q["low"] for q in cells), prompts=dict(who.most_common()))
                print(task, f"sT={sT:g}", {k: v for k, v in by[f"{sT:g}"].items() if k != "prompts"}, flush=True)
            res[task] = dict(n_pools=len(dirs), K=K, by_sT=by)
            if task == "ambrosia":
                flagged = [(i, x) for i, p in enumerate(per) for x in p[1.0]["exact"]]
                prompts = sorted({x for _, x in flagged})
                span = W.map(span_pool, [(d, K, emb, prompts) for d in dirs], chunksize=1)
                n, bad, used = 0, 0, Counter()
                for i, x in flagged:
                    first = next((i + d) % len(dirs) for d in range(1, len(dirs)) if span[(i + d) % len(dirs)][x][0])
                    used[f"{i}->{first}"] += 1
                    for f in glob.glob(os.path.join(rows(dirs[i]), "est3minilm", "*", f"{MN}_{emb}-samp", "*_s1.pkl")):
                        rec = [r for r in pickle.load(open(f, "rb")) if r[0] == x]
                        n += 1
                        if len(rec) != 1 or len(rec[0][1]) != K or not {str(s) for s in rec[0][1]} <= span[first][x][1]:
                            bad += 1
                res[task]["samp_redraw_sT1"] = dict(cells=len(flagged), records_checked=n, wrong_source=bad, pool_to_source=dict(used))
                print(res[task]["samp_redraw_sT1"], flush=True)
    dest = os.path.join(SCRIPT_DIR, "zhat_zero_counts.json")
    with open(dest, "w") as f:
        json.dump(res, f, indent=2)
    print("Saved", dest)


if __name__ == "__main__":
    main()
