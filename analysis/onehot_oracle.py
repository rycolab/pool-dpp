"""
GSM8K one-hot answer oracle (app:embedder)
==========================================
PoolDPP-MAP with a one-hot encoding of each chain's extracted final answer appended to the text
embedding, on the GSM8K pools of the paper (Gemma-2-9B-Instruct at local sampling temperature 1.2,
Qwen3-Embedding-0.6B, K=10, 50 questions, 100 pools).

Feature of a chain: [sqrt(1-a) * unit(embedding), sqrt(a) * onehot(answer)], where the answer is the
chain's extracted final number, the extractor that the pass@K metric uses. Selection is greedy MAP
over the rows exp(lp1 / (2 T)) * feature, lp1 the log-probability at temperature 1. The setting fixed
before the run is a=0.99, T=25. The other settings are reported next to it: T in {10, 500}, a=0.9,
and MAP on the text embedding alone (T in {25, 500}). Ancestral sampling draws 10 sets of K chains
per pool and question without replacement from the pool counts (seed 0).

Pool ceiling: fraction of (question, pool) cells that contain any correct chain. Paired tests:
question-paired sign-flip permutation test against ancestral sampling (100,000 flips, p floored at
1e-5) and a 95% percentile bootstrap over questions (10,000 resamples).

Usage:  python analysis/onehot_oracle.py --cache cache-gsm8k [--cache more-pools-dir ...]
        Pool i is read from the first --cache directory that has pool{i}.
Output: analysis/onehot_oracle.json
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from lm_dpp.estimation import greedy_map  # noqa: E402
from lm_dpp.utils.math_utils import (  # noqa: E402
    build_math_prompt, extract_model_num, is_correct, load_gsm8k_dataset,
    normalize_num,
)

ST = 1.2
K = 10
N_QUESTIONS = 50
BASE = "google__gemma-2-9b-it_qwen_embeddings.pkl"
CONFIGS = [("hyb0.99", 0.99, 25), ("hyb0.99", 0.99, 10), ("hyb0.99", 0.99, 500),
           ("hyb0.9", 0.9, 25), ("text", None, 25), ("text", None, 500)]


def unit(E):
    E = np.asarray(E, np.float64)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1
    return E / n


def pool_path(caches, pi):
    for c in caches:
        p = os.path.join(c, f"pool{pi}", BASE)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"pool{pi}: no {BASE} in {caches}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--n-pools", type=int, default=100)
    a = ap.parse_args()
    n_pools = a.n_pools

    df = load_gsm8k_dataset().head(N_QUESTIONS)
    gold_of = {build_math_prompt(r["question"]): r["answer"] for _, r in df.iterrows()}
    q_of = {pt: i for i, pt in enumerate(gold_of)}
    nq = len(q_of)
    cell = defaultdict(lambda: np.full((nq, n_pools), np.nan))
    chain = defaultdict(list)
    uniq = defaultdict(list)
    ceil = np.full((nq, n_pools), np.nan)
    rng = np.random.default_rng(0)
    anc = np.zeros((nq, n_pools))
    anc_n = np.zeros((nq, n_pools))

    for pi in range(n_pools):
        with open(pool_path(a.cache, pi), "rb") as f:
            emb = pickle.load(f)
        for pt, d in emb[ST].items():
            gold = gold_of.get(pt)
            if gold is None:
                continue
            qi = q_of[pt]
            u = list(d["u_sents"])
            cnt = np.asarray(d["counts"], float)
            lp1 = np.asarray(d["logps_t1"], float)
            E = unit(d["embeddings"])
            n = len(u)
            if n < K:
                continue
            corr = np.array([1.0 if is_correct(s, gold) else 0.0 for s in u])
            key = [normalize_num(extract_model_num(s)) for s in u]
            ids = {x: j for j, x in enumerate(dict.fromkeys(key))}
            A = np.zeros((n, len(ids)))
            for i, x in enumerate(key):
                A[i, ids[x]] = 1.0
            ceil[qi, pi] = 1.0 if corr.sum() > 0 else 0.0
            pr = cnt / cnt.sum()
            for _ in range(10):
                idx = rng.choice(n, K, False, pr)
                anc[qi, pi] += 1.0 if corr[idx].sum() > 0 else 0.0
                anc_n[qi, pi] += 1
                chain["ancestral"].append(float(corr[idx].mean()))
                uniq["ancestral"].append(len({key[i] for i in idx}))
            for name, al, T in CONFIGS:
                F = E if al is None else np.hstack([np.sqrt(1 - al) * E, np.sqrt(al) * A])
                idx = list(greedy_map(lp1 / (2.0 * T), F, K))
                label = f"MAP {name} T={T}"
                cell[label][qi, pi] = 1.0 if corr[idx].sum() > 0 else 0.0
                chain[label].append(float(corr[idx].mean()))
                uniq[label].append(len({key[i] for i in idx}))
        print(f"pool{pi} done", flush=True)

    anc_cell = anc / np.maximum(anc_n, 1)
    anc_q = np.nanmean(anc_cell, axis=1)
    out = {"n_pools": n_pools, "n_questions": nq, "K": K, "local_sampling_temperature": ST,
           "pool_ceiling": float(np.nanmean(ceil)),
           "ancestral": {"pass_at_K": float(np.nanmean(anc_cell)),
                         "per_chain": float(np.mean(chain["ancestral"])),
                         "uniq_answers": float(np.mean(uniq["ancestral"]))}}
    r = np.random.default_rng(1)
    for label, M in cell.items():
        q = np.nanmean(M, axis=1)
        dq = q - anc_q
        signs = r.choice([-1.0, 1.0], (100000, nq))
        null = (signs * np.abs(dq)).mean(1)
        p = float(max((np.abs(null) >= abs(dq.mean())).mean(), 1e-5))
        bs = r.integers(0, nq, (10000, nq))
        out[label] = {"pass_at_K": float(np.nanmean(M)),
                      "per_chain": float(np.mean(chain[label])),
                      "uniq_answers": float(np.mean(uniq[label])),
                      "delta_pass_vs_ancestral": float(dq.mean()),
                      "p_signflip_q": p,
                      "delta_ci95_q": [float(np.quantile(dq[bs].mean(1), x)) for x in (0.025, 0.975)]}
    for k, v in out.items():
        print(k, v)
    with open(os.path.join(SCRIPT_DIR, "onehot_oracle.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved onehot_oracle.json")


if __name__ == "__main__":
    main()
