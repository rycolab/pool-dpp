"""
LLM-based selection baseline on FineWeb (sec:ablations)
=======================================================
Qwen3-8B replaces the DPP: it sees 40 strings of a pool drawn at local sampling temperature 1 and is
asked for K=10 diverse, high-quality ones. The 40 strings are evenly spaced in the pool's ranking by
log-probability, after dropping strings shorter than 10 characters, and each is cut to 200 characters
for display. The model sees neither the log-probabilities nor the embeddings.

k-means, greedy MAP, and random subsets run on the same 40 strings and on the full pool. Quality is the
mean log-probability at T=1 of the selected strings. Diversity is the log-determinant of their Gram
matrix. Intervals are 95% t intervals over pools.

Failures stop the run with an error: a missing pool (the default is cache-fineweb and
cache-fineweb-pools/pool1..pool9, as in run_fineweb.sh), pools with different prompts, a prompt with
fewer than 40 strings of at least 10 characters or fewer than K strings, a method that returns other
than K distinct strings (PoolDPP-MAP does so exactly when the strings span fewer than K dimensions),
a non-finite log-determinant, or fewer than 2 pools for an interval. A reply from which K distinct
indices cannot be read is part of what is measured: it counts in unparsed_replies (and per pool in
unparsed_replies_per_pool), and the LLM row averages the other prompts of the pool. The parser takes
the first K distinct integers below 40 anywhere in the reply.

Needs a GPU. Plain vLLM is enough here, without the dual-logprob patch.

Usage: python analysis/llm_selection.py [--pools cache-fineweb cache-fineweb-pools/pool1 ...]
Output: analysis/llm_selection.json
"""

import argparse
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np
from scipy import stats

from lm_dpp.estimation import (
    greedy_map,
    iter_prompt_data,
    quality_log_scales,
    sample_kmeans_for_prompt,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

EMBEDDINGS = "google__gemma-2-9b-it_minilm_embeddings.pkl"
MODEL = "Qwen/Qwen3-8B"
K = 10
N_SHOW = 40
ST = 1.0
MAP_T = 50
N_RANDOM = 5
N_EXTRA_POOLS = 9   # cache-fineweb-pools/pool1..pool9 (N_EXTRA in scripts/run_fineweb.sh)

PROMPT = """You are given {n} text completions of a passage. Your task is to select exactly {k} completions that are:
1. High quality (coherent, relevant, well-written)
2. Diverse (covering different topics, perspectives, or continuations)

The original passage starts with:
"{context}"

Here are the candidate completions (each preceded by its index number):

{items}
Select exactly {k} completions by listing their index numbers. Output ONLY a JSON list of {k} integers, like [3, 7, 12, ...]. No other text."""


def logdet(F):
    F = np.asarray(F, dtype=np.float64)
    s, ld = np.linalg.slogdet(F @ F.T + 1e-6 * np.eye(len(F)))
    return ld if s > 0 else np.nan


def spaced_by_rank(u_sents, logps_t1, where=""):
    """Indices of N_SHOW strings evenly spaced in the ranking by log-probability, best first."""
    keep = np.array([i for i, s in enumerate(u_sents) if len(str(s).strip()) >= 10], dtype=int)
    if len(keep) < N_SHOW:
        raise ValueError(f"{where}: {len(keep)} strings of at least 10 characters, fewer than {N_SHOW} to show")
    order = keep[np.argsort(logps_t1[keep])[::-1]]
    if len(order) <= N_SHOW:
        return order
    return order[np.linspace(0, len(order) - 1, N_SHOW, dtype=int)]


def build_prompt(context, shown):
    items = ""
    for i, s in enumerate(shown):
        t = str(s).strip()
        items += f"[{i}] {t[:200] + '...' if len(t) > 200 else t}\n\n"
    return PROMPT.format(n=len(shown), k=K, context=context[:200], items=items)


def parse_indices(reply, n):
    """The first K distinct integers below n in the reply, or None if there are fewer."""
    seen = dict.fromkeys(i for i in map(int, re.findall(r"\d+", reply)) if i < n)
    return list(seen)[:K] if len(seen) >= K else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=None,
                    help="pool dirs (default cache-fineweb + cache-fineweb-pools/pool*)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "llm_selection.json"))
    a = ap.parse_args()
    pools = a.pools or [os.path.join(ROOT_DIR, "cache-fineweb")] + [
        os.path.join(ROOT_DIR, "cache-fineweb-pools", f"pool{i}") for i in range(1, N_EXTRA_POOLS + 1)]
    missing = [p for p in pools if not os.path.exists(os.path.join(ROOT_DIR, p, EMBEDDINGS))]
    if missing:
        raise FileNotFoundError(f"fineweb: {len(missing)} of {len(pools)} pools have no {EMBEDDINGS}: {missing}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=a.gpu_mem_util,
              trust_remote_code=True, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.3, max_tokens=256, stop=["\n\n"])
    tok = llm.get_tokenizer()

    per_pool = defaultdict(list)
    failures, n_cells, replies, failures_per_pool, prompts0 = 0, 0, [], [], None
    for p, pool in enumerate(pools):
        with open(os.path.join(ROOT_DIR, pool, EMBEDDINGS), "rb") as f:
            bucket = pickle.load(f)[ST]
        if prompts0 is None:
            prompts0 = set(bucket)
        elif set(bucket) != prompts0:
            raise ValueError(f"{pool}: prompts differ from {pools[0]}")
        rng = np.random.default_rng(p)
        cells, n_seen, failures0 = [], 0, failures
        for prompt, u_sents, _, _, logps_t1, emb in iter_prompt_data(bucket):
            n_seen += 1
            if len(u_sents) < K:
                raise ValueError(f"{pool} prompt {prompt[:60]!r}: {len(u_sents)} distinct strings, fewer than K={K}")
            cells.append((prompt, np.asarray(u_sents, dtype=object), logps_t1, emb,
                          spaced_by_rank(u_sents, logps_t1, f"{pool} prompt {prompt[:60]!r}")))
        if n_seen != len(bucket):
            raise ValueError(f"{pool}: iter_prompt_data yielded {n_seen} of {len(bucket)} prompts")
        chats = [tok.apply_chat_template(
            [{"role": "user", "content": build_prompt(c[0], c[1][c[4]])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in cells]
        outputs = llm.generate(chats, params, use_tqdm=False)

        scores = defaultdict(list)
        for (prompt, _, lp1, emb, shown), out in zip(cells, outputs):
            def add(method, idx):
                idx = np.asarray(idx, dtype=int)
                if len(set(idx.tolist())) != K or len(idx) != K:
                    raise RuntimeError(f"{pool} prompt {prompt[:60]!r}: {method} returned "
                                       f"{len(idx)} strings ({len(set(idx.tolist()))} distinct), expected K={K}")
                ld = logdet(emb[idx])
                if not np.isfinite(ld):
                    raise ValueError(f"{pool} prompt {prompt[:60]!r}: {method} log-determinant {ld}")
                scores[method].append((lp1[idx].mean(), ld))

            reply = out.outputs[0].text
            replies.append(reply[:200])
            n_cells += 1
            picked = parse_indices(reply, len(shown))
            if picked is None:
                failures += 1
            else:
                add("LLM", shown[picked])

            log_s = quality_log_scales(lp1, 1.0 / (2.0 * MAP_T))
            add("k-Means, full pool", sample_kmeans_for_prompt(emb, lp1, K, random_state=p))
            add("k-Means, same 40", shown[sample_kmeans_for_prompt(emb[shown], lp1[shown], K, random_state=p)])
            add("PoolDPP-MAP, full pool", greedy_map(log_s, emb, K))
            add("PoolDPP-MAP, same 40", shown[greedy_map(log_s[shown], emb[shown], K)])
            for _ in range(N_RANDOM):
                add("Random, full pool", rng.choice(len(lp1), K, replace=False))
                add("Random, same 40", shown[rng.choice(len(shown), K, replace=False)])
        for method, v in scores.items():
            per_pool[method].append(np.mean(np.array(v), axis=0))
        failures_per_pool.append(failures - failures0)
        print(f"{pool}: {len(cells)} prompts, {failures} unparsed replies so far", flush=True)

    def summary(values):
        v = np.asarray(values)
        if len(v) < 2:
            raise ValueError(f"{len(v)} pool(s), need at least 2 for an interval")
        half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return float(v.mean()), float(half)

    results = {}
    print(f"\n{'method':<24s} {'quality':>16s} {'log-det':>14s}")
    for method, v in per_pool.items():
        v = np.array(v)
        q, q_ci = summary(v[:, 0])
        ld, ld_ci = summary(v[:, 1])
        results[method] = {"quality": q, "quality_ci": q_ci, "logdet": ld, "logdet_ci": ld_ci, "n_pools": len(v)}
        print(f"{method:<24s} {q:>9.2f} ±{q_ci:>5.2f} {ld:>7.2f} ±{ld_ci:>4.2f}")

    with open(a.out, "w") as f:
        json.dump({"model": MODEL, "k": K, "n_show": N_SHOW, "sT": ST, "map_T": MAP_T, "pools": pools,
                   "n_cells": n_cells, "unparsed_replies": failures,
                   "unparsed_replies_per_pool": failures_per_pool, "first_replies": replies[:5],
                   "results": results}, f, indent=2)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
