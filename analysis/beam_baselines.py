"""
Decoding-time SWOR baselines: SBS and Conditional Poisson SBS (tab:beam_baselines)
====================================================================
analysis/baselines.py has the pool-reranking comparators (MMR, MBR). SBS (Kool et al.,
2019) and CPSBS (Meister et al., 2021) require new decoding runs; those runs
are produced by scripts/run_beam_baselines.sh (same models, prompts, K and
temperature ranges as the pool experiments; implementation and validation in
src/lm_dpp/sbs_decoding.py and tests/test_sbs.py). Each run of either decoder
yields one set of K sequences sampled without replacement from the annealed
model, i.e. exactly the object the pool-selection methods output.

This script scores those sets on each task's headline axes, matching
analysis/baselines.py:

  fineweb   quality = mean log p_t1        diversity = log-det of MiniLM Gram
  ambrosia  precision = golden accuracy    recall = golden query coverage
            (execution-based, since generated SQL strings are new)
  gsm8k     pass@K=10, mean per-chain accuracy, log-det of Qwen Gram

Aggregation: metrics are computed per run (one decoded set), averaged over
prompts per run index, then mean +/- 95% CI across the n_runs independent
runs.

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/beam_baselines.py --task fineweb
    OPENBLAS_NUM_THREADS=1 python analysis/beam_baselines.py --task ambrosia
    OPENBLAS_NUM_THREADS=1 python analysis/beam_baselines.py --task gsm8k
"""

import argparse
import json
import os
import pickle
import sqlite3
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from baselines import logdet  # noqa: E402

# method suffix -> (label, knob name for the cache key: sampling temperature
# for the SWOR methods, diversity weight w for deterministic DetBS)
METHODS = {"sbs": ("SBS", "sT"), "cps": ("CPSBS", "sT"), "detbs": ("DetBS", "w"),
           "setty": ("SetSMC", "T"),  # setty-smc: SIS + inclusion-kernel end readout
           "settysmc": ("SetSMC-online", "T"),  # setty-smc: online tau=.25 arm
           "setty100": ("SetSMC-100", "T"), "setty200": ("SetSMC-200", "T")}
GEMMA = "google__gemma-2-9b-it"
QWEN_LM = "Qwen__Qwen3-8B"


def load_runs(cache, T, prompt, k):
    """-> list of runs, each a dict(sents, lps_t1) of length k, or None."""
    entry = cache.get(T, {}).get(prompt)
    if entry is None:
        return None
    sents, lps, lps_t1, lengths = entry
    n = (len(sents) // k) * k
    return [
        {"sents": sents[r: r + k], "lps_t1": lps_t1[r: r + k]}
        for r in range(0, n, k)
    ]


def load_cache(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_embeddings(samples_file, model_name, output_file):
    if os.path.exists(output_file):
        return
    print(f"  computing embeddings: {output_file}")
    from lm_dpp.embeddings import precompute_embeddings
    precompute_embeddings(model_name, samples_file, output_file)


def emb_lookup(emb, T, prompt):
    d = emb.get(T, {}).get(prompt)
    if d is None:
        return {}
    return dict(zip(list(d["u_sents"]), np.asarray(d["embeddings"], dtype=np.float64)))


def ci95(runs_means):
    a = np.asarray(runs_means, dtype=np.float64)
    if len(a) < 2:
        return float("nan")
    return float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def aggregate(per_run_per_prompt):
    """{run_idx: [prompt values]} -> (mean, ci95 across run means)."""
    run_means = [float(np.nanmean(v)) for v in per_run_per_prompt.values() if len(v)]
    return float(np.nanmean(run_means)), ci95(run_means)


# ---------------------------------------------------------------------------
def run_fineweb(k=10):
    from lm_dpp.utils.fineweb_prompts import prompts as fineweb_prompts

    out = {}
    for lm in (GEMMA, QWEN_LM):
        for method, (label, knob) in METHODS.items():
            samples_file = os.path.join(ROOT_DIR, "cache-fineweb-swor", f"{lm}_{method}.pkl")
            cache = load_cache(samples_file)
            if cache is None:
                continue
            emb_file = samples_file.replace(".pkl", "_minilm_embeddings.pkl")
            ensure_embeddings(samples_file, "sentence-transformers/all-MiniLM-L6-v2", emb_file)
            emb = load_cache(emb_file)

            for T in sorted(cache):
                q_acc, d_acc = defaultdict(list), defaultdict(list)
                for prompt in fineweb_prompts:
                    runs = load_runs(cache, T, prompt, k)
                    if not runs:
                        continue
                    lookup = emb_lookup(emb, T, prompt)
                    for ri, run in enumerate(runs):
                        q_acc[ri].append(float(np.mean(run["lps_t1"])))
                        E = np.stack([lookup[s] for s in run["sents"]])
                        d_acc[ri].append(logdet(E))
                if not q_acc:
                    continue
                q, qc = aggregate(q_acc)
                d, dc = aggregate(d_acc)
                out[f"{label}|{lm}|{knob}={T}"] = {
                    "quality_mean_logp": q, "quality_ci95": qc,
                    "kernel_logdet_minilm": d, "logdet_ci95": dc,
                    "n_runs": len(q_acc), "n_prompts": len(q_acc[0]),
                }
    return out


# ---------------------------------------------------------------------------
def _exec_sigs(sql_dump, queries):
    """Execute queries against the dump; -> list of signature-or-None."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.executescript(sql_dump)
    sigs = []
    for q in queries:
        try:
            cur.execute(q)
            sigs.append(tuple(cur.fetchall()))
        except Exception:
            sigs.append(None)
    conn.close()
    return sigs


def _golden_sigs(sql_dump, golden_queries_str):
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.executescript(sql_dump)
    sigs = set()
    for q in eval(golden_queries_str):
        try:
            cur.execute(q)
            sigs.add(tuple(cur.fetchall()))
        except Exception:
            pass
    conn.close()
    return sigs


def _norm_sql(x):
    return x.strip().replace("```sql\n", "").replace("\n```", "")


def run_ambrosia(k=3):
    from lm_dpp.utils.ambrosia_utils import get_ambrosia_prompts, load_ambrosia_df

    df = load_ambrosia_df(os.path.join(ROOT_DIR, "ambrosia.csv"))
    prompt_list = get_ambrosia_prompts("ambrosia-top100", df)
    row_of = {p: df.loc[pid] for p, pid in prompt_list}

    # execution cache: (prompt, query) -> signature
    sig_cache = {}
    gold_cache = {}

    out = {}
    for method, (label, knob) in METHODS.items():
        cache = load_cache(os.path.join(ROOT_DIR, "cache-ambrosia-swor", f"{GEMMA}_{method}.pkl"))
        if cache is None:
            continue
        for T in sorted(cache):
            cov_acc, prec_acc = defaultdict(list), defaultdict(list)
            for prompt, row in row_of.items():
                runs = load_runs(cache, T, prompt, k)
                if not runs:
                    continue
                if prompt not in gold_cache:
                    gold_cache[prompt] = _golden_sigs(row["db_dump"], row["ambig_queries"])
                gs = gold_cache[prompt]
                if not gs:
                    continue
                # execute all unique queries for this prompt once
                uniq = {(_norm_sql(s)) for run in runs for s in run["sents"]}
                todo = [q for q in uniq if (prompt, q) not in sig_cache]
                if todo:
                    for q, sig in zip(todo, _exec_sigs(row["db_dump"], todo)):
                        sig_cache[(prompt, q)] = sig
                for ri, run in enumerate(runs):
                    ded = list({_norm_sql(s) for s in run["sents"]})
                    sigs = [sig_cache[(prompt, q)] for q in ded]
                    cov = len({s for s in sigs if s is not None and s in gs}) / len(gs)
                    prec = sum(1 for s in sigs if s is not None and s in gs) / len(ded)
                    cov_acc[ri].append(cov)
                    prec_acc[ri].append(prec)
            if not cov_acc:
                continue
            c, cc = aggregate(cov_acc)
            p, pc = aggregate(prec_acc)
            out[f"{label}|{knob}={T}"] = {
                "recall_coverage": c, "recall_ci95": cc,
                "precision_golden_accuracy": p, "precision_ci95": pc,
                "n_runs": len(cov_acc), "n_prompts": len(cov_acc[0]),
            }
            print(f"  {label} {knob}={T} done", flush=True)
    return out


# ---------------------------------------------------------------------------
def run_gsm8k(k=10):
    from lm_dpp.utils.math_utils import (build_math_prompt, extract_model_num, is_correct,
                                         load_gsm8k_dataset, normalize_num)

    df = load_gsm8k_dataset().head(50)
    gold_of = {build_math_prompt(r["question"]): r["answer"] for _, r in df.iterrows()}

    out = {}
    for method, (label, knob) in METHODS.items():
        samples_file = os.path.join(ROOT_DIR, "cache-gsm8k-swor", f"{GEMMA}_{method}.pkl")
        cache = load_cache(samples_file)
        if cache is None:
            continue
        emb_file = samples_file.replace(".pkl", "_qwen_embeddings.pkl")
        ensure_embeddings(samples_file, "Qwen/Qwen3-Embedding-0.6B", emb_file)
        emb = load_cache(emb_file)

        corr_cache = {}
        for T in sorted(cache):
            pass_acc, chain_acc, d_acc, uniq_acc = (defaultdict(list), defaultdict(list),
                                                    defaultdict(list), defaultdict(list))
            for prompt, gold in gold_of.items():
                runs = load_runs(cache, T, prompt, k)
                if not runs:
                    continue
                lookup = emb_lookup(emb, T, prompt)
                for ri, run in enumerate(runs):
                    c = []
                    for s in run["sents"]:
                        key = (prompt, s)
                        if key not in corr_cache:
                            corr_cache[key] = 1.0 if is_correct(s, gold) else 0.0
                        c.append(corr_cache[key])
                    c = np.asarray(c)
                    pass_acc[ri].append(1.0 if c.sum() > 0 else 0.0)
                    chain_acc[ri].append(float(c.mean()))
                    uniq_acc[ri].append(float(len({normalize_num(extract_model_num(s))
                                                   for s in run["sents"]})))
                    E = np.stack([lookup[s] for s in run["sents"]])
                    d_acc[ri].append(logdet(E))
            if not pass_acc:
                continue
            pk, pkc = aggregate(pass_acc)
            ch, chc = aggregate(chain_acc)
            d, dc = aggregate(d_acc)
            uq, uqc = aggregate(uniq_acc)
            out[f"{label}|{knob}={T}"] = {
                "pass_at_K": pk, "pass_ci95": pkc,
                "mean_per_chain_accuracy": ch, "chain_ci95": chc,
                "uniq_answers_per_set": uq, "uniq_ci95": uqc,
                "kernel_logdet_qwen": d, "logdet_ci95": dc,
                "n_runs": len(pass_acc), "n_prompts": len(pass_acc[0]),
            }
            print(f"  {label} {knob}={T} done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fineweb", "ambrosia", "gsm8k"], required=True)
    a = ap.parse_args()

    if a.task == "fineweb":
        res = run_fineweb()
        hdr = ("quality_mean_logp", "kernel_logdet_minilm")
    elif a.task == "ambrosia":
        res = run_ambrosia()
        hdr = ("recall_coverage", "precision_golden_accuracy")
    else:
        res = run_gsm8k()
        hdr = ("pass_at_K", "mean_per_chain_accuracy")

    print(f"\n=== SWOR baselines: {a.task} ===")
    for m in sorted(res):
        v = res[m]
        print(f"  {m:<40s} {hdr[0]}={v[hdr[0]]:+9.4f}  {hdr[1]}={v[hdr[1]]:+9.4f}")

    dest = os.path.join(SCRIPT_DIR, f"beam_baselines_{a.task}.json")
    with open(dest, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
