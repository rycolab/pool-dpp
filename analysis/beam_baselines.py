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
prompts per run index, then mean +/- 95% t-interval half-width across the
n_runs independent runs (t_{0.975, n_runs-1} * sd / sqrt(n_runs)). DetBS and
DivBS are deterministic, one run per prompt, so their *_ci95 fields are null.
prompt_sd is the standard deviation over prompts in run 0 with ddof=0.

Failures stop the run with an error; nothing is skipped. Every method of
EXPECTED[task] must have its decoding cache (FileNotFoundError otherwise). Every
prompt of the task must be in the cache at every temperature (or weight) the
cache holds, with a whole number of K-string runs, the same number of runs for
every prompt (exactly 1 for DetBS/DivBS, at least 2 otherwise), and an
embedding for every string; a golden query that fails to execute, a prompt
with no golden result, or a non-finite metric (e.g. a log-det whose Gram
matrix plus 1e-6 I is not positive definite) raises ValueError. A generated
SQL query that fails to execute is scored as not runnable (that is the
metric).

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
from heldout_diversity import distinct_n  # noqa: E402

# method suffix -> (label, knob name for the cache key: sampling temperature
# for the SWOR methods, diversity weight w for deterministic DetBS)
METHODS = {"sbs": ("SBS", "sT"), "cps": ("CPSBS", "sT"), "detbs": ("DetBS", "w"),
           "divbs": ("DivBS", "penalty"),  # deterministic; knob is the Hamming diversity penalty
           "setty": ("SetSMC", "T"),  # setty-smc: SIS + inclusion-kernel end readout
           "settysmc": ("SetSMC-online", "T"),  # setty-smc: online tau=.25 arm
           "setty100": ("SetSMC-100", "T"), "setty200": ("SetSMC-200", "T")}
GEMMA = "google__gemma-2-9b-it"
QWEN_LM = "Qwen__Qwen3-8B"
# decoding caches each task must have, per LM: those scripts/run_beam_baselines.sh
# writes (sbs, cps, detbs; divbs on FineWeb with Gemma), plus the SetSMC caches
# of the committed outputs (their decoding code is not in this repository)
EXPECTED = {
    "fineweb": {GEMMA: ["sbs", "cps", "detbs", "divbs"], QWEN_LM: ["sbs", "cps", "detbs"]},
    "ambrosia": {GEMMA: ["sbs", "cps", "detbs", "setty", "settysmc", "setty100", "setty200"]},
    "gsm8k": {GEMMA: ["sbs", "cps", "detbs", "setty", "settysmc"]},
}
DETERMINISTIC = {"detbs", "divbs"}   # one decode per prompt, so one run


def load_runs(cache, T, prompt, k, where):
    """-> list of runs, each a dict(sents, lps_t1) of length k. Raises ValueError if the
    prompt is missing or its strings do not split into whole runs of k."""
    entry = cache[T].get(prompt)
    if entry is None:
        raise ValueError(f"{where}: prompt {prompt[:60]!r} is not in the cache")
    sents, lps, lps_t1, lengths = entry
    if len(sents) % k or len(sents) != len(lps_t1) or not sents:
        raise ValueError(f"{where}: prompt {prompt[:60]!r} has {len(sents)} strings and "
                         f"{len(lps_t1)} log-probs, not a whole number of runs of K={k}")
    return [
        {"sents": sents[r: r + k], "lps_t1": lps_t1[r: r + k]}
        for r in range(0, len(sents), k)
    ]


def check_n_runs(runs_of_prompt, method, where):
    """All prompts must have the same number of runs: exactly 1 for a deterministic
    method, at least 2 otherwise (for the interval across runs)."""
    counts = {len(r) for r in runs_of_prompt.values()}
    if len(counts) != 1:
        raise ValueError(f"{where}: prompts have different numbers of runs {sorted(counts)}")
    n = counts.pop()
    if (method in DETERMINISTIC) != (n == 1) or n < 1:
        raise ValueError(f"{where}: {n} runs per prompt; expected "
                         f"{'1 (deterministic)' if method in DETERMINISTIC else 'at least 2'}")


def load_cache(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"decoding cache {path} missing; run scripts/run_beam_baselines.sh")
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_embeddings(samples_file, model_name, output_file):
    if os.path.exists(output_file):
        return
    print(f"  computing embeddings: {output_file}")
    from lm_dpp.embeddings import precompute_embeddings
    precompute_embeddings(model_name, samples_file, output_file)


def emb_lookup(emb, T, prompt, where):
    d = emb.get(T, {}).get(prompt)
    if d is None:
        raise ValueError(f"{where}: no embeddings for prompt {prompt[:60]!r}")
    return dict(zip(list(d["u_sents"]), np.asarray(d["embeddings"], dtype=np.float64)))


def embed_run(lookup, sents, where):
    absent = [s for s in sents if s not in lookup]
    if absent:
        raise ValueError(f"{where}: {len(absent)} strings have no embedding (first: {absent[0][:60]!r})")
    return np.stack([lookup[s] for s in sents])


def ci95(runs_means):
    """95% t-interval half-width, t_{0.975, n-1} * sd / sqrt(n); raises for n < 2."""
    from scipy import stats
    a = np.asarray(runs_means, dtype=np.float64)
    if len(a) < 2:
        raise ValueError(f"a 95% t-interval needs at least 2 runs, got {len(a)}")
    return float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def aggregate(per_run_per_prompt, deterministic=False, where=""):
    """{run_idx: [prompt values]} -> (mean, ci95 across run means). A deterministic
    method has one run and no interval (None). Raises on an empty run or a
    non-finite value."""
    from baselines import finite_mean
    run_means = [finite_mean(v, f"{where} run {r}") for r, v in per_run_per_prompt.items()]
    if not run_means:
        raise ValueError(f"{where}: no runs")
    if deterministic:
        if len(run_means) != 1:
            raise ValueError(f"{where}: deterministic method with {len(run_means)} runs")
        return run_means[0], None
    return float(np.mean(run_means)), ci95(run_means)


# ---------------------------------------------------------------------------
def run_fineweb(k=10):
    from lm_dpp.utils.fineweb_prompts import prompts as fineweb_prompts

    out = {}
    for lm, methods in EXPECTED["fineweb"].items():
        for method in methods:
            label, knob = METHODS[method]
            det = method in DETERMINISTIC
            samples_file = os.path.join(ROOT_DIR, "cache-fineweb-swor", f"{lm}_{method}.pkl")
            cache = load_cache(samples_file)
            emb_file = samples_file.replace(".pkl", "_minilm_embeddings.pkl")
            ensure_embeddings(samples_file, "sentence-transformers/all-MiniLM-L6-v2", emb_file)
            emb = load_cache(emb_file)

            for T in sorted(cache):
                where = f"fineweb {label}|{lm}|{knob}={T} ({samples_file})"
                runs_of = {pr: load_runs(cache, T, pr, k, where) for pr in fineweb_prompts}
                check_n_runs(runs_of, method, where)
                q_acc, d_acc, d2_acc = defaultdict(list), defaultdict(list), defaultdict(list)
                for prompt in fineweb_prompts:
                    runs = runs_of[prompt]
                    lookup = emb_lookup(emb, T, prompt, where)
                    for ri, run in enumerate(runs):
                        q_acc[ri].append(float(np.mean(run["lps_t1"])))
                        E = embed_run(lookup, run["sents"], f"{where} prompt {prompt[:40]!r}")
                        d_acc[ri].append(logdet(E))
                        d2_acc[ri].append(distinct_n(run["sents"], 2))
                q, qc = aggregate(q_acc, det, f"{where} quality")
                d, dc = aggregate(d_acc, det, f"{where} logdet")
                out[f"{label}|{lm}|{knob}={T}"] = {
                    "quality_mean_logp": q, "quality_ci95": qc,
                    "kernel_logdet_minilm": d, "logdet_ci95": dc,
                    "distinct_2": aggregate(d2_acc, det, f"{where} distinct_2")[0],
                    # standard deviations over prompts in run 0, as in tab:beam_baselines
                    "prompt_sd": {"quality": float(np.std(q_acc[0])), "logdet": float(np.std(d_acc[0])),
                                  "distinct_2": float(np.std(d2_acc[0]))},
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
        except Exception as e:
            conn.close()
            raise ValueError(f"golden query {q[:80]!r} fails to execute: {e}") from e
        sigs.add(tuple(cur.fetchall()))
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
    for method in EXPECTED["ambrosia"][GEMMA]:
        label, knob = METHODS[method]
        det = method in DETERMINISTIC
        path = os.path.join(ROOT_DIR, "cache-ambrosia-swor", f"{GEMMA}_{method}.pkl")
        cache = load_cache(path)
        for T in sorted(cache):
            where = f"ambrosia {label}|{knob}={T} ({path})"
            runs_of = {pr: load_runs(cache, T, pr, k, where) for pr in row_of}
            check_n_runs(runs_of, method, where)
            cov_acc, prec_acc = defaultdict(list), defaultdict(list)
            for prompt, row in row_of.items():
                runs = runs_of[prompt]
                if prompt not in gold_cache:
                    try:
                        gold_cache[prompt] = _golden_sigs(row["db_dump"], row["ambig_queries"])
                    except ValueError as e:
                        raise ValueError(f"ambrosia prompt {prompt[:60]!r}: {e}") from e
                gs = gold_cache[prompt]
                if not gs:
                    raise ValueError(f"ambrosia prompt {prompt[:60]!r}: no golden query result")
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
            c, cc = aggregate(cov_acc, det, f"{where} recall")
            p, pc = aggregate(prec_acc, det, f"{where} precision")
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
    if len(df) < 50:
        raise SystemExit("GSM8K dataset did not load (HF offline?); refusing to score")
    gold_of = {build_math_prompt(r["question"]): r["answer"] for _, r in df.iterrows()}

    out = {}
    for method in EXPECTED["gsm8k"][GEMMA]:
        label, knob = METHODS[method]
        det = method in DETERMINISTIC
        samples_file = os.path.join(ROOT_DIR, "cache-gsm8k-swor", f"{GEMMA}_{method}.pkl")
        cache = load_cache(samples_file)
        emb_file = samples_file.replace(".pkl", "_qwen_embeddings.pkl")
        ensure_embeddings(samples_file, "Qwen/Qwen3-Embedding-0.6B", emb_file)
        emb = load_cache(emb_file)

        corr_cache = {}
        for T in sorted(cache):
            pass_acc, chain_acc, d_acc, uniq_acc = (defaultdict(list), defaultdict(list),
                                                    defaultdict(list), defaultdict(list))
            where = f"gsm8k {label}|{knob}={T} ({samples_file})"
            runs_of = {pr: load_runs(cache, T, pr, k, where) for pr in gold_of}
            check_n_runs(runs_of, method, where)
            for prompt, gold in gold_of.items():
                runs = runs_of[prompt]
                lookup = emb_lookup(emb, T, prompt, where)
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
                    E = embed_run(lookup, run["sents"], f"{where} prompt {prompt[:40]!r}")
                    d_acc[ri].append(logdet(E))
            pk, pkc = aggregate(pass_acc, det, f"{where} pass@K")
            ch, chc = aggregate(chain_acc, det, f"{where} per-chain")
            d, dc = aggregate(d_acc, det, f"{where} logdet")
            uq, uqc = aggregate(uniq_acc, det, f"{where} unique answers")
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
