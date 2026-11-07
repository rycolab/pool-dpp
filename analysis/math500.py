"""
MATH-500 saturation check (app:math500, fig:math500)
====================================================
Reruns the selection comparison on a harder benchmark with the same language model: 20 problems per
level of HuggingFaceH4/MATH-500, 10 pools of N=100.

  sample  draws the pools with google/gemma-2-9b-it. Needs a GPU and the patched vLLM.
  verify  extracts each string's final answer (the '####' line, else the last \\boxed{}, else the last
          number) and checks it against the dataset's answer with math-verify. Each answer also gets a
          canonical key for counting distinct answers.
  embed   embeds the unique strings with Qwen3-Embedding-0.6B and a task instruction. Needs a GPU.
  score   runs selection with the functions of the main experiments (10 runs per cell, one for greedy
          MAP) and writes pass@10, distinct answers, and log-determinant per method, with question-paired
          and pool-paired sign-flip tests against ancestral sampling at local sampling temperature 1.2,
          per-level differences, the pool ceiling, and the figure.

Exclusions are counted, never silent. Every (pool, sT, problem) cell gets the pool ceiling and Ancestral
sampling, scored as drawn (duplicates included). A cell whose unique strings span fewer than K dimensions
(fewer than K unique strings, or rank < K; Zhat = 0) has no K-set for k-Means, Naive-DPP, PoolDPP-Samp or
PoolDPP-MAP; those methods leave such cells out (the main experiments redraw the pool instead), and
math500.json records, per method and knob, n_cells and n_cells_excluded_rank_lt_K, plus
_exclusions with the counts per sT. A missing pool, a pool without every problem at every sT, a
verification that times out or fails, a NaN log-determinant, or a missing ancestral reference cell stops
the run.

PoolDPP-Samp and PoolDPP-MAP both use w = plm^(1/T) here, with no division by the proposal. The pools behind
app:math500 were drawn with one raw seed per pool and share 53% of their unique strings. This script seeds each
request separately, so it draws different pools.

Usage: bash scripts/run_math500.sh, or one stage at a time:
  python analysis/math500.py {sample,verify,embed,score}
Output: cache-math500/, analysis/math500.json, analysis/math500_cells.pkl, analysis/math500_plot.pdf
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import re
import signal
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL = "google/gemma-2-9b-it"
TEMPS = [0.5, 0.8, 1.2]
N_PER_LEVEL = 20
K = 10
ANCHOR_ST = 1.2
TARGET_TS = [500, 100, 50, 25, 10, 5]
N_RUNS = 10

PROMPT = ("Solve the following math problem. Think step by step, then write your final answer on the "
          "last line, in the exact format:\n#### <answer>\n\nProblem: {p}")
INSTRUCTION = ("Given a step-by-step solution to a math problem, represent it by its final answer and "
               "the sequence of mathematical operations it performs, ignoring wording, phrasing, and "
               "formatting.")

BOXED_RE = re.compile(r"\\boxed\s*\{")
HASH_RE = re.compile(r"####\s*(.+)")

COLORS = {"Ancestral": "#1f77b4", "$k$-Means": "#9467bd", "Naive-DPP": "#2ca02c",
          "PoolDPP-Samp": "#ff7f0e", "PoolDPP-MAP": "#d62728"}
MARKERS = {"Ancestral": "^", "$k$-Means": "s", "Naive-DPP": "v", "PoolDPP-Samp": "o", "PoolDPP-MAP": "D"}


def math500_problems(n_problems=None):
    """The fixed problem set: N_PER_LEVEL per level, rng(0)."""
    from datasets import load_dataset
    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    rng = np.random.default_rng(0)
    picked = []
    for level in (1, 2, 3, 4, 5):
        idx = [i for i, ex in enumerate(ds) if ex["level"] == level]
        take = rng.choice(len(idx), min(N_PER_LEVEL, len(idx)), replace=False)
        picked += [ds[idx[j]] for j in sorted(take)]
    return picked[:n_problems]


def build_prompt(ex):
    return PROMPT.format(p=ex["problem"])


def pool_path(a, pool):
    return os.path.join(a.cache, f"pool{pool}.pkl")


def read_pools(a):
    missing = [pool for pool in range(a.n_pools) if not os.path.exists(pool_path(a, pool))]
    if missing:
        raise FileNotFoundError(f"MATH-500: {len(missing)} of {a.n_pools} pools missing: {missing} under {a.cache}; "
                                f"run the sample stage for them (scripts/run_math500.sh), or pass --n-pools for the "
                                f"pools that exist and report that count; pools are not skipped")
    for pool in range(a.n_pools):
        with open(pool_path(a, pool), "rb") as f:
            yield pool, pickle.load(f)


# ---------------------------------------------------------------------------
# sample: raw[sT][prompt] = (strings, logp at sT, logp at T=1, lengths)
# ---------------------------------------------------------------------------
def cmd_sample(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from lm_dpp.sample_vllm import at_temperature_logprob, derive_request_seed

    prompts = [build_prompt(ex) for ex in math500_problems(a.n_problems)]
    tok = AutoTokenizer.from_pretrained(MODEL)
    chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True) for p in prompts]
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util, max_model_len=2048,
              enforce_eager=False, logprobs_mode="raw_processed_logprobs")
    os.makedirs(a.cache, exist_ok=True)
    for pool in range(a.first_pool, a.n_pools):
        raw = {}
        if os.path.exists(pool_path(a, pool)):
            with open(pool_path(a, pool), "rb") as f:
                raw = pickle.load(f)
        for sT in TEMPS:
            if sT in raw:
                continue
            params = [SamplingParams(n=a.n, temperature=sT, top_p=1.0, max_tokens=a.max_tokens, logprobs=0,
                                     seed=derive_request_seed(pool, p, sT, 0)) for p in prompts]
            bucket = {}
            for p, o in zip(prompts, llm.generate(chat, params)):
                if len(o.outputs) != a.n:
                    raise RuntimeError(f"pool{pool} sT={sT}: vLLM returned {len(o.outputs)} completions, not n={a.n}")
                bucket[p] = ([c.text for c in o.outputs],
                             [at_temperature_logprob(c) for c in o.outputs],
                             [float(c.cumulative_logprob) for c in o.outputs],
                             [len(c.token_ids) for c in o.outputs])
            raw[sT] = bucket
            with open(pool_path(a, pool), "wb") as f:
                pickle.dump(raw, f)
            print(f"pool{pool} sT={sT} done", flush=True)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def _boxed_last(text):
    """Content of the last balanced \\boxed{...}."""
    m = None
    for m in BOXED_RE.finditer(text):
        pass
    if m is None:
        return None
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1] if depth == 0 else None


def extract_answer(text):
    found = HASH_RE.findall(text)
    s = found[-1].strip() if found else None
    if not s:
        s = _boxed_last(text)
    if not s:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        s = nums[-1] if nums else ""
    s = re.sub(r"\.$", "", s.strip().strip("*").strip())
    return s[:200]


def _canon_key(parsed, fallback):
    """Key for counting distinct answers: the numeric value when possible. No simplify, because it has
    no timeout and can run for minutes on long expressions."""
    import sympy
    for obj in parsed:
        if isinstance(obj, str):
            continue
        try:
            if isinstance(obj, (sympy.Tuple, tuple, list)):
                return "(" + ",".join(_canon_key([o], str(o)) for o in obj) + ")"
            if hasattr(obj, "free_symbols") and not obj.free_symbols:
                v = complex(obj.evalf(15))
                return f"num:{v.real:.8g}{v.imag:+.8g}j"
            return f"sym:{sympy.srepr(obj)}"
        except Exception:
            try:
                return "str:" + str(obj)
            except Exception:
                continue
    return "raw:" + re.sub(r"\s+", "", fallback).lower()


class _Deadline(Exception):
    pass


def _with_deadline(fn, seconds):
    """evalf and equals inside math-verify can run for an unbounded time. A timeout or an exception
    returns {"error": ...} (it returned a default verdict "incorrect" before); cmd_verify stops on it."""
    def handler(signum, frame):
        raise _Deadline()

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return fn()
    except _Deadline:
        return {"error": f"timeout after {seconds} s"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def cmd_verify(a):
    from math_verify import parse, verify
    problems = math500_problems(a.n_problems)
    by_prompt = {build_prompt(ex): ex for ex in problems}
    by_id = {ex["unique_id"]: ex for ex in problems}
    path = os.path.join(a.cache, "verify_cache.pkl")
    cache = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)
    todo = defaultdict(set)
    for _, raw in read_pools(a):
        for bucket in raw.values():
            for p, (sents, _, _, _) in bucket.items():
                qid = by_prompt[p]["unique_id"]
                todo[qid] |= {ans for ans in map(extract_answer, sents) if (qid, ans) not in cache}
    print(f"{sum(map(len, todo.values()))} (problem, answer) pairs to verify, {len(cache)} on file", flush=True)
    for qid, answers in todo.items():
        gold = parse("$" + by_id[qid]["answer"] + "$")
        for ans in answers:
            def check(ans=ans):
                parsed = parse("$" + ans + "$") if ans else []
                return {"correct": bool(verify(gold, parsed)) if parsed else False,
                        "key": _canon_key(parsed, ans)}
            cache[(qid, ans)] = _with_deadline(check, 10)
        with open(path, "wb") as f:
            pickle.dump(cache, f)
    print(f"verified, {len(cache)} on file", flush=True)
    failed = {k: v["error"] for k, v in cache.items() if "error" in v}
    if failed:
        (qid, ans), err = next(iter(failed.items()))
        raise RuntimeError(f"math-verify failed on {len(failed)} (problem, answer) pairs (first: {qid}, {ans[:60]!r}: "
                           f"{err}). They are kept in {path} with their error, not scored as incorrect; how to score "
                           f"them (longer deadline, by hand) is a decision for timv/Manuel")


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------
def cmd_embed(a):
    path = os.path.join(a.cache, f"qwenins_embeddings_shard{a.shard}.pkl")
    emb = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            emb = pickle.load(f)
    todo = {}
    for _, raw in read_pools(a):
        for bucket in raw.values():
            for sents, _, _, _ in bucket.values():
                for s in sents:
                    h = hashlib.md5(s.encode()).hexdigest()
                    if h not in emb and int(h, 16) % a.nshards == a.shard:
                        todo[h] = s
    print(f"shard {a.shard}/{a.nshards}: {len(todo)} unique strings to embed", flush=True)
    if not todo:
        return

    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda",
                                model_kwargs={"torch_dtype": torch.float16})
    todo = list(todo.items())
    for i in range(0, len(todo), 512):
        chunk = todo[i:i + 512]
        V = model.encode([f"Instruct: {INSTRUCTION}\nQuery:{s}" for _, s in chunk], batch_size=32,
                         convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        for (h, _), v in zip(chunk, V):
            emb[h] = v.astype(np.float16)
        with open(path, "wb") as f:
            pickle.dump(emb, f)
    print(f"shard {a.shard} done", flush=True)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def unit(E):
    E = np.asarray(E, np.float64)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1
    return E / n


def logdet(F):
    s, ld = np.linalg.slogdet(F @ F.T + 1e-6 * np.eye(len(F)))
    if not s > 0:
        raise RuntimeError(f"log-determinant of a {len(F)}-row Gram matrix + 1e-6 I has sign {s}")
    return ld


def signflip_p(d, n_perm=100000, seed=1):
    d = np.asarray(d, np.float64)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], (n_perm, len(d)))
    null = (signs * np.abs(d)).mean(1)
    return float(max((np.abs(null) >= abs(d.mean())).mean(), 1.0 / n_perm))


_shared = {}


def _score_pool(item):
    """Rows (method, knob, (pool, sT, qid), metrics) for one pool."""
    from lm_dpp.estimation import ancestral_draw, greedy_map, quality_log_scales, sample_dpp, sample_kmeans_for_prompt

    from dpp.stable import RANK_TOL

    pool, raw = item
    by_prompt, verdicts, emb = _shared["by_prompt"], _shared["verdicts"], _shared["emb"]
    np.random.seed(1000 + pool)
    rows = []
    for sT in sorted(raw):
        if set(raw[sT]) != set(by_prompt):
            raise ValueError(f"pool{pool} sT={sT}: has {len(set(raw[sT]) & set(by_prompt))} of the "
                             f"{len(by_prompt)} problems; sample the missing ones")
        for p, (sents, _, lps_t1, _) in raw[sT].items():
            qid = by_prompt[p]["unique_id"]
            unique = {}
            for s, lp in zip(sents, lps_t1):
                e = unique.setdefault(hashlib.md5(s.encode()).hexdigest(), [s, lp, 0])
                e[1] = max(e[1], lp)
                e[2] += 1
            hashes = list(unique)
            n = len(hashes)
            lp1 = np.array([unique[h][1] for h in hashes])
            counts = np.array([unique[h][2] for h in hashes])
            E = unit(np.stack([emb[h] for h in hashes]))
            # Zhat = 0 exactly when the unique strings span fewer than K dimensions (fewer than K strings
            # included); then k-Means, Naive-DPP and PoolDPP have no K-set, and the cell is counted, not scored
            rank_ok = n >= K and np.linalg.matrix_rank(E, tol=RANK_TOL) >= K
            judged = [verdicts[(qid, extract_answer(unique[h][0]))] for h in hashes]
            bad = [v["error"] for v in judged if "error" in v]
            if bad:
                raise RuntimeError(f"pool{pool} sT={sT} {qid}: {len(bad)} answers failed verification ({bad[0]}); "
                                   f"see cmd_verify")
            correct = np.array([v["correct"] for v in judged], dtype=float)
            keys = [v["key"] for v in judged]
            cell = (pool, sT, qid)

            def add(method, knob, idx):
                idx = list(idx)
                if method not in ("Ancestral",) and len(idx) != K:
                    raise RuntimeError(f"{method} {knob} pool{pool} sT={sT} {qid}: {len(idx)} of K={K} items on a "
                                       f"cell whose strings span K dimensions")
                rows.append((method, knob, cell, {"pass": float(correct[idx].sum() > 0),
                                                  "ukey": len({keys[i] for i in idx}),
                                                  "ld": logdet(E[idx])}))

            def excluded(method, knob):
                rows.append((method, knob, cell, None))

            rows.append(("_ceiling", sT, cell, {"pass": float(correct.sum() > 0), "ukey": len(set(keys)),
                                                "ld": None}))
            for run in range(N_RUNS):
                add("Ancestral", sT, ancestral_draw(counts, K))
                if rank_ok:
                    add("$k$-Means", sT, sample_kmeans_for_prompt(E, lp1, K, random_state=run))
            if not rank_ok:
                excluded("$k$-Means", sT)
            # sample_dpp returns [] exactly when Zhat = 0 and raises on a numerical failure
            draws = sample_dpp(np.zeros(n), E, K, N=N_RUNS) if n >= K else [[]] * N_RUNS
            if any(len(sel) != (K if rank_ok else 0) for sel in draws):
                raise RuntimeError(f"Naive-DPP pool{pool} sT={sT} {qid}: draw sizes {[len(d) for d in draws]} "
                                   f"disagree with the rank test")
            for sel in draws if rank_ok else []:
                add("Naive-DPP", sT, sel)
            if not rank_ok:
                excluded("Naive-DPP", sT)
            if abs(sT - ANCHOR_ST) < 1e-9:
                for T in TARGET_TS:
                    if not rank_ok:
                        excluded("PoolDPP-MAP", T)
                        excluded("PoolDPP-Samp", T)
                        continue
                    log_s = quality_log_scales(lp1, 1.0 / (2.0 * T))
                    add("PoolDPP-MAP", T, greedy_map(log_s, E, K))
                    for sel in sample_dpp(log_s, E, K, N=N_RUNS):
                        add("PoolDPP-Samp", T, sel)
    print(f"pool{pool} scored", flush=True)
    return rows


def _pool_ci(xs, ys):
    """Means and half-widths of 95% bootstrap intervals over pools."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    idx = np.random.default_rng(0).integers(0, len(xs), size=(10000, len(xs)))
    half = lambda v: float(np.subtract(*np.quantile(v[idx].mean(1), [0.975, 0.025])) / 2)
    return float(xs.mean()), float(ys.mean()), half(xs), half(ys)


def _plot(cells, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for name in ("Ancestral", "Naive-DPP", "$k$-Means", "PoolDPP-Samp", "PoolDPP-MAP"):
        pts = []
        for knob, by_cell in cells[name].items():
            per_pool = defaultdict(list)
            for (pool, _, _), draws in by_cell.items():
                per_pool[pool].append((np.mean([d["ld"] for d in draws]), np.mean([d["pass"] for d in draws])))
            xs = [np.mean([c[0] for c in v]) for v in per_pool.values()]
            ys = [np.mean([c[1] for c in v]) for v in per_pool.values()]
            pts.append((*_pool_ci(xs, ys), knob))
        pts.sort()
        x, y, xerr, yerr, knobs = zip(*pts)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, color=COLORS[name], marker=MARKERS[name], markersize=4, capsize=2,
                    capthick=0.5, elinewidth=0.6, linewidth=1.2, label=name)
        for xi, yi, knob in zip(x, y, knobs):
            ax.annotate(f"{knob:g}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6,
                        color=COLORS[name])
    ax.set_xlabel("Diversity (log-determinant of Gram matrix)")
    ax.set_ylabel(f"Accuracy (pass@$K{{=}}{K}$)")
    ax.set_title(f"MATH-500 subset (gemma-2-9b-it, instructed Qwen3-0.6B kernel, {a.n_pools} pools)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.4)
    path = os.path.join(a.out_dir, "math500_plot.pdf")
    fig.savefig(path, bbox_inches="tight")
    print("Saved", path, flush=True)


def cmd_score(a):
    problems = math500_problems(a.n_problems)
    _shared["by_prompt"] = {build_prompt(ex): ex for ex in problems}
    level_of = {ex["unique_id"]: ex["level"] for ex in problems}
    with open(os.path.join(a.cache, "verify_cache.pkl"), "rb") as f:
        _shared["verdicts"] = pickle.load(f)
    _shared["emb"] = {}
    for name in sorted(os.listdir(a.cache)):
        if name.startswith("qwenins_embeddings"):
            with open(os.path.join(a.cache, name), "rb") as f:
                _shared["emb"].update(pickle.load(f))
    print(f"{len(_shared['emb'])} embeddings loaded", flush=True)

    # keep only the problems in play, so --n-problems works on a partial pool
    pools = [(i, {sT: {p: v for p, v in b.items() if p in _shared["by_prompt"]} for sT, b in raw.items()})
             for i, raw in read_pools(a)]
    with mp.get_context("fork").Pool(a.workers) as workers:
        results = workers.map(_score_pool, pools, chunksize=1)

    # cells[method][knob][(pool, sT, qid)] = per-draw metrics; excluded[method][knob] = cells with rank < K
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    excluded = defaultdict(lambda: defaultdict(int))
    for rows in results:
        for method, knob, cell, metrics in rows:
            if metrics is None:
                excluded[method][knob] += 1
            else:
                cells[method][knob][cell].append(metrics)
    with open(os.path.join(a.out_dir, "math500_cells.pkl"), "wb") as f:
        pickle.dump({m: {k: dict(v) for k, v in kn.items()} for m, kn in cells.items()}, f)

    def cell_means(method, knob):
        return {c: float(np.mean([d["pass"] for d in draws])) for c, draws in cells[method][knob].items()}

    reference = cell_means("Ancestral", ANCHOR_ST)

    def paired(method_cells):
        """Differences from ancestral sampling at the anchor temperature on the same pool and problem."""
        by_q, by_pool = defaultdict(list), defaultdict(list)
        for (pool, _, qid), v in method_cells.items():
            ref = reference[(pool, ANCHOR_ST, qid)]  # Ancestral scores every cell, so this always exists
            by_q[qid].append(v - ref)
            by_pool[pool].append(v - ref)
        dq = [np.mean(v) for v in by_q.values()]
        dp = [np.mean(v) for v in by_pool.values()]
        by_level = defaultdict(list)
        for qid, v in by_q.items():
            by_level[level_of[qid]].append(np.mean(v))
        return {"delta": float(np.mean(dq)), "p_question": signflip_p(dq), "p_pool": signflip_p(dp),
                "delta_by_level": {str(k): float(np.mean(v)) for k, v in sorted(by_level.items())}}

    out = {}
    print(f"\n{'method':<16s} {'knob':>6s} {'pass@10':>8s} {'answers':>8s} {'log-det':>8s} {'delta':>8s} "
          f"{'p_q':>8s} {'p_pool':>8s}")
    for method in sorted(set(cells) | set(excluded)):
        for knob in sorted(set(cells[method]) | set(excluded[method])):
            draws = cells[method][knob]
            if not draws:
                raise ValueError(f"{method} {knob}: all {excluded[method][knob]} cells excluded (rank < K)")
            o = {m: float(np.mean([np.mean([d[m] for d in v]) for v in draws.values()]))
                 if method != "_ceiling" or m != "ld" else None for m in ("pass", "ukey", "ld")}
            o["n_cells"] = len(draws)
            o["n_cells_excluded_rank_lt_K"] = excluded[method][knob]
            if method != "_ceiling" and not (method == "Ancestral" and knob == ANCHOR_ST):
                o.update(paired(cell_means(method, knob)))
            out[f"{method}|{knob}"] = o
            nan = float("nan")
            print(f"{method:<16s} {knob:>6g} {o['pass']:>8.4f} {o['ukey']:>8.2f} {o['ld'] or nan:>8.2f} "
                  f"{o.get('delta', nan):>8.4f} {o.get('p_question', nan):>8.5f} {o.get('p_pool', nan):>8.5f}")
    out["_exclusions"] = {
        "note": ("cells (pool, sT, problem) whose unique strings span fewer than K dimensions; k-Means, "
                 "Naive-DPP and PoolDPP leave them out, Ancestral and the ceiling score them"),
        "n_cells_by_sT": {f"{sT:g}": len(cells["_ceiling"][sT]) for sT in sorted(cells["_ceiling"])},
        "n_cells_rank_lt_K_by_sT": {f"{sT:g}": excluded["$k$-Means"][sT] for sT in sorted(cells["_ceiling"])},
    }
    path = os.path.join(a.out_dir, "math500.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", path, flush=True)
    _plot(cells, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sample", "verify", "embed", "score"))
    ap.add_argument("--cache", default=os.path.join(ROOT_DIR, "cache-math500"))
    ap.add_argument("--out-dir", default=SCRIPT_DIR)
    ap.add_argument("--n-pools", type=int, default=10)
    ap.add_argument("--n-problems", type=int, default=None, help="use only the first problems of the set")
    ap.add_argument("--first-pool", type=int, default=0, help="sample: skip earlier pools, to split across GPUs")
    ap.add_argument("--n", type=int, default=100, help="sample: pool size")
    ap.add_argument("--max-tokens", type=int, default=1536, help="sample: length cap")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90, help="sample")
    ap.add_argument("--shard", type=int, default=0, help="embed: one shard per GPU")
    ap.add_argument("--nshards", type=int, default=1, help="embed")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="score")
    a = ap.parse_args()
    {"sample": cmd_sample, "verify": cmd_verify, "embed": cmd_embed, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
