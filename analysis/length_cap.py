"""
Generation cap on the pools (app:length-cap)
============================================
cap     Share of pool draws per task and local temperature that reach the cap: 50 new tokens on FineWeb, 800 on
        Ambrosia (at most 2,300 tokens of prompt and output), and 4,000 tokens of prompt and output on GSM8K. The
        cache stores no finish reason, so a draw whose stop token falls exactly at the cap counts as reaching it;
        one_below_cap, the draws that stop one token earlier, estimates how many those are. Also scores the draws
        at the cap: is_correct on GSM8K, execution and golden match on Ambrosia, both as in the figures.
probe   Fresh draws with the pools' vLLM settings, which record what the cache drops: the finish reason and the
        token log-probabilities. Is the last entry of a stopped string its stop token, and does it enter
        cumulative_logprob?
stored  Forward pass over stored pool strings: does the stored T=1 score equal the sum over the text with the
        tokens the text does not show (<eos>, often after <end_of_turn>, which does not stop generation) or without
        them? Strings that do not re-encode to their token count are skipped.

Usage:
    python analysis/length_cap.py cap --root DIR [--sigtable analysis/pool_ceiling_sigtable.pkl]
    python analysis/length_cap.py probe --csv-path ambrosia.csv
    python analysis/length_cap.py stored --root DIR

DIR holds the pools as scripts/run_{fineweb,ambrosia,gsm8k}.sh write them.
"""

import argparse
import asyncio
import glob
import json
import os
import pickle
import random
import re
import statistics
from collections import Counter, defaultdict
from uuid import uuid4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "google/gemma-2-9b-it"
RAW_PKL = "google__gemma-2-9b-it.pkl"
OUT = os.path.join(SCRIPT_DIR, "length_cap.json")

# new: --max-new-tokens; total: --max-model-len (scripts/run_*.sh, sample_vllm.py default 2,300)
TASKS = {
    "fineweb": {"new": 50, "total": 2300, "chat": False},
    "ambrosia": {"new": 800, "total": 2300, "chat": True},
    "gsm8k": {"new": 4000, "total": 4000, "chat": False},
}


def save(section, value):
    out = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            out = json.load(f)
    out[section] = value
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved {OUT}:{section}", flush=True)


def pool_paths(root, task):
    """FineWeb's pool 0 sits in cache-fineweb, its other pools in cache-fineweb-pools."""
    if task == "fineweb":
        return [os.path.join(root, "cache-fineweb", RAW_PKL),
                *sorted(glob.glob(os.path.join(root, "cache-fineweb-pools", "pool*", RAW_PKL)))]
    return sorted(glob.glob(os.path.join(root, f"cache-{task}", "pool*", RAW_PKL)))


def prompt_ids(tok, prompt, chat):
    """Prompt tokens as sample_vllm.py sends them."""
    if chat:
        return list(tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True))
    return tok(prompt)["input_ids"]


def limit_of(task, plen):
    return min(TASKS[task]["new"], TASKS[task]["total"] - plen)


def stop_token_ids(tok):
    from transformers import GenerationConfig
    eos = GenerationConfig.from_pretrained(MODEL).eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    return sorted(set(eos) | {tok.eos_token_id})


# ---------------------------------------------------------------------------
# cap
# ---------------------------------------------------------------------------
def scorer_gsm8k():
    from lm_dpp.utils.math_utils import build_math_prompt, extract_model_num, is_correct, load_gsm8k_dataset
    gold = {build_math_prompt(r["question"]): r["answer"] for _, r in load_gsm8k_dataset().head(50).iterrows()}
    assert gold, "GSM8K did not load"

    def score(prompt, s):
        if prompt not in gold:
            return {"unknown": True}
        return {"correct": is_correct(s, gold[prompt]), "final_line": re.search(r"####\s*-?\d", s) is not None,
                "no_number": extract_model_num(s) == ""}
    return score


def scorer_ambrosia(path):
    from pool_ceiling import normalize, parse_question
    with open(path, "rb") as f:
        d = pickle.load(f)
    golden, tables = d["golden"], d["sig_tables"]

    def score(prompt, s):
        q, nq = parse_question(prompt), normalize(s)
        if q not in tables or nq not in tables[q]:
            return {"unknown": True}
        sig = tables[q][nq]
        return {"executes": sig is not None, "golden": sig is not None and sig in golden[q]}
    return score


def cap(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    scorers = {"gsm8k": scorer_gsm8k(), "ambrosia": scorer_ambrosia(a.sigtable), "fineweb": lambda p, s: {}}
    out = {}
    for task in TASKS:
        cells, plen = defaultdict(Counter), {}
        per_pool = defaultdict(list)
        paths = pool_paths(a.root, task)
        for path in paths:
            with open(path, "rb") as f:
                raw = pickle.load(f)
            memo = {}
            for sT, per_prompt in raw.items():
                c, draws, capped = cells[sT], 0, 0
                for prompt, entry in per_prompt.items():
                    sents, lengths = entry[0], entry[3]
                    if prompt not in plen:
                        plen[prompt] = len(prompt_ids(tok, prompt, TASKS[task]["chat"]))
                    limit = limit_of(task, plen[prompt])
                    capped_here = set()
                    for s, n in zip(sents, lengths):
                        draws += 1
                        c["max_output_tokens"] = max(c["max_output_tokens"], n)
                        c["max_total_tokens"] = max(c["max_total_tokens"], n + plen[prompt])
                        c["one_below_cap"] += int(n == limit - 1)
                        is_cap = n >= limit
                        side = "capped" if is_cap else "ended"
                        c[f"{side}_draws"] += 1
                        if is_cap:
                            capped += 1
                            capped_here.add(s)
                        key = (prompt, s)
                        if key not in memo:
                            memo[key] = scorers[task](prompt, s)
                        for k, v in memo[key].items():
                            c[f"{side}_{k}"] += int(v)
                    c["unique"] += len(set(sents))
                    c["unique_capped"] += len(capped_here)
                    c["min_limit"] = min(c["min_limit"] or limit, limit)
                per_pool[sT].append(capped / max(draws, 1))
            print(f"{task} {os.path.dirname(path)} done", flush=True)
        out[task] = {"n_pools": len(paths), "n_prompts": len(plen),
                     "n_prompts_total_limit_binds": sum(limit_of(task, v) < TASKS[task]["new"] for v in plen.values())}
        for sT in sorted(cells):
            capped_n, total = cells[sT]["capped_draws"], cells[sT]["capped_draws"] + cells[sT]["ended_draws"]
            c = dict(cells[sT]) | {"capped_draws": capped_n, "draws": total, "capped_share": capped_n / total,
                                   "capped_share_pool_min": min(per_pool[sT]),
                                   "capped_share_pool_max": max(per_pool[sT]),
                                   "unique_capped_share": cells[sT]["unique_capped"] / cells[sT]["unique"]}
            out[task][f"{sT:g}"] = c
            print(task, f"{sT:g}", json.dumps(c), flush=True)
    save("cap", out)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def probe_task(task, sT, rows, tok, stops):
    reasons, last_ids = Counter(), Counter()
    checks = Counter()
    worst_gap, last_lps, example = 0.0, [], None
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    for plen, outs in rows:
        limit = min(TASKS[task]["new"], 4000 - plen)
        for c in outs:
            ids, entries = list(c.token_ids), c.logprobs
            reason = str(c.finish_reason)
            reasons[reason] += 1
            lps = [e[i].logprob for e, i in zip(entries, ids)]
            worst_gap = max(worst_gap, abs(sum(lps) - float(c.cumulative_logprob)))
            checks["entries_equal_tokens"] += int(len(entries) == len(ids))
            if reason == "stop":
                last_ids[tok.convert_ids_to_tokens(ids[-1])] += 1
                checks["stopped_last_is_stop_token"] += int(ids[-1] in stops)
                checks["stopped_after_end_of_turn"] += int(len(ids) > 1 and ids[-2] == eot)
                checks["stopped_below_limit"] += int(len(ids) < limit)
                last_lps.append(lps[-1])
                if example is None or lps[-1] < example["last_logprobs"][-1]:
                    example = {"text_tail": c.text[-80:], "n_tokens": len(ids), "finish_reason": reason,
                               "last_tokens": tok.convert_ids_to_tokens(ids[-3:]), "last_logprobs": lps[-3:],
                               "sum_logprobs": sum(lps), "cumulative_logprob": float(c.cumulative_logprob)}
            else:
                checks["capped_at_limit"] += int(len(ids) == limit)
                checks["capped_last_not_stop_token"] += int(ids[-1] not in stops)
    return {"local_temperature": sT, "finish_reasons": dict(reasons), "stop_tokens": dict(last_ids),
            "stop_token_ids": stops, "checks": dict(checks), "max_abs_sum_minus_cumulative": worst_gap,
            "stop_logprob_median": statistics.median(last_lps) if last_lps else None, "example": example}


async def probe_async(a):
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    from lm_dpp.utils.ambrosia_utils import get_ambrosia_prompts, load_ambrosia_df
    from lm_dpp.utils.fineweb_prompts import prompts as fineweb_prompts
    from lm_dpp.utils.math_utils import build_math_prompt, load_gsm8k_dataset

    prompts = {"fineweb": list(fineweb_prompts),
               "ambrosia": [p for p, _ in get_ambrosia_prompts("ambrosia-top100", load_ambrosia_df(a.csv_path))],
               "gsm8k": [build_math_prompt(q) for q in load_gsm8k_dataset().head(50)["question"]]}
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=MODEL, dtype="auto", max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
        max_model_len=4000, gpu_memory_utilization=a.gpu_mem_util, logprobs_mode="raw_processed_logprobs"))
    tok = await engine.get_tokenizer()
    stops = stop_token_ids(tok)

    async def draw(task, prompt, sT):
        inputs = prompt
        if TASKS[task]["chat"]:
            inputs = TokensPrompt(prompt_token_ids=prompt_ids(tok, prompt, True))
        params = SamplingParams(temperature=sT, max_tokens=TASKS[task]["new"], n=a.n, logprobs=1,
                                output_kind=RequestOutputKind.CUMULATIVE)
        plen, outs = None, []
        async for req in engine.generate(inputs, params, request_id=f"s-{uuid4()}"):
            plen = len(req.prompt_token_ids)
            outs += [o for o in req.outputs if o.finished()]
        return plen, outs

    out = {}
    for task, sT in (("fineweb", 1.0), ("ambrosia", 1.5), ("gsm8k", 1.2)):
        rows = await asyncio.gather(*(draw(task, p, sT) for p in prompts[task][:a.n_prompts]))
        out[task] = probe_task(task, sT, rows, tok, stops)
        print(task, json.dumps(out[task]), flush=True)
    save("probe", out | {"n_prompts": a.n_prompts, "n_per_prompt": a.n})
    os._exit(0)  # the engine's background loop keeps the process alive


# ---------------------------------------------------------------------------
# stored
# ---------------------------------------------------------------------------
def stored(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    eos, eot = tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")
    rng = random.Random(0)
    out = {"stop_token_ids": stop_token_ids(tok)}
    for task, sT in (("fineweb", 1.0), ("ambrosia", 1.0), ("gsm8k", 1.0), ("gsm8k", 1.2)):
        with open(pool_paths(a.root, task)[0], "rb") as f:
            per_prompt = pickle.load(f)[sT]
        groups = {"ended": [], "capped": []}
        for prompt, (sents, _, lp1, lengths) in per_prompt.items():
            ids = prompt_ids(tok, prompt, TASKS[task]["chat"])
            limit = limit_of(task, len(ids))
            for s, lp, n in zip(sents, lp1, lengths):
                groups["capped" if n >= limit else "ended"].append((ids, s, float(lp), n))
        cell = {}
        for name, items in groups.items():
            rng.shuffle(items)
            tails = {0: []} if name == "capped" else {1: [eos], 2: [eot, eos]}
            without, with_tail, tail_kinds = [], [], Counter()
            for ids, s, lp, n in items:
                if len(without) == a.n:
                    break
                text = tok(s, add_special_tokens=False)["input_ids"]
                tail = tails.get(n - len(text))
                if tail is None:
                    continue
                tail_kinds[len(tail)] += 1
                x = torch.tensor([ids + text + tail], device="cuda")
                with torch.no_grad():
                    logp = torch.log_softmax(model(x).logits[0].float(), -1)
                vals = [logp[len(ids) - 1 + t, i].item() for t, i in enumerate(text + tail)]
                body, hidden = sum(vals[:len(text)]), sum(vals[len(text):])
                # only strings whose hidden tokens have log-probability away from zero separate the two sums
                if name == "capped" or abs(hidden) > 0.5:
                    without.append(abs(lp - body))
                    with_tail.append(abs(lp - body - hidden))
            cell[name] = {"n": len(without), "hidden_tokens": {str(k): v for k, v in tail_kinds.items()},
                          "median_abs_stored_minus_text": statistics.median(without) if without else None}
            if name == "ended":
                cell[name]["median_abs_stored_minus_text_and_hidden"] = statistics.median(with_tail) if with_tail else None
        out[f"{task}|{sT:g}"] = cell
        print(task, sT, json.dumps(cell), flush=True)
    save("stored", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["cap", "probe", "stored"])
    ap.add_argument("--root", default=os.path.dirname(SCRIPT_DIR))
    ap.add_argument("--sigtable", default=os.path.join(SCRIPT_DIR, "pool_ceiling_sigtable.pkl"))
    ap.add_argument("--csv-path", default="ambrosia.csv")
    ap.add_argument("--n", type=int, default=100, help="probe: draws per prompt; stored: strings per group")
    ap.add_argument("--n-prompts", type=int, default=5, help="probe")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90, help="probe")
    ap.add_argument("--max-num-seqs", type=int, default=64, help="probe")
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096, help="probe")
    a = ap.parse_args()
    if a.command == "cap":
        cap(a)
    elif a.command == "probe":
        asyncio.run(probe_async(a))
    else:
        stored(a)


if __name__ == "__main__":
    main()
