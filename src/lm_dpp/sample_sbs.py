#!/usr/bin/env python3
"""
Decoding-time SWOR baselines: Stochastic Beam Search (Kool et al., 2019) and
Conditional Poisson Stochastic Beam Search (Meister et al., 2021).

Unlike the pool-selection methods (which choose K items out of an N-sample
pool), each run of these decoders directly produces one set of K sequences
sampled without replacement from the (annealed) model. `--n-runs` independent
sets are decoded per (temperature, prompt) for run-level spread.

Output cache mirrors sample_vllm.py: results[T][prompt] = (sents, lps,
lps_t1, lengths), except the flat lists are run-major: run r occupies the
slice [r*K : (r+1)*K]. lps accumulates log p_tau (locally renormalized
annealed conditionals, the vLLM `cumulative_logprob_at_temperature`
convention) and lps_t1 the raw T=1 log-probability.

Usage (one method per invocation):
    python -m lm_dpp.sample_sbs --mode gsm8k --method sbs \
        --model google/gemma-2-9b-it --k 10 --n-runs 10 \
        --temperatures 0.5 0.8 1.0 1.2 --max-new-tokens 2000 \
        --cache-dir cache-gsm8k-swor --group-batch 5 --seed 0
"""
import argparse
import time
import zlib
from pathlib import Path

import torch

from lm_dpp.sbs_decoding import HFBackend, swor_beam_generate
from lm_dpp.utils.cached import cached_ddict


def parse_args():
    p = argparse.ArgumentParser(description="SBS / CPSBS sampling baselines")
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--method", choices=("sbs", "cps"), required=True)
    p.add_argument("--mode", default="fineweb", choices=("fineweb", "ambrosia-top100", "gsm8k"))
    p.add_argument("--k", type=int, default=10, help="Set size (beam width)")
    p.add_argument("--n-runs", type=int, default=10, help="Independent sets per (T, prompt)")
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0])
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--csv-path", default="ambrosia.csv")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Limit number of prompts")
    p.add_argument("--group-batch", type=int, default=None,
                   help="Sets decoded per batch (default: n-runs)")
    p.add_argument("--cp-head-size", type=int, default=8192)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_prompts(args):
    if args.mode == "fineweb":
        from lm_dpp.utils.fineweb_prompts import prompts as fineweb_prompts
        return [(p, None) for p in fineweb_prompts]
    if args.mode == "gsm8k":
        from lm_dpp.utils.math_utils import build_math_prompt, load_gsm8k_dataset
        df = load_gsm8k_dataset().head(50)
        return [(build_math_prompt(row["question"]), str(i)) for i, row in df.iterrows()]
    from lm_dpp.utils.ambrosia_utils import get_ambrosia_prompts, load_ambrosia_df
    return get_ambrosia_prompts(args.mode, load_ambrosia_df(args.csv_path))


def get_eos_ids(model, tokenizer) -> set[int]:
    ids = set()
    for src in (model.generation_config.eos_token_id, tokenizer.eos_token_id):
        if src is None:
            continue
        if isinstance(src, int):
            ids.add(src)
        else:
            ids.update(int(i) for i in src)
    return ids


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt_list = load_prompts(args)
    if args.limit:
        prompt_list = prompt_list[: args.limit]

    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "__").replace(":", "_")
    cache_path = cache_root / f"{model_safe}_{args.method}.pkl"
    results = cached_ddict(str(cache_path))

    target = args.n_runs * args.k
    work = []
    for T in args.temperatures:
        for prompt_text, _ in prompt_list:
            entry = results[T].get(prompt_text)
            have = 0 if entry is None else len(entry[0])
            if have < target:
                work.append((T, prompt_text, have))
    print(f"{len(prompt_list)} prompts, {len(work)} (T, prompt) jobs remaining. Cache: {cache_path}")
    if not work:
        return

    is_gemma = "gemma-2" in args.model.lower()
    print(f"Loading {args.model} ({'eager' if is_gemma else 'sdpa'} attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        attn_implementation="eager" if is_gemma else "sdpa",
        device_map=args.device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos_ids = get_eos_ids(model, tokenizer)
    print(f"EOS ids: {sorted(eos_ids)}")

    G_max = args.group_batch or args.n_runs
    t0 = time.perf_counter()
    total_tokens = 0

    for ji, (T, prompt_text, have) in enumerate(work):
        if args.chat_template:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True, enable_thinking=False)
        else:
            ids = tokenizer(prompt_text)["input_ids"]
        prompt_ids = torch.tensor(ids, dtype=torch.long)

        runs_missing = (target - have + args.k - 1) // args.k
        chunk_idx = have // args.k
        while runs_missing > 0:
            G = min(G_max, runs_missing)
            seed = (args.seed * 1000003
                    + zlib.crc32(f"{args.method}|{T}|{prompt_text}|{chunk_idx}".encode())) % (2**63)
            gen = torch.Generator(device=args.device).manual_seed(seed)
            backend = HFBackend(model, prompt_ids)
            groups = swor_beam_generate(
                backend, method=args.method, K=args.k, G=G,
                max_new_tokens=args.max_new_tokens, temperature=T,
                eos_ids=eos_ids, generator=gen, cp_head_size=args.cp_head_size)

            entry = results[T].get(prompt_text)
            sents, lps, lps_t1, lengths = ([], [], [], []) if entry is None else map(list, entry)
            for g in groups:
                for s in g:
                    sents.append(tokenizer.decode(s.tokens, skip_special_tokens=True))
                    lps.append(float(s.logp))
                    lps_t1.append(float(s.logp_t1))
                    lengths.append(len(s.tokens))
                    total_tokens += len(s.tokens)
            results[T][prompt_text] = (sents, lps, lps_t1, lengths)
            results._save()
            runs_missing -= G
            chunk_idx += G

        rate = total_tokens / (time.perf_counter() - t0)
        print(f"[{ji + 1}/{len(work)}] T={T} prompt#{ji % len(prompt_list)} done "
              f"({rate:.0f} tok/s cumulative)", flush=True)

    print(f"Done. {total_tokens} tokens in {(time.perf_counter() - t0) / 60:.1f} min.")


def cli():
    main()


if __name__ == "__main__":
    main()
