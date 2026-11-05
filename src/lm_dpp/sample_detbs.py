#!/usr/bin/env python3
"""
Determinantal Beam Search baseline (Meister et al., 2021). Deterministic: one
beam of K sequences per (diversity weight w, prompt); w is the diversity knob
(w = 0 recovers standard beam search at T=1). All weights are decoded in one
batch per prompt, sharing model forward passes.

Output cache mirrors sample_vllm.py but is keyed by w instead of temperature:
results[w][prompt] = (sents, lps, lps_t1, lengths), exactly K entries.

Usage:
    python -m lm_dpp.sample_detbs --mode gsm8k --model google/gemma-2-9b-it \
        --k 10 --weights 0.0 0.01 0.03 0.06 0.1 0.15 \
        --max-new-tokens 2000 --cache-dir cache-gsm8k-swor
"""
import argparse
import time
from pathlib import Path

import torch

from lm_dpp.det_beam import detbs_generate
from lm_dpp.sample_sbs import get_eos_ids, load_prompts
from lm_dpp.sbs_decoding import HFBackend
from lm_dpp.utils.cached import cached_ddict


def parse_args():
    p = argparse.ArgumentParser(description="Determinantal Beam Search baseline")
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--mode", default="fineweb", choices=("fineweb", "ambrosia-top100", "gsm8k"))
    p.add_argument("--k", type=int, default=10, help="Beam size / set size")
    p.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.06, 0.1, 0.15],
                   help="Diversity weights w (w=0 is standard beam search)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--csv-path", default="ambrosia.csv")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt_list = load_prompts(args)
    if args.limit:
        prompt_list = prompt_list[: args.limit]

    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "__").replace(":", "_")
    cache_path = cache_root / f"{model_safe}_detbs.pkl"
    results = cached_ddict(str(cache_path))

    work = []  # (prompt, [missing weights])
    for prompt_text, _ in prompt_list:
        missing = [w for w in args.weights
                   if (e := results[w].get(prompt_text)) is None or len(e[0]) < args.k]
        if missing:
            work.append((prompt_text, missing))
    print(f"{len(prompt_list)} prompts, {len(work)} with missing weights. Cache: {cache_path}")
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

    t0 = time.perf_counter()
    total_tokens = 0
    for ji, (prompt_text, weights) in enumerate(work):
        if args.chat_template:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True, enable_thinking=False)
        else:
            ids = tokenizer(prompt_text)["input_ids"]

        backend = HFBackend(model, torch.tensor(ids, dtype=torch.long))
        groups = detbs_generate(backend, K=args.k, weights=weights,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, eos_ids=eos_ids)
        for w, g in zip(weights, groups):
            sents, lps, lps_t1, lengths = [], [], [], []
            for s in g:
                sents.append(tokenizer.decode(s.tokens, skip_special_tokens=True))
                lps.append(float(s.logp))
                lps_t1.append(float(s.logp_t1))
                lengths.append(len(s.tokens))
                total_tokens += len(s.tokens)
            results[w][prompt_text] = (sents, lps, lps_t1, lengths)
        results._save()
        rate = total_tokens / (time.perf_counter() - t0)
        print(f"[{ji + 1}/{len(work)}] prompt done, {len(weights)} weights "
              f"({rate:.0f} tok/s cumulative)", flush=True)

    print(f"Done. {total_tokens} tokens in {(time.perf_counter() - t0) / 60:.1f} min.")


def cli():
    main()


if __name__ == "__main__":
    main()
