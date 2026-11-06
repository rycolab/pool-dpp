#!/usr/bin/env python3
"""
Diverse Beam Search baseline (Vijayakumar et al., 2018; tab:beam_baselines), through the
transformers-community/group-beam-search generation module. Deterministic: one beam of K sequences per
(diversity penalty, prompt), with K groups of one beam each.

Output cache mirrors sample_detbs.py, keyed by the penalty instead of temperature:
results[penalty][prompt] = (sents, lps, lps_t1, lengths), exactly K entries. lps and lps_t1 are both the
sum of log p at T=1 over the generated tokens, from one forward pass, because the beam scores include the
diversity penalty.

Usage:
    python -m lm_dpp.sample_divbs --mode fineweb --model google/gemma-2-9b-it --chat-template \
        --k 10 --penalties 1.0 --max-new-tokens 50 --cache-dir cache-fineweb-swor
"""
import argparse
import time
from pathlib import Path

import torch

from lm_dpp.sample_sbs import load_prompts
from lm_dpp.utils.cached import cached_ddict


def parse_args():
    p = argparse.ArgumentParser(description="Diverse Beam Search baseline")
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--mode", default="fineweb", choices=("fineweb", "ambrosia-top100", "gsm8k"))
    p.add_argument("--k", type=int, default=10, help="Beam size, set size, and number of groups")
    p.add_argument("--penalties", type=float, nargs="+", default=[1.0], help="Hamming diversity penalties")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--csv-path", default="ambrosia.csv")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sum_logp_t1(model, sequences, prompt_len, pad_id):
    """Sum of log p at T=1 over the generated tokens of each sequence, and its token count."""
    with torch.no_grad():
        logits = model(input_ids=sequences, attention_mask=(sequences != pad_id).long()).logits
    log_p = torch.log_softmax(logits.float(), dim=-1)
    generated = sequences[:, prompt_len:]
    token_lp = log_p[:, prompt_len - 1:-1].gather(-1, generated.unsqueeze(-1)).squeeze(-1)
    mask = (generated != pad_id).float()
    return (token_lp * mask).sum(1).tolist(), mask.sum(1).long().tolist()


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt_list = load_prompts(args)[: args.limit]
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "__").replace(":", "_")
    cache_path = cache_root / f"{model_safe}_divbs.pkl"
    results = cached_ddict(str(cache_path))

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.perf_counter()
    for i, (prompt_text, _) in enumerate(prompt_list):
        text = prompt_text
        if args.chat_template:
            text = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False,
                                                 add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        for penalty in args.penalties:
            if prompt_text in results[penalty]:
                continue
            with torch.no_grad():
                sequences = model.generate(
                    **inputs,
                    num_beams=args.k,
                    num_beam_groups=args.k,
                    num_return_sequences=args.k,
                    diversity_penalty=penalty,
                    do_sample=False,
                    custom_generate="transformers-community/group-beam-search",
                    trust_remote_code=True,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                ).sequences
            lps, lengths = sum_logp_t1(model, sequences, prompt_len, tokenizer.pad_token_id)
            sents = [tokenizer.decode(s[prompt_len:], skip_special_tokens=True) for s in sequences]
            results[penalty][prompt_text] = (sents, lps, lps, lengths)
            results._save()
        print(f"[{i + 1}/{len(prompt_list)}] done ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
