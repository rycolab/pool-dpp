"""
Run DetBS and Diverse Beam Search on FineWeb intrinsic prompts and save
per-prompt (sents, logps_t1, embeddings) tuples in the same format used by
the rest of the pipeline (a list of 4-tuples per pkl file).

Usage:
    python baselines/detbs_ssk/run_intrinsic_baselines.py \
        --method detbs --beam-size 10 --kernel-weight 0.1 \
        --max-new-tokens 50 --limit 50

Output:
    cache-fineweb-ssk/{tag}.pkl  — list of (prompt, sents, logps, embeddings)
"""

import argparse, os, sys, time, pickle, math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lm_dpp.utils.fineweb_prompts import prompts as fineweb_prompts
from generate import generate as detbs_generate

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling, Normalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["detbs", "dbs"])
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--beam-size", type=int, default=10)
    p.add_argument("--num-beam-groups", type=int, default=None,
                   help="DBS only — number of diversity groups (defaults to beam-size)")
    p.add_argument("--diversity-penalty", type=float, default=1.0,
                   help="DBS only — Hamming penalty between groups")
    p.add_argument("--kernel-weight", type=float, default=0.1,
                   help="DetBS only — DPP diversity weight")
    p.add_argument("--kernel-decay", type=float, default=0.5)
    p.add_argument("--kernel-n", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="DetBS only — softmax temperature for candidate scoring")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", type=int, default=0,
                   help="Process only prompts with index % n-shards == shard")
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--cache-dir", default="cache-fineweb-ssk")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--tag", default=None,
                   help="Output filename stem (defaults derived from method + params)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--no-chat-template", action="store_true")
    return p.parse_args()


def build_embedder(name):
    if "Qwen" in name:
        m = SentenceTransformer(name,
                                model_kwargs={"device_map": "auto", "dtype": "auto"},
                                tokenizer_kwargs={"padding_side": "left"})
        bs = 32
    else:
        emb = Transformer(name)
        pool = Pooling(emb.get_word_embedding_dimension(), pooling_mode="mean")
        m = SentenceTransformer(modules=[emb, pool, Normalize()])
        bs = 32
    return m, bs


def sum_logp_at_t1(model, tokenizer, prompt_ids, full_ids):
    """Sum of log p(token | context) under the model at T=1 over generated suffix.

    full_ids: [B, L_full]   prompt_ids: [1, L_prompt]
    """
    L_prompt = prompt_ids.shape[1]
    pad_id = tokenizer.pad_token_id
    # Mask out padding so we don't count post-EOS tokens
    attn = (full_ids != pad_id).long()
    with torch.no_grad():
        logits = model(input_ids=full_ids, attention_mask=attn,
                       return_dict=True).logits  # [B, L, V]
    log_ps = torch.log_softmax(logits.float(), dim=-1)
    # Token at position t was predicted from logits at position t-1
    # Generated tokens span positions [L_prompt, L_full - 1].
    # We sum log_ps[:, t-1, full_ids[:, t]] for t in [L_prompt, L_full-1].
    gen = full_ids[:, L_prompt:]                     # [B, L_gen]
    src = log_ps[:, L_prompt - 1: -1, :]              # [B, L_gen, V]
    tok_lp = src.gather(-1, gen.unsqueeze(-1)).squeeze(-1)   # [B, L_gen]
    # Mask: only count tokens up to and including first EOS, ignore padding.
    mask = (gen != pad_id).float()
    return (tok_lp * mask).sum(dim=1).cpu().numpy(), mask.sum(dim=1).cpu().numpy()


def run_dbs(model, tokenizer, formatted_prompt, args):
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    L_prompt = inputs.input_ids.shape[1]
    nbg = args.num_beam_groups or args.beam_size
    if args.beam_size % nbg != 0:
        raise SystemExit(f"beam_size ({args.beam_size}) must be divisible by num_beam_groups ({nbg})")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            num_beams=args.beam_size,
            num_beam_groups=nbg,
            num_return_sequences=args.beam_size,
            diversity_penalty=args.diversity_penalty,
            do_sample=False,
            custom_generate="transformers-community/group-beam-search",
            trust_remote_code=True,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    seqs = out.sequences          # [K, L_full]
    return seqs, L_prompt


def run_detbs(model, tokenizer, formatted_prompt, args):
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    L_prompt = inputs.input_ids.shape[1]
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        eos_ids = torch.tensor(eos, device=model.device)
    else:
        eos_ids = torch.tensor([eos], device=model.device)
    seqs = detbs_generate(
        model,
        input_ids=inputs.input_ids,
        eos_token_ids=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
        beam_size=args.beam_size,
        sub_beam_size=args.beam_size,
        kernel_n=args.kernel_n,
        kernel_decay=args.kernel_decay,
        kernel_weight=args.kernel_weight,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    ).to(model.device)
    return seqs, L_prompt


def main():
    args = parse_args()

    if args.tag is None:
        if args.method == "detbs":
            args.tag = f"detbs_k{args.beam_size}_w{args.kernel_weight}_n{args.kernel_n}"
        else:
            nbg = args.num_beam_groups or args.beam_size
            args.tag = f"dbs_k{args.beam_size}_g{nbg}_p{args.diversity_penalty}"

    out_dir = REPO / args.cache_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{args.shard}of{args.n_shards}" if args.n_shards > 1 else ""
    out_path = out_dir / f"{args.tag}{suffix}.pkl"
    print(f"[run_intrinsic_baselines] method={args.method} tag={args.tag}")
    print(f"  out: {out_path}")

    # Resume support: load existing list and skip processed prompts.
    if out_path.exists():
        with open(out_path, "rb") as f:
            results = pickle.load(f)
        done = {x[0] for x in results}
        print(f"  resuming with {len(results)} already-processed prompts")
    else:
        results, done = [], set()

    prompts = fineweb_prompts[: args.limit] if args.limit else fineweb_prompts
    if args.n_shards > 1:
        prompts = [p for i, p in enumerate(prompts) if i % args.n_shards == args.shard]
        print(f"  shard {args.shard}/{args.n_shards}: {len(prompts)} prompts")

    print(f"Loading {args.model} ({args.dtype})…")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="eager" if args.method == "detbs" else None,
    )
    model.eval()

    print(f"Loading embedder {args.embedder}…")
    embedder, emb_bs = build_embedder(args.embedder)

    t0 = time.time()
    for prompt_text in tqdm(prompts, desc=args.tag):
        if prompt_text in done:
            continue

        if args.no_chat_template:
            formatted = prompt_text
        else:
            messages = [{"role": "user", "content": prompt_text}]
            try:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False,
                                                          add_generation_prompt=True)
            except Exception:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False)

        if args.method == "dbs":
            seqs, L_prompt = run_dbs(model, tokenizer, formatted, args)
        else:
            seqs, L_prompt = run_detbs(model, tokenizer, formatted, args)

        # Compute sum log-prob at T=1 with a single forward pass on the full sequences.
        lps, lengths = sum_logp_at_t1(model, tokenizer,
                                       seqs[:1, :L_prompt], seqs)
        sents = [tokenizer.decode(seqs[i, L_prompt:], skip_special_tokens=True)
                 for i in range(seqs.shape[0])]

        # Embed (no dedup — match the K=10 paper convention even if duplicates appear)
        embeds = embedder.encode(sents, batch_size=emb_bs, show_progress_bar=False)

        results.append((prompt_text, sents, [float(x) for x in lps], np.asarray(embeds)))

        # Persist after every prompt for crash-safety
        with open(out_path, "wb") as f:
            pickle.dump(results, f)

    print(f"\nDone in {time.time()-t0:.1f}s. Wrote {len(results)} prompts to {out_path}.")


if __name__ == "__main__":
    main()
