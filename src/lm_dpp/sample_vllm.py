#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import time
from pathlib import Path
from uuid import uuid4

import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TokensPrompt
from vllm.sampling_params import RequestOutputKind


from lm_dpp.utils.cached import cached_ddict
from lm_dpp.utils.fineweb_prompts import prompts as fineweb_prompts

from lm_dpp.utils.ambrosia_utils import (
    load_ambrosia_df,
    init_stats_file,
    get_ambrosia_prompts,
    compute_ambrosia_stats,
)
from lm_dpp.utils.math_utils import load_gsm8k_dataset, build_math_prompt


def derive_request_seed(base_seed: int | None, prompt_text: str, T: float, offset: int) -> int | None:
    """Unique per-request seed from (pool seed, prompt, temperature, resume offset).

    Passing one raw seed to every request makes all pools consume the same
    RNG stream: pools sampled that way shared ~73% of their strings pairwise
    (measured on GSM8K). Hashing guarantees a distinct Philox
    stream per (pool, prompt, temperature) and per resume, while staying
    reproducible given --seed.
    """
    if base_seed is None:
        return None
    h = hashlib.blake2b(
        f"{base_seed}|{T}|{offset}|{prompt_text}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(h, "little") >> 1  # 63-bit non-negative


def extract_results(outputs: list) -> tuple[list[str], list[float], list[float], list[int]]:
    texts = [c.text for c in outputs]
    t1_lps = [float(c.cumulative_logprob) for c in outputs]
    lps = [float(getattr(c, "cumulative_logprob_at_temperature", float("nan"))) for c in outputs]
    comp_ids_list = [list(c.token_ids) for c in outputs]
    lengths = [len(ids) for ids in comp_ids_list]
    return texts, t1_lps, lps, lengths


def cli() -> None:
    asyncio.run(async_main())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async vLLM sampling + T=1 logprob capture.")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--cache-dir", default="use_mode")
    p.add_argument("--mode", default="fineweb", choices=("fineweb", "fineweb-short", "ambrosia", "ambrosia-short", "ambrosia-selected","ambrosia-top5", "ambrosia-top25", "ambrosia-top100", "ambrosia-top500", "gsm8k"))
    p.add_argument("--csv-path", default="ambrosia.csv", help="Path to CSV for ambrosia mode")
    p.add_argument("--chat-template", action="store_true", help="Apply chat template to prompts")
    p.add_argument(
        "--stop-token-ids",
        type=int,
        nargs="+",
        default=None,
        help="Token IDs at which to stop generation",
    )

    p.add_argument("--sample-concurrency", type=int, default=2, help="Max concurrent SAMPLING jobs")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--pipeline-parallel-size", type=int, default=1)
    p.add_argument(
        "--data-parallel-size-local",
        default="auto",
        help="'auto' (all visible GPUs) or an int >=1. Number of replicas on this node",
    )
    p.add_argument(
        "--distributed-executor-backend",
        default="uni",
        choices=("mp", "ray", "uni", "external_launcher"),
        help="Process-per-GPU backend; 'mp' is great for single-node multi-GPU",
    )
    p.add_argument("--gpu-mem-util", type=float, default=0.94)
    p.add_argument("--ambrosia-stats", action="store_true", help="Show stats for ambrosia prompts")
    p.add_argument("--stats-output-file", default=None, help="Path to save ambrosia stats to")
    p.add_argument("--max-model-len", type=int, default=2300, help="Max model length (context window)")
    p.add_argument("--max-num-seqs", type=int, default=450, help="vLLM max_num_seqs (lower it on 24 GB cards)")
    p.add_argument("--max-num-batched-tokens", type=int, default=4096, help="vLLM max_num_batched_tokens (lower it on 24 GB cards)")
    p.add_argument("--dtype", default="auto", help="Data type for model weights and activations")
    p.add_argument("--seed", type=int, default=None, help="vLLM SamplingParams seed for reproducible pools")
    return p.parse_args()


async def async_main() -> None:
    args = parse_args()
    df = load_ambrosia_df(args.csv_path)

    if args.stats_output_file:
        init_stats_file(args.stats_output_file)

    if args.mode == "fineweb":
        prompt_list = [(p, None) for p in fineweb_prompts]
    elif args.mode == "fineweb-short":
        prompt_list = [(p, None) for p in fineweb_prompts[-6:]]
    elif args.mode.startswith("ambrosia"):
        prompt_list = get_ambrosia_prompts(args.mode, df)
    elif args.mode == "gsm8k":
        gsm_df = load_gsm8k_dataset()
        # Limit to first 50 for experiment
        gsm_df = gsm_df.head(50)
        prompt_list = [(build_math_prompt(row['question']), str(i)) for i, row in gsm_df.iterrows()]
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    cache_dir = f"cache-{args.mode}" if args.cache_dir == "use_mode" else args.cache_dir
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    model_cache_path = cache_root / (args.model.replace("/", "__").replace(":", "_") + ".pkl")
    print(f"{len(prompt_list)} prompts. Cache: {model_cache_path}")
    results = cached_ddict(str(model_cache_path))

    # Check for remaining work before expensive engine init
    work: list[tuple[float, str, int]] = []
    for T in args.temperatures:
        temp_bucket = results[T]
        for i, (prompt_text, name) in enumerate(prompt_list):
            entry = temp_bucket.get(prompt_text)
            have = 0 if entry is None else len(entry[0])
            if (remaining := args.n_samples - have) > 0:
                work.append((T, (prompt_text, name), remaining))
    if not work:
        print("Nothing to do; all prompts/temperatures complete.")
        return

    print("Creating engine...")

    # tp = int(args.tensor_parallel_size)
    # pp = int(args.pipeline_parallel_size)
    dpl = torch.cuda.device_count() if args.data_parallel_size_local == "auto" else args.data_parallel_size_local
    # de = args.distributed_executor_backend

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            # tensor_parallel_size=tp,
            # pipeline_parallel_size=pp,
            # dtype=torch.float16,
            dtype=args.dtype,
            data_parallel_size=dpl,
            data_parallel_size_local=dpl,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            # distributed_executor_backend=de,
            gpu_memory_utilization=args.gpu_mem_util,
            logprobs_mode="raw_processed_logprobs",
        )
    )
    tokenizer = await engine.get_tokenizer()
    print(f"Engine ready. Temperatures: {args.temperatures}")
    print(f"Worklist: {len(work)} jobs across {len(args.temperatures)} temperatures.")

    sample_sem = asyncio.Semaphore(max(1, args.sample_concurrency))
    lock = asyncio.Lock()

    sampling_total = len(work)
    sampling_done = 0
    sampling_active = 0
    total_tokens = 0
    wall_t0 = time.perf_counter()
    per_T_done = {T: 0 for T in args.temperatures}
    per_T_total = {T: sum(1 for t, _, _ in work if t == T) for T in args.temperatures}

    def sampling_params_for(T: float, n: int, seed: int | None) -> SamplingParams:
        return SamplingParams(
            temperature=T,
            max_tokens=args.max_new_tokens,
            n=n,
            logprobs=1,
            output_kind=RequestOutputKind.CUMULATIVE,
            stop_token_ids=args.stop_token_ids,
            seed=seed,
        )

    async def run_one(T: float, prompt: tuple[str, str | None], remaining: int, prompt_idx: int) -> None:
        nonlocal sampling_done, sampling_active, total_tokens
        async with sample_sem:
            sampling_active += 1
            t0 = time.perf_counter()
            prompt_text, prompt_name = prompt
            pid = prompt_name if prompt_name is not None else prompt_idx
            outputs = []
            if args.chat_template:
                messages = [{"role": "user", "content": prompt_text}]
                tokens = tokenizer.apply_chat_template(messages, enable_thinking=False, add_generation_prompt=True)
                inputs = TokensPrompt(prompt_token_ids=tokens)
            else:
                inputs = prompt_text

            req_seed = derive_request_seed(
                args.seed, prompt_text, T, offset=args.n_samples - remaining
            )
            async for req in engine.generate(
                inputs,
                sampling_params_for(T, remaining, req_seed),
                request_id=f"s-{uuid4()}",
            ):
                for out in req.outputs:
                    if out.finished():
                        outputs.append(out)

            elapsed = time.perf_counter() - t0
            sampling_active -= 1

        texts, t1_lps, lps, lengths = extract_results(outputs)

        temp_bucket = results[T]
        entry = temp_bucket.get(prompt_text)
        if entry is None:
            raw_sents, raw_lps, raw_t1_lps, raw_lengths = [], [], [], []
        else:
            raw_sents, raw_lps, raw_t1_lps, raw_lengths = map(list, entry)

        raw_sents.extend(texts)
        raw_lps.extend(lps)
        raw_t1_lps.extend(t1_lps)
        raw_lengths.extend(lengths)

        temp_bucket[prompt_text] = (raw_sents, raw_lps, raw_t1_lps, raw_lengths)
        if hasattr(results, "_save"):
            results._save()

        async with lock:
            sampling_done += 1
            total_tokens += sum(lengths)
            per_T_done[T] += 1
            t_done = per_T_done[T]
            t_total = per_T_total[T]
            toks = total_tokens / (time.perf_counter() - wall_t0)
            if t_done == t_total:
                # Permanent line for completed temperature
                print(f"\rT={T}: {t_total}/{t_total} done  ({toks:.0f} tok/s)                        ")
            else:
                t_str = "  ".join(
                    f"T={t}:{per_T_done[t]}/{per_T_total[t]}" for t in sorted(per_T_done) if per_T_total[t] > 0
                )
                print(f"\r[{sampling_done}/{sampling_total} | {sampling_active} active]  {t_str}  ({toks:.0f} tok/s)   ", end="", flush=True)

    print(f"Scheduling {len(work)} jobs...")
    tasks = [asyncio.create_task(run_one(T, p, rem, prompt_list.index(p))) for (T, p, rem) in work]
    await asyncio.gather(*tasks)
    print(f"[all done] {sampling_done}/{sampling_total}")

    if args.ambrosia_stats:
        compute_ambrosia_stats(
            results,
            args.temperatures,
            prompt_list,
            df,
            args.model,
            args.stats_output_file
        )


if __name__ == "__main__":
    asyncio.run(async_main())
