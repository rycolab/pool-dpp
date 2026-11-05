def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Precompute embeddings for samples", prog="dpp-embeddings")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2", help="Model name")
    parser.add_argument("--samples_file", default="cache/gpt2-xl.pkl", help="Results pkl file directory")
    parser.add_argument("--output_file", default=None, help="Embeddings pkl file")

    args = parser.parse_args()
    precompute_embeddings(args.model_name, args.samples_file, args.output_file)

def precompute_embeddings(model_name, samples_file, output_file):
    import os, numpy as np
    from lm_dpp.utils.cached import cached_ddict
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Transformer, Pooling, Normalize

    results = cached_ddict(samples_file, read_only=True)
    if "Qwen" in model_name:
        embedding_model = SentenceTransformer(model_name,
                                              model_kwargs={#"attn_implementation": "flash_attention_2",
                                                            "device_map": "auto", "dtype": "auto"},
                                              tokenizer_kwargs={"padding_side": "left"}
        )
        # batch_size=1 was the original default (legacy memory caution).
        # On 96GB Blackwell, Qwen3-Embedding-0.6B easily handles batch_size=32
        # for the typical reasoning-chain length (200-1000 tokens).
        batch_size = int(os.environ.get("EMB_BATCH_SIZE", 32))
    else:
        emb_model = Transformer(model_name)
        pooling = Pooling(emb_model.get_word_embedding_dimension(), pooling_mode="mean")
        embedding_model = SentenceTransformer(modules=[emb_model, pooling, Normalize()])
        batch_size = 32

    output_file = samples_file.replace(".pkl", "_embeddings.pkl") if output_file is None else output_file
    embeddings_dict = cached_ddict(output_file)
    embeddings_dict.cow = False

    all_Ts = list(results.keys())
    for sampling_T in all_Ts:
        prompts_items = list(results[sampling_T].items())
        todo = [(p, v) for p, v in prompts_items if p not in embeddings_dict[sampling_T]]
        if not todo:
            continue
        n_todo = len(todo)
        prefix = f"T={sampling_T}({n_todo}/{len(prompts_items)})"
        for i, (prompt, (raw_sents, raw_lps, raw_lps_t1, raw_lengths)) in enumerate(todo):
            prompt_ids = None
            if isinstance(raw_sents[0], tuple):
                prompt_ids = [s[1] for s in raw_sents]
                raw_sents = [s[0] for s in raw_sents]
            u_sents, idx, counts = np.unique(raw_sents, return_counts=True, return_index=True)
            ii = np.argsort(idx)
            u_sents, idx, counts, logps, logps_t1 = (
                u_sents[ii], idx[ii], counts[ii],
                np.array(raw_lps)[idx[ii]], np.array(raw_lps_t1)[idx[ii]],
            )
            u_prompt_ids = np.array(prompt_ids)[idx[ii]] if prompt_ids is not None else None
            print(f"\r{prefix}: {i+1}/{n_todo}", end="", flush=True)
            embeddings = embedding_model.encode(u_sents, batch_size=batch_size, show_progress_bar=False)
            embeddings_dict[sampling_T][prompt] = {
                "embeddings": embeddings, "u_sents": u_sents, "counts": counts,
                "logps": logps, "logps_t1": logps_t1,
            }
            if u_prompt_ids is not None:
                embeddings_dict[sampling_T][prompt]["prompt_ids"] = u_prompt_ids
        print(f"\r{prefix}: done.      ", flush=True)
        embeddings_dict._save()
