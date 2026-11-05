"""
Instructed-kernel embeddings for GSM8K (tab:gsm8k-kernels, tab:significance)
============================================================================
Qwen3-Embedding-0.6B with a task-instruction prefix. Writes per pool
google__gemma-2-9b-it_qwenins_embeddings.pkl with the same
{sT: {prompt: {u_sents, counts, logps, logps_t1, embeddings}}} structure as the
plain qwen file (all fields copied, embeddings recomputed).

Usage (one shard per GPU):
  CUDA_VISIBLE_DEVICES=0 python analysis/embed_instructed.py --pool-start 0 --pool-end 20
"""

import argparse
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from embedder_probe import GSM_INSTR, with_instruct  # noqa: E402

BASE = "google__gemma-2-9b-it_qwen_embeddings.pkl"
DEST = "google__gemma-2-9b-it_qwenins_embeddings.pkl"
TEMPS = [0.5, 0.8, 1.0, 1.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-start", type=int, required=True)
    ap.add_argument("--pool-end", type=int, required=True)
    ap.add_argument("--cache-root", default="cache-gsm8k", help="pool cache root relative to the repo")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda",
                            model_kwargs={"torch_dtype": torch.float16})

    for pi in range(a.pool_start, a.pool_end):
        pdir = os.path.join(ROOT_DIR, a.cache_root, f"pool{pi}")
        dest = os.path.join(pdir, DEST)
        out = pickle.load(open(dest, "rb")) if os.path.exists(dest) else {}
        emb = pickle.load(open(os.path.join(pdir, BASE), "rb"))
        changed = False
        for sT in TEMPS:
            if sT in out or sT not in emb:
                continue
            out[sT] = {}
            for pt, d in emb[sT].items():
                texts = [with_instruct(GSM_INSTR, s) for s in d["u_sents"]]
                V = m.encode(texts, batch_size=a.batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
                out[sT][pt] = {
                    "u_sents": d["u_sents"], "counts": d["counts"],
                    "logps": d["logps"], "logps_t1": d["logps_t1"],
                    "embeddings": np.asarray(V, dtype=np.float32),
                }
            changed = True
            print(f"pool{pi} sT={sT} done", flush=True)
        if changed:
            with open(dest, "wb") as f:
                pickle.dump(out, f)
        print(f"pool{pi} saved", flush=True)


if __name__ == "__main__":
    main()
