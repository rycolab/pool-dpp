"""
DivBS and string-subsequence-kernel DetBS on FineWeb (tab:beam_baselines)
=========================================================================
Scores the decoding runs of baselines/detbs_ssk (run_intrinsic_baselines.py) on the 50 FineWeb
prompts: Gemma-2-9B-Instruct with the chat template, K=10, max_new_tokens=50, MiniLM-L6 embeddings.
Each pickle is a list of (prompt, sents, logps, embeddings) tuples, one per prompt.

Per prompt: quality = mean over the K strings of log p(y | x) at T=1; semantic diversity =
log det(F F^T + 1e-6 I) with F the K unit-norm embeddings; token diversity = Distinct-2, unique
whitespace bigrams over all bigrams of the K strings. Prompts with fewer than 2 strings or a
non-positive determinant sign are counted in n_skipped. Mean and standard deviation over prompts.

Rows: DivBS (G=10 groups of one beam, penalty 1.0), DetBS-SSK at w=1.0 and w=0.1 (T=1.0, kernel
n=2, decay 0.5), and the DetBS sweep over T in {0.7, 1.0, 1.3} and w on a subset of the prompts
(limit=8 in sweep_detbs.sh).

Usage:  python analysis/beam_baselines_intrinsic.py [--dir cache-fineweb-ssk]
Output: analysis/beam_baselines_intrinsic.json
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ROWS = {
    "DivBS|G=10|penalty=1.0": "dbs_k10_g10_p1.0",
    "DivBS|G=3|penalty=1.0|K=6": "dbs_k6_g3_p1.0",
    "DetBS-SSK|T=1.0|w=1.0": "detbs_k10_w1.0",
    "DetBS-SSK|T=1.0|w=0.1": "detbs_k10_w0.1_shard",
}
SWEEP = [(1.0, 0.05), (1.0, 0.3), (1.0, 0.5), (1.0, 2.0), (1.3, 0.1), (1.3, 0.3), (1.3, 1.0),
         (0.7, 0.3), (0.7, 1.0)]
for T, w in SWEEP:
    ROWS[f"DetBS-SSK|T={T}|w={w}|subset"] = f"sweep_detbs_T{T}_w{w}"


def distinct_n(sents, n=2):
    total, uniq = 0, set()
    for s in sents:
        toks = s.split()
        for i in range(len(toks) - n + 1):
            uniq.add(tuple(toks[i:i + n]))
            total += 1
    return len(uniq) / total if total else 0.0


def score(paths):
    lp, ld, d2, skipped, seen = [], [], [], 0, set()
    for p in paths:
        with open(p, "rb") as f:
            rows = pickle.load(f)
        for prompt, sents, lps, emb in rows:
            if prompt in seen:
                continue
            seen.add(prompt)
            F = np.asarray(emb)
            ok = bool(lps) and len(sents) >= 2 and F.ndim == 2 and F.shape[0] == len(sents)
            if ok:
                sign, logdet = np.linalg.slogdet(F @ F.T + 1e-6 * np.eye(len(sents)))
                ok = sign > 0
            if not ok:
                skipped += 1
                continue
            lp.append(float(np.mean(lps)))
            ld.append(float(logdet))
            d2.append(distinct_n(sents))
    f = lambda x: (float(np.mean(x)), float(np.std(x)))
    (mq, sq), (ml, sl), (md, sd) = f(lp), f(ld), f(d2)
    return {"quality_mean_logp": mq, "quality_sd": sq, "kernel_logdet_minilm": ml, "logdet_sd": sl,
            "distinct_2": md, "distinct_2_sd": sd, "n_prompts": len(lp), "n_skipped": skipped,
            "n_files": len(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="cache-fineweb-ssk")
    args = ap.parse_args()
    out = {}
    for key, stem in ROWS.items():
        paths = sorted(glob.glob(os.path.join(args.dir, stem + "*.pkl")))
        if not paths:
            raise FileNotFoundError(f"{key}: no {stem}*.pkl in {args.dir}")
        out[key] = score(paths)
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[key].items()})
    with open(os.path.join(SCRIPT_DIR, "beam_baselines_intrinsic.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
