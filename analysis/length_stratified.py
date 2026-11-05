"""
Does log p carry quality signal beyond length? (tab:length)
=====================================================
length_analysis.py shows a strong negative correlation
between sequence length and log p, strongest at high sampling temperature.
Hypothesis to test: the correlation is driven by a tail of very long, low
quality generations; within the short generations log p still ranks quality.

Per prompt, items are the unique pool generations with their T=1 sequence
log probabilities. Quality labels: GSM8K = final answer correct; Ambrosia =
executes successfully (runnable) and matches a gold interpretation (golden),
both via the execution table of pool_ceiling.py.

Reported, averaged over prompts:
  - quality by within-prompt length quintile (Q1 = shortest): is the long
    tail disproportionately wrong or non-executable?
  - AUC(log p ranks correct above incorrect), on the full pool and within the
    shortest half of each prompt's pool: does the quality signal survive when
    length variation is removed?
  - Spearman(log p, length), full vs shortest half: how much of the length
    coupling is the tail?

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/length_stratified.py
"""

import json
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
N_POOLS = 10


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def analyze(items_iter, label_name):
    """items_iter yields (lengths, logps, labels) per prompt."""
    quint_q = defaultdict(list)          # quality by length quintile
    quint_len = defaultdict(list)
    auc_full, auc_short, sp_full, sp_short = [], [], [], []
    for L, P, Y in items_iter:
        L, P, Y = np.asarray(L, float), np.asarray(P, float), np.asarray(Y, float)
        if len(L) < 10:
            continue
        # length quintiles within the prompt
        qs = np.quantile(L, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(L, qs)
        for b in range(5):
            m = bins == b
            if m.sum():
                quint_q[b].append(Y[m].mean())
                quint_len[b].append(L[m].mean())
        # AUC full
        a = auc(P[Y == 1], P[Y == 0])
        if a is not None:
            auc_full.append(a)
        # shortest half
        m = L <= np.median(L)
        if m.sum() >= 5:
            a = auc(P[m & (Y == 1)], P[m & (Y == 0)])
            if a is not None:
                auc_short.append(a)
            if len(np.unique(L[m])) > 2:
                sp_short.append(spearmanr(P[m], L[m])[0])
        if len(np.unique(L)) > 2:
            sp_full.append(spearmanr(P, L)[0])

    out = {
        f"{label_name}_by_length_quintile":
            [float(np.mean(quint_q[b])) for b in range(5)],
        "mean_chars_by_quintile":
            [float(np.mean(quint_len[b])) for b in range(5)],
        "auc_logp_full_pool": float(np.mean(auc_full)) if auc_full else None,
        "auc_logp_shortest_half": float(np.mean(auc_short)) if auc_short else None,
        "n_prompts_auc_full": len(auc_full),
        "n_prompts_auc_short": len(auc_short),
        "spearman_logp_length_full": float(np.nanmean(sp_full)) if sp_full else None,
        "spearman_logp_length_shortest_half":
            float(np.nanmean(sp_short)) if sp_short else None,
    }
    return out


def gsm8k_items(sT=1.2):
    from lm_dpp.utils.math_utils import build_math_prompt, is_correct, \
        load_gsm8k_dataset
    df = load_gsm8k_dataset().head(50)
    gold_of = {build_math_prompt(r["question"]): r["answer"]
               for _, r in df.iterrows()}
    for pi in range(N_POOLS):
        p = os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{pi}",
                         "google__gemma-2-9b-it_qwen_embeddings.pkl")
        if not os.path.exists(p):
            continue
        emb = pickle.load(open(p, "rb"))
        if sT not in emb:
            continue
        for pt, d in emb[sT].items():
            gold = gold_of.get(pt)
            if gold is None:
                continue
            u, lp = list(d["u_sents"]), np.asarray(d["logps_t1"], float)
            if len(u) != len(lp):
                continue
            yield ([len(s) for s in u], lp,
                   [1.0 if is_correct(s, gold) else 0.0 for s in u])


def ambrosia_items(sT, label):
    """label in {'golden', 'runnable'}."""
    with open(os.path.join(SCRIPT_DIR, "pool_ceiling_sigtable.pkl"), "rb") as f:
        d = pickle.load(f)
    golden, tables = d["golden"], d["sig_tables"]

    def parse_q(pt):
        s = pt.find("and the following question:\n")
        if s < 0:
            return None
        s += len("and the following question:\n")
        e = pt.find("\ngenerate the SQL query", s)
        return pt[s:e if e != -1 else len(pt)].strip()

    def norm(x):
        return x.strip().replace("```sql\n", "").replace("\n```", "")

    for pi in range(N_POOLS):
        p = os.path.join(ROOT_DIR, "cache-ambrosia", f"pool{pi}",
                         "google__gemma-2-9b-it_minilm_embeddings.pkl")
        if not os.path.exists(p):
            continue
        emb = pickle.load(open(p, "rb"))
        if sT not in emb:
            continue
        for pt, d2 in emb[sT].items():
            q = parse_q(pt)
            if q is None or q not in golden:
                continue
            gs, tbl = golden[q], tables[q]
            u, lp = list(d2["u_sents"]), np.asarray(d2["logps_t1"], float)
            if len(u) != len(lp):
                continue
            sigs = [tbl.get(norm(s)) for s in u]
            if label == "golden":
                y = [1.0 if s in gs else 0.0 for s in sigs]
            else:
                y = [1.0 if s is not None else 0.0 for s in sigs]
            yield ([len(s) for s in u], lp, y)


def main():
    out = {}
    print("=== GSM8K sT=1.2 (label: correct) ===", flush=True)
    out["gsm8k_sT1.2_correct"] = analyze(gsm8k_items(1.2), "correct")
    for k, v in out["gsm8k_sT1.2_correct"].items():
        print(f"  {k}: {v}")

    for sT in (1.0, 1.5):
        for label in ("runnable", "golden"):
            key = f"ambrosia_sT{sT}_{label}"
            print(f"=== Ambrosia sT={sT} (label: {label}) ===", flush=True)
            out[key] = analyze(ambrosia_items(sT, label), label)
            for k, v in out[key].items():
                print(f"  {k}: {v}")

    dest = os.path.join(SCRIPT_DIR, "length_stratified.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
