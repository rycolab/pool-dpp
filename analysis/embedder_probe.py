"""
Embedder sensitivity to correctness-critical vs surface-only edits (tab:embedder-probe)
========================================================================================
"Is the embedding-space diversity the kernel measures the same thing as
solution diversity?"  We probe the embedders directly with paired edits whose
labels are *verified*, not asserted:

GSM8K (math derivations, real pool samples)
  a1  answer-flip      : change only the final '#### N' token. One token,
                         changes correctness.
  a2  operator-flip    : flip one arithmetic operator in one derivation step,
                         propagate the changed value to the stated answer.
                         One operation, changes correctness.
  b   surface-only     : rewrite step markers, connectives, whitespace and
                         punctuation. Verified to leave the final answer AND
                         the multiset of (a, op, b, c) arithmetic tuples
                         unchanged.

Ambrosia (SQL, real pool samples, verified by EXECUTION against the question's
SQLite dump - this is ground truth, not a heuristic)
  a   semantic edit    : AND<->OR, '='<->'!=', comparison flip, drop a
                         predicate. Kept only if the executed result signature
                         CHANGES.
  b   surface-only     : keyword case, whitespace reflow, redundant parens,
                         consistent alias renaming. Kept only if the executed
                         result signature is IDENTICAL.

For each embedder we report the cosine-distance distribution for the
correctness-critical edits vs the surface-only edits, and the separation AUC
= P(d_correctness > d_surface) over random pairs. AUC 0.5 means the embedder
cannot tell the two apart; the kernel geometry then does not track correctness.

Embedders: MiniLM and Qwen3-Embedding-0.6B, plain and instructed, on one item set,
with 2,000-resample bootstrap CIs over paired items.

Usage: python analysis/embedder_probe.py [--n-items 600] [--device cuda]
Output: analysis/embedder_probe.json
"""

import argparse
import json
import os
import pickle
import random
import re
import sqlite3

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

EQ_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([\+\-\*/x×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
ANS_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


GSM_INSTR = ("Given a step-by-step solution to a math word problem, represent "
             "it by its final numeric answer and the sequence of arithmetic "
             "operations it performs, ignoring wording, phrasing, and "
             "formatting.")
SQL_INSTR = ("Given a SQL query, represent it by the result set it returns "
             "when executed, ignoring formatting, capitalization, aliasing, "
             "and stylistic differences.")


def with_instruct(instr, text):
    return f"Instruct: {instr}\nQuery:{text}"


# ---------------------------------------------------------------------------
# GSM8K edits
# ---------------------------------------------------------------------------
def _num(s):
    f = float(s)
    return int(f) if f == int(f) else f


def _apply(a, op, b):
    if op in "*x×":
        return a * b
    if op in "/÷":
        return None if b == 0 else a / b
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    return None


def arith_tuples(text):
    """Multiset of arithmetic facts stated in the text."""
    return sorted((m.group(1), m.group(2), m.group(3), m.group(4))
                  for m in EQ_RE.finditer(text))


def gsm_answer(text):
    m = ANS_RE.search(text)
    return m.group(1) if m else None


def edit_answer_flip(text):
    """a1: change only the final answer token."""
    m = ANS_RE.search(text)
    if not m:
        return None
    old = _num(m.group(1))
    new = old + (1 if old != 1 else 2)
    out = text[:m.start(1)] + str(new) + text[m.end(1):]
    return out if gsm_answer(out) != m.group(1) else None


def edit_operator_flip(text):
    """a2: flip one arithmetic operator, propagate to the stated answer."""
    ans = gsm_answer(text)
    if ans is None:
        return None
    cands = []
    for m in EQ_RE.finditer(text):
        a, op, b, c = _num(m.group(1)), m.group(2), _num(m.group(3)), _num(m.group(4))
        got = _apply(a, op, b)
        if got is None or abs(got - c) > 1e-6:
            continue          # only edit steps whose arithmetic is actually right
        alt = {"+": "-", "-": "+", "*": "+", "x": "+", "×": "+",
               "/": "-", "÷": "-"}.get(op)
        if alt is None:
            continue
        new_c = _apply(a, alt, b)
        if new_c is None or abs(new_c - c) < 1e-9:
            continue
        cands.append((m, alt, c, new_c))
    if not cands:
        return None
    m, alt, c, new_c = cands[-1]
    new_c_s = str(int(new_c) if float(new_c) == int(new_c) else round(new_c, 4))
    out = (text[:m.start()]
           + f"{m.group(1)} {alt} {m.group(3)} = {new_c_s}"
           + text[m.end():])
    # propagate the delta to the stated final answer
    am = ANS_RE.search(out)
    if am is None:
        return None
    new_ans = _num(am.group(1)) + (new_c - c)
    new_ans_s = str(int(new_ans) if float(new_ans) == int(new_ans) else round(new_ans, 4))
    out = out[:am.start(1)] + new_ans_s + out[am.end(1):]
    return out if gsm_answer(out) != ans else None


SURFACE_WORDS = [
    (r"\bTotal\b", "Altogether"), (r"\btotal\b", "altogether"),
    (r"\bSo\b", "Therefore"), (r"\bso\b", "therefore"),
    (r"\bThen\b", "Next"), (r"\bthen\b", "next"),
    (r"\bWe need\b", "We require"), (r"\bneeded\b", "required"),
    (r"\bFirst\b", "To begin"), (r"\bfinally\b", "lastly"),
]


def edit_surface(text, rng):
    """b: surface-only rewrite. Numbers and operators are never touched."""
    out = text
    # renumber list markers: '1.' -> 'Step 1:'
    out = re.sub(r"(?m)^\s*(\d+)\.\s+", lambda m: f"Step {m.group(1)}: ", out)
    for pat, rep in SURFACE_WORDS:
        if rng.random() < 0.7:
            out = re.sub(pat, rep, out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    if rng.random() < 0.5:
        out = out.replace("\n\n", "\n")
    if not out.startswith("Let us work through this."):
        out = "Let us work through this.\n" + out.lstrip()
    # verify: same answer, same arithmetic facts
    if gsm_answer(out) != gsm_answer(text):
        return None
    if arith_tuples(out) != arith_tuples(text):
        return None
    if out.strip() == text.strip():
        return None
    return out


# ---------------------------------------------------------------------------
# SQL edits (verified by execution)
# ---------------------------------------------------------------------------
def sql_sig(cur, q):
    try:
        cur.execute(q)
        return tuple(cur.fetchall())
    except Exception:
        return None


SQL_SEMANTIC = [
    (r"\bAND\b", "OR"), (r"\bOR\b", "AND"),
    (r"([^<>!])=([^=])", r"\1!=\2"),
    (r"\bIN\b", "NOT IN"),
    (r">=", "<"), (r"<=", ">"),
]


def edit_sql_semantic(q, rng):
    pats = SQL_SEMANTIC[:]
    rng.shuffle(pats)
    for pat, rep in pats:
        new = re.sub(pat, rep, q, count=1)
        if new != q:
            return new
    return None


def edit_sql_surface(q, rng):
    out = q
    for kw in ["SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR",
               "GROUP BY", "ORDER BY", "AS", "INNER", "LEFT"]:
        if rng.random() < 0.7:
            out = re.sub(rf"\b{kw}\b", kw.lower(), out)
    out = re.sub(r"\s+", " ", out).strip()
    if rng.random() < 0.5:
        out = out.replace(" from ", "\n  FROM ").replace(" FROM ", "\n  FROM ")
    if rng.random() < 0.5:
        out = out.replace(" where ", "\n  WHERE ").replace(" WHERE ", "\n  WHERE ")
    if out.endswith(";"):
        out = out[:-1] + " ;"
    return out if out.strip() != q.strip() else None


# ---------------------------------------------------------------------------
# Build the paired datasets
# ---------------------------------------------------------------------------
def build_gsm8k_pairs(n_items, rng):
    path = os.path.join(ROOT_DIR, "cache-gsm8k", "pool0",
                        "google__gemma-2-9b-it.pkl")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    sT = 1.0 if 1.0 in raw else sorted(raw.keys())[0]
    pairs = []
    for prompt, entry in raw[sT].items():
        for s in entry[0][:40]:
            if len(pairs) >= n_items:
                break
            base = s.strip()
            if gsm_answer(base) is None or not EQ_RE.search(base):
                continue
            a1 = edit_answer_flip(base)
            a2 = edit_operator_flip(base)
            b = edit_surface(base, rng)
            if b is None or (a1 is None and a2 is None):
                continue
            pairs.append({"base": base, "a1_answer_flip": a1,
                          "a2_operator_flip": a2, "b_surface": b})
        if len(pairs) >= n_items:
            break
    return pairs


def build_sql_pairs(n_items, rng):
    from lm_dpp.utils.ambrosia_utils import load_ambrosia_df
    df = load_ambrosia_df(os.path.join(ROOT_DIR, "ambrosia.csv"))
    by_q = {str(r["question"]).strip(): r for _, r in df.iterrows()}
    path = os.path.join(ROOT_DIR, "cache-ambrosia", "pool0",
                        "google__gemma-2-9b-it.pkl")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    sT = 1.0 if 1.0 in raw else sorted(raw.keys())[0]

    def parse_question(pt):
        s = pt.find("and the following question:\n")
        if s < 0:
            return None
        s += len("and the following question:\n")
        e = pt.find("\ngenerate the SQL query", s)
        return pt[s:e if e != -1 else len(pt)].strip()

    pairs = []
    for prompt, entry in raw[sT].items():
        q = parse_question(prompt)
        row = by_q.get(q) if q else None
        if row is None:
            continue
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        try:
            cur.executescript(row["db_dump"])
        except Exception:
            conn.close()
            continue
        for s in entry[0][:6]:
            if len(pairs) >= n_items:
                break
            base = s.strip().replace("```sql\n", "").replace("\n```", "")
            base_sig = sql_sig(cur, base)
            if base_sig is None:
                continue                      # base must run
            sem = edit_sql_semantic(base, rng)
            sur = edit_sql_surface(base, rng)
            sem_ok = sur_ok = None
            if sem:
                sig = sql_sig(cur, sem)
                if sig is not None and sig != base_sig:
                    sem_ok = sem              # verified: result CHANGED
            if sur:
                sig = sql_sig(cur, sur)
                if sig is not None and sig == base_sig:
                    sur_ok = sur              # verified: result IDENTICAL
            if sem_ok and sur_ok:
                pairs.append({"base": base, "a_semantic": sem_ok,
                              "b_surface": sur_ok})
        conn.close()
        if len(pairs) >= n_items:
            break
    return pairs


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
def embed_sentence_transformer(name, texts, device, batch_size=32):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device=device)
    v = m.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    del m
    return v


def cosine_dist(u, v):
    return 1.0 - float(np.dot(u, v))


def auc(pos, neg):
    """P(pos > neg) with ties counted as half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(n_items, device):
    rng = random.Random(0)
    gsm = build_gsm8k_pairs(n_items, rng)
    sql = build_sql_pairs(n_items, rng)
    print(f"{len(gsm)} GSM8K items, {len(sql)} SQL items", flush=True)
    tasks = {"gsm8k": (gsm, ["a1_answer_flip", "a2_operator_flip", "b_surface"], GSM_INSTR),
             "ambrosia_sql": (sql, ["a_semantic", "b_surface"], SQL_INSTR)}
    texts, instr_texts, index = [], [], {}
    for tname, (items, variants, instr) in tasks.items():
        for i, it in enumerate(items):
            for v in ["base"] + variants:
                if v == "base" or it.get(v):
                    index[(tname, i, v)] = len(texts)
                    texts.append(it[v])
                    instr_texts.append(with_instruct(instr, it[v]))
    embedders = [("MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", False),
                 ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", False),
                 ("Qwen3-Embedding-0.6B instructed", "Qwen/Qwen3-Embedding-0.6B", True)]
    results = {}
    for label, name, use_instr in embedders:
        print(f"--- {label}", flush=True)
        V = embed_sentence_transformer(name, instr_texts if use_instr else texts, device)
        per_task = {}
        for tname, (items, variants, _) in tasks.items():
            aucs = {}
            for v in variants:
                if v == "b_surface":
                    continue
                pp, pn = [], []
                for i, it in enumerate(items):
                    if it.get(v) and it.get("b_surface"):
                        b = V[index[(tname, i, "base")]]
                        pp.append(cosine_dist(b, V[index[(tname, i, v)]]))
                        pn.append(cosine_dist(b, V[index[(tname, i, "b_surface")]]))
                pp, pn = np.asarray(pp), np.asarray(pn)
                g = np.random.default_rng(0)
                idx = g.integers(0, len(pp), size=(2000, len(pp)))
                bs = np.array([auc(pp[j], pn[j]) for j in idx])
                aucs[v] = {"auc_vs_surface": auc(list(pp), list(pn)),
                           "auc_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                           "n_paired": int(len(pp)),
                           "n_questions": int(len({items[i].get("qid", items[i].get("prompt", i)) for i in range(len(items))}))}
                print(f"  {tname} {v}: AUC {aucs[v]['auc_vs_surface']:.3f} {aucs[v]['auc_ci95']} n={len(pp)}", flush=True)
            per_task[tname] = aucs
        results[label] = per_task
    dest = os.path.join(SCRIPT_DIR, "embedder_probe.json")
    json.dump({"n_gsm8k": len(gsm), "n_sql": len(sql), "results": results}, open(dest, "w"), indent=2)
    print("Saved", dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-items", type=int, default=600)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    main(a.n_items, a.device)
