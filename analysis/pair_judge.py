"""
Pair decomposition of selected GSM8K sets (tab:gsm8k-pairs)
===========================================================
Splits each selected set into its pairs and labels every pair in which both chains state a final answer
in the '####' format. A parser decides 'different answer'. Gemini 3.6 Flash labels each same-answer pair
as 'same answer, different derivation' or 'surface only'.

  validate  compares the judge with the deterministic labels on a sample stratified by label. The
            deterministic rule calls a same-answer pair 'surface only' when both chains state the same
            multiset of arithmetic facts (a, op, b, c).
  score     writes the decomposition. A judge reply that contradicts the parser, or no reply after
            retries, falls back to the deterministic rule and is counted under 'anomalies'.

Reads selections from run_gsm8k.sh and run_gsm8k_instructed.sh. Sends chain text to the Gemini API, so it
needs GEMINI_API_KEY. Verdicts are kept in a file, so a rerun only pays for new pairs.

Usage:
  GEMINI_API_KEY=... python analysis/pair_judge.py validate
  GEMINI_API_KEY=... python analysis/pair_judge.py score [--selection LABEL PATH ...]
Output: analysis/pair_judge_validation.json, analysis/pair_judge.json
"""

import argparse
import hashlib
import http.client
import json
import os
import pickle
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

JUDGE = "gemini-3.6-flash"
LM = "google__gemma-2-9b-it"
SCORE_SELECTIONS = [
    ("Ancestral", f"{LM}_qwenins-ancestral/s1.pkl"),
    ("k-Means", f"{LM}_qwenins-kmeans/s1.pkl"),
    ("PoolDPP-Samp T=1.05", f"{LM}_qwenins-samp/T1.05_s1.2.pkl"),
    ("PoolDPP-MAP T=500", f"{LM}_qwenins-map/T500_s1.2.pkl"),
]
VALIDATION_SELECTIONS = [
    f"{LM}_qwen-ancestral/s1.2.pkl",
    f"{LM}_qwen-kmeans/s1.2.pkl",
    f"{LM}_qwen-map/T500_s1.2.pkl",
    f"{LM}_qwen-map/T10_s1.2.pkl",
]

NUM = r"-?\d[\d,]*(?:\.\d+)?"
EQ_RE = re.compile(rf"({NUM})\s*([\+\-\*/x×÷])\s*({NUM})\s*=\s*({NUM})")
ANS_RE = re.compile(rf"####\s*\$?\s*({NUM})")

LABELS = ["different_answer", "same_answer_diff_derivation", "surface_only"]
LETTER = dict(zip("ABC", LABELS))

PROMPT = """Two chain-of-thought solutions to the same grade-school math problem are shown.

Classify their relationship as exactly one of:
A = different_answer: they state different final numeric answers.
B = same_answer_diff_derivation: same final answer, but genuinely different reasoning (different intermediate quantities, or a different solution strategy).
C = surface_only: same final answer AND the same underlying arithmetic; they differ only in wording, formatting, or phrasing.

SOLUTION 1:
{x}

SOLUTION 2:
{y}

Answer with exactly one letter (A, B, or C) and nothing else."""


def canon_num(s):
    """'243.00' == '243' and '72,500' == '72500'."""
    t = s.replace(",", "").rstrip(".")
    try:
        f = float(t)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return t


def answer_of(text):
    m = ANS_RE.search(text)
    return canon_num(m.group(1)) if m else None


def derivation_of(text):
    return tuple(sorted((canon_num(m.group(1)),
                         m.group(2).replace("x", "*").replace("×", "*").replace("÷", "/"),
                         canon_num(m.group(3)), canon_num(m.group(4)))
                        for m in EQ_RE.finditer(text)))


def classify_pair(x, y):
    """The deterministic label, or None if a chain has no '####' answer."""
    ax, ay = answer_of(x), answer_of(y)
    if ax is None or ay is None:
        return None
    if ax != ay:
        return "different_answer"
    return "surface_only" if derivation_of(x) == derivation_of(y) else "same_answer_diff_derivation"


def pair_key(x, y):
    return hashlib.md5((x + "\x00" + y).encode()).hexdigest()


def judge_pairs(pairs, model, workers=16, tries=4):
    """A label from LETTER for each (x, y), or None if no parseable reply came back. Stops on an HTTP
    error that a retry cannot fix, such as a rejected key."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(f"{len(pairs)} pairs have no verdict and GEMINI_API_KEY is not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}

    def one(pair):
        body = json.dumps({
            "contents": [{"parts": [{"text": PROMPT.format(x=pair[0], y=pair[1])}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
        }).encode()
        for attempt in range(tries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as r:
                    reply = json.loads(r.read())
                parts = reply["candidates"][0]["content"].get("parts", [])
                text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip().upper()
                # the reply should be one letter; take the last so a preamble cannot win
                letters = re.findall(r"\b([ABC])\b", text) or re.findall(r"[ABC]", text)
                if letters:
                    return LETTER[letters[-1]]
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:
                    raise SystemExit(f"Gemini API error {e.code}: {e.reason}")
                time.sleep(2 ** attempt)
            except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError):
                time.sleep(2 ** attempt)
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, pairs))


def load_sets(path):
    with open(path, "rb") as f:
        return [item[1] for item in pickle.load(f)]


def cmd_validate(a):
    buckets = defaultdict(list)
    for pool in range(a.n_pools):
        for sub in VALIDATION_SELECTIONS:
            path = os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{pool}", "est10qwen", "0", sub)
            for sents in load_sets(path):
                for x, y in combinations(sents, 2):
                    label = classify_pair(x, y)
                    if label and len(x) < 2500 and len(y) < 2500:
                        buckets[label].append((x, y))
    print({k: len(v) for k, v in buckets.items()}, flush=True)

    rng = np.random.default_rng(0)
    sample = []
    for label in LABELS:
        v = buckets[label]
        for i in rng.choice(len(v), size=min(a.per_class, len(v)), replace=False):
            sample.append((label, *v[i]))
    verdicts = judge_pairs([(x, y) for _, x, y in sample], a.model)

    confusion = Counter()
    agree, n_class = Counter(), Counter()
    for (label, _, _), got in zip(sample, verdicts):
        confusion[f"{label}->{got}"] += 1
        if got is not None:
            n_class[label] += 1
            agree[label] += got == label
    out = {"judge_model": a.model, "per_class_n": dict(n_class),
           "per_class_agreement": {k: agree[k] / n_class[k] for k in n_class},
           "confusion": dict(confusion)}
    print(json.dumps(out, indent=2))
    with open(a.out or os.path.join(SCRIPT_DIR, "pair_judge_validation.json"), "w") as f:
        json.dump(out, f, indent=2)


def cmd_score(a):
    root = os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{a.pool}", "est10qwenins", str(a.run))
    selections = a.selection or [(label, os.path.join(root, sub)) for label, sub in SCORE_SELECTIONS]
    pairs_by_method = {}
    for label, path in selections:
        pairs = []
        for sents in load_sets(path):
            for x, y in combinations(sents, 2):
                det = classify_pair(x, y)
                if det is not None:
                    pairs.append((det, x, y))
        pairs_by_method[label] = pairs
        print(f"{label}: {len(pairs)} labeled pairs", flush=True)

    cache = {}
    if os.path.exists(a.verdicts):
        with open(a.verdicts) as f:
            cache = json.load(f)
    todo = {}
    for pairs in pairs_by_method.values():
        for det, x, y in pairs:
            k = pair_key(x, y)
            if det != "different_answer" and k not in cache:
                todo[k] = (x, y)
    print(f"{len(todo)} same-answer pairs to judge ({len(cache)} verdicts on file)", flush=True)
    todo = list(todo.items())
    for i in range(0, len(todo), 400):
        chunk = todo[i:i + 400]
        for (k, _), verdict in zip(chunk, judge_pairs([pair for _, pair in chunk], a.model)):
            cache[k] = verdict
        with open(a.verdicts, "w") as f:
            json.dump(cache, f)
        print(f"{i + len(chunk)}/{len(todo)} judged", flush=True)

    out = {}
    print(f"\n{'method':<24s} {'diff ans':>9s} {'diff deriv':>11s} {'surface':>8s} {'pairs':>6s} {'agree':>6s}")
    for label, pairs in pairs_by_method.items():
        final, anomalies = [], Counter()
        n_judged = agree = 0
        for det, x, y in pairs:
            if det == "different_answer":
                final.append(det)
                continue
            v = cache.get(pair_key(x, y))
            if v in LABELS[1:]:
                final.append(v)
                n_judged += 1
                agree += v == det
            else:
                anomalies["judge_says_different_answer" if v == LABELS[0] else "no_verdict"] += 1
                final.append(det)
        c = Counter(final)
        n = len(final)
        out[label] = {**{k: c[k] / n for k in LABELS}, "n_pairs": n, "n_judge_verdicts": n_judged,
                      "judge_agrees_with_rule": agree / n_judged if n_judged else None,
                      "anomalies": dict(anomalies)}
        o = out[label]
        print(f"{label:<24s} {o[LABELS[0]]:>9.3f} {o[LABELS[1]]:>11.3f} {o[LABELS[2]]:>8.3f} {n:>6d} "
              f"{o['judge_agrees_with_rule'] or float('nan'):>6.2f} {dict(anomalies) or ''}", flush=True)
    with open(a.out or os.path.join(SCRIPT_DIR, "pair_judge.json"), "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("validate", "score"))
    ap.add_argument("--model", default=JUDGE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-pools", type=int, default=4, help="validate: pools to sample pairs from")
    ap.add_argument("--per-class", type=int, default=100, help="validate: pairs per deterministic label")
    ap.add_argument("--pool", type=int, default=0, help="score: pool of the selections")
    ap.add_argument("--run", type=int, default=0, help="score: selection run")
    ap.add_argument("--selection", nargs=2, action="append", metavar=("LABEL", "PATH"),
                    help="score: selection file to use in place of the default rows; repeatable")
    ap.add_argument("--verdicts", default=os.path.join(SCRIPT_DIR, "pair_judge_verdicts.json"))
    a = ap.parse_args()
    {"validate": cmd_validate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
