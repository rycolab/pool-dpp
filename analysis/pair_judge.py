"""
Pair decomposition of selected GSM8K sets (tab:gsm8k-pairs)
===========================================================
Splits each selected set into its pairs and labels every pair in which both chains state a final answer
in the '####' format. A parser decides 'different answer'. Gemini 3.6 Flash labels each same-answer pair
as 'same answer, different derivation' or 'surface only'.

  validate  compares the judge with the deterministic labels on a sample stratified by label. The
            deterministic rule calls a same-answer pair 'surface only' when both chains state the same
            multiset of arithmetic facts (a, op, b, c).
  score     writes the decomposition. A judge verdict 'different answer' on a pair the parser calls
            same-answer, or a reply without a standalone A, B or C, stops the run with an error that
            counts them per method (the verdicts on file are kept); no pair takes the deterministic
            label in place of the judge's.

A request that fails with HTTP 429 or 5xx, or a network error, is retried up to 4 times with
backoff 1, 2, 4, 8 s, and then stops the run (verdicts already received are saved first in score).
A malformed API response stops the run. Pairs in which a chain has no '####' answer are left out
(counted as n_pairs_no_answer); validate also leaves out pairs with a chain of 2,500 characters or
more (counted as n_pairs_too_long). A selection file must hold 50 prompts with 10 strings each.

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
UNPARSEABLE = "unparseable"   # verdict of a reply without a standalone A, B or C
K = 10                        # strings per selected set (run_gsm8k.sh)
N_QUESTIONS = 50              # GSM8K questions per selection file

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


class JudgeError(RuntimeError):
    """A request failed for good; .verdicts has the verdicts received (None for the failed ones)."""

    def __init__(self, msg, verdicts):
        super().__init__(msg)
        self.verdicts = verdicts


def judge_pairs(pairs, model, workers=16, tries=4):
    """A label from LETTER, or UNPARSEABLE, for each (x, y). Retries HTTP 429/5xx and network errors
    up to `tries` times, then raises JudgeError. Stops at once on another HTTP error, such as a
    rejected key, and on a malformed response."""
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
        last = None
        for attempt in range(tries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as r:
                    raw = r.read()
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:
                    raise SystemExit(f"Gemini API error {e.code}: {e.reason}")
                last = f"HTTP {e.code} {e.reason}"
                time.sleep(2 ** attempt)
                continue
            except (OSError, http.client.HTTPException) as e:
                last = repr(e)
                time.sleep(2 ** attempt)
                continue
            try:
                reply = json.loads(raw)
                parts = reply["candidates"][0]["content"].get("parts", [])
            except (ValueError, KeyError, IndexError) as e:
                raise RuntimeError(f"malformed Gemini response ({e!r}): {raw[:300]!r}") from e
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip().upper()
            # the reply should be one letter; take the last so a preamble cannot win
            letters = re.findall(r"\b([ABC])\b", text)
            return LETTER[letters[-1]] if letters else UNPARSEABLE
        raise RuntimeError(f"no reply after {tries} tries (last error: {last})")

    def safe(pair):
        try:
            return one(pair), None
        except RuntimeError as e:
            return None, e

    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(safe, pairs))
    errors = [e for _, e in got if e is not None]
    if errors:
        raise JudgeError(f"{len(errors)} of {len(pairs)} judge requests failed; first: {errors[0]}",
                         [v for v, _ in got])
    return [v for v, _ in got]


def require_files(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} selection files missing: {missing}")


def load_sets(path):
    """The selected sets of a selection file: N_QUESTIONS distinct prompts, K strings each."""
    with open(path, "rb") as f:
        items = pickle.load(f)
    prompts = set()
    for item in items:
        if len(item) != 4:
            raise ValueError(f"{path}: record of length {len(item)}, expected 4")
        if len(item[1]) != K:
            raise ValueError(f"{path}: prompt {item[0][:60]!r} has {len(item[1])} strings, expected K={K}")
        prompts.add(item[0])
    if len(prompts) != N_QUESTIONS or len(items) != N_QUESTIONS:
        raise ValueError(f"{path}: {len(items)} records for {len(prompts)} prompts, expected {N_QUESTIONS}")
    return [item[1] for item in items]


def cmd_validate(a):
    buckets = defaultdict(list)
    left_out = Counter()
    paths = [os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{pool}", "est10qwen", "0", sub)
             for pool in range(a.n_pools) for sub in VALIDATION_SELECTIONS]
    require_files(paths)
    for path in paths:
        for sents in load_sets(path):
            for x, y in combinations(sents, 2):
                label = classify_pair(x, y)
                if label is None:
                    left_out["n_pairs_no_answer"] += 1
                elif len(x) >= 2500 or len(y) >= 2500:
                    left_out["n_pairs_too_long"] += 1
                else:
                    buckets[label].append((x, y))
    print({k: len(v) for k, v in buckets.items()}, dict(left_out), flush=True)

    rng = np.random.default_rng(0)
    sample = []
    for label in LABELS:
        v = buckets[label]
        for i in rng.choice(len(v), size=min(a.per_class, len(v)), replace=False):
            sample.append((label, *v[i]))
    verdicts = judge_pairs([(x, y) for _, x, y in sample], a.model)

    confusion = Counter()
    agree, n_class, n_unparseable = Counter(), Counter(), Counter()
    for (label, _, _), got in zip(sample, verdicts):
        confusion[f"{label}->{got}"] += 1
        if got == UNPARSEABLE:
            n_unparseable[label] += 1   # the judge gave no verdict: counted, not scored
        else:
            n_class[label] += 1
            agree[label] += got == label
    out = {"judge_model": a.model, "per_class_n": dict(n_class),
           "per_class_unparseable": dict(n_unparseable),
           "per_class_agreement": {k: agree[k] / n_class[k] for k in n_class},
           "confusion": dict(confusion), "left_out": dict(left_out),
           "pool_pairs_per_class": {k: len(v) for k, v in buckets.items()}}
    print(json.dumps(out, indent=2))
    with open(a.out or os.path.join(SCRIPT_DIR, "pair_judge_validation.json"), "w") as f:
        json.dump(out, f, indent=2)


def cmd_score(a):
    root = os.path.join(ROOT_DIR, "cache-gsm8k", f"pool{a.pool}", "est10qwenins", str(a.run))
    selections = a.selection or [(label, os.path.join(root, sub)) for label, sub in SCORE_SELECTIONS]
    require_files([path for _, path in selections])
    pairs_by_method, no_answer = {}, {}
    for label, path in selections:
        pairs, no_answer[label] = [], 0
        for sents in load_sets(path):
            for x, y in combinations(sents, 2):
                det = classify_pair(x, y)
                if det is None:
                    no_answer[label] += 1
                else:
                    pairs.append((det, x, y))
        pairs_by_method[label] = pairs
        print(f"{label}: {len(pairs)} labeled pairs, {no_answer[label]} without a '####' answer", flush=True)

    cache = {}
    if os.path.exists(a.verdicts):
        with open(a.verdicts) as f:
            cache = json.load(f)
    todo = {}
    for pairs in pairs_by_method.values():
        for det, x, y in pairs:
            k = pair_key(x, y)
            if det != "different_answer" and cache.get(k) is None:   # None: an older run got no reply
                todo[k] = (x, y)
    print(f"{len(todo)} same-answer pairs to judge ({len(cache)} verdicts on file)", flush=True)
    todo = list(todo.items())
    for i in range(0, len(todo), 400):
        chunk = todo[i:i + 400]
        try:
            verdicts, err = judge_pairs([pair for _, pair in chunk], a.model), None
        except JudgeError as e:
            verdicts, err = e.verdicts, e
        for (k, _), verdict in zip(chunk, verdicts):
            if verdict is not None:
                cache[k] = verdict
        with open(a.verdicts, "w") as f:
            json.dump(cache, f)
        if err is not None:
            raise err
        print(f"{i + len(chunk)}/{len(todo)} judged", flush=True)

    out, anomalies_by_method = {}, {}
    print(f"\n{'method':<24s} {'diff ans':>9s} {'diff deriv':>11s} {'surface':>8s} {'pairs':>6s} {'agree':>6s}")
    for label, pairs in pairs_by_method.items():
        final, anomalies = [], Counter()
        n_judged = agree = 0
        for det, x, y in pairs:
            if det == "different_answer":
                final.append(det)
                continue
            v = cache[pair_key(x, y)]
            if v in LABELS[1:]:
                final.append(v)
                n_judged += 1
                agree += v == det
            elif v == LABELS[0]:
                anomalies["judge_says_different_answer"] += 1
            elif v == UNPARSEABLE:
                anomalies["unparseable_reply"] += 1
            else:
                raise ValueError(f"{a.verdicts}: verdict {v!r} for pair {pair_key(x, y)}")
        if anomalies:
            anomalies_by_method[label] = dict(anomalies)
            continue
        c = Counter(final)
        n = len(final)
        out[label] = {**{k: c[k] / n for k in LABELS}, "n_pairs": n, "n_judge_verdicts": n_judged,
                      "n_pairs_no_answer": no_answer[label],
                      "judge_agrees_with_rule": agree / n_judged if n_judged else None,
                      "anomalies": dict(anomalies)}
        o = out[label]
        print(f"{label:<24s} {o[LABELS[0]]:>9.3f} {o[LABELS[1]]:>11.3f} {o[LABELS[2]]:>8.3f} {n:>6d} "
              f"{o['judge_agrees_with_rule'] or float('nan'):>6.2f} {dict(anomalies) or ''}", flush=True)
    if anomalies_by_method:
        raise ValueError(f"judge verdicts without a label to score, per method: {anomalies_by_method}. "
                         f"pair_judge.json is not written; the verdicts are kept in {a.verdicts}")
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
