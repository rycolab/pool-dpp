"""GSM8K at N=2500 from two independently seeded workers (tab:ndial, tab:weights)
==============================================================================
Two workers (seeds 303 and 404, 1,250 draws each at sT=1.2) merged to one pool of
N=2,500 draws per question.

Kernels as in the figures: PoolDPP-Samp w = plm^(1/T)/q, PoolDPP-MAP w = plm^(1/T).
The merged pool holds unique strings with summed counts.

Reports:
  - the bound on the KL at N in {100, 1250, 2500} from the per-question bound rows
    of the same T (bound_raw_gsm8k.json, built from the N=100 pools;
    chi2_1 = Var_q(u_1) and r_1 do not depend on N), median over questions:
      key "bound_m1|{setting}" (primary): second-moment hybrid
          log(1 + a_1 chi2_1 + sum_{c>=2} a_c ((K^c/(K)_c)^2 m_1^c - 1)),
          m_1 = 1 + chi2_1;
      key "cert|{setting}" (secondary): r_1 hybrid
          log(1 + a_1 chi2_1 + sum_{c>=2} a_c (K^c/(K)_c r_1^c - 1));
    formulas in bound_formulas.py. These values come from the bound rows, not
    from the N=2,500 pools;
  - pass@10 of PoolDPP-MAP per worker (N=1250) and merged (N=2500);
  - pass@10 of PoolDPP-Samp at N=2500;
  - ESS of the PoolDPP-Samp weights at N=2500;
  - pool ceiling (any correct chain among the 2,500 draws);
  - cross/within collision ratio between the two workers (independence).

Failures stop the run with an error, and nothing is written: the two workers or the GSM8K
questions with different prompt sets, a missing bound_raw file or a bound_raw entry that is not a
non-empty list of rows (keys starting with "_" are metadata), a non-finite log-weight, a
PoolDPP-MAP or PoolDPP-Samp set with fewer than K strings (the pool spans fewer than K
dimensions, Zhat = 0), or workers that fail the independence check. So n_sets_lt_K is 0 in every
output this script writes.

Output: analysis/gsm8k_n2500.json

Usage:
    OPENBLAS_NUM_THREADS=8 python analysis/gsm8k_n2500.py
"""

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, SCRIPT_DIR)

import bound_formulas as bf  # noqa: E402
from lm_dpp.estimation import fmt, greedy_map, quality_log_scales, sample_dpp, target_log_scales  # noqa: E402
from lm_dpp.utils.math_utils import build_math_prompt, is_correct, load_gsm8k_dataset  # noqa: E402

ST = 1.2
K = 10
MAP_TS = [10, 50, 100, 500]
SAMP_TS = [1, 1.05]
ESS_TS = [0.428571, 0.666667, 0.818182, 1, 1.03, 1.05, 1.07, 1.1, 1.22222]
SAMP_DRAWS = 5
CACHES = ["cache-gsm8k-n2500-a", "cache-gsm8k-n2500-b"]
EMB_FNAME = "google__gemma-2-9b-it_qwen_embeddings.pkl"
BOUND_RAW = "bound_raw_gsm8k.json"


def bound_m1_at(N, chi2_1):
    """Second-moment hybrid bound on the KL at pool size N (primary)."""
    return bf.kl_hybrid_m1(N, K, chi2_1)


def cert_at(N, chi2_1, r1):
    """r_1 hybrid bound on the KL at pool size N (secondary)."""
    return bf.kl_hybrid_r1(N, K, chi2_1, r1)


def bound_rows(path):
    """(setting, rows) of a bound_raw file; keys starting with "_" are metadata. Raises on an
    entry that is not a non-empty list of rows with a finite chi2_1 and r1."""
    out = []
    for key, rows in json.load(open(path)).items():
        if key.startswith("_"):
            continue
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: entry {key!r} is not a non-empty list of rows")
        for i, r in enumerate(rows):
            bad = [f for f in ("chi2_1", "r1") if not np.isfinite(r.get(f, np.nan))]
            if bad:
                raise ValueError(f"{path}: {key!r} row {i} has no finite {bad}")
            bf.m1_of_row(r)  # rows that store m1 must have m1 = 1 + chi2_1
        out.append((key, rows))
    return out


def merge_halves(ha, hb):
    """Unique strings across both workers, counts summed (int64)."""
    if set(ha) != set(hb):
        raise ValueError(f"gsm8k N=2500: the workers have different prompts "
                         f"({len(set(ha) - set(hb))} only in A, {len(set(hb) - set(ha))} only in B)")
    out = {}
    for pt in ha:
        idx, sents, phi, lp, lp1, cnt = {}, [], [], [], [], []
        for d in (ha[pt], hb[pt]):
            E = np.asarray(d["embeddings"])
            L0 = np.asarray(d["logps"], dtype=np.float64)
            L1 = np.asarray(d["logps_t1"], dtype=np.float64)
            C = np.asarray(d["counts"], dtype=np.int64)
            for j, s in enumerate(d["u_sents"]):
                k = idx.get(s)
                if k is None:
                    k = idx[s] = len(sents)
                    sents.append(s)
                    phi.append(E[j])
                    lp.append(L0[j])
                    lp1.append(L1[j])
                    cnt.append(0)
                cnt[k] += int(C[j])
        out[pt] = dict(u_sents=sents, embeddings=np.asarray(phi, dtype=np.float64),
                       logps=np.asarray(lp), logps_t1=np.asarray(lp1),
                       counts=np.asarray(cnt, dtype=np.int64))
    return out


def as_pool(d):
    return dict(u_sents=list(d["u_sents"]),
                embeddings=np.asarray(d["embeddings"], dtype=np.float64),
                logps=np.asarray(d["logps"], dtype=np.float64),
                logps_t1=np.asarray(d["logps_t1"], dtype=np.float64),
                counts=np.asarray(d["counts"], dtype=np.int64))


def cross_within_ratio(ha, hb):
    """Matched draw pairs across workers vs. their expectation under
    independence (see analysis/check_pool_independence.py)."""
    if set(ha) != set(hb):
        raise ValueError("gsm8k N=2500: the workers have different prompts")
    X = E = 0.0
    for pt in ha:
        ca = dict(zip(ha[pt]["u_sents"], np.asarray(ha[pt]["counts"], float)))
        cb = dict(zip(hb[pt]["u_sents"], np.asarray(hb[pt]["counts"], float)))
        X += sum(c * cb[s] for s, c in ca.items() if s in cb)
        Na, Nb = sum(ca.values()), sum(cb.values())
        Qa = sum(c * (c - 1) / 2 for c in ca.values()) / (Na * (Na - 1) / 2)
        Qb = sum(c * (c - 1) / 2 for c in cb.values()) / (Nb * (Nb - 1) / 2)
        E += Na * Nb * (Qa + Qb) / 2
    return dict(cross_matched_pairs=float(X), expected_if_independent=float(E),
                ratio=(float(X / E) if E > 0 else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs=2, default=CACHES)
    ap.add_argument("--bound-raw", default=os.path.join(SCRIPT_DIR, BOUND_RAW))
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "gsm8k_n2500.json"))
    ap.add_argument("--max-prompts", type=int, default=None, help="smoke tests")
    a = ap.parse_args()
    np.random.seed(0)

    df = load_gsm8k_dataset().head(50)

    if len(df) < 50:  # offline load returns an empty frame

        raise SystemExit("GSM8K dataset did not load (HF offline?); refusing to score")
    prompt_to_gold = {build_math_prompt(r["question"]): r["answer"]
                      for _, r in df.iterrows()}

    halves = []
    for c in a.caches:
        with open(os.path.join(ROOT_DIR, c, EMB_FNAME), "rb") as f:
            halves.append(pickle.load(f)[ST])
    for c, h in zip(a.caches, halves):
        if set(h) != set(prompt_to_gold):
            raise ValueError(f"gsm8k {c} sT={ST}: {len(set(prompt_to_gold) - set(h))} of the "
                             f"{len(prompt_to_gold)} GSM8K questions missing, "
                             f"{len(set(h) - set(prompt_to_gold))} unexpected prompts")
    prompts = list(halves[0])
    if a.max_prompts:
        prompts = prompts[:a.max_prompts]
    print(f"{len(prompts)} prompts", flush=True)
    merged = merge_halves({p: halves[0][p] for p in prompts},
                          {p: halves[1][p] for p in prompts})

    results = {"_meta": dict(
        caches=a.caches, K=K, sT=ST, kernel="Samp w = plm^(1/T)/q, MAP w = plm^(1/T)",
        n_prompts=len(prompts))}
    results["independence_between_workers"] = cross_within_ratio(halves[0], halves[1])
    print("independence:", results["independence_between_workers"], flush=True)
    Ns = [int(merged[pt]["counts"].sum()) for pt in prompts]
    results["_meta"]["draws_per_prompt"] = [min(Ns), max(Ns)]
    results["_meta"]["unique_per_prompt_median"] = float(np.median(
        [len(merged[pt]["u_sents"]) for pt in prompts]))

    corr_cache = {}

    def correct_any(pt, sents):
        gold = prompt_to_gold[pt]
        for s in sents:
            key = (pt, s)
            if key not in corr_cache:
                corr_cache[key] = bool(is_correct(s, gold))
            if corr_cache[key]:
                return True
        return False

    # bounds on the KL from the bound rows (same anchors)
    if not os.path.exists(a.bound_raw):
        raise FileNotFoundError(f"{a.bound_raw} missing; run bound_empirical.py --task gsm8k first")
    for key, rows in bound_rows(a.bound_raw):
        results[f"bound_m1|{key}"] = {
            str(N): float(np.median([bound_m1_at(N, r["chi2_1"]) for r in rows]))
            for N in (100, 1250, 2500)}
        results[f"cert|{key}"] = {
            str(N): float(np.median([cert_at(N, r["chi2_1"], r["r1"]) for r in rows]))
            for N in (100, 1250, 2500)}
        print(f"bound on the KL (m1) {key}: {results[f'bound_m1|{key}']}", flush=True)
        print(f"  r1 version (secondary) {key}: {results[f'cert|{key}']}", flush=True)

    # ESS of the PoolDPP-Samp weights at N=2500
    ess_out = {}
    for T in ESS_TS:
        ess = []
        for pt in prompts:
            d = merged[pt]
            c = d["counts"].astype(np.float64)
            lw = d["logps_t1"] / T - d["logps"]
            if not np.all(np.isfinite(lw)):
                raise ValueError(f"gsm8k N=2500 T={fmt(T)} prompt {pt[:60]!r}: non-finite log-weight")
            w = np.exp(lw - lw.max())
            wbar = w / (w * c).sum()
            ess.append(1.0 / (c * wbar ** 2).sum())
        ess_out[fmt(T)] = {"median": float(np.median(ess)), "q25": float(np.quantile(ess, 0.25)),
                           "q75": float(np.quantile(ess, 0.75)), "min": float(np.min(ess)),
                             "n_lt_K": int((np.asarray(ess) < K).sum())}
    results["L_weight_ess_N2500"] = ess_out

    def map_pass(pool, T, where):
        sel = greedy_map(quality_log_scales(pool["logps_t1"], 1.0 / (2.0 * T)), pool["embeddings"], K)
        if len(sel) != K:
            raise RuntimeError(f"gsm8k {where} PoolDPP-MAP T={fmt(T)}: {len(sel)} strings selected, "
                               f"K={K}: the pool spans fewer than K dimensions (Zhat = 0)")
        return sel

    # per-worker N=1250 and merged N=2500 MAP
    for name, src in [("A_N1250", {p: as_pool(halves[0][p]) for p in prompts}),
                      ("B_N1250", {p: as_pool(halves[1][p]) for p in prompts}),
                      ("merged_N2500", merged)]:
        vals = {}
        for T in MAP_TS:
            hits = [float(correct_any(pt, [src[pt]["u_sents"][i]
                                           for i in map_pass(src[pt], T, f"{name} prompt {pt[:60]!r}")]))
                    for pt in prompts]
            vals[fmt(T)] = float(np.mean(hits))
        results[f"map_pass_at_10|{name}"] = vals
        print(f"MAP {name}: {vals}", flush=True)

    # merged N=2500 Samp
    samp = {}
    for T in SAMP_TS:
        vals, n_short = [], 0
        for pt in prompts:
            d = merged[pt]
            draws = sample_dpp(target_log_scales(d["logps"], d["logps_t1"], d["counts"], T),
                               d["embeddings"], K, N=SAMP_DRAWS)
            n_short += sum(len(s) < K for s in draws)
            if n_short:
                raise RuntimeError(f"gsm8k N=2500 PoolDPP-Samp T={fmt(T)} prompt {pt[:60]!r}: "
                                   f"{n_short} of {SAMP_DRAWS} draws have fewer than K={K} strings: "
                                   f"the pool spans fewer than K dimensions (Zhat = 0)")
            hits = [float(correct_any(pt, [d["u_sents"][i] for i in s])) for s in draws]
            vals.append(float(np.mean(hits)))
        samp[fmt(T)] = {"pass_at_10": float(np.mean(vals)), "n_prompts": len(vals),
                          "n_draws_per_prompt": SAMP_DRAWS, "n_sets_lt_K": n_short}
        print(f"Samp N=2500 T={fmt(T)}: {samp[fmt(T)]}", flush=True)
    results["samp_N2500"] = samp

    results["ceiling_pass_at_2500"] = float(np.mean(
        [float(correct_any(pt, merged[pt]["u_sents"])) for pt in prompts]))

    ind = results["independence_between_workers"]
    X, E = ind["cross_matched_pairs"], ind["expected_if_independent"]
    if X > 1.5 * E + 3 * math.sqrt(E) + 5:   # checked before saving, so no output is written
        print(f"ERROR: workers are not independent (X={X:.0f}, E={E:.1f}); "
              "these halves must not back the N-dial claim")
        sys.exit(2)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(json.dumps(results, indent=1, default=float))
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
