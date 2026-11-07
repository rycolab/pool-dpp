"""Fail-loudly checks for the set-level baseline scripts in analysis/: baselines.py,
baselines_gsm8k.py, heldout_diversity.py, beam_baselines.py, length_stratified.py and
length_analysis.py. Each test builds a tiny fake pool or decoding cache under tmp_path and
checks that a missing or incomplete input raises instead of being skipped. No GPU, LM,
network or real pools needed.

Run: PYTHONPATH=src pytest tests/test_fail_loudly_baselines.py
"""
import json
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ANALYSIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
sys.path.insert(0, ANALYSIS)

import baselines as bl  # noqa: E402
import baselines_gsm8k as bg  # noqa: E402
import beam_baselines as bb  # noqa: E402
import heldout_diversity as hd  # noqa: E402
import length_analysis as la  # noqa: E402
import length_stratified as ls  # noqa: E402
import lm_dpp.utils.fineweb_prompts as fwp  # noqa: E402
import lm_dpp.utils.math_utils as mu  # noqa: E402

RNG = np.random.default_rng(0)


def unit_rows(n, d=8):
    E = RNG.standard_normal((n, d))
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pool_row(sents, lp=None):
    n = len(sents)
    return {"u_sents": np.array(sents), "logps_t1": -np.arange(1.0, n + 1) if lp is None else lp,
            "embeddings": unit_rows(n), "counts": np.ones(n, int)}


def selection(prompt, k):
    return (prompt, [f"{prompt} s{i}" for i in range(k)], list(-np.arange(1.0, k + 1)), list(unit_rows(k)))


# ---------------------------------------------------------------------------
# baselines.py: shared checks
# ---------------------------------------------------------------------------
def test_require_pools_lists_every_missing_pool(tmp_path):
    paths = {i: [str(tmp_path / f"pool{i}" / "x.pkl")] for i in range(6)}
    for i in (0, 4):
        dump(paths[i][0], {})
    with pytest.raises(FileNotFoundError, match=r"gsm8k: 4 of 6 pools missing: pool 1\.\.3, 5"):
        bl.require_pools("gsm8k", paths)


def test_run_dirs_must_be_contiguous(tmp_path):
    for r in ("0", "1", "3"):
        os.makedirs(tmp_path / r)
    with pytest.raises(ValueError, match="run dirs are"):
        bl.run_dirs(str(tmp_path))


def test_selection_files_stochastic_needs_every_run_deterministic_only_run0(tmp_path):
    est = tmp_path / "est"
    for r in range(3):
        os.makedirs(est / str(r) / "m-samp")
        os.makedirs(est / str(r) / "m-map")
    for r in (0, 1):
        dump(str(est / str(r) / "m-samp" / "T1_s1.pkl"), [])
    dump(str(est / "0" / "m-map" / "T1_s1.pkl"), [])
    with pytest.raises(FileNotFoundError, match="1 of 3 files missing"):
        bl.selection_files(str(est), "m-samp", "T1_s1.pkl", deterministic=False)
    assert bl.selection_files(str(est), "m-map", "T1_s1.pkl", deterministic=True) == [
        str(est / "0" / "m-map" / "T1_s1.pkl")]


@pytest.mark.parametrize("items, msg", [
    ([selection("a", 3)], "1 of 2 prompts missing"),
    ([selection("a", 3), selection("b", 3), selection("a", 3)], "appears twice"),
    ([selection("a", 3), selection("c", 3)], "not one of the task's prompts"),
    ([selection("a", 3), selection("b", 2)], "expected K=3"),
    ([selection("a", 3), ("b", ["x"], [0.0])], "3 fields"),
])
def test_check_selections(items, msg):
    with pytest.raises(ValueError, match=msg):
        bl.check_selections(items, "f.pkl", {"a", "b"}, 3)
    bl.check_selections([selection("a", 3), selection("b", 3)], "f.pkl", {"a", "b"}, 3)


def test_logdet_raises_instead_of_nan():
    assert np.isfinite(bl.logdet(unit_rows(3)))
    with pytest.raises(ValueError, match="at least 2"):
        bl.logdet(unit_rows(1))
    E = unit_rows(3)
    E[1, 0] = np.nan
    with pytest.raises(ValueError, match="log-det failed"):
        bl.logdet(E)


def test_t975_and_summarize_pools():
    with pytest.raises(ValueError, match="at least 2 pools"):
        bl.t975(1)
    acc = {"m": {0: {"x": [1.0, 2.0]}, 1: {"x": [3.0]}}}
    r = bl.summarize_pools(acc, ["x"])["m"]
    assert r["x"] == pytest.approx(2.25) and r["n_pools"] == 2 and r["n"] == 3
    assert r["x_ci95"] == pytest.approx(stats.t.ppf(0.975, 1) * np.std([1.5, 3.0], ddof=1) / math.sqrt(2))
    with pytest.raises(ValueError, match="not finite"):
        bl.summarize_pools({"m": {0: {"x": [1.0, np.nan]}, 1: {"x": [3.0]}}}, ["x"])
    with pytest.raises(ValueError, match="no values"):
        bl.summarize_pools({"m": {0: {"x": []}, 1: {"x": [3.0]}}}, ["x"])
    with pytest.raises(ValueError, match="at least 2 pools"):
        bl.summarize_pools({"m": {0: {"x": [1.0]}}}, ["x"])
    with pytest.raises(ValueError, match="different pools"):
        bl.summarize_pools({"m": {0: {"x": [1.0]}, 1: {"x": [2.0]}},
                            "n": {0: {"x": [1.0]}, 2: {"x": [2.0]}}}, ["x"])


def test_selections_rejects_small_pool_and_nan_logp():
    with pytest.raises(ValueError, match="fewer than K=3"):
        bl.selections(unit_rows(2), [-1.0, -2.0], 3)
    with pytest.raises(ValueError, match="not finite"):
        bl.selections(unit_rows(3), [-1.0, np.nan, -2.0], 3)


# ---------------------------------------------------------------------------
# GSM8K fixtures: 50 questions with answer 3
# ---------------------------------------------------------------------------
@pytest.fixture
def gsm8k_df(monkeypatch):
    df = pd.DataFrame({"question": [f"q{i}" for i in range(50)], "answer": ["work #### 3"] * 50})
    monkeypatch.setattr(mu, "load_gsm8k_dataset", lambda split="test": df)
    return [mu.build_math_prompt(q) for q in df["question"]]


def gsm8k_strings(n):
    return [f"chain {j} #### {3 if j % 2 == 0 else 4}" for j in range(n)]


def write_gsm8k_pool(root, pi, prompts, n_strings, sT=1.2, model="google__gemma-2-9b-it_qwen", drop=None):
    rows = {pt: pool_row(gsm8k_strings(n_strings)) for pt in prompts if pt != drop}
    dump(os.path.join(root, "cache-gsm8k", f"pool{pi}", f"{model}_embeddings.pkl"), {sT: rows})


def test_baselines_gsm8k_missing_pools_raise(tmp_path, monkeypatch, gsm8k_df):
    monkeypatch.setattr(bl, "ROOT_DIR", str(tmp_path))
    write_gsm8k_pool(str(tmp_path), 0, gsm8k_df, 4)
    with pytest.raises(FileNotFoundError, match=r"gsm8k: 2 of 3 pools missing: pool 1\.\.2"):
        bl.run_gsm8k(n_pools=3, k=2, est_rows=False)


def test_baselines_gsm8k_pool_missing_prompt_or_small_pool_raises(tmp_path, monkeypatch, gsm8k_df):
    monkeypatch.setattr(bl, "ROOT_DIR", str(tmp_path))
    write_gsm8k_pool(str(tmp_path), 0, gsm8k_df, 4)
    write_gsm8k_pool(str(tmp_path), 1, gsm8k_df, 4, drop=gsm8k_df[7])
    with pytest.raises(ValueError, match="1 of the task's 50 prompts missing"):
        bl.run_gsm8k(n_pools=2, k=2, est_rows=False)
    write_gsm8k_pool(str(tmp_path), 1, gsm8k_df, 1)
    with pytest.raises(ValueError, match="fewer than K=2"):
        bl.run_gsm8k(n_pools=2, k=2, est_rows=False)
    write_gsm8k_pool(str(tmp_path), 1, gsm8k_df, 4)
    res = bl.run_gsm8k(n_pools=2, k=2, est_rows=False)
    assert res["quality-topK"]["n_pools"] == 2 and res["quality-topK"]["n"] == 100


def write_gsm8k_est(root, pi, prompts, k, runs, variant="qwen", skip=None):
    model = f"google__gemma-2-9b-it_{variant}"
    est = os.path.join(root, "cache-gsm8k", f"pool{pi}", f"est10{variant}")
    strings = gsm8k_strings(k)
    for name, sub, fname in bg.est_methods(model):
        for r in (["0"] if sub.endswith("-map") else [str(i) for i in range(runs)]):
            if (sub, r) == skip:
                continue
            items = [(pt, strings, list(-np.arange(1.0, k + 1)), list(unit_rows(k))) for pt in prompts]
            dump(os.path.join(est, r, sub, fname), items)
    for r in range(runs):
        os.makedirs(os.path.join(est, str(r)), exist_ok=True)


def run_bg_main(monkeypatch, tmp_path, *args):
    monkeypatch.setattr(bg, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(bg, "K", 2)
    out = str(tmp_path / "out.json")
    monkeypatch.setattr(sys, "argv", ["baselines_gsm8k.py", "--n-pools", "2", "--out", out, *args])
    bg.main()
    with open(out) as f:
        return json.load(f)


def test_baselines_gsm8k_script_raises_and_counts(tmp_path, monkeypatch, gsm8k_df):
    root = str(tmp_path)
    for pi in (0, 1):
        write_gsm8k_pool(root, pi, gsm8k_df, 4)
    write_gsm8k_est(root, 0, gsm8k_df, 2, runs=2)
    # pool 1 has no est dir
    with pytest.raises(FileNotFoundError, match=r"1 of 2 pools missing: pool 1"):
        run_bg_main(monkeypatch, tmp_path)
    # a stochastic method lacks run dir 1
    write_gsm8k_est(root, 1, gsm8k_df, 2, runs=2, skip=("google__gemma-2-9b-it_qwen-samp", "1"))
    with pytest.raises(FileNotFoundError, match="every run dir"):
        run_bg_main(monkeypatch, tmp_path)
    write_gsm8k_est(root, 1, gsm8k_df, 2, runs=2)
    out = run_bg_main(monkeypatch, tmp_path)
    assert out["PoolDPP-MAP T=10"]["n"] == 100          # run dir 0 only
    assert out["PoolDPP-Samp T=1"]["n"] == 200          # both run dirs
    assert out["Ancestral"]["n_chains_no_final_answer"] == 0
    assert all(np.isfinite(v["logdet_ci95"]) for k, v in out.items() if not k.startswith("_"))
    # a selection with a prompt that is not one of the 50 questions
    bad = os.path.join(root, "cache-gsm8k", "pool1", "est10qwen", "0", "google__gemma-2-9b-it_qwen-naive",
                       "s1.2.pkl")
    dump(bad, [selection("not a gsm8k prompt", 2)])
    with pytest.raises(ValueError, match="not one of the task's prompts"):
        run_bg_main(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# heldout_diversity.py
# ---------------------------------------------------------------------------
def test_distinct_n_and_jaccard_raise_when_undefined():
    assert hd.distinct_n(["a b c", "a b d"], 2) == pytest.approx(3 / 4)
    with pytest.raises(ValueError, match="distinct-3 is undefined"):
        hd.distinct_n(["a b", "c"], 3)
    with pytest.raises(ValueError, match="both empty"):
        hd.mean_pairwise_jaccard(["a b", " ", ""])


def test_heldout_missing_embedding_raises(tmp_path, monkeypatch):
    prompts = ["p0", "p1"]
    monkeypatch.setattr(fwp, "prompts", prompts)
    monkeypatch.setattr(hd, "K", 2)
    est = tmp_path / "est10minilm"
    name, knob, sub, fname = hd.METHODS[0]
    sents = {pt: [f"{pt} one two three", f"{pt} four five six"] for pt in prompts}
    for r in range(2):
        dump(str(est / str(r) / sub / fname),
             [(pt, sents[pt], [-1.0, -2.0], list(unit_rows(2))) for pt in prompts])
    qwen = {0.6: {pt: {"u_sents": sents[pt][:1] if pt == "p1" else sents[pt],
                       "embeddings": unit_rows(1 if pt == "p1" else 2)} for pt in prompts}}
    dump(str(tmp_path / "qwen.pkl"), qwen)
    monkeypatch.setattr(hd, "EST_DIR", str(est))
    monkeypatch.setattr(hd, "QWEN_EMB", str(tmp_path / "qwen.pkl"))
    monkeypatch.setattr(hd, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="no held-out embedding"):
        hd.main()


# ---------------------------------------------------------------------------
# beam_baselines.py
# ---------------------------------------------------------------------------
def test_beam_ci95_is_a_t_interval():
    a = [1.0, 2.0, 4.0]
    assert bb.ci95(a) == pytest.approx(stats.t.ppf(0.975, 2) * np.std(a, ddof=1) / math.sqrt(3))
    with pytest.raises(ValueError, match="at least 2 runs"):
        bb.ci95([1.0])


def test_beam_aggregate():
    assert bb.aggregate({0: [1.0, 3.0]}, deterministic=True) == (2.0, None)
    m, c = bb.aggregate({0: [1.0, 3.0], 1: [3.0, 5.0]})
    assert m == pytest.approx(3.0) and c == pytest.approx(bb.ci95([2.0, 4.0]))
    with pytest.raises(ValueError, match="not finite"):
        bb.aggregate({0: [1.0, np.nan], 1: [3.0]})
    with pytest.raises(ValueError, match="no values"):
        bb.aggregate({0: [], 1: [3.0]})
    with pytest.raises(ValueError, match="deterministic method with 2 runs"):
        bb.aggregate({0: [1.0], 1: [3.0]}, deterministic=True)


def test_beam_load_runs_and_counts():
    cache = {1.0: {"a": (["x", "y", "z", "w"], [0] * 4, [-1.0] * 4, [1] * 4),
                   "b": (["x", "y", "z"], [0] * 3, [-1.0] * 3, [1] * 3)}}
    assert len(bb.load_runs(cache, 1.0, "a", 2, "t")) == 2
    with pytest.raises(ValueError, match="not a whole number of runs"):
        bb.load_runs(cache, 1.0, "b", 2, "t")
    with pytest.raises(ValueError, match="not in the cache"):
        bb.load_runs(cache, 1.0, "c", 2, "t")
    with pytest.raises(ValueError, match="different numbers of runs"):
        bb.check_n_runs({"a": [1, 2], "b": [1]}, "sbs", "t")
    with pytest.raises(ValueError, match="expected at least 2"):
        bb.check_n_runs({"a": [1], "b": [1]}, "sbs", "t")
    with pytest.raises(ValueError, match="expected 1"):
        bb.check_n_runs({"a": [1, 2]}, "detbs", "t")
    bb.check_n_runs({"a": [1], "b": [1]}, "detbs", "t")


def test_beam_missing_cache_and_failing_golden_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="decoding cache"):
        bb.load_cache(str(tmp_path / "nope.pkl"))
    dump_sql = "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1);"
    assert bb._golden_sigs(dump_sql, "['SELECT a FROM t']") == {((1,),)}
    with pytest.raises(ValueError, match="fails to execute"):
        bb._golden_sigs(dump_sql, "['SELECT a FROM t', 'SELECT b FROM nope']")


def test_beam_fineweb_missing_prompt_raises(tmp_path, monkeypatch):
    prompts = ["p0", "p1"]
    monkeypatch.setattr(fwp, "prompts", prompts)
    monkeypatch.setattr(bb, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(bb, "EXPECTED", {"fineweb": {bb.GEMMA: ["sbs"]}})
    swor = tmp_path / "cache-fineweb-swor"
    sents = ["one two", "three four"]
    dump(str(swor / f"{bb.GEMMA}_sbs.pkl"), {1.0: {"p0": (sents * 2, [0] * 4, [-1.0] * 4, [1] * 4)}})
    dump(str(swor / f"{bb.GEMMA}_sbs_minilm_embeddings.pkl"),
         {1.0: {"p0": {"u_sents": np.array(sents), "embeddings": unit_rows(2)}}})
    with pytest.raises(ValueError, match="'p1' is not in the cache"):
        bb.run_fineweb(k=2)


# ---------------------------------------------------------------------------
# length_stratified.py
# ---------------------------------------------------------------------------
def test_length_stratified_counts_exclusions():
    L = np.arange(1.0, 13.0)
    Y = np.array([1, 0] * 6, float)
    items = [(L, -L, Y),                       # used everywhere
             (L[:5], -L[:5], Y[:5]),           # fewer than 10 strings: excluded
             (L, np.full(12, -3.0), np.ones(12))]  # no negatives (no AUC), constant log p (no Spearman)
    out = ls.analyze(iter(items), "correct")
    assert out["n_prompts"] == 3 and out["n_prompts_lt10_strings"] == 1
    assert out["n_prompts_auc_full"] == 1 and out["n_prompts_spearman_full"] == 1
    assert out["n_prompts_by_quintile"] == [2] * 5
    assert out["spearman_logp_length_full"] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="AUC, full pool: no values"):
        ls.analyze(iter([(L, -L, np.ones(12))]), "correct")


def test_length_stratified_missing_pool_raises(tmp_path, monkeypatch, gsm8k_df):
    monkeypatch.setattr(ls, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(ls, "N_POOLS", 3)
    write_gsm8k_pool(str(tmp_path), 0, gsm8k_df, 12)
    with pytest.raises(FileNotFoundError, match=r"gsm8k: 2 of 3 pools missing"):
        list(ls.gsm8k_items(1.2))


# ---------------------------------------------------------------------------
# length_analysis.py
# ---------------------------------------------------------------------------
def test_length_analysis_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(la, "ROOT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="3 of 3 files missing"):
        la.pool_length_vs_logp()


def test_length_baseline_counts_short_length_kmeans_sets(tmp_path, monkeypatch):
    prompts = ["p0", "p1", "p2"]
    monkeypatch.setattr(fwp, "prompts", prompts)
    monkeypatch.setattr(la, "ROOT_DIR", str(tmp_path))
    rows = {"p0": pool_row([f"{'x' * i}" for i in range(1, 7)]),      # 6 lengths: 3 bins all filled
            "p1": pool_row(["a", "bb", "cccccccccc", "dddddddddd"]),  # middle bin empty: 2 picks
            "p2": pool_row(["aa", "bb", "cc", "dd"])}                   # one length: excluded
    dump(str(tmp_path / la.TASKS["fineweb"]["emb"]), {1.0: rows})
    out = la.length_baseline(k=3)
    km = out["length-kmeans"]
    assert (km["n_prompts"], km["n_prompts_excluded"], km["n_prompts_fewer_than_K"], km["min_set_size"]) == \
        (2, 1, 1, 2)
    assert out["quality-topk"]["n_prompts"] == 3 and out["quality-topk"]["n_prompts_fewer_than_K"] == 0
    rows["p2"] = pool_row(["aa", "bb"])
    dump(str(tmp_path / la.TASKS["fineweb"]["emb"]), {1.0: rows})
    with pytest.raises(ValueError, match="fewer than K=3"):
        la.length_baseline(k=3)
