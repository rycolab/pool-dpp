"""Failures in src/, the vLLM patch, the bound scripts and math500.py stop with an error instead of being
skipped, and the exclusions the method keeps are exact and counted. No GPU, LM, vLLM or network.
Run: pytest tests/test_fail_loudly_src.py"""
import itertools
import json
import math
import os
import pickle
import sys
import types
from unittest import mock

import numpy as np
import pytest

from dpp import stable
from dpp.stable import StableKDPP, kernel_info
from lm_dpp import estimation as est

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import test_estimation as te  # noqa: E402  (fake pools and CLI helpers)


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


RNG = np.random.default_rng(5)


# --------------------------------------------------------------------------- dpp.stable
def test_nan_or_inf_log_scale_raises():
    Phi = unit(RNG.normal(size=(5, 4)))
    for bad in (np.nan, np.inf):
        ls = np.zeros(5)
        ls[2] = bad
        with pytest.raises(ValueError, match="NaN or \\+inf log scale"):
            StableKDPP(ls, Phi, 2)
    Phi[1, 0] = np.nan
    with pytest.raises(ValueError, match="embedding rows"):
        StableKDPP(np.zeros(5), Phi, 2)


def test_minus_inf_scale_and_zero_row_have_no_mass():
    Phi = unit(RNG.normal(size=(5, 4)))
    ls = np.zeros(5)
    ls[0] = -np.inf
    Phi[1] = 0.0
    np.random.seed(0)
    draws = [set(StableKDPP(ls, Phi, 2).sample()) for _ in range(50)]
    assert all(len(d) == 2 and not d & {0, 1} for d in draws)


def test_numerical_failure_on_full_rank_rows_raises():
    Phi = unit(RNG.normal(size=(6, 4)))
    with mock.patch.object(stable, "_sample_flat", lambda ls, U, k: [0]):
        with pytest.raises(RuntimeError, match="numerical failure"):
            StableKDPP(np.zeros(6), Phi, 3).sample()


def test_tiered_draw_with_a_small_gap_raises():
    """The counterexample of the dpp.stable docstring: the old tiered draw always returned both top
    rows, which the k-DPP does with probability 0.497."""
    log_s = np.array([0.0, -299.99, -300.01, -300.01, -300.01])
    Phi = unit(np.random.default_rng(1).normal(size=(5, 5)))
    info = kernel_info(log_s, Phi, 3)
    assert info["path"] == "tiered" and info["min_gap_nats"] == pytest.approx(0.02)
    np.random.seed(0)
    with pytest.raises(RuntimeError, match="not a draw from the k-DPP"):
        StableKDPP(log_s, Phi, 3).sample()


def test_tiered_draw_with_separated_tiers_runs():
    log_s = np.array([0.0, -1.0, -700.0, -700.0, -701.0])
    Phi = unit(np.random.default_rng(1).normal(size=(5, 5)))
    assert kernel_info(log_s, Phi, 3)["path"] == "tiered"
    np.random.seed(0)
    assert all({0, 1} <= set(StableKDPP(log_s, Phi, 3).sample()) for _ in range(20))


def exact_dropped_mass(log_s, Phi, k, cut):
    lw, dropped = [], []
    for S in itertools.combinations(range(len(log_s)), k):
        S = list(S)
        lw.append(2 * log_s[S].sum() + np.log(np.linalg.det(Phi[S] @ Phi[S].T)))
        dropped.append((log_s[S] < log_s.max() - cut).any())
    p = np.exp(np.array(lw) - max(lw))
    return p[np.array(dropped)].sum() / p.sum()


@pytest.mark.parametrize("log_s", [[0, -1, -2, -3, -299.99, -300.01, -300.01],
                                   [0, -299.99, -299.99, -299.99, -300.01, -300.01],
                                   [0, -0.5, -1, -2, -310, -350, -400]])
def test_drop_bound_is_an_upper_bound_on_the_dropped_mass(log_s):
    log_s = np.array(log_s, float)
    Phi = unit(np.random.default_rng(3).normal(size=(len(log_s), 4)))
    info = kernel_info(log_s, Phi, 3)
    assert info["path"] == "drop" and info["n_dropped"] == int((log_s < -300).sum())
    exact = exact_dropped_mass(log_s, Phi, 3, stable.TIER_NATS)
    assert math.log10(max(exact, 1e-300)) <= info["log10_drop_bound"] + 1e-9


def test_flat_kernel_info():
    info = kernel_info(np.zeros(6), unit(RNG.normal(size=(6, 4))), 3)
    assert info == dict(path="flat", n_rows=6, n_dropped=0, log10_drop_bound=None, min_gap_nats=None)


# --------------------------------------------------------------------------- estimation
def test_greedy_map_rejects_nan_and_raises_when_its_tolerance_disagrees_with_the_rank():
    Phi = unit(RNG.normal(size=(6, 4)))
    with pytest.raises(ValueError, match="NaN"):
        est.greedy_map(np.array([0, np.nan, 0, 0, 0, 0.0]), Phi, 3)
    with pytest.raises(RuntimeError, match="numerical failure"):
        est.greedy_map(np.zeros(6), Phi, 4, tol=0.99)
    # rows that span 2 dimensions: short, and no error
    assert len(est.greedy_map(np.zeros(6), np.repeat(Phi[:2], 3, axis=0), 3)) == 2


def test_iter_prompt_data_rejects_arrays_of_different_lengths():
    p = te.fake_pool()
    p["logps_t1"] = p["logps_t1"][:-1]
    with pytest.raises(ValueError, match="different lengths"):
        list(est.iter_prompt_data({"prompt": p}))


def test_calibration_rejects_a_prompt_with_one_draw():
    p = te.fake_pool(n_unique=1)
    p["counts"] = np.array([1])
    with pytest.raises(ValueError, match="two or more finite draws"):
        est.sigma_hat_from_embeddings({1.0: {"prompt": p}})


def test_kmeans_with_fewer_unique_strings_than_k_raises(tmp_path):
    p = te.fake_pool(n_unique=2)
    with pytest.raises(ValueError, match="fewer than k=3"):
        est.sample_kmeans_for_prompt(p["embeddings"], p["logps_t1"], 3)
    path = te.fake_cache(tmp_path, sampling_Ts=[1.0], n_prompts=2)
    data = pickle.load(open(path, "rb"))
    one = data[1.0]["prompt_1"]
    for key in ("u_sents", "counts", "logps", "logps_t1", "embeddings"):
        one[key] = one[key][:1]
    pickle.dump(data, open(path, "wb"))
    with pytest.raises(ValueError, match="1 prompts have no selection"):
        te.run_cli("--output_folder", tmp_path / "out", "--model_name", te.MODEL, "--cache_dir", tmp_path,
                   "--k", 3, "--kmeans")


def test_a_short_draw_on_a_full_rank_pool_stops_the_run_and_is_not_redrawn(tmp_path):
    (pool0, pool1), _ = te.degenerate_pools(tmp_path)
    short = lambda log_s, embeddings, k: est.sample_dpp(log_s, embeddings, k)[0][:k - 1]  # noqa: E731
    with mock.patch.object(est, "draw_one", short):
        with pytest.raises(RuntimeError, match="numerical failure, not Zhat = 0"):
            te.selections(pool0, "--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1)


def distinct_pools(tmp_path):
    """te.degenerate_pools, with pool1's strings renamed so no string is in both pools."""
    (pool0, pool1), bad = te.degenerate_pools(tmp_path)
    path = pool1 / f"{te.MODEL}_embeddings.pkl"
    data = pickle.load(open(path, "rb"))
    for row in data[1.0].values():
        row["u_sents"] = np.array([f"pool1 {s}" for s in row["u_sents"]])
    pickle.dump(data, open(path, "wb"))
    return (pool0, pool1), bad


def test_redraws_are_counted_in_the_sidecar(tmp_path):
    (pool0, pool1), bad = distinct_pools(tmp_path)
    te.selections(pool0, "--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1)
    side = json.load(open(pool0 / "out" / "0" / f"{te.MODEL}-samp" / "T1_s1.redraws.json"))
    assert (side["n_prompts"], side["n_redrawn"], side["k"], side["sampling_T"]) == (5, 1, 3, 1.0)
    (r,) = side["redrawn"]
    assert r["prompt"] == bad and r["rank"] == 2 and r["from"].startswith(str(pool1))
    assert side["sampler"]["n_flat"] == 5
    # a resumed run rebuilds the same sidecar from the stored selections
    os.remove(pool0 / "out" / "0" / f"{te.MODEL}-samp" / "T1_s1.redraws.json")
    te.selections(pool0, "--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1)
    again = json.load(open(pool0 / "out" / "0" / f"{te.MODEL}-samp" / "T1_s1.redraws.json"))
    assert again == side


def test_a_stored_selection_swapped_by_the_old_trigger_is_detected(tmp_path):
    """A full-rank prompt whose stored selection came from another pool: written by the trigger
    len(selected) < k after a numerical failure."""
    (pool0, pool1), bad = distinct_pools(tmp_path)
    te.selections(pool0, "--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1)
    path = pool0 / "out" / "0" / f"{te.MODEL}-samp" / "T1_s1.pkl"
    stored = pickle.load(open(path, "rb"))
    other = pickle.load(open(pool1 / f"{te.MODEL}_embeddings.pkl", "rb"))[1.0]
    good = next(x[0] for x in stored if x[0] != bad)
    o = other[good]
    stored = [x if x[0] != good else (good, list(o["u_sents"][:3]), list(o["logps_t1"][:3]),
                                       list(o["embeddings"][:3])) for x in stored]
    pickle.dump(stored, open(path, "wb"))
    with pytest.raises(RuntimeError, match="swapped for another pool"):
        te.selections(pool0, "--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1)


def test_a_redraw_pool_without_the_prompt_raises(tmp_path):
    (pool0, pool1), bad = te.degenerate_pools(tmp_path)
    path = pool1 / f"{te.MODEL}_embeddings.pkl"
    data = pickle.load(open(path, "rb"))
    del data[1.0][bad]
    pickle.dump(data, open(path, "wb"))
    with pytest.raises(KeyError, match="has no row"):
        te.selections(pool0, "--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1)


# --------------------------------------------------------------------------- sample_vllm, vLLM patch
@pytest.fixture
def sample_vllm(monkeypatch):
    """lm_dpp.sample_vllm with stub vllm modules (vLLM is not installed for the tests)."""
    vllm = types.ModuleType("vllm")
    for name in ("AsyncEngineArgs", "AsyncLLMEngine", "SamplingParams", "TokensPrompt"):
        setattr(vllm, name, object)
    sp = types.ModuleType("vllm.sampling_params")
    sp.RequestOutputKind = object
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)
    monkeypatch.delitem(sys.modules, "lm_dpp.sample_vllm", raising=False)
    import lm_dpp.sample_vllm as sv
    return sv


def completion(**kw):
    c = types.SimpleNamespace(text="x", cumulative_logprob=-3.0, token_ids=[1, 2])
    c.__dict__.update(kw)
    return c


def test_missing_or_unknown_at_temperature_logprob_raises(sample_vllm):
    with pytest.raises(RuntimeError, match="apply vllm_dual_logprobs.patch"):
        sample_vllm.extract_results([completion()])
    with pytest.raises(RuntimeError, match="unknown \\(None\\)"):
        sample_vllm.extract_results([completion(cumulative_logprob_at_temperature=None)])
    with pytest.raises(RuntimeError, match="must be a positive"):
        sample_vllm.extract_results([completion(cumulative_logprob_at_temperature=float("nan"))])
    texts, t1, lps, lengths = sample_vllm.extract_results([completion(cumulative_logprob_at_temperature=-2.5)])
    assert (t1, lps, lengths) == ([-3.0], [-2.5], [2])


def test_the_patch_never_adds_a_default_log_prob():
    patch = open(os.path.join(ROOT, "vllm_dual_logprobs.patch")).read()
    added = [line for line in patch.splitlines() if line.startswith("+")]
    assert not any("else 0.0" in line for line in added)
    assert any("self.cumulative_logprob_at_temperature = None" in line for line in added)
    # FlashInfer returns no processed log-probs, so the dual mode must not select it
    assert '+                                 "raw_processed_logprobs"' in added


# --------------------------------------------------------------------------- utils
def test_gsm8k_loader_raises_instead_of_returning_an_empty_frame(monkeypatch):
    import datasets

    from lm_dpp.utils import math_utils

    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(math_utils, "load_dataset", boom)
    with pytest.raises(ConnectionError):
        math_utils.load_gsm8k_dataset()
    assert datasets  # imported only to make sure the module patched is the real one


def test_detbs_greedy_map_raises_on_a_zero_residual():
    pytest.importorskip("torch")
    from lm_dpp.det_beam import greedy_map_logdet
    assert greedy_map_logdet(np.diag([3.0, 2.0, 1.0]), 2) == [0, 1]
    with pytest.raises(RuntimeError, match="rank < k"):
        greedy_map_logdet(np.diag([1.0, 0.0, 0.0]), 2)


def test_ambrosia_golden_query_that_fails_raises():
    from lm_dpp.utils.ambrosia_utils import compute_ambrosia_coverage, test_ambrosia_sql_queries
    dump = "CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1), (2);"
    golden = repr(["SELECT x FROM t", "SELECT y FROM nowhere"])
    with pytest.raises(ValueError, match="golden query fails to run"):
        compute_ambrosia_coverage(dump, ["SELECT x FROM t"], golden)
    with pytest.raises(ValueError, match="golden query fails to run"):
        test_ambrosia_sql_queries(dump, ["SELECT x FROM t"], golden)
    assert compute_ambrosia_coverage(dump, ["SELECT x FROM t", "BAD SQL"], repr(["SELECT x FROM t"])) == 1.0


# --------------------------------------------------------------------------- bound scripts
import bound_empirical as be  # noqa: E402
import bound_tables as bt  # noqa: E402


def test_exact_target_raises_when_the_kernel_needs_tiers():
    Phi = unit(np.random.default_rng(1).normal(size=(5, 5)))
    kappa = np.exp(2 * np.array([0.0, -299.99, -300.01, -300.01, -300.01]))
    with pytest.raises(RuntimeError, match="tiered path"):
        be.exact_target(Phi, kappa, 3)
    tgt = be.exact_target(Phi, np.full(5, 0.2), 3)
    assert tgt["tier_path"] == "flat" and tgt["sumP1"] == pytest.approx(3)


def test_a_pool_with_zhat_zero_enters_the_variance_as_a_zero():
    Phi = unit(RNG.normal(size=(6, 4)))
    rho = np.ones(6)
    assert be.pool_logzhat(Phi, rho, np.array([5.0, 5, 0, 0, 0, 0]), 3) == -np.inf   # 2 strings, K = 3
    Cnt = np.array([[4.0, 4, 0, 0, 0, 0], [1, 1, 2, 2, 1, 1]] * 6)
    rv, n_blocks, n_zero = be.block_relvar(Phi, rho, Cnt, 3, 8)
    assert (n_blocks, n_zero) == (12, 6)
    logs = [be.pool_logzhat(Phi, rho, row, 3) for row in Cnt]
    v = np.exp(np.array(logs) - max(logs))
    assert rv == pytest.approx(v.var(ddof=1) / v.mean() ** 2)
    with pytest.raises(ValueError, match="needs a positive Zhat"):
        be.relvar_from_logs([-np.inf, -np.inf])
    with pytest.raises(ValueError, match="exceeds pool"):
        be.block_relvar(Phi, rho, Cnt, 3, 100)


def test_extract_stops_on_missing_pools(tmp_path, monkeypatch):
    monkeypatch.setattr(be, "SCRATCH", str(tmp_path / "work"))
    cfg = dict(be.CONFIGS["gsm8k"], pool_files=[str(tmp_path / "pool0.pkl"), str(tmp_path / "pool1.pkl")])
    monkeypatch.setitem(be.CONFIGS, "gsm8k", cfg)
    with pytest.raises(FileNotFoundError, match="2 of 2 pools missing"):
        be.extract("gsm8k")


def test_bound_medians_raise_on_nan():
    with pytest.raises(ValueError, match="non-finite"):
        bt.q([1.0, float("nan"), 2.0], 0.5)
    with pytest.raises(ValueError, match="None"):
        bt.q([1.0, None], 0.5)
    assert bt.med([1.0, 3.0, 2.0]) == 2.0


# --------------------------------------------------------------------------- math500
import math500 as m5  # noqa: E402


def test_math500_missing_pools_raise(tmp_path):
    a = types.SimpleNamespace(cache=str(tmp_path), n_pools=3)
    pickle.dump({}, open(tmp_path / "pool1.pkl", "wb"))
    with pytest.raises(FileNotFoundError, match="2 of 3 pools missing: \\[0, 2\\]"):
        list(m5.read_pools(a))


def test_math500_verification_failure_is_an_error_not_incorrect():
    assert m5._with_deadline(lambda: {"correct": True, "key": "num:1"}, 5) == {"correct": True, "key": "num:1"}

    def crash():
        raise ZeroDivisionError("x")
    assert m5._with_deadline(crash, 5) == {"error": "ZeroDivisionError: x"}


def test_math500_scores_degenerate_cells_for_ancestral_and_counts_them_for_the_rest(monkeypatch):
    import hashlib
    rng = np.random.default_rng(0)
    K = m5.K
    full = [f"solution {i} #### {i % 3}" for i in range(K + 2)]
    few = ["a #### 1", "b #### 2", "c #### 1"]
    prompts = {"P full": dict(unique_id="q1"), "P few": dict(unique_id="q2")}
    sents = {"P full": full * 3, "P few": few * 10}
    raw = {m5.ANCHOR_ST: {p: (s, [-1.0] * len(s), list(-rng.random(len(s)) * 5), [3] * len(s))
                          for p, s in sents.items()}}
    emb = {hashlib.md5(s.encode()).hexdigest(): unit(rng.normal(size=(1, 16)))[0] for s in full + few}
    verdicts = {(prompts[p]["unique_id"], m5.extract_answer(s)): {"correct": s.endswith("1"), "key": s[-1]}
                for p, ss in sents.items() for s in ss}
    monkeypatch.setitem(m5._shared, "by_prompt", prompts)
    monkeypatch.setitem(m5._shared, "verdicts", verdicts)
    monkeypatch.setitem(m5._shared, "emb", emb)
    rows = m5._score_pool((0, raw))
    by = {}
    for method, knob, cell, metrics in rows:
        by.setdefault((method, cell[2]), []).append(metrics)
    assert len(by[("Ancestral", "q2")]) == m5.N_RUNS and by[("_ceiling", "q2")][0]["pass"] == 1.0
    for method in ("$k$-Means", "Naive-DPP"):
        assert by[(method, "q2")] == [None] and len(by[(method, "q1")]) == m5.N_RUNS
    assert by[("PoolDPP-MAP", "q2")] == [None] * len(m5.TARGET_TS)
    assert all(m is not None for m in by[("PoolDPP-Samp", "q1")])
