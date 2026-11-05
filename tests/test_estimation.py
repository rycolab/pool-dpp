"""Checks for lm_dpp.estimation on synthetic pools. No GPU, models, or vLLM needed.
Run: pytest tests/test_estimation.py"""
import collections
import itertools
import json
import os
import pickle
import sys
from unittest import mock

import numpy as np
import pytest

from lm_dpp.estimation import (cli, greedy_map, quality_log_scales, sample_dpp, sample_kmeans_for_prompt,
                               sigma_hat_from_embeddings, target_log_scales)
from lm_dpp.utils.cached import cached_ddict

RNG = np.random.default_rng(42)
MODEL = "test_model_minilm"


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def fake_pool(n_unique=30, dim=16):
    return dict(u_sents=np.array([f"sentence_{i}" for i in range(n_unique)]),
                counts=RNG.integers(1, 8, size=n_unique),
                logps=-RNG.exponential(5, size=n_unique) - 10.0,
                logps_t1=-RNG.exponential(3, size=n_unique) - 8.0,
                embeddings=unit(RNG.standard_normal((n_unique, dim))))


def fake_cache(cache_dir, sampling_Ts, n_prompts=4):
    path = os.path.join(cache_dir, f"{MODEL}_embeddings.pkl")
    emb = cached_ddict(path)
    for sT in sampling_Ts:
        for i in range(n_prompts):
            emb[sT][f"prompt_{i}"] = fake_pool()
    emb._save()
    return path


def run_cli(*argv):
    with mock.patch.object(sys, "argv", ["dpp-estimation", *map(str, argv)]):
        cli()


def kdpp_law(log_s, Phi, k, labels):
    """Exact law over label sets of the k-DPP with rows exp(log_s_i) * Phi_i."""
    law = collections.defaultdict(float)
    for S in itertools.combinations(range(len(log_s)), k):
        S = list(S)
        G = Phi[S] @ Phi[S].T
        law[frozenset(labels[i] for i in S)] += np.linalg.det(G) * np.exp(2 * log_s[S].sum())
    total = sum(law.values())
    return {s: v / total for s, v in law.items() if v > 1e-14 * total}


def test_quality_scales():
    p = fake_pool()
    assert np.allclose(quality_log_scales(p["logps_t1"], 0.2), 0.2 * p["logps_t1"])
    assert np.allclose(quality_log_scales(p["logps_t1"], 0.2, p["counts"]),
                       0.2 * p["logps_t1"] + 0.5 * np.log(p["counts"]))


def test_target_T_relabels_tilt_at_sT_1():
    """At sT=1, q = plm, so w = plm^(1/T) / q = plm^(1/T - 1)."""
    p = fake_pool()
    T = 0.5
    a = target_log_scales(p["logps_t1"], p["logps_t1"], p["counts"], T)
    b = quality_log_scales(p["logps_t1"], (1 / T - 1) / 2, p["counts"])
    assert np.allclose(a, b)


def test_collapsed_rows_match_one_column_per_draw():
    """Unique rows scaled by sqrt(count) and one column per draw give one law over string sets."""
    Phi = unit(RNG.standard_normal((5, 6)))
    log_s = RNG.normal(size=5)
    counts = np.array([1, 3, 2, 1, 2])
    draws = np.repeat(np.arange(5), counts)
    per_draw = kdpp_law(log_s[draws], Phi[draws], 3, draws)
    collapsed = kdpp_law(log_s + 0.5 * np.log(counts), Phi, 3, np.arange(5))
    assert per_draw.keys() == collapsed.keys()
    for s in per_draw:
        assert per_draw[s] == pytest.approx(collapsed[s], rel=1e-9)


def test_greedy_over_draws_matches_unique_rows():
    Phi = unit(RNG.standard_normal((12, 8)))
    log_s = RNG.normal(size=12)
    counts = RNG.integers(1, 5, size=12)
    draws = np.repeat(np.arange(12), counts)
    over_draws = {draws[i] for i in greedy_map(log_s[draws], Phi[draws], 5)}
    assert over_draws == set(greedy_map(log_s, Phi, 5))


def test_greedy_map_maximizes_each_gain():
    """Each pick has the largest residual gain, checked by determinant ratios."""
    Phi = unit(RNG.standard_normal((10, 8)))
    log_s = RNG.normal(size=10)
    selected = greedy_map(log_s, Phi, 4)

    def log_det(S):
        return np.linalg.slogdet(Phi[S] @ Phi[S].T)[1] + 2 * log_s[S].sum()
    for step in range(1, len(selected) + 1):
        head = selected[:step - 1]
        best = max((i for i in range(10) if i not in head), key=lambda i: log_det(head + [i]))
        assert selected[step - 1] == best


def test_greedy_map_with_wide_scales():
    """Scales span 8,000 nats. A near-copy of the top row has gain about -46 nats here, so it
    follows rows 2, 3, and 4 (gains about -20, -30, -40) and precedes row 5 (about -50)."""
    rng = np.random.default_rng(0)
    Phi = unit(rng.standard_normal((12, 8)))
    Phi[1] = unit(Phi[0][None] + 1e-10 * rng.standard_normal((1, 8)))[0]
    log_s = -5.0 * np.arange(12.0)
    log_s[1] = -1.0
    log_s[-1] = -8000.0
    assert greedy_map(log_s, Phi, 5) == [0, 2, 3, 4, 1]


def test_greedy_map_stops_at_rank():
    Phi = unit(RNG.standard_normal((2, 8)))
    assert len(greedy_map(np.zeros(6), np.repeat(Phi, 3, axis=0), 3)) == 2


def test_sample_dpp():
    p = fake_pool()
    draws = sample_dpp(quality_log_scales(p["logps_t1"], 0.2, p["counts"]), p["embeddings"], 5, N=3)
    assert len(draws) == 3
    assert all(len(set(s)) == 5 for s in draws)


def test_kmeans():
    p = fake_pool()
    assert len(set(sample_kmeans_for_prompt(p["embeddings"], p["logps_t1"], k=5))) == 5


def test_cli_end_to_end(tmp_path):
    cache_dir, out, cal = tmp_path / "cache", tmp_path / "output", tmp_path / "calibration.json"
    cache_dir.mkdir()
    path = fake_cache(cache_dir, sampling_Ts=[0.6, 1.0, 1.5])

    base = ["--model_name", MODEL, "--cache_dir", cache_dir]
    run_cli("--calibrate", "--calibration", cal, *base)
    sigma = json.load(open(cal))["sigma_hat"]
    assert set(sigma) == {"0.6", "1", "1.5"}
    assert sigma_hat_from_embeddings(cached_ddict(path, read_only=True))[1.0] == pytest.approx(sigma["1"])

    est = ["--output_folder", out, *base, "--k", 5, "--N", 2, "--sampling_T", 1.0]
    run_cli(*est, "--samp", "--taus", 0.03, 1, "--calibration", cal)
    run_cli(*est, "--samp", "--target_Ts", 0.5)
    run_cli(*est, "--map", "--taus", 0.03, "--calibration", cal)
    run_cli(*est, "--map", "--target_Ts", 10)
    run_cli(*est, "--ancestral", "--naive", "--kmeans")

    assert sorted(os.listdir(out)) == ["0", "1"]
    run0 = out / "0"
    assert sorted(os.listdir(run0)) == sorted(f"{MODEL}-{m}" for m in ["samp", "map", "ancestral", "naive", "kmeans"])
    assert sorted(os.listdir(run0 / f"{MODEL}-samp")) == ["T0.5_s1.pkl", "tau0.03_s1.pkl", "tau1_s1.pkl"]
    assert sorted(os.listdir(run0 / f"{MODEL}-map")) == ["T10_s1.pkl", "tau0.03_s1.pkl"]
    assert os.listdir(run0 / f"{MODEL}-ancestral") == ["s1.pkl"]
    assert not (out / "1" / f"{MODEL}-map").exists()

    data = pickle.load(open(run0 / f"{MODEL}-samp" / "tau1_s1.pkl", "rb"))
    assert len(data) == 4
    assert len(data[0][1]) == 5


def test_cli_rejects_unknown_sampling_T(tmp_path):
    fake_cache(tmp_path, sampling_Ts=[0.6, 1.0], n_prompts=2)
    with pytest.raises(ValueError, match="not in cache"):
        run_cli("--output_folder", tmp_path / "out", "--model_name", MODEL, "--cache_dir", tmp_path,
                "--k", 3, "--ancestral", "--sampling_T", 99.0)
