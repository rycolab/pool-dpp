"""Checks for lm_dpp.estimation on synthetic pools. No GPU, models, or vLLM needed.
Run: pytest tests/test_estimation.py"""
import collections
import itertools
import json
import os
import pickle
import shutil
import sys
from unittest import mock

import numpy as np
import pytest

from lm_dpp.estimation import (PoolRedraw, ancestral_draw, cli, greedy_map, method_seed, quality_log_scales, sample_dpp,
                               sample_kmeans_for_prompt, sigma_hat_from_embeddings, target_log_scales)
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


def test_ancestral_draw_follows_counts():
    # Taking every position of the pool returns each string exactly count times.
    for counts in ([3, 2], [1, 1, 1], [5, 1, 4], [200]):
        drawn = ancestral_draw(np.array(counts), sum(counts))
        assert np.bincount(drawn, minlength=len(counts)).tolist() == counts


def test_pools_get_their_own_streams():
    assert method_seed("ancestral", 0, "pool0") != method_seed("ancestral", 0, "pool1")
    assert method_seed("ancestral", 0, "pool0") != method_seed("ancestral", 1, "pool0")
    assert method_seed("ancestral", 0, "pool0") != method_seed("samp", 0, "pool0")
    assert method_seed("ancestral", 0, "pool0") == method_seed("ancestral", 0, "pool0")


def test_cli_seed_depends_on_the_pool_name_only(tmp_path):
    src = tmp_path / "a" / "pool0"
    src.mkdir(parents=True)
    fake_cache(src, sampling_Ts=[1.0], n_prompts=6)
    for name in ("a/pool1", "b/pool0"):
        shutil.copytree(src, tmp_path / name)

    def sets(name):
        cache = tmp_path / name
        run_cli("--output_folder", cache / "out", "--model_name", MODEL, "--cache_dir", cache, "--k", 3, "--ancestral")
        return [list(x[1]) for x in pickle.load(open(cache / "out" / "0" / f"{MODEL}-ancestral" / "s1.pkl", "rb"))]

    assert sets("a/pool0") == sets("b/pool0")
    assert sets("a/pool0") != sets("a/pool1")


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
    def files(method):
        return sorted(os.listdir(run0 / f"{MODEL}-{method}"))

    def with_sidecars(*stems):
        return sorted([f"{s}.pkl" for s in stems] + [f"{s}.redraws.json" for s in stems])

    assert files("samp") == with_sidecars("T0.5_s1", "tau0.03_s1", "tau1_s1")
    assert files("map") == with_sidecars("T10_s1", "tau0.03_s1")
    assert files("naive") == with_sidecars("s1")
    assert files("ancestral") == ["s1.pkl"]
    side = json.load(open(run0 / f"{MODEL}-samp" / "T0.5_s1.redraws.json"))
    assert (side["n_prompts"], side["n_redrawn"], side["k"]) == (4, 0, 5)
    assert side["sampler"]["n_flat"] + side["sampler"]["n_drop"] + side["sampler"]["n_tiered"] == 4
    assert json.load(open(run0 / f"{MODEL}-map" / "T10_s1.redraws.json"))["sampler"] is None
    assert not (out / "1" / f"{MODEL}-map").exists()

    data = pickle.load(open(run0 / f"{MODEL}-samp" / "tau1_s1.pkl", "rb"))
    assert len(data) == 4
    assert len(data[0][1]) == 5


def test_cli_rejects_unknown_sampling_T(tmp_path):
    fake_cache(tmp_path, sampling_Ts=[0.6, 1.0], n_prompts=2)
    with pytest.raises(ValueError, match="not in cache"):
        run_cli("--output_folder", tmp_path / "out", "--model_name", MODEL, "--cache_dir", tmp_path,
                "--k", 3, "--ancestral", "--sampling_T", 99.0)


def degenerate_pools(root, n_prompts=5, k=3):
    """pool0 and pool1 with the same prompts. In pool0, one prompt spans 2 directions and has no size-k set."""
    caches = []
    for name in ("pool0", "pool1"):
        cache = root / name
        cache.mkdir(parents=True)
        fake_cache(cache, sampling_Ts=[1.0], n_prompts=n_prompts)
        caches.append(cache)
    path = caches[0] / f"{MODEL}_embeddings.pkl"
    data = pickle.load(open(path, "rb"))
    bad = sorted(data[1.0])[1]
    row = data[1.0][bad]
    E = np.array(row["embeddings"], float)
    E[1::2], E[::2] = E[1], E[0]
    row["embeddings"] = E
    pickle.dump(data, open(path, "wb"))
    return caches, bad


def selections(cache, *args, out="out", method="samp", name="T1_s1.pkl", run=0, k=3):
    run_cli("--output_folder", cache / out, "--model_name", MODEL, "--cache_dir", cache, "--k", str(k), *args)
    return {x[0]: list(x[1]) for x in pickle.load(open(cache / out / str(run) / f"{MODEL}-{method}" / name, "rb"))}


def test_zhat_zero_without_redraw_pools_stops_the_run(tmp_path):
    (pool0, _), _ = degenerate_pools(tmp_path)
    with pytest.raises(RuntimeError, match="redraw_dirs"):
        selections(pool0, "--N", "1", "--samp", "--target_Ts", "1")


@pytest.mark.parametrize("flags,method,name", [(["--N", "2", "--samp", "--target_Ts", "1"], "samp", "T1_s1.pkl"),
                                               (["--N", "1", "--map", "--target_Ts", "2"], "map", "T2_s1.pkl"),
                                               (["--N", "2", "--naive"], "naive", "s1.pkl")])
def test_redraw_fills_the_prompt_from_the_other_pool(tmp_path, flags, method, name):
    (pool0, pool1), bad = degenerate_pools(tmp_path)
    got = selections(pool0, *flags, "--redraw_dirs", pool1, method=method, name=name)
    other = pickle.load(open(pool1 / f"{MODEL}_embeddings.pkl", "rb"))[1.0]
    own = pickle.load(open(pool0 / f"{MODEL}_embeddings.pkl", "rb"))[1.0]
    assert set(got) == set(own)
    assert len(got[bad]) == 3
    assert set(got[bad]) <= set(other[bad]["u_sents"].tolist())


def test_redraw_leaves_other_prompts_unchanged(tmp_path):
    (pool0, pool1), bad = degenerate_pools(tmp_path)
    flags = ["--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1]
    real = selections(pool0, *flags, out="real")
    other = pickle.load(open(pool1 / f"{MODEL}_embeddings.pkl", "rb"))[1.0]

    def no_randomness(self, sT, prompt, log_scales, select, k, label, pool_size=None):
        d = other[prompt]
        return "stub", d["u_sents"], d["logps_t1"], d["embeddings"], [0, 1, 2]

    with mock.patch.object(PoolRedraw, "select", no_randomness):
        stub = selections(pool0, *flags, out="stub")
    assert real[bad] != stub[bad]
    assert {p: v for p, v in real.items() if p != bad} == {p: v for p, v in stub.items() if p != bad}


def test_a_resumed_fill_matches_a_full_run(tmp_path):
    (pool0, pool1), bad = degenerate_pools(tmp_path)
    flags = ["--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1]
    full = selections(pool0, *flags, out="full")
    done = pool0 / "fill" / "0" / f"{MODEL}-samp" / "T1_s1.pkl"
    done.parent.mkdir(parents=True)
    kept = [x for x in pickle.load(open(pool0 / "full" / "0" / f"{MODEL}-samp" / "T1_s1.pkl", "rb")) if x[0] != bad]
    pickle.dump(kept, open(done, "wb"))
    assert selections(pool0, *flags, out="fill") == full


def test_redraw_skips_pools_that_are_also_degenerate(tmp_path):
    (pool0, pool1), bad = degenerate_pools(tmp_path)
    pool2 = tmp_path / "pool2"
    shutil.copytree(pool1, pool2)
    d = pickle.load(open(pool0 / f"{MODEL}_embeddings.pkl", "rb"))
    other = pickle.load(open(pool1 / f"{MODEL}_embeddings.pkl", "rb"))
    other[1.0][bad] = d[1.0][bad]
    pickle.dump(other, open(pool1 / f"{MODEL}_embeddings.pkl", "wb"))
    good = pickle.load(open(pool2 / f"{MODEL}_embeddings.pkl", "rb"))[1.0][bad]
    got = selections(pool0, "--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1, pool2)
    assert set(got[bad]) <= set(good["u_sents"].tolist())


def test_redraw_stops_when_no_pool_is_valid(tmp_path):
    (pool0, pool1), bad = degenerate_pools(tmp_path)
    shutil.copy(pool0 / f"{MODEL}_embeddings.pkl", pool1 / f"{MODEL}_embeddings.pkl")
    with pytest.raises(RuntimeError, match="no redraw pool"):
        selections(pool0, "--N", "1", "--samp", "--target_Ts", "1", "--redraw_dirs", pool1)

