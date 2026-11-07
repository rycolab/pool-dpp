"""The frontier, ablation, significance and N-dial scripts stop on missing or broken data instead
of skipping it. Tiny fake pools under tmp_path; no caches, network, GPU or LM.

Run: PYTHONPATH=src pytest tests/test_fail_loudly_plots.py
"""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ANALYSIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
sys.path.insert(0, ANALYSIS)

import plot_ablations as pab  # noqa: E402
import plot_gsm8k as pg  # noqa: E402
import plot_intrinsic as pi  # noqa: E402
import plot_ndial as pn  # noqa: E402
import significance as sig  # noqa: E402


@pytest.fixture(scope="module")
def pa():
    """plot_ambrosia reads ambrosia.csv at import; give it an empty frame instead."""
    import lm_dpp.utils.ambrosia_utils as au
    orig = au.load_ambrosia_df
    au.load_ambrosia_df = lambda path: pd.DataFrame(columns=["question", "db_dump", "ambig_queries"])
    try:
        import plot_ambrosia
    finally:
        au.load_ambrosia_df = orig
    return plot_ambrosia


def write_sel(path, prompts, k, seed=0, sents=None):
    """A selection pkl: (prompt, sents, logps_t1, embs) per prompt, k unit-norm rows each."""
    rng = np.random.default_rng(seed)
    items = []
    for pt in prompts:
        emb = rng.standard_normal((k, k + 2))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        ss = sents(pt) if sents else [f"{pt} answer {j}" for j in range(k)]
        items.append((pt, ss, list(-1.0 - rng.random(k)), list(emb)))
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(items, f)
    return path


def write_sidecar(pkl, k, n_prompts, redrawn=(), sampler=None):
    """Sidecar as src/lm_dpp/estimation.py:write_redraws writes it; sampler=None as for greedy MAP."""
    with open(os.path.splitext(pkl)[0] + ".redraws.json", "w") as f:
        json.dump({"selection_file": os.path.basename(pkl), "method": "test", "k": k,
                   "sampling_T": 1.0, "pool_size": None, "n_prompts": n_prompts,
                   "n_redrawn": len(redrawn),
                   "redrawn": [{"prompt": p, "rank": 1, "n_unique": 1, "from": "other"}
                               for p in redrawn], "sampler": sampler}, f)


def sampler_counts(n_flat, n_drop, n_tiered, bound):
    return dict(n_flat=n_flat, n_drop=n_drop, n_tiered=n_tiered, n_unknown=0, n_rows_dropped=n_drop,
                max_log10_drop_bound=bound, min_tier_gap_nats=None)


GSM_PROMPTS = {f"Question: q{i}\nAnswer:": f"work #### {i}" for i in range(3)}


def gsm_pools(root, n_pools, runs, files, k=10, skip=()):
    """cache-gsm8k/pool{i}/est10qwen/{run}/{sub}/{fname} for (sub, fname) in files."""
    for i in range(n_pools):
        for r in runs:
            os.makedirs(os.path.join(root, "cache-gsm8k", f"pool{i}", "est10qwen", str(r)), exist_ok=True)
            for sub, fname in files:
                if (i, r, fname) in skip:
                    continue
                write_sel(os.path.join(root, "cache-gsm8k", f"pool{i}", "est10qwen", str(r), sub, fname),
                          GSM_PROMPTS, k, seed=100 * i + r)


# --------------------------------------------------------------------------- plot_gsm8k
def test_gsm8k_missing_pools_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "ROOT_DIR", str(tmp_path))
    gsm_pools(tmp_path, 1, [0], [("m-ancestral", "s1.pkl")])
    with pytest.raises(FileNotFoundError, match=r"2 of 3 pools missing.*pool1\.\.pool2"):
        pg.pool_dirs(3, "est10qwen")


def test_gsm8k_missing_run_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "ROOT_DIR", str(tmp_path))
    gsm_pools(tmp_path, 2, [0, 1], [("m-ancestral", "s1.pkl")], skip={(1, 1, "s1.pkl")})
    with pytest.raises(FileNotFoundError, match="1 of 2 run files missing"):
        pg.collect_pool_level(pg.pool_dirs(2, "est10qwen"), "m-ancestral", [1],
                              lambda v: "s1.pkl", pg.make_metrics(GSM_PROMPTS), "pool_t")


def test_gsm8k_unequal_runs_across_pools_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "ROOT_DIR", str(tmp_path))
    gsm_pools(tmp_path, 2, [0, 1], [("m-ancestral", "s1.pkl")])
    gsm_pools(tmp_path, 1, [2], [("m-ancestral", "s1.pkl")])  # pool0 has runs 0..2, pool1 0..1
    with pytest.raises(ValueError, match="different numbers of run dirs"):
        pg.collect_pool_level(pg.pool_dirs(2, "est10qwen"), "m-ancestral", [1],
                              lambda v: "s1.pkl", pg.make_metrics(GSM_PROMPTS), "pool_t")


def test_gsm8k_map_reads_run0_and_redraw_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "ROOT_DIR", str(tmp_path))
    gsm_pools(tmp_path, 2, [0, 1], [("m-ancestral", "s1.pkl")])      # run dirs 0, 1
    gsm_pools(tmp_path, 2, [0], [("m-map", "T10_s1.2.pkl")])          # MAP only in run 0
    pdirs = pg.pool_dirs(2, "est10qwen")
    for d in pdirs:
        write_sidecar(os.path.join(d, "0", "m-map", "T10_s1.2.pkl"), 10, 3,
                      redrawn=["Question: q0\nAnswer:"] if d == pdirs[1] else ())
    out = pg.collect_pool_level(pdirs, "m-map", [10], lambda v: "T10_s1.2.pkl",
                                pg.make_metrics(GSM_PROMPTS), "pool_t", deterministic=True,
                                redraws=(10, 3))
    assert out[10]["n_runs"] == 1 and out[10]["n"] == 2
    assert out[10]["n_redrawn"] == 1 and out[10]["n_prompt_pools"] == 6
    assert out[10]["sampler_tier_paths"] is None   # greedy MAP: no sampler
    os.remove(os.path.join(pdirs[0], "0", "m-map", "T10_s1.2.redraws.json"))
    with pytest.raises(FileNotFoundError, match="rerun dpp-estimation"):
        pg.collect_pool_level(pdirs, "m-map", [10], lambda v: "T10_s1.2.pkl",
                              pg.make_metrics(GSM_PROMPTS), "pool_t", deterministic=True,
                              redraws=(10, 3))


def test_gsm8k_one_pool_ci_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "ROOT_DIR", str(tmp_path))
    gsm_pools(tmp_path, 1, [0, 1], [("m-ancestral", "s1.pkl")])
    with pytest.raises(ValueError, match="needs at least 2"):
        pg.collect_pool_level(pg.pool_dirs(1, "est10qwen"), "m-ancestral", [1],
                              lambda v: "s1.pkl", pg.make_metrics(GSM_PROMPTS), "pool_t")


@pytest.mark.parametrize("prompts,k,match", [
    (list(GSM_PROMPTS)[:2], 10, "1 expected prompts missing"),           # a dropped question
    (list(GSM_PROMPTS) + ["Question: other\nAnswer:"], 10, "1 unexpected"),  # used to be skipped
    (list(GSM_PROMPTS), 9, "expected K=10"),                             # a short selection
])
def test_gsm8k_selection_must_cover_questions_with_k(tmp_path, prompts, k, match):
    p = write_sel(tmp_path / "s.pkl", prompts, k)
    with open(p, "rb") as f:
        data = pickle.load(f)
    with pytest.raises(ValueError, match=match):
        pg.make_metrics(GSM_PROMPTS)(data, p)


def test_gsm8k_malformed_record_raises():
    with pytest.raises(ValueError, match="3 fields, expected 4"):
        pg.check_selection([("p", ["a"], [0.0])], {"p"}, 1, "f.pkl")


def test_logdet_failure_raises():
    bad = np.full((3, 4), np.nan)
    for mod in (pg, pi):
        with pytest.raises(ValueError, match="log-determinant failed"):
            mod.compute_logdet(bad)


# --------------------------------------------------------------------------- plot_ambrosia
AMB_ROW = pd.Series({"question": "Which x?",
                     "db_dump": "CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1), (2), (3);",
                     "ambig_queries": "['SELECT x FROM t WHERE x = 1', 'SELECT x FROM t WHERE x = 2']"})
AMB_PT = "Given the following database dump:\n...\nand the following question:\nWhich x?\ngenerate the SQL query"


@pytest.fixture
def amb(pa, monkeypatch):
    monkeypatch.setattr(pa, "_PROMPT_ROWS", {AMB_PT: AMB_ROW})
    monkeypatch.setattr(pa, "_PROMPT_ENV", {})
    return pa


def amb_items(sents):
    return [(AMB_PT, sents, [-1.0] * len(sents), [np.eye(4)[j] for j in range(len(sents))])]


def test_ambrosia_failed_query_is_a_miss(amb):
    # one golden hit, one wrong result, one query that fails to run: precision 1/3, recall 1/2
    pp = amb.per_run_prompt_metrics(amb_items(["SELECT x FROM t WHERE x = 1", "SELECT x FROM t",
                                               "SELEC broken"]))
    assert pp[AMB_PT] == pytest.approx((1 / 3, 1 / 2))


def test_ambrosia_failing_golden_query_raises(amb, monkeypatch):
    row = AMB_ROW.copy()
    row["ambig_queries"] = "['SELECT x FROM t WHERE x = 1', 'SELECT nope FROM t']"
    monkeypatch.setattr(amb, "_PROMPT_ROWS", {AMB_PT: row})
    with pytest.raises(ValueError, match="golden query 'SELECT nope FROM t' failed to run"):
        amb.per_run_prompt_metrics(amb_items(["SELECT 1", "SELECT 2", "SELECT 3"]))


def test_ambrosia_question_without_golden_raises(amb, monkeypatch):
    row = AMB_ROW.copy()
    row["ambig_queries"] = "[]"
    monkeypatch.setattr(amb, "_PROMPT_ROWS", {AMB_PT: row})
    with pytest.raises(ValueError, match="no golden queries"):
        amb.per_run_prompt_metrics(amb_items(["SELECT 1", "SELECT 2", "SELECT 3"]))


def test_ambrosia_unexpected_prompt_raises(amb):
    items = amb_items(["SELECT 1", "SELECT 2", "SELECT 3"]) + [("other prompt", ["a", "b", "c"], [0.0] * 3,
                                                                [np.eye(4)[j] for j in range(3)])]
    with pytest.raises(ValueError, match="1 unexpected"):
        amb.per_run_prompt_metrics(items)


def test_ambrosia_missing_pools_and_redraw_counts(amb, tmp_path, monkeypatch):
    monkeypatch.setattr(amb, "ROOT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"ambrosia: 2 of 2 pools missing.*pool0\.\.pool1"):
        amb.pool_dirs(2)
    sents = lambda pt: ["SELECT x FROM t WHERE x = 1", "SELECT x FROM t WHERE x = 2", "SELECT 3"]  # noqa: E731
    for i in range(2):
        for r in (0, 1):
            p = write_sel(tmp_path / "cache-ambrosia" / f"pool{i}" / "est3minilm" / str(r) / "m-samp" / "T1_s1.pkl",
                          [AMB_PT], 3, sents=sents)
            if r == 0:
                write_sidecar(p, 3, 1, redrawn=[AMB_PT] if i == 0 else (),
                              sampler=sampler_counts(1 - i, i, 0, -40.0 if i else None))
    out = amb.collect_pool_level(amb.pool_dirs(2), "m-samp", [1], lambda t: "T1_s1.pkl", "pool_t",
                                 redraws=True)
    assert out[1]["x_mean"] == pytest.approx(1.0) and out[1]["y_mean"] == pytest.approx(2 / 3)
    assert (out[1]["n_redrawn"], out[1]["n_prompt_pools"], out[1]["n_runs"]) == (1, 2, 2)
    assert out[1]["sampler_tier_paths"] == dict(n_flat=1, n_drop=1, n_tiered=0, n_unknown=0,
                                                max_log10_drop_bound=-40.0)


# --------------------------------------------------------------------------- plot_intrinsic


def fw_prompts():
    from lm_dpp.utils.fineweb_prompts import prompts
    return prompts


def test_intrinsic_default_pools_missing_or_extra_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(pi, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(pi, "EST_DIR", str(tmp_path / "cache-fineweb" / "est10minilm"))
    with pytest.raises(FileNotFoundError, match="10 of 10 est dirs missing"):
        pi.default_est_dirs()
    os.makedirs(tmp_path / "cache-fineweb" / "est10minilm")
    for i in range(1, 11):   # pool10 is not one of pool1..pool9
        os.makedirs(tmp_path / "cache-fineweb-pools" / f"pool{i}" / "est10minilm")
    with pytest.raises(ValueError, match="outside pool1..pool9"):
        pi.default_est_dirs()


def test_intrinsic_short_set_raises(tmp_path):
    # sets with fewer than 2 strings used to be skipped; any set with != K strings now raises
    p = write_sel(tmp_path / "s.pkl", fw_prompts(), 1)
    with open(p, "rb") as f:
        data = pickle.load(f)
    with pytest.raises(ValueError, match="expected K=10"):
        pi.metrics(data, p)


def test_intrinsic_run_mode_ddof1_and_deterministic_map(tmp_path):
    est = tmp_path / "est10minilm"
    for r in range(3):
        write_sel(est / str(r) / "m-samp" / "T1_s1.pkl", fw_prompts(), 10, seed=r)
        write_sidecar(str(est / str(r) / "m-samp" / "T1_s1.pkl"), 10, 50, sampler=sampler_counts(50, 0, 0, None))
    write_sel(est / "0" / "m-map" / "T50_s1.pkl", fw_prompts(), 10, seed=9)
    write_sidecar(str(est / "0" / "m-map" / "T50_s1.pkl"), 10, 50)
    samp = pi.collect(str(est), "m-samp", [1], lambda t: "T1_s1.pkl", redraws=True)[1]
    xs = [pi.metrics(pickle.load(open(est / str(r) / "m-samp" / "T1_s1.pkl", "rb")))[1] for r in range(3)]
    assert samp["x_std"] == pytest.approx(np.std(xs, ddof=1))
    assert samp["x_ci"] == pytest.approx(pi.ci95(np.std(xs, ddof=1), 3))
    assert samp["n_redrawn"] == 0 and samp["n_prompt_pools"] == 50
    assert samp["sampler_tier_paths"]["n_flat"] == 50
    mp = pi.collect(str(est), "m-map", [50], lambda t: "T50_s1.pkl", deterministic=True)[50]
    assert mp["deterministic"] == 1 and mp["n"] == 1 and mp["x_ci"] == 0.0
    with pytest.raises(ValueError, match="at least 2"):
        pi.ci95(1.0, 1)


# --------------------------------------------------------------------------- plot_ablations
def abl_est(tmp_path, n_runs, k=10, model="M", skip_run=None):
    est = tmp_path / "est"
    for r in range(n_runs):
        os.makedirs(est / str(r), exist_ok=True)
        if r != skip_run:
            write_sel(est / str(r) / f"{model}-samp" / "T0.818182_s1.pkl", fw_prompts(), k, seed=r)
    write_sel(est / "0" / f"{model}-map" / "T50_s1.pkl", fw_prompts(), k, seed=7)
    return est


def test_ablations_ddof1_and_map_without_interval(tmp_path):
    est = abl_est(tmp_path, 3)
    qm, qs, dm, ds, n = pab.collect(est, "samp", model="M", n_runs=3)
    runs = [pab.per_run_metrics(pickle.load(open(est / str(r) / "M-samp" / "T0.818182_s1.pkl", "rb")), 10)
            for r in range(3)]
    assert n == 3 and qs == pytest.approx(np.std([q for q, _ in runs], ddof=1))
    assert ds == pytest.approx(np.std([d for _, d in runs], ddof=1))
    qm, qs, dm, ds, n = pab.collect(est, "map", model="M", n_runs=3, T=50)
    assert n == 1 and qs is None and ds is None and pab.half_width(qs, n) == 0.0


def test_ablations_missing_run_file_raises(tmp_path):
    est = abl_est(tmp_path, 3, skip_run=2)
    with pytest.raises(FileNotFoundError, match="missing"):
        pab.collect(est, "samp", model="M", n_runs=3)


def test_ablations_run_dirs_must_match(tmp_path):
    est = abl_est(tmp_path, 3)
    with pytest.raises(FileNotFoundError, match=r"run dirs are \[0, 1, 2\], expected 0..9"):
        pab.collect(est, "samp", model="M")


def test_ablations_multimodel_missing_lm_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pab, "BASE", tmp_path)
    os.makedirs(tmp_path / "est10minilm" / "0" / "google__gemma-2-9b-it_minilm-samp")
    with pytest.raises(FileNotFoundError, match="Qwen__Qwen3-8B"):
        pab.multimodel()


# --------------------------------------------------------------------------- significance
def test_significance_gsm8k_missing_pool_raises(tmp_path, monkeypatch):
    import lm_dpp.utils.math_utils as mu
    monkeypatch.setattr(sig, "ROOT_DIR", str(tmp_path))
    df = pd.DataFrame({"question": [f"q{i}" for i in range(50)], "answer": ["#### 1"] * 50})
    monkeypatch.setattr(mu, "load_gsm8k_dataset", lambda: df)
    os.makedirs(tmp_path / "cache-gsm8k" / "pool0" / sig.GSM_EST)
    with pytest.raises(FileNotFoundError, match="gsm8k: 1 of 2 pools missing"):
        sig.collect_gsm8k(2)


def test_significance_ambrosia_query_not_in_table_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sig, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(sig, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.setattr(sig, "AMB_METHODS", [("Ancestral", 1.0, "m-ancestral", "s1.pkl")])
    gs = {((1,),)}
    with open(tmp_path / "pool_ceiling_sigtable.pkl", "wb") as f:
        pickle.dump({"golden": {"Which x?": gs},
                     "sig_tables": {"Which x?": {"SELECT 1": ((1,),), "SELECT 2": None}}}, f)
    write_sel(tmp_path / "cache-ambrosia" / "pool0" / sig.AMB_EST / "0" / "m-ancestral" / "s1.pkl",
              [AMB_PT], 3, sents=lambda pt: ["SELECT 1", "SELECT 2", "SELECT 3"])
    with pytest.raises(KeyError, match="not in the signature table"):
        sig.collect_ambrosia(1)


def test_significance_analyze_missing_or_unmatched_cells_raise():
    rows = [(m, "1", p, 0, q, 1.0, 0.5) for m in ("A", "B") for p in (0, 1) for q in ("x", "y")]
    cells = sig.to_cells(rows)
    with pytest.raises(KeyError, match="no cells"):
        sig.analyze("t", cells, [("A vs C", ("A", "1"), ("C", "1"))], "")
    del cells[("B", "1")][(1, "y")]
    with pytest.raises(ValueError, match="different"):
        sig.analyze("t", cells, [("A vs B", ("A", "1"), ("B", "1"))], "")
    with pytest.raises(ValueError, match="at least 2 pairs"):
        sig.paired_test([1.0], [0.0])


# --------------------------------------------------------------------------- plot_ndial
def test_ndial_missing_n2500_raises(tmp_path, monkeypatch):
    with open(os.path.join(ANALYSIS, "bound_raw_gsm8k.json")) as f:
        raw = f.read()
    (tmp_path / "bound_raw_gsm8k.json").write_text(raw)
    monkeypatch.setattr(pn, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["plot_ndial.py", "--out", str(tmp_path / "n.pdf")])
    with pytest.raises(FileNotFoundError, match="gsm8k_n2500.json"):
        pn.main()
    (tmp_path / "gsm8k_n2500.json").write_text(json.dumps({"bound_m1|gsm8k|T=1": {"2500": 0.1}}))
    with pytest.raises(KeyError, match=r"bound_m1\|gsm8k\|T=1.05"):
        pn.main()
