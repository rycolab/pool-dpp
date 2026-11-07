"""The diagnostics scripts stop with an error instead of skipping missing pools, prompts, files,
or failed requests, and they count the exclusions that are part of what they measure.
Tiny synthetic pools in tmp_path; no GPU, network, models, ambrosia.csv or GSM8K download.

Covers analysis/pool_ceiling.py, weight_diagnostics.py, ess_fineweb_n8000.py, ess_ambrosia_n6000.py,
gsm8k_n2500.py, embedder_probe.py, llm_selection.py, pair_judge.py and dual_vs_primal.py.
embed_instructed.py needs a GPU embedder and has no test here.

Run: PYTHONPATH=src pytest tests/test_fail_loudly_diagnostics.py
"""
import io
import json
import os
import pickle
import sys
import types
import urllib.error
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ANALYSIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
sys.path.insert(0, ANALYSIS)

import dual_vs_primal as dvp  # noqa: E402
import embedder_probe as ep  # noqa: E402
import ess_ambrosia_n6000 as ea  # noqa: E402
import ess_fineweb_n8000 as ef  # noqa: E402
import gsm8k_n2500 as g25  # noqa: E402
import llm_selection as ls  # noqa: E402
import pair_judge as pj  # noqa: E402
import pool_ceiling as pc  # noqa: E402
import weight_diagnostics as wd  # noqa: E402

RNG = np.random.default_rng(0)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def emb_row(n, counts=None, dim=16, prefix="s"):
    E = RNG.normal(size=(n, dim))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return dict(u_sents=np.array([f"{prefix}{i} is a string" for i in range(n)]),
                counts=np.asarray(counts if counts is not None else np.ones(n), dtype=np.int64),
                logps=-RNG.exponential(5, size=n) - 1.0, logps_t1=-RNG.exponential(5, size=n) - 1.0,
                embeddings=E)


# ---------------------------------------------------------------------------
# pool_ceiling.py
# ---------------------------------------------------------------------------
def ambrosia_prompt(q):
    return f"Given the following database dump:\nDUMP\nand the following question:\n{q}\ngenerate the SQL query..."


def test_pool_ceiling_lists_every_missing_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(pc, "load_ambrosia_df", lambda path: pd.DataFrame({"question": ["q"]}))
    monkeypatch.setattr(pc, "expected_questions", lambda df: {"q"})
    dump(str(tmp_path / pc.CACHE / "pool0" / pc.RAW_PKL), {})
    with pytest.raises(FileNotFoundError, match=r"2 of 3 pools .*pool1, pool2"):
        pc.part_a(3, workers=1)


def test_pool_ceiling_pool_missing_a_temperature(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "ROOT_DIR", str(tmp_path))
    dump(str(tmp_path / pc.CACHE / "pool0" / pc.RAW_PKL), {1.0: {}})
    with pytest.raises(ValueError, match="no draws at sT"):
        pc.load_pool(0)


def test_pool_ceiling_gold_query_must_execute():
    dump_sql = "CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (1), (2);"
    q, gs, tbl = pc.signatures_for_question(
        ("q", dump_sql, repr(["SELECT x FROM t"]), ["SELECT x FROM t", "SELEC nonsense"]))
    assert gs == {((1,), (2,))}
    assert tbl["SELEC nonsense"] is None          # not runnable: part of the metric, kept
    with pytest.raises(RuntimeError, match="gold query does not execute"):
        pc.signatures_for_question(("q", dump_sql, repr(["SELECT x FROM t", "SELECT y FROM nope"]), []))
    with pytest.raises(RuntimeError, match="database dump does not load"):
        pc.signatures_for_question(("q", "CREATE TABL broken", repr(["SELECT 1"]), []))


def _part_b_setup(tmp_path, monkeypatch, skip=None):
    """Two pools, two questions, N_RUNS=2, one sT per method; returns the expected output path."""
    monkeypatch.setattr(pc, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(pc, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.setattr(pc, "SIGTABLE", str(tmp_path / "sig.pkl"))
    monkeypatch.setattr(pc, "N_RUNS", 2)
    monkeypatch.setattr(pc, "ST_SWEEP_BASE", [1.0])
    monkeypatch.setattr(pc, "SAMP_TS", [1])
    monkeypatch.setattr(pc, "MAP_TS", [1])
    sqls = ["A", "B", "C", "D"]
    golden = {"q1": {("g1",), ("g2",)}, "q2": {("g3",)}}
    sig_tables = {"q1": {"A": ("g1",), "B": None, "C": ("x",), "D": ("g2",)},
                  "q2": {"A": ("x",), "B": ("y",), "C": None, "D": ("z",)}}
    dump(pc.SIGTABLE, {"golden": golden, "sig_tables": sig_tables})
    ceil = {str(sT): {"q1": {"ceiling_mean": 1.0, "n_golden_sigs": 2},
                      "q2": {"ceiling_mean": 0.0, "n_golden_sigs": 1}} for sT in (1.0, 1.5)}
    with open(tmp_path / "pool_ceiling.json", "w") as f:
        json.dump({"per_question_full_pool": ceil}, f)
    files = [("ancestral", "s1.pkl", [0, 1]), ("kmeans", "s1.pkl", [0, 1]),
             ("samp", "T1_s1.pkl", [0, 1]), ("map", "T1_s1.5.pkl", [0])]
    for i in range(2):
        for r in range(2):
            os.makedirs(tmp_path / pc.CACHE / f"pool{i}" / pc.EST_SUBDIR / str(r), exist_ok=True)
        for m, fname, runs in files:
            for r in runs:
                if skip == (i, m, r):
                    continue
                sel = [(ambrosia_prompt("q1"), sqls[:3], None, None), (ambrosia_prompt("q2"), sqls[1:], None, None)]
                dump(str(tmp_path / pc.CACHE / f"pool{i}" / pc.EST_SUBDIR / str(r)
                         / f"{pc.MODEL}-{m}" / fname), sel)
    return tmp_path / "recall_vs_ceiling.json"


def test_pool_ceiling_part_b_counts_zero_ceiling_questions(tmp_path, monkeypatch):
    out = _part_b_setup(tmp_path, monkeypatch)
    pc.part_b(2)
    res = json.load(open(out))
    row = res["PoolDPP-MAP"][0]
    assert row["n_pools"] == 2 and row["n_runs"] == 1
    assert row["n_questions_zero_ceiling"] == 1          # q2 has ceiling 0: out of the ratio only
    assert row["recall_vs_ceiling"] == pytest.approx(0.5)  # q1: A covers g1 of {g1, g2}
    assert res["PoolDPP-Samp"][0]["n_runs"] == 2


def test_pool_ceiling_part_b_missing_run_file(tmp_path, monkeypatch):
    _part_b_setup(tmp_path, monkeypatch, skip=(1, "kmeans", 1))
    with pytest.raises(FileNotFoundError, match=r"k-Means 1.*1 of 4 selection files missing.*pool1 run 1"):
        pc.part_b(2)


def test_pool_ceiling_selection_must_have_K_strings(tmp_path):
    path = str(tmp_path / "sel.pkl")
    golden = {"q1": {("g1",)}}
    tbl = {"q1": {"A": ("g1",), "B": None}}
    dump(path, [(ambrosia_prompt("q1"), ["A", "B"], None, None)])
    with pytest.raises(ValueError, match="2 selected strings, expected K=3"):
        pc._ratio_for_pkl(path, golden, tbl, {("q1", 1.0): 1.0}, 1.0)
    dump(path, [(ambrosia_prompt("q1"), ["A", "B", "Z"], None, None)])
    with pytest.raises(ValueError, match="not in the signature table"):
        pc._ratio_for_pkl(path, golden, tbl, {("q1", 1.0): 1.0}, 1.0)


# ---------------------------------------------------------------------------
# weight_diagnostics.py
# ---------------------------------------------------------------------------
def test_weight_diagnostics_missing_pool_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(wd, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="gsm8k: missing"):
        wd.main()
    assert not (tmp_path / "weight_diagnostics.json").exists()


def test_weight_diagnostics_counts_single_string_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(wd, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.setattr(wd, "CONFIGS", [("t", "p.pkl", 1.0, 2, True, [1])])
    dump(str(tmp_path / "p.pkl"), {1.0: {"a": emb_row(3, [2, 1, 1]), "b": emb_row(1, [4]), "c": emb_row(2, [3, 1])}})
    wd.main()
    o = json.load(open(tmp_path / "weight_diagnostics.json"))["t|sT=1|T=1"]
    assert (o["n_prompts"], o["n_prompts_in_pool"], o["n_prompts_excluded_lt2_unique"]) == (2, 3, 1)
    bad = emb_row(2, [3, 1])
    bad["logps"] = np.array([np.nan, -1.0])
    dump(str(tmp_path / "p.pkl"), {1.0: {"a": bad}})
    with pytest.raises(ValueError, match="not finite"):
        wd.main()


# ---------------------------------------------------------------------------
# ess_fineweb_n8000.py and ess_ambrosia_n6000.py
# ---------------------------------------------------------------------------
def test_ess_fineweb_checks_draws_and_counts_exclusions():
    per = {"a": emb_row(2, [3, 1]), "b": emb_row(1, [4])}
    ess, n_lt2 = ef.ess_rows(per, 1.0, 4, "x")
    assert len(ess) == 1 and n_lt2 == 1
    assert ef.summarize((ess, n_lt2), 4)["n_prompts_excluded_lt2_unique"] == 1
    with pytest.raises(ValueError, match="expected N=5"):
        ef.ess_rows(per, 1.0, 5, "x")
    with pytest.raises(ValueError, match="different prompts"):
        ef.merge_two({"a": per["a"]}, {"b": per["b"]})


def test_bound_rows_reject_empty_or_nonfinite(tmp_path):
    p = str(tmp_path / "raw.json")
    for mod in (ef, ea, g25):
        json.dump({"_meta": {"x": 1}, "t|T=1": [{"chi2_1": 0.5, "r1": 2.0}]}, open(p, "w"))
        assert [k for k, _ in mod.bound_rows(p)] == ["t|T=1"]
        json.dump({"t|T=1": []}, open(p, "w"))
        with pytest.raises(ValueError, match="non-empty list"):
            mod.bound_rows(p)
        json.dump({"t|T=1": [{"chi2_1": float("nan"), "r1": 2.0}]}, open(p, "w"))
        with pytest.raises(ValueError, match="no finite"):
            mod.bound_rows(p)


def test_ess_fineweb_missing_bound_raw_raises_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(ef, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(ef, "SCRIPT_DIR", str(tmp_path))
    rows = {"pool0": [399, 1], "ha": [3999, 1], "hb": [3999, 1]}
    for cache, counts in rows.items():
        r = emb_row(2, counts)
        r["u_sents"] = np.array(["x one", "x two"])
        dump(str(tmp_path / cache / ef.FNAME), {1.0: {"p": r}})
    a = SimpleNamespace(halves=["ha", "hb"], pool0_cache="pool0", bound_raw="nope.json")
    with pytest.raises(FileNotFoundError, match="nope.json missing"):
        ef.run(a)
    assert not (tmp_path / "ess_fineweb_n8000.json").exists()


def test_ess_ambrosia_lists_missing_pools_and_checks_draws(tmp_path, monkeypatch):
    monkeypatch.setattr(ea, "ROOT_DIR", str(tmp_path))
    dump(str(tmp_path / "c" / "pool0" / ea.FNAME), {1.0: {"p": emb_row(2, [3, 1])}})
    with pytest.raises(FileNotFoundError, match=r"2 of 3 pools missing: pool1, pool2"):
        ea.merge("c", 1.0, 3)
    dump(str(tmp_path / "c" / "pool1" / ea.FNAME), {1.0: {"q": emb_row(2, [3, 1])}})
    with pytest.raises(ValueError, match="prompts differ from pool0"):
        ea.merge("c", 1.0, 2)
    per = {"a": emb_row(2, [3, 1]), "b": emb_row(2, [2, 1]), "c": emb_row(1, [4])}
    with pytest.raises(ValueError, match="different numbers of draws"):
        ea.ess_summary(per, 1.0, True, "x")
    del per["b"]
    assert ea.ess_summary(per, 1.0, True, "x")["n_prompts_excluded_lt2_unique"] == 1


# ---------------------------------------------------------------------------
# gsm8k_n2500.py
# ---------------------------------------------------------------------------
def _gsm_setup(tmp_path, monkeypatch, n_questions=50, drop=0):
    qs = [f"question {i}" for i in range(n_questions)]
    monkeypatch.setattr(g25, "load_gsm8k_dataset",
                        lambda: pd.DataFrame({"question": qs, "answer": ["#### 1"] * n_questions}))
    monkeypatch.setattr(g25, "is_correct", lambda s, gold: False)
    monkeypatch.setattr(g25, "ROOT_DIR", str(tmp_path))
    prompts = [g25.build_math_prompt(q) for q in qs]
    for c in g25.CACHES:
        dump(str(tmp_path / c / g25.EMB_FNAME),
             {g25.ST: {p: emb_row(12, np.full(12, 2)) for p in prompts[drop:]}})
    raw = str(tmp_path / "raw.json")
    json.dump({"gsm8k|T=1": [{"chi2_1": 0.5, "r1": 2.0}]}, open(raw, "w"))
    out = str(tmp_path / "out.json")
    monkeypatch.setattr(sys, "argv", ["gsm8k_n2500.py", "--bound-raw", raw, "--out", out])
    return out


def test_gsm8k_n2500_missing_question_raises(tmp_path, monkeypatch):
    _gsm_setup(tmp_path, monkeypatch, drop=1)
    with pytest.raises(ValueError, match="1 of the 50 GSM8K questions missing"):
        g25.main()


def test_gsm8k_n2500_short_map_set_raises(tmp_path, monkeypatch):
    out = _gsm_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(g25, "greedy_map", lambda log_s, emb, k: [0])
    with pytest.raises(RuntimeError, match="PoolDPP-MAP .*Zhat = 0"):
        g25.main()
    assert not os.path.exists(out)


def test_gsm8k_n2500_short_samp_draw_raises(tmp_path, monkeypatch):
    out = _gsm_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(g25, "sample_dpp", lambda log_s, emb, k, N=1: [list(range(k))] * (N - 1) + [[]])
    with pytest.raises(RuntimeError, match="PoolDPP-Samp .*1 of 5 draws"):
        g25.main()
    assert not os.path.exists(out)


# ---------------------------------------------------------------------------
# embedder_probe.py
# ---------------------------------------------------------------------------
GSM_CHAIN = "1. We need 2 + 3 = 5 apples.\nSo the total is 5.\n#### 5"


def test_embedder_probe_requires_sT1_and_counts_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, "ROOT_DIR", str(tmp_path))
    path = str(tmp_path / "cache-gsm8k" / "pool0" / "google__gemma-2-9b-it.pkl")
    dump(path, {0.5: {"p": ([GSM_CHAIN], [], [], [])}})
    with pytest.raises(ValueError, match="no draws at sT=1.0"):
        ep.build_gsm8k_pairs(5, ep.random.Random(0))
    dump(path, {1.0: {"p1": ([GSM_CHAIN, "no answer here"], [], [], []),
                      "p2": ([GSM_CHAIN], [], [], [])}})
    filters = Counter()
    pairs = ep.build_gsm8k_pairs(5, ep.random.Random(0), filters)
    assert len(pairs) == 2 and filters["no_answer_or_no_arithmetic"] == 1
    assert {p["prompt"] for p in pairs} == {"p1", "p2"}   # n_questions counts these, not items


def test_embedder_probe_sql_prompt_without_question_raises(tmp_path, monkeypatch):
    import lm_dpp.utils.ambrosia_utils as au
    monkeypatch.setattr(ep, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(au, "load_ambrosia_df", lambda p: pd.DataFrame({"question": ["other"], "db_dump": [""]}))
    dump(str(tmp_path / "cache-ambrosia" / "pool0" / "google__gemma-2-9b-it.pkl"),
         {1.0: {ambrosia_prompt("q1"): (["SELECT 1"], [], [], [])}})
    with pytest.raises(ValueError, match="no question in ambrosia.csv"):
        ep.build_sql_pairs(5, ep.random.Random(0))


def test_embedder_probe_auc_of_nothing_raises():
    assert ep.auc([2.0, 3.0], [1.0]) == 1.0
    with pytest.raises(ValueError, match="AUC of 0"):
        ep.auc([], [1.0])


# ---------------------------------------------------------------------------
# llm_selection.py (vLLM replaced by a stub)
# ---------------------------------------------------------------------------
class _StubLLM:
    def __init__(self, **kw):
        pass

    def get_tokenizer(self):
        return SimpleNamespace(apply_chat_template=lambda msgs, **kw: msgs[0]["content"])

    def generate(self, chats, params, use_tqdm=False):
        reply = SimpleNamespace(text=json.dumps(list(range(ls.K))))
        return [SimpleNamespace(outputs=[reply]) for _ in chats]


def _llm_setup(tmp_path, monkeypatch, pools):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_StubLLM, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(ls, "ROOT_DIR", str(tmp_path))
    for p in pools:
        dump(str(tmp_path / p / ls.EMBEDDINGS), {ls.ST: {"prompt a": emb_row(45, prefix="a"),
                                                         "prompt b": emb_row(45, prefix="b")}})
    out = str(tmp_path / "out.json")
    monkeypatch.setattr(sys, "argv", ["llm_selection.py", "--out", out, "--pools"] + pools)
    return out


def test_llm_selection_default_pools_must_all_exist(tmp_path, monkeypatch):
    _llm_setup(tmp_path, monkeypatch, ["cache-fineweb"])
    monkeypatch.setattr(sys, "argv", ["llm_selection.py"])
    with pytest.raises(FileNotFoundError, match="9 of 10 pools"):
        ls.main()


def test_llm_selection_short_map_set_raises(tmp_path, monkeypatch):
    out = _llm_setup(tmp_path, monkeypatch, ["p0", "p1"])
    ls.main()
    res = json.load(open(out))
    assert res["unparsed_replies_per_pool"] == [0, 0] and res["results"]["LLM"]["n_pools"] == 2
    monkeypatch.setattr(ls, "greedy_map", lambda log_s, emb, k: [0, 1])
    with pytest.raises(RuntimeError, match="PoolDPP-MAP, full pool returned 2 strings"):
        ls.main()


def test_llm_selection_no_fallback_to_short_strings():
    u = np.array(["short"] * 5 + [f"a long string {i}" for i in range(39)], dtype=object)
    with pytest.raises(ValueError, match="39 strings of at least 10 characters"):
        ls.spaced_by_rank(u, np.zeros(len(u)))


# ---------------------------------------------------------------------------
# pair_judge.py (the Gemini API replaced by a stub)
# ---------------------------------------------------------------------------
def _stub_urlopen(monkeypatch, reply=None, http_code=None):
    calls = []

    def urlopen(req, timeout=None):
        calls.append(1)
        if http_code is not None:
            raise urllib.error.HTTPError(req.full_url, http_code, "busy", {}, None)
        body = json.dumps({"candidates": [{"content": {"parts": [{"text": reply}]}}]}).encode()
        return io.BytesIO(body)

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(pj.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(pj.time, "sleep", lambda s: None)
    return calls


def test_pair_judge_gives_up_with_an_error(monkeypatch):
    calls = _stub_urlopen(monkeypatch, http_code=503)
    with pytest.raises(pj.JudgeError, match="1 of 1 judge requests failed.*no reply after 4 tries"):
        pj.judge_pairs([("x", "y")], "m", workers=1)
    assert len(calls) == 4


def test_pair_judge_unparseable_reply_is_not_guessed(monkeypatch):
    _stub_urlopen(monkeypatch, reply="BECAUSE")       # used to read as a letter inside the word
    assert pj.judge_pairs([("x", "y")], "m", workers=1) == [pj.UNPARSEABLE]
    _stub_urlopen(monkeypatch, reply="**B**")
    assert pj.judge_pairs([("x", "y")], "m", workers=1) == ["same_answer_diff_derivation"]


def _selection(path, n=pj.N_QUESTIONS, k=pj.K):
    sets = [(f"q{i}", [f"{j} + 1 = {j + 1}\n#### 7" for j in range(k)], None, None) for i in range(n)]
    dump(path, sets)
    return sets


def test_pair_judge_load_sets_checks_shape(tmp_path):
    p = str(tmp_path / "s.pkl")
    _selection(p, k=9)
    with pytest.raises(ValueError, match="9 strings, expected K=10"):
        pj.load_sets(p)
    _selection(p, n=49)
    with pytest.raises(ValueError, match="49 records for 49 prompts"):
        pj.load_sets(p)


def test_pair_judge_score_stops_on_contradicting_verdict(tmp_path, monkeypatch):
    p = str(tmp_path / "s.pkl")
    sets = _selection(p)
    keys = [pj.pair_key(x, y) for _, s, _, _ in sets for x, y in pj.combinations(s, 2)]
    cache = {k: "surface_only" for k in keys}
    verdicts, out = str(tmp_path / "v.json"), str(tmp_path / "out.json")
    a = SimpleNamespace(pool=0, run=0, selection=[("M", p)], verdicts=verdicts, model="m", out=out)
    json.dump(cache, open(verdicts, "w"))
    pj.cmd_score(a)
    res = json.load(open(out))["M"]
    assert res["n_pairs"] == len(keys) and res["n_pairs_no_answer"] == 0
    os.remove(out)
    cache[keys[0]] = "different_answer"
    cache[keys[1]] = pj.UNPARSEABLE
    json.dump(cache, open(verdicts, "w"))
    with pytest.raises(ValueError, match=r"judge_says_different_answer.: 50, .unparseable_reply.: 50"):
        pj.cmd_score(a)
    assert not os.path.exists(out)


# ---------------------------------------------------------------------------
# dual_vs_primal.py
# ---------------------------------------------------------------------------
def test_dual_vs_primal_setting_needs_its_own_grid_row(tmp_path, monkeypatch):
    monkeypatch.setattr(dvp, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.setattr(dvp, "N_GRID", [25])
    monkeypatch.setattr(dvp, "D_GRID", [16])
    monkeypatch.setattr(dvp, "SETTINGS", [("x", "x", 16, 25)])
    dvp.main()
    assert json.load(open(tmp_path / "dual_vs_primal.json"))["settings"]["x"]["N"] == 25
    monkeypatch.setattr(dvp, "SETTINGS", [("x", "x", 16, 30)])   # used to take the nearest N
    with pytest.raises(ValueError, match="0 rows at D=16, N=30"):
        dvp.main()
    with pytest.raises(RuntimeError, match="returned 3 items"):
        dvp.full([0, 1, 2], "stable", 16, 25)
