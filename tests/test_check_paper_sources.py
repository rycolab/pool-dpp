"""Checks for analysis/check_paper_sources.py on a throwaway git repository.

Run: pytest tests/test_check_paper_sources.py
"""
import io
import json
import os
import subprocess
import sys

ANALYSIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
sys.path.insert(0, ANALYSIS)

import check_paper_sources as cps  # noqa: E402

TEX = r"""
%% SOURCE: analysis/committed.json:{"a|T=1"}.mean and changed.json (bare name).
%% SOURCE: analysis/missing.json. FINAL.
See \texttt{analysis/untracked\_file.json} and git da8fb05:analysis/staged.json:key.
%% SOURCE: analysis/x_{a,b}_all.json and analysis/<task>.json and bound_raw_*.json
cache-fineweb/est10/out.json is not an analysis output; analysis/plot.py is not JSON.
%% SOURCE: analysis/e8_baselines_fineweb.json:{"est:A|sT=1","x"}; analysis/x23_probe_full.json:results.{a}
%% SOURCE: analysis/e4b_length_split.json:{gsm8k_sT1.2_correct, b}.x and analysis/x9_other.json:"<t>|a"
"""

COMMITTED = {
    "committed.json": {"a|T=1": {}},
    "changed.json": {},
    "baselines_fineweb.json": {"est:A|sT=1": {}},
    "beam_baselines_fineweb.json": {"est:A|sT=1": {}},
    "embedder_probe.json": {"results": {}},
    "length_split.json": {"other": {}},
}


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    for name, data in COMMITTED.items():
        (repo / "analysis" / name).write_text(json.dumps(data))
    git(repo, "init", "-q")
    git(repo, "add", "analysis")
    git(repo, "commit", "-q", "-m", "init")
    (repo / "analysis" / "changed.json").write_text('{"new": 1}')
    (repo / "analysis" / "untracked_file.json").write_text("{}")
    (repo / "analysis" / "staged.json").write_text("{}")
    git(repo, "add", "analysis/staged.json")
    tex = tmp_path / "paper.tex"
    tex.write_text(TEX)
    return repo, tex


def test_citations(tmp_path):
    _, tex = make_repo(tmp_path)
    found = cps.citations([tex])
    assert set(found) == {
        "analysis/committed.json", "analysis/changed.json", "analysis/missing.json",
        "analysis/untracked_file.json", "analysis/staged.json",
        "analysis/x_{a,b}_all.json", "analysis/<task>.json", "analysis/bound_raw_*.json",
        "analysis/e8_baselines_fineweb.json", "analysis/x23_probe_full.json",
        "analysis/e4b_length_split.json", "analysis/x9_other.json",
    }
    assert found["analysis/committed.json"] == [("paper.tex", 2, False, "a|T=1")]
    assert found["analysis/changed.json"] == [("paper.tex", 2, True, None)]
    assert found["analysis/missing.json"] == [("paper.tex", 3, False, None)]
    assert found["analysis/staged.json"] == [("paper.tex", 4, False, "key")]
    assert found["analysis/x23_probe_full.json"][0][3] == "results"
    assert found["analysis/e4b_length_split.json"][0][3] == "gsm8k_sT1.2_correct"
    assert found["analysis/x9_other.json"][0][3] is None  # templated key


def test_status_and_report(tmp_path):
    repo, tex = make_repo(tmp_path)
    found = cps.citations([tex])
    checkable = [p for p in found if not cps.is_template(p)]
    status = cps.classify(checkable, repo)
    assert status["analysis/committed.json"] == ("committed", "")
    assert status["analysis/changed.json"] == ("uncommitted", "changed since last commit")
    assert status["analysis/missing.json"] == ("missing", "")
    assert status["analysis/untracked_file.json"] == ("uncommitted", "untracked")
    assert status["analysis/staged.json"] == ("uncommitted", "staged, not committed")

    committed = cps.committed_json(repo)
    assert "analysis/changed.json" not in committed  # changed since HEAD
    cand = lambda p: cps.candidates(p, found[p], committed)  # noqa: E731
    # same words once the experiment ID is dropped; only the first key ("est:A|sT=1",
    # not "x") is read, and beam_baselines_fineweb.json, which has it too, is not listed
    assert cand("analysis/e8_baselines_fineweb.json") == [
        ("analysis/baselines_fineweb.json", 'by name; has cited key "est:A|sT=1"')]
    # no name match, so the first cited key decides
    assert cand("analysis/x23_probe_full.json") == [
        ("analysis/embedder_probe.json", 'by key; has cited key "results"')]
    assert cand("analysis/e4b_length_split.json") == [
        ("analysis/length_split.json", 'by name; lacks cited key "gsm8k_sT1.2_correct"')]
    assert cand("analysis/missing.json") == []

    out = io.StringIO()
    assert cps.report(found, status, committed, out) == 1
    text = out.getvalue()
    assert ("12 analysis/*.json paths cited: 5 not found under the cited name (3 of them with a "
            "candidate under another name), 3 uncommitted, 1 committed, 3 templated") in text
    assert "candidate: analysis/embedder_probe.json" in text


def test_exit_zero_when_all_committed(tmp_path):
    repo, _ = make_repo(tmp_path)
    tex = tmp_path / "ok.tex"
    tex.write_text("%% SOURCE: analysis/committed.json:{x}\n")
    assert cps.main([str(tex), "--repo", str(repo)]) == 0


def test_deleted_file_is_missing(tmp_path):
    repo, _ = make_repo(tmp_path)
    (repo / "analysis" / "committed.json").unlink()
    status = cps.classify(["analysis/committed.json"], repo)
    assert status == {"analysis/committed.json": ("missing", "deleted from working tree")}
