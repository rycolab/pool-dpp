"""Check that every analysis/*.json file the paper cites is committed to this repository.

Usage:
    python analysis/check_paper_sources.py PAPER.tex [MORE.tex ...]

Reads the given .tex files (the paper is a separate repository; pass its paths) and
collects every JSON file they name, in %% SOURCE: comments or anywhere else. Each
cited path is reported as one of:

  missing      no file of that name in the working tree
  uncommitted  in the working tree, but untracked, staged only, or changed since HEAD
  committed    in HEAD and unchanged in the working tree

Exit status 1 if any cited file is missing or uncommitted, else 0.

What counts as a citation:
  - `analysis/<name>.json`, after unescaping LaTeX `\\_` to `_`.
  - A bare `<name>.json` with no directory, read as `analysis/<name>.json`; SOURCE
    comments shorten names this way ("analysis/a.json and b.json"). The report
    marks these as bare.
  - A JSON path under any other directory (`cache-fineweb/x.json`) is ignored.
  - A path containing `{`, `<`, or `*` is a template (`x_{a,b}.json`). It is not
    expanded or checked; it is listed under "templated, not checked".

Renamed outputs. For a missing name, the report lists committed analysis/*.json
files that may hold the same output under a new name. These are guesses for a
person to confirm, never treated as a match (the file still counts as missing):
  - by name: the cited name without its experiment ID (`e8_`, `x11c_`) and without
    a `tau` token has the same words as the committed name, or (if no file has the
    same words) all of its words appear in the committed name;
  - by key, only if no file matches by name: the committed file has, as a top-level
    key, the first key the SOURCE comment names right after `<name>.json:`
    (`x.json:{"a|T=1"}.mean` names `a|T=1`).
Each candidate says which cited keys it has and which it lacks.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# A file name: word characters, '-', '.', '*', '<', '>', or a '{...}' group. The
# lookbehind rejects a name that follows '/', i.e. one under another directory.
_PART = r"(?:[\w\-.*<>]|\{[^{}\s]*\})+"
CITE = re.compile(r"(?<![\w/.\-])(?P<dir>analysis/)?(?P<name>" + _PART + r"\.json)\b")
TEMPLATE_CHARS = "{<*"
EXPERIMENT_ID = re.compile(r"^[a-z]\d+[a-z]?_")


def is_template(path):
    return any(c in path for c in TEMPLATE_CHARS)


def first_key(rest):
    """The first key named in `rest`, the text right after `<name>.json:`; None if templated."""
    rest = rest.lstrip("{")
    if rest.startswith('"'):
        end = rest.find('"', 1)
        key = rest[1:end] if end > 0 else ""
    else:
        # a '.' before a digit is part of the key (`gsm8k_sT1.2_correct`); others nest
        key = re.match(r'(?:[^,.{}\[\]:\s"]|\.(?=\d))*', rest).group()
    return key if key and not is_template(key) else None


def citations(tex_paths):
    """Return {path: [(tex_file, line_no, bare, key), ...]} for every JSON path cited."""
    found = {}
    for tex in tex_paths:
        text = Path(tex).read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.replace("\\_", "_")
            for m in CITE.finditer(line):
                rest = line[m.end():]
                key = first_key(rest[1:]) if rest.startswith(":") else None
                path = "analysis/" + m.group("name")
                found.setdefault(path, []).append(
                    (Path(tex).name, line_no, m.group("dir") is None, key))
    return found


def git_lines(repo, *args):
    out = subprocess.run(["git", "-C", str(repo), *args], check=True,
                         capture_output=True, text=True).stdout
    return set(out.splitlines())


def classify(paths, repo=REPO):
    """Return {path: (status, detail)} with status in missing/uncommitted/committed."""
    in_head = git_lines(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", "analysis")
    in_index = git_lines(repo, "ls-files", "--", "analysis")
    changed = git_lines(repo, "diff", "--name-only", "HEAD", "--", "analysis")
    result = {}
    for p in paths:
        if not (Path(repo) / p).is_file():
            result[p] = ("missing", "deleted from working tree" if p in in_head else "")
        elif p not in in_index:
            result[p] = ("uncommitted", "untracked")
        elif p not in in_head:
            result[p] = ("uncommitted", "staged, not committed")
        elif p in changed:
            result[p] = ("uncommitted", "changed since last commit")
        else:
            result[p] = ("committed", "")
    return result


def words(path):
    stem = EXPERIMENT_ID.sub("", Path(path).stem.lower())
    return {w for w in stem.split("_") if w and w != "tau"}


def committed_json(repo=REPO):
    """Return {path: set of top-level keys} for every analysis/*.json committed and unchanged."""
    in_head = git_lines(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", "analysis")
    paths = [p for p in in_head if p.endswith(".json") and p.count("/") == 1]
    status = classify(paths, repo)
    out = {}
    for p in sorted(paths):
        if status[p][0] == "committed":
            data = json.loads((Path(repo) / p).read_text(encoding="utf-8"))
            out[p] = set(data) if isinstance(data, dict) else set()
    return out


def candidates(path, cites, committed):
    """Return [(committed path, reason)] for files that may hold `path` under a new name."""
    w = words(path)
    keys = sorted({c[3] for c in cites if c[3]})
    same = [p for p in committed if words(p) == w]
    by_name = same or [p for p in committed if w and w <= words(p)]
    by_key = [] if by_name else [p for p in committed if any(k in committed[p] for k in keys)]
    out = []
    for p, how in [(p, "name") for p in by_name] + [(p, "key") for p in by_key]:
        has = [k for k in keys if k in committed[p]]
        lacks = [k for k in keys if k not in committed[p]]
        reason = f"by {how}"
        if has:
            reason += "; has cited key " + ", ".join(f'"{k}"' for k in has)
        if lacks:
            reason += "; lacks cited key " + ", ".join(f'"{k}"' for k in lacks)
        out.append((p, reason))
    return out


def where(cites):
    by_file = {}
    for tex, line_no, _, _ in cites:
        by_file.setdefault(tex, []).append(str(line_no))
    s = "; ".join(f"{tex}:{','.join(lines)}" for tex, lines in by_file.items())
    if all(c[2] for c in cites):
        s += "  (bare name in the paper)"
    return s


def report(found, status, committed, out=sys.stdout):
    templated = sorted(p for p in found if is_template(p))
    groups = {"missing": [], "uncommitted": [], "committed": []}
    for p in sorted(status):
        groups[status[p][0]].append(p)
    cands = {p: candidates(p, found[p], committed) for p in groups["missing"]}
    n_cand = sum(1 for c in cands.values() if c)
    width = max((len(p) for p in found), default=0)
    print(f"{len(found)} analysis/*.json paths cited: {len(groups['missing'])} not found under the "
          f"cited name ({n_cand} of them with a candidate under another name), "
          f"{len(groups['uncommitted'])} uncommitted, {len(groups['committed'])} committed, "
          f"{len(templated)} templated (not checked).", file=out)
    titles = {"missing": "Not found under the cited name",
              "uncommitted": "Present but not committed",
              "committed": "Committed"}
    for key, title in titles.items():
        print(f"\n{title} ({len(groups[key])}):", file=out)
        for p in groups[key]:
            detail = f"  [{status[p][1]}]" if status[p][1] else ""
            print(f"  {p:<{width}}  {where(found[p])}{detail}", file=out)
            for c, reason in cands.get(p, []):
                print(f"      candidate: {c} ({reason})", file=out)
    print(f"\nTemplated, not checked ({len(templated)}):", file=out)
    for p in templated:
        print(f"  {p:<{width}}  {where(found[p])}", file=out)
    return 1 if groups["missing"] or groups["uncommitted"] else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tex", nargs="+", help="the paper's .tex files")
    ap.add_argument("--repo", default=REPO, help="this repository's root (default: %(default)s)")
    args = ap.parse_args(argv)
    found = citations(args.tex)
    checkable = [p for p in found if not is_template(p)]
    return report(found, classify(checkable, args.repo), committed_json(args.repo))


if __name__ == "__main__":
    sys.exit(main())
