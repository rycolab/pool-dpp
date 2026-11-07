# CLAUDE.md

The paper (`neurips_2026.tex`, on Overleaf) quotes numbers from the JSON files in `analysis/`. Each
quoted number has a comment next to it, `%% SOURCE: analysis/<file>.json:<key> (<commit>)`. These rules
keep every quoted number checkable and regenerable from this repository.

## Commit every output the paper quotes

- Commit every analysis output the paper quotes as JSON under `analysis/`, in the same push as the run
  that produced it. Outputs are small: commit them, don't gitignore them.
- Keep large intermediates ignored, as `.gitignore` does now: pools, embeddings, and selections
  (`cache-*`), logs, `analysis/*.pkl`, `analysis/bound_work/`.
- Give every number quoted in the paper a SOURCE comment: file, key, and the commit that holds the file.

## Keep the paper in sync

- When an output changes, update every number in the paper that quotes it right away, in the same work
  session, and its SOURCE comment (file, key, commit hash). A stale number in the paper is a bug.
- When you rename or replace an output, or change its keys (e.g. `tau=` to `T=`), update every SOURCE
  comment that names the old file or key.
- Before pushing results, run the checker on the paper and commit what it reports:

  ```bash
  python analysis/check_paper_sources.py path/to/paper/neurips_2026.tex path/to/paper/checklist.tex
  ```

  It lists each `analysis/*.json` the paper cites as missing, uncommitted, or committed, and exits
  nonzero if any is missing or uncommitted.

## No silent workarounds

- A run that hits a failure (NaN or inf, non-convergence, a skipped prompt, a fallback, clipping) stops
  with a clear error by default.
- Agree on any workaround with the authors first. Then disclose it in the script's docstring, in the run
  log when it fires (what failed, how often), and in the paper's description of the experiment.
- Disclose approximations that change results (iteration caps, tolerances, data filters) the same way.
