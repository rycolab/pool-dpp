# PoolDPP

Code for *A Model of Diverse Sampling from Language Models* (NeurIPS 2026).

PoolDPP draws a pool of strings from a language model and selects a diverse, high-quality subset of
size K with a K-DPP over the pool. PoolDPP-Samp draws an exact sample. PoolDPP-MAP runs the greedy
algorithm.

## Install

```bash
pip install -e ".[dev]"
```

Sampling also needs vLLM with a patch that reports log-probabilities at the sampling temperature and at
T=1. Selection and all figures run on CPU without it.

```bash
pip install -e ".[vllm]"
patch -p1 -d "$(python -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')" \
    < vllm_dual_logprobs.patch
```

The Ambrosia experiments read `ambrosia.csv` from the repository root. Download it from the
[AMBROSIA benchmark](https://arxiv.org/abs/2406.19073). The MATH-500 check also needs `pip install -e ".[math]"`.

## Reproduce the paper

| Step | Command | Needs |
|---|---|---|
| FineWeb pools and selections | `bash scripts/run_fineweb.sh` | GPU, then CPU |
| FineWeb pool-size and K ablations | `bash scripts/run_fineweb_ablations.sh` | CPU |
| Ambrosia pools and selections | `bash scripts/run_ambrosia.sh` | GPU, then CPU |
| GSM8K pools and selections | `bash scripts/run_gsm8k.sh` | GPU, then CPU |
| GSM8K instructed kernel | `python analysis/embed_instructed.py --pool-start 0 --pool-end 100`, then `bash scripts/run_gsm8k_instructed.sh` | GPU, then CPU |
| Larger pools | `bash scripts/run_large_pools.sh` | GPU |
| Beam-search baselines | `METHOD=sbs`, `cps`, `detbs`, or `divbs` `bash scripts/run_beam_baselines.sh` | GPU |
| MATH-500 pools, selections, and figure | `bash scripts/run_math500.sh` | GPU, then CPU |
| Generation cap: stop token in the scores | `python analysis/length_cap.py stored`, then `probe --csv-path ambrosia.csv`, after the pools | GPU |
| LLM-based selection | `python analysis/llm_selection.py`, after the FineWeb pools | GPU |
| Pair judge | `GEMINI_API_KEY=... python analysis/pair_judge.py validate`, then `score`, after the GSM8K selections | Gemini API |
| All other figures and tables | `bash scripts/make_figures.sh` | CPU |

To rerun selection on existing pools, use `STAGES=estimate J=16 bash scripts/run_gsm8k.sh`. Every step
resumes where it stopped.

| Paper | Script in `analysis/` |
|---|---|
| FineWeb, Ambrosia, and GSM8K frontiers | `plot_intrinsic.py`, `plot_ambrosia.py`, `plot_gsm8k.py` |
| Pool-size, K, LM and embedder ablations | `plot_ablations.py` |
| Bound on the KL, required pool size, N-dial | `bound_empirical.py`, `bound_required_n.py`, `bound_tables.py`, `gsm8k_n2500.py`, `plot_ndial.py` (formulas shared in `bound_formulas.py`) |
| Weight diagnostics | `weight_diagnostics.py`, `ess_fineweb_n8000.py`, `ess_ambrosia_n6000.py` |
| Significance tests | `significance.py` |
| Set-level baselines, GSM8K kernels | `baselines.py`, `baselines_gsm8k.py` |
| Beam-search baselines | `beam_baselines.py` |
| Held-out diversity metrics | `heldout_diversity.py` |
| Pool ceiling | `pool_ceiling.py` |
| Length and log-probability | `length_analysis.py`, `length_stratified.py` |
| Generation cap | `length_cap.py` |
| Embedder probe | `embedder_probe.py` |
| Pair decomposition of GSM8K sets | `pair_judge.py` |
| LLM-based selection | `llm_selection.py` |
| MATH-500 saturation check | `math500.py` |
| Dual versus primal cost | `dual_vs_primal.py` |
| Prompts with Zhat = 0 and the redraw rule | `zhat_zero_counts.py` |
| GSM8K pools from different runs, embedding batch size | `pool_checks.py`, `embed_batch_check.py` |

## Lean proofs

`lean/` has machine-checked Lean 4 proofs of the paper's variance-bias bound, finite-sample bound, and sharper KL bound. See `lean/README.md` to build and check them.

## Pools

Ambrosia and GSM8K use 100 pools and FineWeb uses 10, with the pool index as the seed. At 10 pools, FineWeb's largest
half-width is 2.3% of the plotted range of log-det and 0.4% of the plotted range of quality. The pools come from separate runs:

| Pools | Card | `dpp-sample` settings |
|---|---|---|
| GSM8K 0-19, Ambrosia 0-29 | first run, not recorded here | not recorded here |
| GSM8K 20, 22-24, 26-28 | RTX PRO 6000 | the default of `scripts/run_gsm8k.sh` |
| GSM8K 21, 25, 29 | RTX 3090 | 24 GB settings |
| GSM8K 30-99 | RTX 4090, and RTX 3090 for 79, 81, 83, 88, 89, 90, 95, 96, 98, 99 | 24 GB settings |
| Ambrosia 30-99 | 24 GB cards | 24 GB settings |

The 24 GB settings are the `SAMPLE_ARGS` value in the comment of each run script. The instructed GSM8K embeddings use
`--batch-size 64` on pools 0-29 and `--batch-size 8` on pools 30-99.

`analysis/pool_checks.json` compares these runs. With the plain GSM8K kernel, log-det is lower on pools 30-99 than on pools
0-29 at all 29 plotted settings, by up to 0.26 nats for PoolDPP-MAP (T=50: -14.02 vs. -13.76, z = -4.5) and by 0.01 for
Ancestral at sT=1.2. One of the 29 pass@10 comparisons has |z| above 1.96, at z = 2.09. Within pools 30-99, RTX 3090 and RTX 4090 pools show
no systematic difference (mean z 0.01). The cause of the log-det difference is not established. `analysis/embed_batch_check.json` re-embeds one pool at batch 8: mean PoolDPP-MAP
log-det moves by 0.005 and the control at batch 64 reproduces the stored selections exactly.

## Layout

```
src/dpp/stable.py          exact K-DPP sampling for row weights that span many orders of magnitude
src/lm_dpp/sample_vllm.py  dpp-sample: draw the pool
src/lm_dpp/embeddings.py   dpp-embeddings: embed the unique strings
src/lm_dpp/estimation.py   dpp-estimation: PoolDPP-Samp, PoolDPP-MAP, and pool baselines
src/lm_dpp/sample_sbs.py, sample_detbs.py, sample_divbs.py, sbs_decoding.py, det_beam.py   beam-search baselines
scripts/                   one script per experiment
analysis/                  figures and tables
```

Selections are written to `cache-{task}/.../est{K}{embedder}/{run}/{lm}_{embedder}-{method}/`, as
`tau{tau}_s{T}.pkl` or `T{T}_s{T}.pkl` for PoolDPP and `s{T}.pkl` for baselines.

## Tests

```bash
pytest tests/test_estimation.py tests/test_sbs.py tests/test_bounds_m1.py
python tests/test_stable.py
```
