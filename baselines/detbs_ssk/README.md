# DivBS and string-subsequence-kernel DetBS on FineWeb

Code and commands for `tab:beam_baselines` of the paper: Diverse Beam Search (DivBS) and a
HuggingFace port of Determinantal Beam Search (DetBS) with the string-subsequence kernel of
Meister et al. (2021), scored on the 50 FineWeb prompts with Gemma-2-9B-Instruct (chat template,
K=10, 50 new tokens) and MiniLM-L6 embeddings.

This DetBS is not the one in `src/lm_dpp/det_beam.py`, which uses a cosine Gram matrix over hashed
unigram and bigram counts and gives the DetBS rows of `analysis/beam_baselines.py`
(`tab:baselines`). Here the kernel is the dynamic-programming string-subsequence kernel (n=2,
decay 0.5), a port of the fairseq code at rycolab/swor (branch `diverse_decoding`). It does a full
forward pass per beam per step, so a full run takes about 25 minutes on four GPUs.

| File | Role |
|---|---|
| `generate.py` | the port (greedy Cholesky MAP step in log space, kernel DP) |
| `run_intrinsic_baselines.py` | runs DivBS (`--method dbs`) or DetBS (`--method detbs`) and saves `(prompt, sents, logps, embeddings)` per prompt |
| `sweep_detbs.sh` | DetBS over T in {0.7, 1.0, 1.3} and w on the first 8 prompts |
| `test_det_beam_search.py` | tests of the kernel and the greedy MAP step |

```bash
for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$SHARD python baselines/detbs_ssk/run_intrinsic_baselines.py \
    --method detbs --beam-size 10 --kernel-weight 0.1 --max-new-tokens 50 \
    --shard $SHARD --n-shards 4 --tag detbs_k10_w0.1 &
done; wait
python baselines/detbs_ssk/run_intrinsic_baselines.py --method detbs --beam-size 10 \
  --kernel-weight 1.0 --max-new-tokens 50 --tag detbs_k10_w1.0
python baselines/detbs_ssk/run_intrinsic_baselines.py --method dbs --beam-size 10 \
  --num-beam-groups 10 --diversity-penalty 1.0 --max-new-tokens 50 --tag dbs_k10_g10_p1.0
python baselines/detbs_ssk/run_intrinsic_baselines.py --method dbs --beam-size 6 \
  --num-beam-groups 3 --diversity-penalty 1.0 --max-new-tokens 50 --tag dbs_k6_g3_p1.0
bash baselines/detbs_ssk/sweep_detbs.sh
python analysis/beam_baselines_intrinsic.py     # writes analysis/beam_baselines_intrinsic.json
```

Outputs go to `cache-fineweb-ssk/`. `analysis/beam_baselines_intrinsic.json` has the mean and the
standard deviation over prompts of quality, log-det, and Distinct-2 for every run.
