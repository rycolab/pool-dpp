"""PoolDPP estimation: subset selection and baselines on a precomputed pool.

One column per draw. The analyzed kernel gives every draw of the pool its own
column sqrt(w/N) * emb, with w = kappa/q. Collapsed onto the pool's unique
strings this is the row sqrt(w_i * count_i / N) * emb_i: same dual kernel and
same law over string sets. PoolDPP-Samp and the naive DPP use this form. Greedy
MAP does not need the count factor: a second copy of a selected column adds zero
gain, so greedy over the N draws and greedy over the unique rows pick the same
strings.

Kernels are passed as log row scales, so weights that span thousands of nats stay
exact. dpp.stable.StableKDPP samples from them and greedy_map compares gains in
log space. Neither forms a Gram matrix.

Two knobs:
  --target_Ts T
      --samp: kappa = plm^(1/T), q = p_sT, w = kappa/q. At sT=1 this is
      w = plm^(1/T - 1).
      --map: w = plm^(1/T), i.e. beta = 1/(2*T).
  --taus tau
      w = plm^(2*beta), beta = tau / sigma_hat, where sigma_hat is a robust
      per-(task, LM, sT) scale of logp_t1 (1.4826 x count-weighted MAD, median over
      prompts) frozen as a calibration constant before selection (see --calibrate).
      No proposal division.

Output layout (one scheme for all methods):
    {output_folder}/{run}/{model_name}-{method}/
        samp, map        -> tau{tau}_s{sT}.pkl or T{T}_s{sT}.pkl
        ancestral, naive, kmeans -> s{sT}.pkl
Deterministic methods (map) are computed once into run dir 0; stochastic
methods write one dir per run 0..N-1 (numpy seeded per (method, run, pool)).
"""
import argparse
import json
import os
import time
import zlib

import numpy as np

from dpp.stable import StableKDPP
from lm_dpp.utils.cached import cached_ddict, cached_list

MAD_TO_SIGMA = 1.4826


def fmt(x):
    """Float -> filesystem token: 1.0 -> '1', 0.03 -> '0.03', 500 -> '500'."""
    return f"{float(x):g}"


def method_seed(method, run, pool):
    """Numpy seed of one (method, run) on one pool. Pools need separate streams: on a shared
    stream every pool draws the same positions for a prompt, so the pools' means share their
    sampling error and intervals across pools come out too narrow."""
    return (run * 1000003 + zlib.crc32(f"{method}|{pool}".encode())) % (2 ** 32)


# ---------------------------------------------------------------------------
# Calibration: sigma_hat per (task cache, LM, sT)
# ---------------------------------------------------------------------------
def sigma_hat_from_embeddings(embeddings_dict):
    """{sT: sigma_hat}: count-weighted 1.4826*MAD of logp_t1 within each
    prompt's pool, median over prompts. Robust to the garbage tail that makes
    the plain std useless on hot pools (Ambrosia sT=1.5: std 3489 vs
    MAD-sigma 56.5)."""
    out = {}
    for sT in sorted(embeddings_dict.keys()):
        per_prompt = []
        for data in embeddings_dict[sT].values():
            lp1 = np.repeat(np.asarray(data["logps_t1"], dtype=np.float64),
                            np.asarray(data["counts"], dtype=np.int64))
            if lp1.size < 2:
                continue
            med = np.median(lp1)
            per_prompt.append(MAD_TO_SIGMA * np.median(np.abs(lp1 - med)))
        if per_prompt:
            out[float(sT)] = float(np.median(per_prompt))
    return out


def load_calibration(path):
    with open(path) as f:
        d = json.load(f)
    return {float(k): float(v) for k, v in d["sigma_hat"].items()}


def write_calibration(path, sigma, source):
    payload = {
        "sigma_hat": {fmt(k): v for k, v in sorted(sigma.items())},
        "definition": ("per sT: median over prompts of count-weighted "
                       "1.4826*MAD of logp_t1; beta = tau / sigma_hat"),
        "source": source,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"calibration -> {path}: {payload['sigma_hat']}")


# ---------------------------------------------------------------------------
# Kernels: log s_i, for rows s_i * emb_i
# ---------------------------------------------------------------------------
def quality_log_scales(logps_t1, beta, counts=None):
    """w = plm^(2*beta). Pass counts for anything that samples: sqrt(count_i) gives
    the per-draw law over string sets. Greedy MAP uses counts=None."""
    log_s = beta * np.asarray(logps_t1, dtype=np.float64)
    if counts is not None:
        log_s = log_s + 0.5 * np.log(np.asarray(counts, dtype=np.float64))
    return log_s


def target_log_scales(logps, logps_t1, counts, target_T):
    """w = plm^(1/T) / p_sT with the per-draw factor sqrt(count_i)."""
    log_w = np.asarray(logps_t1, dtype=np.float64) / target_T - np.asarray(logps, dtype=np.float64)
    return 0.5 * (log_w + np.log(np.asarray(counts, dtype=np.float64)))


def sample_dpp(log_s, embeddings, k, N=1):
    """N exact k-DPP draws over rows exp(log_s_i) * emb_i. A draw is [] when the rows
    span fewer than k dimensions."""
    with np.errstate(divide="ignore"):
        sampler = StableKDPP(log_s, embeddings, k)
    return [sampler.sample() for _ in range(N)]


def greedy_map(log_s, embeddings, k, tol=1e-10):
    """Greedy MAP over rows exp(log_s_i) * emb_i. The gain of row i is the squared norm of
    its residual on the span of the selected rows, compared as 2 log s_i + log ||r_i||^2.
    Returns fewer than k items when the rows span fewer than k dimensions."""
    emb = np.asarray(embeddings, dtype=np.float64)
    norm = np.linalg.norm(emb, axis=1)
    nonzero = norm > 0
    log_s = np.asarray(log_s, dtype=np.float64) + np.log(np.where(nonzero, norm, 1.0))
    log_s[~nonzero] = -np.inf
    residual = emb / np.where(nonzero, norm, 1.0)[:, None]
    res2 = (residual ** 2).sum(1)
    selected = []
    for _ in range(min(k, len(log_s))):
        with np.errstate(divide="ignore"):
            gain = 2 * log_s + np.log(np.maximum(res2, 0.0))
        gain[res2 <= tol ** 2] = -np.inf
        gain[selected] = -np.inf
        i = int(np.argmax(gain))
        if not np.isfinite(gain[i]):
            break
        selected.append(i)
        v = residual[i] / np.sqrt(res2[i])
        residual = residual - np.outer(residual @ v, v)
        res2 = (residual ** 2).sum(1)
    return selected


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------
def subsample_pool(u_sents, counts, logps, logps_t1, embeddings, pool_size):
    """Subsample pool_size items from the pool (without replacement) and recount."""
    N = int(sum(counts))
    if N <= pool_size:
        return u_sents, counts, logps, logps_t1, embeddings
    sample_idx = np.random.choice(N, size=pool_size, replace=False)
    cum_counts = np.cumsum(counts)
    item_indices = np.searchsorted(cum_counts, sample_idx, side="right")
    unique_items, new_counts = np.unique(item_indices, return_counts=True)
    return (
        u_sents[unique_items],
        new_counts,
        logps[unique_items],
        logps_t1[unique_items],
        embeddings[unique_items],
    )


def iter_prompt_data(bucket, pool_size=None):
    """Yield (prompt, u_sents, counts, logps, logps_t1, embeddings) with the
    usual sanity filter and optional pool subsampling."""
    for prompt, data in bucket.items():
        u_sents, counts, logps, logps_t1, embeddings = (
            data["u_sents"], data["counts"], data["logps"],
            data["logps_t1"], data["embeddings"],
        )
        if logps.shape[0] != logps_t1.shape[0] or len(embeddings) < 2:
            continue
        if pool_size is not None:
            u_sents, counts, logps, logps_t1, embeddings = subsample_pool(
                u_sents, counts, logps, logps_t1, embeddings, pool_size
            )
        yield prompt, u_sents, counts, logps, logps_t1, embeddings


def selection_record(prompt, selected, u_sents, logps_t1, embeddings):
    return (prompt,
            [u_sents[i] for i in selected],
            [logps_t1[i] for i in selected],
            [embeddings[i] for i in selected])


def run_kernel_method(embeddings_dict, path_of, log_scales, select, k, label, pool_size=None):
    """One selection per prompt at every pool temperature, resume-safe. A pool whose rows
    span fewer than k dimensions has no valid set, so that prompt is skipped and reported."""
    print(f"  {label}:", end=" ", flush=True)
    for sT in embeddings_dict:
        output_l = cached_list(path_of(sT))
        processed = set(x[0] for x in output_l)
        t0, n_new = time.perf_counter(), 0
        for prompt, u_sents, counts, logps, logps_t1, embeddings in iter_prompt_data(
                embeddings_dict[sT], pool_size):
            if prompt in processed:
                continue
            log_s = log_scales(sT, counts, logps, logps_t1)
            selected = select(log_s, embeddings, k)
            if len(selected) < k:
                print(f"\n  ! {label}: rank < k, skipped s{fmt(sT)} prompt={prompt[:20]}")
                continue
            output_l.append(selection_record(prompt, selected, u_sents, logps_t1, embeddings))
            n_new += 1
        output_l._save()
        if n_new:
            print(f"s{fmt(sT)}({(time.perf_counter() - t0) / n_new * 1000:.0f}ms)", end=" ", flush=True)
    print()


def draw_one(log_s, embeddings, k):
    return sample_dpp(log_s, embeddings, k)[0]


# ---------------------------------------------------------------------------
# Methods (all write {output}/{model}-{method}/... inside one run dir)
# ---------------------------------------------------------------------------
def run_samp(embeddings_dict, output_folder, model_name, k, taus, sigma, pool_size=None):
    """PoolDPP-Samp on w = plm^(2*beta), sweeping tau."""
    method_dir = f"{output_folder}/{model_name}-samp"
    for tau in taus:
        run_kernel_method(
            embeddings_dict,
            lambda sT, tau=tau: f"{method_dir}/tau{fmt(tau)}_s{fmt(sT)}.pkl",
            lambda sT, c, lq, lp, tau=tau: quality_log_scales(lp, tau / sigma[sT], c),
            draw_one, k, f"samp tau={fmt(tau)}", pool_size)


def run_samp_T(embeddings_dict, output_folder, model_name, k, target_Ts, pool_size=None):
    """PoolDPP-Samp on w = plm^(1/T) / p_sT, sweeping T."""
    method_dir = f"{output_folder}/{model_name}-samp"
    for T in target_Ts:
        run_kernel_method(
            embeddings_dict,
            lambda sT, T=T: f"{method_dir}/T{fmt(T)}_s{fmt(sT)}.pkl",
            lambda sT, c, lq, lp, T=T: target_log_scales(lq, lp, c, T),
            draw_one, k, f"samp T={fmt(T)}", pool_size)


def run_map(embeddings_dict, output_folder, model_name, k, taus, sigma, pool_size=None,
            target_Ts=None):
    """PoolDPP-MAP: greedy MAP on w = plm^(2*beta). Deterministic. Sweeps tau
    (beta = tau / sigma_hat) and/or T (beta = 1 / (2*T))."""
    method_dir = f"{output_folder}/{model_name}-map"
    knobs = [(f"tau{fmt(t)}", lambda sT, t=t: t / sigma[sT]) for t in (taus or [])]
    knobs += [(f"T{fmt(T)}", lambda sT, T=T: 1.0 / (2.0 * T)) for T in (target_Ts or [])]
    for name, beta_of in knobs:
        run_kernel_method(
            embeddings_dict,
            lambda sT, name=name: f"{method_dir}/{name}_s{fmt(sT)}.pkl",
            lambda sT, c, lq, lp, beta_of=beta_of: quality_log_scales(lp, beta_of(sT)),
            greedy_map, k, f"map {name}", pool_size)


def run_naive(embeddings_dict, output_folder, model_name, k):
    """Naive DPP baseline: diversity-only kernel, the DPP over the raw pool with
    duplicate rows collapsed, i.e. row sqrt(count_i) * emb_i."""
    method_dir = f"{output_folder}/{model_name}-naive"
    run_kernel_method(
        embeddings_dict,
        lambda sT: f"{method_dir}/s{fmt(sT)}.pkl",
        lambda sT, c, lq, lp: quality_log_scales(lp, 0.0, c),
        draw_one, k, "naive")


def ancestral_draw(counts, k):
    """k pool draws without replacement, as indices into the pool's unique strings."""
    positions = np.random.choice(sum(counts), size=k, replace=False)
    return np.searchsorted(np.cumsum(counts), positions, side="right")


def run_ancestral(embeddings_dict, output_folder, model_name, k):
    """Ancestral baseline: k pool draws without replacement."""
    method_dir = f"{output_folder}/{model_name}-ancestral"
    print("  ancestral:", end=" ", flush=True)
    for sT in embeddings_dict:
        output_l = cached_list(f"{method_dir}/s{fmt(sT)}.pkl")
        processed = set(x[0] for x in output_l)
        for prompt, data in embeddings_dict[sT].items():
            if prompt in processed:
                continue
            u_sents, counts, logps_t1, embeddings = (
                data["u_sents"], data["counts"], data["logps_t1"], data["embeddings"])
            sample = ancestral_draw(counts, k)
            output_l.append(selection_record(prompt, sample, u_sents, logps_t1, embeddings))
        output_l._save()
        print(f"s{fmt(sT)}", end=" ", flush=True)
    print()


def sample_kmeans_for_prompt(embeddings, logps_t1, k, random_state=0, max_dim=128):
    """K-means clustering baseline: cluster pool embeddings, pick highest-logp
    item per cluster.

      - Gaussian random projection D->max_dim if D>max_dim (JL preserves
        distances). Projection RNG is fixed (seed 0) so it is identical across
        runs/pools; only KMeans random_state varies.
      - n_init=3, max_iter=200, random_state=<per-run> for run-level variance.
    """
    from sklearn.cluster import KMeans

    n = len(embeddings)
    k_eff = min(k, n)
    if k_eff == 0:
        return []
    if k_eff == n:
        return list(range(n))
    F = np.asarray(embeddings, dtype=np.float32)
    if F.shape[1] > max_dim:
        rng_p = np.random.default_rng(0)
        P = rng_p.standard_normal((F.shape[1], max_dim)).astype(np.float32) / np.sqrt(max_dim)
        F = F @ P
    km = KMeans(n_clusters=k_eff, n_init=3, random_state=random_state, max_iter=200)
    labels = km.fit_predict(F)
    selected = []
    for c in range(k_eff):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        best = idxs[np.argmax(logps_t1[idxs])]
        selected.append(int(best))
    return selected


def run_kmeans(embeddings_dict, output_folder, model_name, k, random_state=0, pool_size=None):
    """k-Means baseline; random_state = run index for run-level spread."""
    method_dir = f"{output_folder}/{model_name}-kmeans"
    print("  kmeans:", end=" ", flush=True)
    for sT in embeddings_dict:
        output_l = cached_list(f"{method_dir}/s{fmt(sT)}.pkl")
        processed = set(x[0] for x in output_l)
        for prompt, u_sents, counts, logps, logps_t1, embeddings in iter_prompt_data(
                embeddings_dict[sT], pool_size):
            if prompt in processed:
                continue
            selected = sample_kmeans_for_prompt(embeddings, logps_t1, k,
                                                random_state=random_state)
            output_l.append(selection_record(prompt, selected, u_sents, logps_t1, embeddings))
        output_l._save()
        print(f"s{fmt(sT)}", end=" ", flush=True)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli():
    parser = argparse.ArgumentParser(
        description="PoolDPP estimation: subset selection and baselines on a "
                    "precomputed embedding pool")
    parser.add_argument("--model_name", required=True,
                        help="LM_embedder name, filesystem-encoded (e.g. google__gemma-2-9b-it_qwen)")
    parser.add_argument("--cache_dir", default="./cache",
                        help="Directory containing the embeddings cache")
    parser.add_argument("--output_folder", default="./estimation_output",
                        help="Estimation root; run dirs 0..N-1 are created inside")
    parser.add_argument("--k", type=int, default=10, help="Number of items to select")
    parser.add_argument("--N", type=int, default=1,
                        help="Independent estimation runs (run dirs; deterministic methods always use run dir 0)")
    parser.add_argument("--sampling_T", type=float, default=None,
                        help="Pin one pool sampling temperature (default: all in the cache)")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Subsample pool to this many items (ablation only)")
    # PoolDPP
    parser.add_argument("--samp", action="store_true",
                        help="PoolDPP-Samp (exact k-DPP draws); sweep --target_Ts and/or --taus")
    parser.add_argument("--map", dest="map_", action="store_true",
                        help="PoolDPP-MAP (greedy MAP); sweep --target_Ts and/or --taus")
    parser.add_argument("--taus", nargs="+", type=float, default=None,
                        help="Dimensionless tilt values; beta = tau / sigma_hat(sT)")
    parser.add_argument("--target_Ts", nargs="+", type=float, default=None,
                        help="T grid for --samp (kappa = plm^(1/T), q = p_sT) and --map (beta = 1/(2T))")
    parser.add_argument("--calibration", default=None,
                        help="Path to the sigma_hat calibration JSON (required with --taus)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Compute sigma_hat from this cache, write to --calibration, and exit")
    # baselines
    parser.add_argument("--ancestral", action="store_true")
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--kmeans", action="store_true")
    args = parser.parse_args()

    embeddings_path = os.path.join(args.cache_dir, f"{args.model_name}_embeddings.pkl")
    embeddings_dict = cached_ddict(embeddings_path, read_only=True)
    sampling_Ts = sorted(embeddings_dict.keys())
    if args.sampling_T is not None:
        sampling_Ts = [t for t in sampling_Ts if abs(t - args.sampling_T) < 1e-9]
        if not sampling_Ts:
            raise ValueError(f"sampling_T={args.sampling_T} not in cache. "
                             f"Available: {sorted(embeddings_dict.keys())}")
    active = {t: embeddings_dict[t] for t in sampling_Ts}

    if args.calibrate:
        if not args.calibration:
            parser.error("--calibrate requires --calibration <output path>")
        sigma = sigma_hat_from_embeddings(embeddings_dict)  # always the full cache
        write_calibration(args.calibration, sigma,
                          source=f"{args.cache_dir}/{args.model_name}")
        return

    sigma = None
    if args.samp or args.map_:
        if not args.taus and not args.target_Ts:
            parser.error("--samp/--map require --target_Ts and/or --taus")
    if (args.samp or args.map_) and args.taus:
        if not args.calibration:
            parser.error("--taus requires --calibration (see --calibrate)")
        sigma = load_calibration(args.calibration)
        missing = [t for t in sampling_Ts if t not in sigma]
        if missing:
            raise ValueError(f"calibration {args.calibration} lacks sT={missing}")

    print(f"[estimation] {args.model_name} | k={args.k} N={args.N} "
          f"pool={args.pool_size or 'full'} | sT={sampling_Ts} -> {args.output_folder}")

    pool = os.path.basename(os.path.normpath(args.cache_dir))

    for run in range(args.N):
        run_dir = f"{args.output_folder}/{run}"
        if args.N > 1:
            print(f" run {run}")
        if args.samp and args.taus:
            np.random.seed(method_seed("samp", run, pool))
            run_samp(active, run_dir, args.model_name, args.k, args.taus, sigma,
                     pool_size=args.pool_size)
        if args.samp and args.target_Ts:
            np.random.seed(method_seed("samp_T", run, pool))
            run_samp_T(active, run_dir, args.model_name, args.k, args.target_Ts,
                       pool_size=args.pool_size)
        if args.map_ and run == 0:
            np.random.seed(method_seed("map", run, pool))  # only --pool-size draws from it
            run_map(active, run_dir, args.model_name, args.k, args.taus, sigma,
                    pool_size=args.pool_size, target_Ts=args.target_Ts)
        if args.ancestral:
            np.random.seed(method_seed("ancestral", run, pool))
            run_ancestral(active, run_dir, args.model_name, args.k)
        if args.naive:
            np.random.seed(method_seed("naive", run, pool))
            run_naive(active, run_dir, args.model_name, args.k)
        if args.kmeans:
            np.random.seed(method_seed("kmeans", run, pool))
            run_kmeans(active, run_dir, args.model_name, args.k,
                       random_state=run, pool_size=args.pool_size)


if __name__ == "__main__":
    cli()
