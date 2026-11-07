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

A pool whose rows span fewer than k dimensions has Zhat = 0 and no k-DPP. Such a prompt
redraws its pool from --redraw_dirs (see PoolRedraw), so every prompt contributes. The test is
the rank of the pool's embeddings (numpy.linalg.matrix_rank, tol=RANK_TOL), the same test
PoolRedraw applies to the replacement pool; a prompt with fewer than two unique strings has
rank below k for every k >= 2. A selection with fewer than k items from a pool that does span
k dimensions is a numerical failure and stops the run (RuntimeError); it is never swapped for
another pool. Each PoolDPP-Samp, PoolDPP-MAP and Naive-DPP selection file X.pkl gets a sidecar
X.redraws.json with the number of prompts, the number redrawn, and for each redrawn prompt its
rank, its number of unique strings and the pool it was drawn from (see write_redraws). The
sidecar is rebuilt on every call, also for a resumed file: a stored selection whose strings are
not all in the prompt's own pool counts as redrawn, and if that pool spans k dimensions the run
stops, since the selection was swapped by the trigger before this check (len(selected) < k).

Other failures also stop with an error: pool arrays of different lengths, NaN or +inf log
scales, a k-Means pool with fewer than k unique strings or an empty cluster, and a
calibration prompt with fewer than two draws.

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
import hashlib
import json
import os
import time
import zlib

import numpy as np

from dpp.stable import RANK_TOL, TIER_NATS, StableKDPP, kernel_info
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
        for prompt, data in embeddings_dict[sT].items():
            lp1 = np.repeat(np.asarray(data["logps_t1"], dtype=np.float64),
                            np.asarray(data["counts"], dtype=np.int64))
            if lp1.size < 2 or not np.isfinite(lp1).all():
                raise ValueError(f"calibration: sT={fmt(sT)} prompt={prompt[:60]!r} has {lp1.size} draws "
                                 f"({int((~np.isfinite(lp1)).sum())} non-finite logp_t1); the MAD needs two "
                                 f"or more finite draws")
            med = np.median(lp1)
            per_prompt.append(MAD_TO_SIGMA * np.median(np.abs(lp1 - med)))
        if not per_prompt:
            raise ValueError(f"calibration: sT={fmt(sT)} has no prompts")
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
    Returns fewer than k items only when the rows with mass (log s_i > -inf, nonzero emb_i)
    span fewer than k dimensions (matrix_rank, tol=RANK_TOL). Raises ValueError on a NaN or
    +inf log scale or a non-finite embedding, and RuntimeError when it stops short of k on rows
    that span k dimensions (its residual tolerance tol and the rank test disagree)."""
    emb = np.asarray(embeddings, dtype=np.float64)
    log_s_in = np.asarray(log_s, dtype=np.float64)
    if np.isnan(log_s_in).any() or (log_s_in == np.inf).any() or not np.isfinite(emb).all():
        raise ValueError(f"greedy_map: {int(np.isnan(log_s_in).sum())} NaN and {int((log_s_in == np.inf).sum())} "
                         f"+inf log scales, {int((~np.isfinite(emb)).any(axis=1).sum())} non-finite embedding rows")
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
    if len(selected) < k:
        r = span_rank(log_s_in, emb)
        if r >= k:
            raise RuntimeError(f"greedy_map stopped at {len(selected)} of k={k} items, but the rows with mass "
                               f"span {r} >= k dimensions (rank tol {RANK_TOL}, residual tol {tol}): a "
                               f"numerical failure, not Zhat = 0")
    return selected


def span_rank(log_s, embeddings):
    """Rank (tol=RANK_TOL) of the embeddings of the rows with mass (log scale > -inf). The k-DPP
    over these rows has Zhat = 0, and no size-k set, exactly when this is below k. When every row
    has mass this is PoolRedraw.first_valid's test."""
    E = np.asarray(embeddings, dtype=np.float64)
    E = E[np.asarray(log_s, dtype=np.float64) > -np.inf]
    return int(np.linalg.matrix_rank(E, tol=RANK_TOL)) if len(E) else 0


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
    """Yield (prompt, u_sents, counts, logps, logps_t1, embeddings) with optional pool
    subsampling. A prompt with fewer than two unique strings is not yielded (before any
    subsampling, so the random stream is unchanged); every caller checks for such prompts
    (run_kernel_method redraws them, run_kmeans stops). Raises ValueError when the pool's
    arrays have different lengths."""
    for prompt, data in bucket.items():
        u_sents, counts, logps, logps_t1, embeddings = (
            data["u_sents"], data["counts"], data["logps"],
            data["logps_t1"], data["embeddings"],
        )
        lens = {name: len(a) for name, a in (("u_sents", u_sents), ("counts", counts), ("logps", logps),
                                             ("logps_t1", logps_t1), ("embeddings", embeddings))}
        if len(set(lens.values())) != 1:
            raise ValueError(f"prompt={prompt[:60]!r}: pool arrays have different lengths {lens}")
        if len(embeddings) < 2:
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


class PoolRedraw:
    """Redraw of a pool with Zhat = 0: the prompt moves to the first of the given pools whose rows
    span k dimensions. Zhat > 0 does not depend on the weights, so the choice of pool is the same for
    every kernel and uses no randomness. Pool files are large, so only the chosen prompt's row stays
    in memory. The selection has its own numpy stream, keyed by the method's seed, so it leaves
    every other prompt's draws unchanged and a resumed run matches a full run."""

    def __init__(self, cache_dirs, model_name):
        self.paths = [os.path.join(d, f"{model_name}_embeddings.pkl") for d in cache_dirs]
        missing = [p for p in self.paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"redraw pools not found: {missing}")
        self.found = {}
        self.seed = 0

    def first_valid(self, sT, prompt, k):
        """(path, row) of the first pool where this prompt's embeddings span k dimensions. A pool
        without a row for (sT, prompt) raises: skipping it would change which pool the prompt takes."""
        if (sT, prompt) not in self.found:
            for path in self.paths:
                row = dict.get(cached_ddict(path, read_only=True), sT, {}).get(prompt)
                if row is None:
                    raise KeyError(f"redraw pool {path} has no row for sT={fmt(sT)} prompt={prompt[:60]!r}: "
                                   f"finish sampling and embedding that pool, or leave it out of --redraw_dirs; "
                                   f"it is not skipped, since the pool order decides which pool a prompt takes")
                if np.linalg.matrix_rank(np.asarray(row["embeddings"], float), tol=RANK_TOL) >= k:
                    self.found[(sT, prompt)] = path, row
                    break
            else:
                raise RuntimeError(f"no redraw pool spans {k} dimensions at sT={fmt(sT)} for prompt={prompt[:40]!r}: "
                                   f"pass more pools in --redraw_dirs (spare pools, not replicates; see the PR)")
        return self.found[(sT, prompt)]

    def select(self, sT, prompt, log_scales, select, k, label, pool_size=None):
        """(path, u_sents, logps_t1, embeddings, selected) from the first valid pool."""
        path, row = self.first_valid(sT, prompt, k)
        state = np.random.get_state()
        np.random.seed(zlib.crc32(f"{self.seed}|{label}|{fmt(sT)}|{prompt}".encode()))
        try:
            _, u_sents, counts, logps, logps_t1, embeddings = next(iter_prompt_data({prompt: row}, pool_size))
            selected = select(log_scales(sT, counts, logps, logps_t1), embeddings, k)
        finally:
            np.random.set_state(state)
        if len(selected) < k:
            raise RuntimeError(f"{label}: redraw pool {path} spans {k} dimensions but the selection from it has "
                               f"{len(selected)} items at sT={fmt(sT)} prompt={prompt[:40]!r} (with --pool-size, "
                               f"the subsample can span fewer): a decision for timv/Manuel, not a skip")
        return path, u_sents, logps_t1, embeddings, selected


_KERNEL_INFO = {}  # content hash of (log scales, embeddings, k) -> dpp.stable.kernel_info, shared by run dirs
_RANK = {}         # content hash of embeddings -> span_rank, for resumed selections


def _content_key(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def redraws_path(selection_path):
    """Sidecar of a selection file: X.pkl -> X.redraws.json."""
    return os.path.splitext(str(selection_path))[0] + ".redraws.json"


def write_redraws(selection_path, label, k, sT, pool_size, entries, sampler):
    """Write the sidecar X.redraws.json of selection file X.pkl. entries: one dict per prompt of
    X.pkl, in its order, with "redrawn" (bool), "info" (kernel_info or None) and, when redrawn,
    "prompt", "rank", "n_unique", "from". sampler: whether the method draws from StableKDPP."""
    redrawn = [{key: e[key] for key in ("prompt", "rank", "n_unique", "from")} for e in entries if e["redrawn"]]
    out = dict(selection_file=os.path.basename(str(selection_path)), method=label, k=k, sampling_T=float(sT),
               pool_size=pool_size, n_prompts=len(entries), n_redrawn=len(redrawn), redrawn=redrawn, sampler=None)
    if sampler:
        infos = [e["info"] for e in entries if e["info"] is not None]
        drop = [i for i in infos if i["path"] == "drop"]
        gaps = [i["min_gap_nats"] for i in infos if i["path"] == "tiered" and i["min_gap_nats"] is not None]
        out["sampler"] = dict(
            n_flat=sum(i["path"] == "flat" for i in infos), n_drop=len(drop),
            n_tiered=sum(i["path"] == "tiered" for i in infos), n_unknown=len(entries) - len(infos),
            n_rows_dropped=sum(i["n_dropped"] for i in drop),
            max_log10_drop_bound=max((i["log10_drop_bound"] for i in drop), default=None),
            min_tier_gap_nats=min(gaps, default=None),
            note=("per prompt, the tier path of dpp.stable.StableKDPP for its kernel: flat = exact; drop = rows "
                  f"more than {TIER_NATS:g} nats below the top dropped, with log10_drop_bound an upper bound on "
                  "the left-out probability mass; tiered = tier-by-tier draw (stops unless the gap at each cut "
                  f"exceeds {TIER_NATS:g} nats); unknown = resumed --pool-size selections (subsample not kept)"))
    with open(redraws_path(selection_path), "w") as f:
        json.dump(out, f, indent=1)
    return out


def run_kernel_method(embeddings_dict, path_of, log_scales, select, k, label, pool_size=None, redraw=None,
                      sampler=False):
    """One selection per prompt at every pool temperature, resume-safe. A prompt whose pool's embeddings
    span fewer than k dimensions (Zhat = 0, span_rank) redraws its pool, and without --redraw_dirs the run
    stops: no prompt is dropped. A selection with fewer than k items from a pool that spans k dimensions
    stops the run (a numerical failure). Writes the sidecar X.redraws.json for every selection file
    (write_redraws), also for prompts resumed from an earlier run; see the module docstring."""
    print(f"  {label}:", end=" ", flush=True)

    def info_of(log_s, embeddings):
        """kernel_info of the kernel the selection was drawn from (None for greedy MAP)."""
        if not sampler:
            return None
        if pool_size is not None:
            return kernel_info(log_s, embeddings, k)
        key = (_content_key(log_s, embeddings), k)
        if key not in _KERNEL_INFO:
            _KERNEL_INFO[key] = kernel_info(log_s, embeddings, k)
        return _KERNEL_INFO[key]

    def row_info(sT, prompt, row):
        """info_of for a stored selection, from its full pool row (unknown, None, under --pool-size)."""
        if not sampler or pool_size is not None:
            return None
        _, _, counts, logps, logps_t1, embeddings = next(iter_prompt_data({prompt: row}))
        return info_of(log_scales(sT, counts, logps, logps_t1), embeddings)

    def redrawn(sT, prompt, rank, n_unique):
        if redraw is None:
            raise RuntimeError(f"{label}: Zhat = 0 at sT={fmt(sT)} for prompt={prompt[:40]!r} (its pool's embeddings "
                               f"span {rank} < k={k} dimensions): pass --redraw_dirs with pools to redraw from, as "
                               f"scripts/run_ambrosia.sh does; no prompt is dropped")
        path, u_sents, logps_t1, embeddings, selected = redraw.select(
            sT, prompt, log_scales, select, k, label, pool_size)
        print(f"\n  {label}: Zhat = 0 (rank {rank} < k={k}), redrawn from {path} s{fmt(sT)} prompt={prompt[:20]}")
        entry = {"redrawn": True, "prompt": prompt, "rank": rank, "n_unique": n_unique, "from": str(path),
                 "info": row_info(sT, prompt, redraw.first_valid(sT, prompt, k)[1])}
        return selection_record(prompt, selected, u_sents, logps_t1, embeddings), entry

    def stored_entry(sT, prompt, sents, row):
        """Sidecar entry of a selection resumed from an earlier run, checked against its pools. Without
        --pool-size the rank of the prompt's own pool decides whether it was redrawn, and the strings
        must come from that pool or from the first redraw pool that spans k dimensions. With
        --pool-size the subsample is not kept, so the strings decide."""
        where = f"{label} sT={fmt(sT)} prompt={prompt[:40]!r} in {path_of(sT)}"
        own = set(np.asarray(row["u_sents"]).tolist())
        rank = None
        if pool_size is None:
            key = _content_key(row["embeddings"])
            if key not in _RANK:
                _RANK[key] = span_rank(np.zeros(len(row["embeddings"])), row["embeddings"])
            rank = _RANK[key]
            if rank >= k and not set(sents) <= own:
                raise RuntimeError(f"{where}: the stored selection is not from this pool, but this pool spans "
                                   f"{rank} >= k={k} dimensions, so Zhat > 0: it was swapped for another pool by "
                                   f"the redraw trigger before the rank test (len(selected) < k), after a "
                                   f"numerical failure. Delete the file and rerun dpp-estimation")
        if (rank is not None and rank >= k) or (rank is None and set(sents) <= own):
            info = row_info(sT, prompt, row)
            if info is not None and info["path"] == "tiered" and info["min_gap_nats"] is not None \
                    and info["min_gap_nats"] <= TIER_NATS:
                raise RuntimeError(f"{where}: the stored selection was drawn by the tiered k-DPP path with a gap "
                                   f"of {info['min_gap_nats']:.2f} <= {TIER_NATS:g} nats at the first cut, so it is "
                                   f"not a draw from the k-DPP (dpp.stable docstring). Delete the file and rerun "
                                   f"once timv/Manuel have decided the sampler fix (PR 'fail loudly')")
            return dict(redrawn=False, info=info)
        if redraw is None:
            raise RuntimeError(f"{where}: the stored selection is not from this pool; pass the --redraw_dirs it "
                               f"was made with, so its source pool can be checked and recorded")
        path, src_row = redraw.first_valid(sT, prompt, k)
        if not set(sents) <= set(np.asarray(src_row["u_sents"]).tolist()):
            raise RuntimeError(f"{where}: the stored selection is neither from this pool nor from the first "
                               f"redraw pool that spans k dimensions ({path}). Delete the file and rerun")
        return {"redrawn": True, "prompt": prompt, "rank": rank, "n_unique": len(row["u_sents"]), "from": str(path),
                "info": row_info(sT, prompt, src_row)}

    for sT in embeddings_dict:
        bucket = embeddings_dict[sT]
        output_l = cached_list(path_of(sT))
        processed = set(x[0] for x in output_l)
        t0, n_new = time.perf_counter(), 0
        seen, fresh = set(), {}
        for prompt, u_sents, counts, logps, logps_t1, embeddings in iter_prompt_data(bucket, pool_size):
            seen.add(prompt)
            if prompt in processed:
                continue
            log_s = log_scales(sT, counts, logps, logps_t1)
            # drawn before the rank test, also for a pool that is then redrawn, so every other prompt's
            # draw uses the same random numbers as in runs made before the test
            selected = select(log_s, embeddings, k)
            rank = span_rank(log_s, embeddings)
            if rank < k and len(selected) == k:
                raise RuntimeError(f"{label}: sT={fmt(sT)} prompt={prompt[:40]!r}: k={k} items selected from rows "
                                   f"whose rank test gives {rank} < k: the selection's tolerance and RANK_TOL "
                                   f"disagree. Not redrawn; fix the tolerances")
            if rank < k:
                rec, fresh[prompt] = redrawn(sT, prompt, rank, len(u_sents))
            elif len(selected) != k:
                raise RuntimeError(f"{label}: sT={fmt(sT)} prompt={prompt[:40]!r}: {len(selected)} of k={k} items "
                                   f"selected from a pool that spans {rank} >= k dimensions: a numerical failure, "
                                   f"not Zhat = 0, so the pool is not redrawn. Fix the selection for this kernel")
            else:
                rec = selection_record(prompt, selected, u_sents, logps_t1, embeddings)
                fresh[prompt] = dict(redrawn=False, info=info_of(log_s, embeddings))
            output_l.append(rec)
            n_new += 1
        for prompt, row in bucket.items():  # not yielded by iter_prompt_data: fewer than two unique strings
            if prompt in seen or prompt in processed:
                continue
            rank = span_rank(np.zeros(len(row["embeddings"])), row["embeddings"])
            if rank >= k:
                raise ValueError(f"{label}: sT={fmt(sT)} prompt={prompt[:40]!r} has one unique string and k={k}: "
                                 f"run with k >= 2 (a single-string pool has Zhat = 0 only for k >= 2)")
            rec, fresh[prompt] = redrawn(sT, prompt, rank, len(row["u_sents"]))
            output_l.append(rec)
            n_new += 1
        output_l._save()
        order = [x[0] for x in output_l]
        if len(order) != len(set(order)) or set(order) != set(bucket):
            raise ValueError(f"{path_of(sT)}: its prompts do not match the pool's: {len(order)} records, "
                             f"{len(set(order))} distinct, {len(set(order) - set(bucket))} not in the pool, "
                             f"{len(set(bucket) - set(order))} missing. Delete the file and rerun")
        entries = []
        for x in output_l:
            if len(x[1]) != k:
                raise ValueError(f"{path_of(sT)}: prompt={x[0][:40]!r} has {len(x[1])} selected items, not k={k}")
            entries.append(fresh[x[0]] if x[0] in fresh else stored_entry(sT, x[0], x[1], bucket[x[0]]))
        side = write_redraws(path_of(sT), label, k, sT, pool_size, entries, sampler)
        if n_new or side["n_redrawn"]:
            ms = f"{(time.perf_counter() - t0) / n_new * 1000:.0f}ms, " if n_new else ""
            print(f"s{fmt(sT)}({ms}{side['n_redrawn']}/{side['n_prompts']} redrawn)", end=" ", flush=True)
    print()


def draw_one(log_s, embeddings, k):
    return sample_dpp(log_s, embeddings, k)[0]


# ---------------------------------------------------------------------------
# Methods (all write {output}/{model}-{method}/... inside one run dir)
# ---------------------------------------------------------------------------
def run_samp(embeddings_dict, output_folder, model_name, k, taus, sigma, pool_size=None, redraw=None):
    """PoolDPP-Samp on w = plm^(2*beta), sweeping tau."""
    method_dir = f"{output_folder}/{model_name}-samp"
    for tau in taus:
        run_kernel_method(
            embeddings_dict,
            lambda sT, tau=tau: f"{method_dir}/tau{fmt(tau)}_s{fmt(sT)}.pkl",
            lambda sT, c, lq, lp, tau=tau: quality_log_scales(lp, tau / sigma[sT], c),
            draw_one, k, f"samp tau={fmt(tau)}", pool_size, redraw, sampler=True)


def run_samp_T(embeddings_dict, output_folder, model_name, k, target_Ts, pool_size=None, redraw=None):
    """PoolDPP-Samp on w = plm^(1/T) / p_sT, sweeping T."""
    method_dir = f"{output_folder}/{model_name}-samp"
    for T in target_Ts:
        run_kernel_method(
            embeddings_dict,
            lambda sT, T=T: f"{method_dir}/T{fmt(T)}_s{fmt(sT)}.pkl",
            lambda sT, c, lq, lp, T=T: target_log_scales(lq, lp, c, T),
            draw_one, k, f"samp T={fmt(T)}", pool_size, redraw, sampler=True)


def run_map(embeddings_dict, output_folder, model_name, k, taus, sigma, pool_size=None,
            target_Ts=None, redraw=None):
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
            greedy_map, k, f"map {name}", pool_size, redraw)


def run_naive(embeddings_dict, output_folder, model_name, k, redraw=None):
    """Naive DPP baseline: diversity-only kernel, the DPP over the raw pool with
    duplicate rows collapsed, i.e. row sqrt(count_i) * emb_i."""
    method_dir = f"{output_folder}/{model_name}-naive"
    run_kernel_method(
        embeddings_dict,
        lambda sT: f"{method_dir}/s{fmt(sT)}.pkl",
        lambda sT, c, lq, lp: quality_log_scales(lp, 0.0, c),
        draw_one, k, "naive", redraw=redraw, sampler=True)


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
    if n < k:
        raise ValueError(f"k-Means: the pool has {n} unique strings, fewer than k={k}, so k clusters do not "
                         f"exist. Not shrunk or skipped: how the baseline handles such a pool (redraw it as "
                         f"PoolDPP does, or keep the {n} strings and report it) is a decision for timv/Manuel")
    k_eff = k
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
            raise RuntimeError(f"k-Means: cluster {c} of k={k} is empty ({n} strings, {len(np.unique(F, axis=0))} "
                               f"distinct embeddings), so it selects fewer than k strings. Not shrunk: a "
                               f"decision for timv/Manuel")
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
            try:
                selected = sample_kmeans_for_prompt(embeddings, logps_t1, k, random_state=random_state)
            except (ValueError, RuntimeError) as e:
                raise type(e)(f"sT={fmt(sT)} prompt={prompt[:40]!r}: {e}") from e
            output_l.append(selection_record(prompt, selected, u_sents, logps_t1, embeddings))
        output_l._save()
        short = [x[0] for x in output_l if len(x[1]) != k]
        done = set(x[0] for x in output_l)
        missing = [p for p in embeddings_dict[sT] if p not in done]
        if short or missing:
            raise ValueError(f"k-Means sT={fmt(sT)}: {len(missing)} prompts have no selection (pools with fewer "
                             f"than two unique strings; first: {(missing or [''])[0][:40]!r}) and {len(short)} "
                             f"stored selections have fewer than k={k} strings (written before this check). Not "
                             f"skipped: how k-Means handles pools with fewer than k unique strings is a decision "
                             f"for timv/Manuel")
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
    parser.add_argument("--redraw_dirs", nargs="+", default=None,
                        help="Other pools of this task, tried in order for a prompt whose pool has no size-k set")
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
    redraw = PoolRedraw(args.redraw_dirs, args.model_name) if args.redraw_dirs else None

    def seed_method(method, run):
        seed = method_seed(method, run, pool)
        np.random.seed(seed)
        if redraw is not None:
            redraw.seed = seed

    for run in range(args.N):
        run_dir = f"{args.output_folder}/{run}"
        if args.N > 1:
            print(f" run {run}")
        if args.samp and args.taus:
            seed_method("samp", run)
            run_samp(active, run_dir, args.model_name, args.k, args.taus, sigma,
                     pool_size=args.pool_size, redraw=redraw)
        if args.samp and args.target_Ts:
            seed_method("samp_T", run)
            run_samp_T(active, run_dir, args.model_name, args.k, args.target_Ts,
                       pool_size=args.pool_size, redraw=redraw)
        if args.map_ and run == 0:
            seed_method("map", run)  # only --pool-size draws from it
            run_map(active, run_dir, args.model_name, args.k, args.taus, sigma,
                    pool_size=args.pool_size, target_Ts=args.target_Ts, redraw=redraw)
        if args.ancestral:
            np.random.seed(method_seed("ancestral", run, pool))
            run_ancestral(active, run_dir, args.model_name, args.k)
        if args.naive:
            seed_method("naive", run)
            run_naive(active, run_dir, args.model_name, args.k, redraw=redraw)
        if args.kmeans:
            np.random.seed(method_seed("kmeans", run, pool))
            run_kmeans(active, run_dir, args.model_name, args.k,
                       random_state=run, pool_size=args.pool_size)


if __name__ == "__main__":
    cli()
