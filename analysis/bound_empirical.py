"""
Empirical test of the Z-free variance identity (thm:zfree)
================================================================================
The identity under test, in the paper's notation (N = pool size, q = proposal,
w = kappa/q, P = target K-DPP, Z = normalizer):

    Var_omega(Zhat)/Z^2 = sum_{c=1}^K a_c * Var_{q^c}(u_c),
    a_c = C(K,c) C(N-K,K-c) / C(N,K)
    u_c(y_1..y_c) = P({y_1..y_c} subseteq S) / ((K)_c q(y_1)..q(y_c))

u_c is the inclusion ratio of the note; Var(u_c) equals chi2(pi_c || q^c) for
the normalized c-point marginal pi_c, which is why the variables below are named
chi2_*. Corollaries (formulas in bound_formulas.py):
  - lower bounds on Var(Zhat)/Z^2: PRIMARY (K^2/N) Var(u_1) (Hoeffding 1948,
    Thm 5.2: Var(u_c)/c is nondecreasing in c), SECONDARY the exact leading
    term a_1 Var(u_1), from the K-DPP marginal formula;
  - PRIMARY, second-moment bound. With m_1 = E_q[u_1^2] = 1 + Var_q(u_1),
      KL <= log(1 + a_1 Var(u_1) + sum_{c>=2} a_c ((K^c/(K)_c)^2 m_1^c - 1))
         = log sum_{c=0}^K a_c (K^c/(K)_c)^2 m_1^c          (hybrid, cor:kl-sum)
    and, with theta = K^2 m_1/(N-K+1) and a_0 = C(N-K,K)/C(N,K),
      KL <= log(e^theta - 1 + a_0) <= theta            (closed form, thm:zfree);
  - SECONDARY, sup bound (the paper keeps it in a footnote). With
    r_1 = sup_y u_1(y),
      KL <= log(1 + a_1 Var(u_1) + sum_{c>=2} a_c (K^c/(K)_c r_1^c - 1))
    and KL <= K^2 r_1/(N-K+1).
Both use negative association, u_c <= K^c/(K)_c prod_i u_1(y_i). m_1 <= r_1, so
the second-moment closed form is never the larger one; the two hybrids are not
ordered (see bound_formulas.py), so the summary reports how often the
second-moment hybrid is the larger.

Code note: this file predates the switch to the paper's symbols, so the pool
size is spelled M/M_exp/M_grid here; it is the paper's N throughout. The JSON
keys cert_* predate the second-moment bound and hold the r_1 (sup) versions;
their meaning is unchanged.

Per-prompt row keys (bound_raw_{task}.json, analyze stage):
  chi2_1               Var_q(u_1) = E_q[u_1^2] - 1 = sum_y pi_1^2/q - 1
  m1                   E_q[u_1^2] = 1 + chi2_1                       (new)
  r1                   sup_y u_1(y)                                  (secondary)
  lower_hoeffding      (K^2/N) chi2_1, lower bound on Var(Zhat)/Z^2   (new, primary)
  a1, leading          a_1(N), exact leading term a_1 chi2_1          (secondary lower bound)
  na_tail_m1           sum_{c>=2} a_c ((K^c/(K)_c)^2 m_1^c - 1)        (new)
  bound_hybrid_m1_KL   log(1 + leading + na_tail_m1)                 (new, primary)
  bound_simple_m1_KL   log(e^theta - 1 + a_0), theta = K^2 m_1/(N-K+1) (new, primary;
                       the first commit of this change stored theta here)
  bound_theta_m1       theta = K^2 m_1/(N-K+1)                       (new)
  na_tail              sum_{c>=2} a_c (K^c/(K)_c r_1^c - 1)            (secondary)
  cert_hybrid_KL       log(1 + leading + na_tail)                    (secondary)
  cert_simple_KL       K^2 r_1/(N-K+1)                               (secondary)
  pred_lower_hoeffding_by_m  {m: (K^2/m) chi2_1}, lower bound on Var(Zhat)/Z^2 (new, primary)
  pred_leading_by_m    {m: a_1(m) chi2_1}, exact lower bound on Var(Zhat)/Z^2 (secondary)
  pred_upper_m1_by_m   {m: second-moment upper bound on Var(Zhat)/Z^2}   (new)
  pred_upper_by_m      {m: r_1 upper bound on Var(Zhat)/Z^2}         (secondary)
  log10_covered_mass   log10 sum_y q(y) over the prompt's distinct observed
                       strings: the exact proposal mass the pool-built
                       instance covers (see "Covered proposal mass")  (new)
  covered_mass         10**log10_covered_mass; underflows to 0.0 when the
                       strings are long, so read the log10 value      (new)
  missing_mass         Good-Turing ESTIMATE of the uncovered mass, (#strings
                       seen once)/(#draws); kept so earlier outputs stay
                       comparable; superseded by 1 - covered_mass   (unchanged)
  others (mc_*, measured_relvar, sub_*, iid, ...) unchanged.
Summary keys (bound_empirical_{task}_{stage}.json, "{task}|summary"): the
medians of the above (median_m1, median_lower_hoeffding,
median_bound_hybrid_m1_KL, median_bound_simple_m1_KL, median_bound_theta_m1,
median_log10_covered_mass, median_covered_mass are new; median_missing_mass is
still the Good-Turing estimate; median_r1, median_cert_* are the r_1 versions),
median_lower_hoeffding_over_mc (new) next to median_leading_over_mc,
predicted_lower_hoeffding_by_m (new) next to predicted_leading_by_m (a_1),
predicted_upper_m1_by_m (new) next to predicted_upper_by_m (r_1),
frac_below_hoeffding_lower_bound (new) next to frac_below_exact_lower_bound
(a_1; both should be 0 up to block-estimate noise),
frac_hybrid_m1_above_r1 (new: fraction of prompts where bound_hybrid_m1_KL >
cert_hybrid_KL), and frac_above_upper_m1_bound (new: fraction of prompts whose
block relvar at m exceeds the second-moment upper bound; nonzero values come
from Monte Carlo noise in the block estimate or a broken bound).

Three tests, all on the real caches at the experimental K, D and pool size:

  T1  Exactness on real data. Small real sub-instances (real embeddings, real
      LM weights) where all |Y|^M sketch realizations are enumerable: compare
      the exact Var(Zhat)/Z^2 to the identity RHS computed term by term,
      check log(1 + exact) against both hybrids and both closed forms, and
      check Hoeffding's bracket (K^2/M) Var(u_1) <= Var/Z^2 <= (K/M) Var(u_K).

  T2  Experimental scale. Ground set = the pooled distinct strings of one
      prompt across all pools (q = empirical frequency, w = the real importance
      weight, kappa = w*q, so C = E_q[w phi phi^T] exactly). Everything about
      the target is then exactly computable: Z = e_K(eig C), and P(y in S) by
      the leave-one-out elementary symmetric polynomial identity
      P(y in S) Z = kappa(y) sum_d <v_d,phi_y>^2 e_{K-1}(lambda_{-d}),
      giving u_1 = P(y in S)/(K q(y)). Compare: MC estimate of the LHS
      (= the full RHS), the exact leading term, and the bounds (m_1 primary,
      r_1 secondary).

  T3  Rate/structure on the TRUE instance. Disjoint blocks of a pool's N i.i.d.
      draws are independent draws of m i.i.d. points from the true q, so
      blocking real pools measures the true LHS at pool size m with no
      surrogate. The identity predicts relvar(m) = sum_c a_c(m) Var(u_c) with
      m-independent Var(u_c): slope -1 in log-log once m >> K^2, steeper below.

Weights. log w below is the L-ensemble (diagonal) weight: rho = w enters the
pool B-rows as sqrt(rho), so L_yy = rho_y |phi_y|^2. PoolDPP-Samp at T has
log w = logp_t1 / T - logp.
All quantities in the bound are invariant to the global scale of kappa; we fix
the gauge E_q[rho] = 1 (so kappa_Sigma = 1).

Covered proposal mass. The T2 instance replaces the proposal by the empirical
frequencies of the strings it saw, so it leaves out every string no pool drew.
How much proposal mass that is needs no estimator: logps is log q(y) for each
distinct observed string (one entry per column of Cnt), where q is the proposal
the pools were drawn from, i.e. the sampler's log-probabilities at the local
sampling temperature sT (lm_dpp/sample_vllm.py stores vLLM's processed,
at-temperature cumulative log-prob, EOS token included), the same q as in
log_w. So the covered mass is exactly sum_y q(y) = exp(logsumexp(logps)), and
the missing mass is exactly 1 minus it. Three caveats, all unmeasured: logps is
the log-prob of the token sequence first drawn for a string, so a string the
LM can also tokenize another way has q(string) >= exp(logps), which makes the
covered mass a lower bound in that case; a string drawn by several pools
keeps the logps of the first pool that drew it (extract); and a generation cut
off at max_new_tokens has the log-prob of its prefix, so it covers every
string that extends that prefix. The analysis stops
with an error if a log q(y) is not finite, or if the sum exceeds 1 (checked
without tolerance), since then logps are not log-probabilities of distinct
strings under one proposal.

Pools. ambrosia/gsm8k read {cache}/pool{i}/; fineweb reads the flat
{cache}/ pool plus every cache-fineweb-pools/pool*/ (independent seeds) when
present. --pool-files overrides. Extracted ground sets are cached under
$BOUND_SCRATCH (default analysis/bound_work/) keyed by a fingerprint of the input
files, so a changed pool set is never served from a stale extract.

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/bound_empirical.py
    ... --task ambrosia|gsm8k|both --stage extract|analyze|all
"""

import argparse
import glob
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from scipy.special import logsumexp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import bound_formulas as bf  # noqa: E402
from dpp.stable import StableKDPP  # noqa: E402
SCRATCH = os.environ.get("BOUND_SCRATCH", os.path.join(SCRIPT_DIR, "bound_work"))
# e_K from eigvalsh is accurate while the K-th eigenvalue is well above rounding noise (1e-16 of the top).
EIG_FLOOR = 1e-8
_POOLED = None  # prompt -> ground set, shared with the worker processes by fork

CONFIGS = {
    "ambrosia": dict(
        cache="cache-ambrosia", n_pools=100,
        fname="google__gemma-2-9b-it_minilm_embeddings.pkl",
        sT=1.0, K=3, D=384, M_exp=200,
        M_grid=[10, 20, 25, 40, 50, 100, 200],
    ),
    "fineweb": dict(
        cache="cache-fineweb", n_pools=1, flat=True,
        extra_pool_glob="cache-fineweb-pools/pool*",
        fname="google__gemma-2-9b-it_minilm_embeddings.pkl",
        sT=1.0, K=10, D=384, M_exp=400,
        M_grid=[10, 20, 25, 50, 100, 200],
    ),
    "gsm8k": dict(
        cache="cache-gsm8k", n_pools=100,
        fname="google__gemma-2-9b-it_qwen_embeddings.pkl",
        sT=1.2, K=10, D=1024, M_exp=100,
        M_grid=[20, 25, 33, 50, 100],
    ),
}


# --------------------------------------------------------------------------
# elementary symmetric polynomials
# --------------------------------------------------------------------------
def log_esp_prefix_suffix(log_lam, K):
    """Stable log-space e_k DP with leave-one-out values.

    Input log_lam: log of the (positive) eigenvalues, shape (n,).
    Returns (log_eK, log_eK_minus_1_loo) where log_eK = log e_K(lam) and
    log_eK_minus_1_loo[d] = log e_{K-1}(lam with index d removed).
    All terms are positive, so the DP has no cancellation.
    """
    n = len(log_lam)
    NEG = -np.inf
    # prefix A[j, k] = e_k(lam_0..lam_{j-1})
    A = np.full((n + 1, K + 1), NEG)
    A[:, 0] = 0.0
    for j in range(1, n + 1):
        A[j, 1:] = np.logaddexp(A[j - 1, 1:], log_lam[j - 1] + A[j - 1, :-1])
    # suffix B[j, k] = e_k(lam_j..lam_{n-1})
    B = np.full((n + 1, K + 1), NEG)
    B[:, 0] = 0.0
    for j in range(n - 1, -1, -1):
        B[j, 1:] = np.logaddexp(B[j + 1, 1:], log_lam[j] + B[j + 1, :-1])
    log_eK = A[n, K]
    # e_{K-1}(lam_{-d}) = sum_i A[d, i] * B[d+1, K-1-i]
    ks = np.arange(K)                                  # i = 0..K-1
    terms = A[:-1, ks] + B[1:, (K - 1) - ks]           # (n, K)
    loo = np.logaddexp.reduce(terms, axis=1)
    return float(log_eK), loo


def batch_log_e_k(eig_batch, K):
    """log e_K for a batch of eigenvalue vectors, shape (R, M) -> (R,).

    Scales each row by its max eigenvalue so all scaled eigenvalues are <= 1
    (e_K <= C(M,K), no overflow), then corrects by K*log(max).
    """
    E = np.clip(np.asarray(eig_batch, dtype=np.float64), 0.0, None)
    mx = E.max(axis=1)
    ok = mx > 0
    out = np.full(E.shape[0], -np.inf)
    if not ok.any():
        return out
    Es = E[ok] / mx[ok][:, None]
    R, M = Es.shape
    coef = np.zeros((R, K + 1))
    coef[:, 0] = 1.0
    for j in range(M):
        lam = Es[:, j]
        for k in range(min(K, j + 1), 0, -1):
            coef[:, k] += lam * coef[:, k - 1]
    eK = coef[:, K]
    good = eK > 0
    res = np.full(R, -np.inf)
    res[good] = np.log(eK[good]) + K * np.log(mx[ok][good])
    out[ok] = res
    return out


def log_alpha(M, K):
    if K > M:
        return np.inf
    return -sum(math.log1p(-j / M) for j in range(K))


def a_coeffs(M, K):
    """a_c(M) = C(K,c) C(M-K,K-c)/C(M,K) for c=1..K (0 when infeasible)."""
    return bf.a_coeffs(M, K)


# --------------------------------------------------------------------------
# instance construction
# --------------------------------------------------------------------------
def log_w(logps, logps_t1, T):
    """log of the L-ensemble weight rho of PoolDPP-Samp, plm^(1/T) / q."""
    return logps_t1 / T - logps


def build_instance(Phi, cnt_tot, logps, logps_t1, T):
    """Ground-set instance: q empirical, rho = w (gauge E_q[rho]=1), kappa=rho*q."""
    q = cnt_tot / cnt_tot.sum()
    lw = log_w(logps, logps_t1, T)
    lw = lw - lw.max()
    rho = np.exp(lw)
    rho = rho / float(q @ rho)              # gauge: E_q[rho] = 1
    kappa = rho * q                          # kappa_Sigma = 1
    return q, rho, kappa


def stable_log_e_K(log_s, Phi, K):
    """log e_K of the spectrum of B B^T for rows exp(log_s_i) Phi_i, or None if it is zero or the
    rows need tiers. The graded SVD stays accurate however far the scales spread, and rows that
    share a direction are merged, because a pair of them adds a singular value of rounding noise."""
    dpp = StableKDPP(log_s, Phi, K)
    if dpp.tiers is None or len(dpp.tiers[1]) < K:
        return None
    return log_esp_prefix_suffix(2.0 * dpp.tiers[1], K)[0]


def exact_target(Phi, kappa, K):
    """Exact target quantities: log Z, one-point inclusion marginals P_1. Computed from the graded
    SVD: weights plm^(1/T)/q span more nats than a matrix C = B^T B can resolve in double precision."""
    with np.errstate(divide="ignore"):
        dpp = StableKDPP(0.5 * np.log(kappa), Phi, K)
    P1 = dpp.inclusion_probs()
    if P1 is None:
        return None
    ls = dpp.tiers[1]
    return dict(logZ=log_esp_prefix_suffix(2.0 * ls, K)[0], P1=P1, lam=np.exp(2.0 * ls),
                sumP1=float(P1.sum()))


def mc_relvar(Phi, rho, q, K, M, logZ, R, rng, batch=256):
    """MC estimate of Var(Zhat)/Z^2 and E[Zhat]/Z under M i.i.d. draws from q."""
    la = log_alpha(M, K)
    vals = []
    done = 0
    while done < R:
        b = min(batch, R - done)
        idx = rng.choice(len(q), size=(b, M), p=q)
        eigs = np.empty((b, M))
        for r in range(b):
            ii = idx[r]
            Br = Phi[ii] * np.sqrt(rho[ii] / M)[:, None]
            eigs[r] = np.clip(np.linalg.eigvalsh(Br @ Br.T), 0.0, None)
        le = batch_log_e_k(eigs, K)
        for r in np.flatnonzero(eigs[:, -K] <= EIG_FLOOR * eigs[:, -1]):
            ii = idx[r]
            with np.errstate(divide="ignore"):
                lz = stable_log_e_K(0.5 * np.log(rho[ii] / M), Phi[ii], K)
            le[r] = -np.inf if lz is None else lz
        vals.append(np.exp(le + la - logZ))
        done += b
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    mu = float(v.mean())
    rv = float(v.var(ddof=1) / (mu * mu)) if mu > 0 else float("nan")
    # SE of the relative variance (delta method on the 2nd/1st moment ratio)
    m2 = float((v ** 2).mean())
    se_m2 = float(v.__pow__(2).std(ddof=1) / math.sqrt(len(v)))
    se_rv = se_m2 / (mu * mu) if mu > 0 else float("nan")
    return dict(relvar=rv, mean_ratio=mu, se_relvar=se_rv, n=int(len(v)),
                m2_over_Z2=m2)


def pool_logzhat(Phi, rho, cnt_row, K):
    """log Zhat for one realized pool (counts = multiplicities), shared gauge."""
    sel = cnt_row > 0
    if sel.sum() < K:
        return None
    M = float(cnt_row.sum())
    A = Phi[sel] * np.sqrt(rho[sel] * cnt_row[sel] / M)[:, None]
    # nonzero spectra of A A^T and A^T A coincide; eigendecompose the smaller
    G = A @ A.T if A.shape[0] <= A.shape[1] else A.T @ A
    lam = np.clip(np.linalg.eigvalsh(G), 0.0, None)
    if len(lam) < K:
        return None
    if lam[-K] <= EIG_FLOOR * lam[-1]:
        with np.errstate(divide="ignore"):
            log_eK = stable_log_e_K(0.5 * np.log(rho[sel] * cnt_row[sel] / M), Phi[sel], K)
        return None if log_eK is None else log_eK + log_alpha(int(M), K)
    lam = lam[lam > 0]
    log_eK, _ = log_esp_prefix_suffix(np.log(lam), K)
    if not np.isfinite(log_eK):
        return None
    return log_eK + log_alpha(int(M), K)


def log_covered_mass(logps, Cnt, pools, where):
    """Natural log of sum_y q(y) over the distinct observed strings, from their
    log q(y) (rec["logps"], one entry per column of Cnt): the exact proposal
    mass the pool-built instance covers (module docstring, "Covered proposal
    mass"). Raises ValueError, naming `where` and the pools that drew the
    offending string, if a log q(y) is not finite (-inf: q(y) = 0 for a string
    that was drawn; NaN: the sampler reported no at-temperature log-prob) or if
    the sum exceeds 1 (no tolerance)."""
    logps = np.asarray(logps, dtype=np.float64)
    if logps.shape != (Cnt.shape[1],):
        raise ValueError(f"{where}: {logps.shape[0]} log-probs for {Cnt.shape[1]} distinct strings")
    bad = np.flatnonzero(~np.isfinite(logps))
    if bad.size:
        j = int(bad[0])
        drew = [pools[r] for r in np.flatnonzero(Cnt[:, j])]
        raise ValueError(
            f"{where}: {bad.size} of {logps.size} distinct strings have a non-finite "
            f"log q(y) (first: string {j}, log q = {float(logps[j])!r}, drawn by pool(s) {drew}); "
            f"the covered proposal mass and the importance weights are undefined")
    lse = float(logsumexp(logps))
    if lse > 0.0:
        raise ValueError(
            f"{where}: the distinct observed strings have total proposal mass "
            f"exp({lse:.6g}) > 1, so logps are not log-probabilities of distinct "
            f"strings under one proposal")
    return lse


def iid_diagnostics(Cnt, logps):
    """Is the pool an i.i.d. sample from the LM, and are pools independent?

    Two checks against the LM's own log-probabilities q(y) = exp(logps):
      (a) within-pool collisions. E[#colliding pairs] = C(N,2) sum_y q_y^2.
          Summing only over observed y underestimates sum q^2, but the
          collision mass sits on the top-q strings, which are observed.
      (b) cross-pool shared strings. Under independent pools,
          E[#shared between two pools] = sum_y (1-(1-q_y)^N)^2.
    """
    q = np.exp(logps)
    n_pools, _ = Cnt.shape
    N = Cnt.sum(1)
    obs_coll, pred_coll = [], []
    for r in range(n_pools):
        c = Cnt[r]
        obs_coll.append(float((c * (c - 1) / 2).sum()))
        pred_coll.append(float(N[r] * (N[r] - 1) / 2 * (q ** 2).sum()))
    # q_emp: pooled empirical distribution. Consistent for q when the
    # missing mass is small, and it does not depend on the LM's log-probs.
    q_emp = Cnt.sum(0) / Cnt.sum()
    obs_share, pred_share, pred_share_emp = [], [], []
    for r in range(min(n_pools, 8)):
        for s in range(r + 1, min(n_pools, 8)):
            both = ((Cnt[r] > 0) & (Cnt[s] > 0)).sum()
            obs_share.append(float(both))
            for qq, acc in ((q, pred_share), (q_emp, pred_share_emp)):
                qc = np.clip(qq, 0.0, 1 - 1e-16)
                ir = 1.0 - np.exp(np.log1p(-qc) * N[r])
                iss = 1.0 - np.exp(np.log1p(-qc) * N[s])
                acc.append(float((ir * iss).sum()))
    return dict(
        obs_collisions=float(np.mean(obs_coll)),
        pred_collisions=float(np.mean(pred_coll)),
        obs_shared_pairwise=float(np.mean(obs_share)) if obs_share else None,
        pred_shared_pairwise=float(np.mean(pred_share)) if pred_share else None,
        pred_shared_pairwise_emp=(float(np.mean(pred_share_emp))
                                  if pred_share_emp else None),
        max_q=float(q.max()), sum_q_observed=float(q.sum()),
    )


def block_relvar(Phi, rho, Cnt, K, m, min_blocks=10):
    """relvar of Zhat at sketch size m from DISJOINT blocks of each real pool.

    Disjoint blocks of an i.i.d. sample are independent draws of m i.i.d.
    points, so this measures the true-instance Var(Zhat)/Z^2 at M=m without
    assuming the pools themselves are independent of each other.
    """
    logs = []
    for r in range(Cnt.shape[0]):
        cr = Cnt[r].astype(int)
        expand = np.repeat(np.arange(len(cr)), cr)
        n_blk = len(expand) // m
        if n_blk < 1:
            continue
        perm = np.random.default_rng(1000 + r).permutation(expand)
        for b in range(n_blk):
            blk = perm[b * m:(b + 1) * m]
            row = np.bincount(blk, minlength=len(cr)).astype(float)
            lz = pool_logzhat(Phi, rho, row, K)
            if lz is not None:
                logs.append(lz)
    if len(logs) < min_blocks:
        return None, len(logs)
    return relvar_from_logs(logs), len(logs)


def relvar_from_logs(logs):
    L = np.asarray(logs, dtype=np.float64)
    v = np.exp(L - L.max())
    mu = v.mean()
    if mu <= 0:
        return None
    return float(v.var(ddof=1) / (mu * mu))


# --------------------------------------------------------------------------
# stage 1: extract compact per-prompt pooled ground sets
# --------------------------------------------------------------------------
def pool_files(task):
    """Embedding pkls that make up the task's pools, in pool order."""
    cfg = CONFIGS[task]
    if cfg.get("pool_files"):
        return list(cfg["pool_files"])
    if cfg.get("flat"):
        files = [os.path.join(ROOT_DIR, cfg["cache"], cfg["fname"])]
        if cfg.get("extra_pool_glob"):
            extra = glob.glob(os.path.join(ROOT_DIR, cfg["extra_pool_glob"]))
            extra = sorted(extra, key=lambda d: int("".join(
                ch for ch in os.path.basename(d) if ch.isdigit()) or 0))
            files += [os.path.join(d, cfg["fname"]) for d in extra]
        return files
    return [os.path.join(ROOT_DIR, cfg["cache"], f"pool{pi}", cfg["fname"])
            for pi in range(cfg["n_pools"])]


def pooled_path(task):
    """Scratch path of the extracted ground set, keyed by the input files
    (path, size, mtime) so a changed or extended pool set re-extracts."""
    h = hashlib.sha1()
    for p in pool_files(task):
        try:
            st = os.stat(p)
            h.update(f"{os.path.realpath(p)}|{st.st_size}|{int(st.st_mtime)}".encode())
        except FileNotFoundError:
            h.update(f"{p}|missing".encode())
    h.update(f"sT={CONFIGS[task]['sT']}".encode())
    return os.path.join(SCRATCH, f"{task}_pooled_{h.hexdigest()[:12]}.pkl")


def extract(task):
    cfg = CONFIGS[task]
    os.makedirs(SCRATCH, exist_ok=True)
    dest = pooled_path(task)
    if os.path.exists(dest):
        print(f"[extract] {dest} exists, skipping")
        return dest
    store = {}           # prompt -> dict(sents, Phi, logps, logps_t1, Cnt rows)
    order = {}
    for pi, p in enumerate(pool_files(task)):
        if not os.path.exists(p):
            print(f"[extract] missing {p}")
            continue
        with open(p, "rb") as f:
            emb = pickle.load(f)
        if cfg["sT"] not in emb:
            print(f"[extract] pool{pi}: no sT={cfg['sT']}")
            continue
        for pt, d in emb[cfg["sT"]].items():
            s = np.asarray(d["u_sents"])
            cnt = np.asarray(d["counts"], dtype=np.float64)
            lp = np.asarray(d["logps"], dtype=np.float64)
            lp1 = np.asarray(d["logps_t1"], dtype=np.float64)
            ph = np.asarray(d["embeddings"], dtype=np.float64)
            if not (len(s) == len(cnt) == len(lp) == len(lp1) == len(ph)):
                continue
            st = store.setdefault(pt, dict(idx={}, sents=[], Phi=[], lp=[],
                                           lp1=[], rows={}))
            row = defaultdict(float)
            for j, sj in enumerate(s):
                k = st["idx"].get(sj)
                if k is None:
                    k = len(st["sents"])
                    st["idx"][sj] = k
                    st["sents"].append(sj)
                    st["Phi"].append(ph[j])
                    st["lp"].append(lp[j])
                    st["lp1"].append(lp1[j])
                row[k] += cnt[j]
            st["rows"][pi] = dict(row)
        order[pi] = True
        del emb
        print(f"[extract] pool{pi} done ({len(store)} prompts) <- {p}", flush=True)

    out = {}
    for pt, st in store.items():
        n = len(st["sents"])
        pools = sorted(st["rows"])
        Cnt = np.zeros((len(pools), n))
        for r, pi in enumerate(pools):
            for k, v in st["rows"][pi].items():
                Cnt[r, k] = v
        out[pt] = dict(
            Phi=np.asarray(st["Phi"]), logps=np.asarray(st["lp"]),
            logps_t1=np.asarray(st["lp1"]), Cnt=Cnt, pools=pools,
            n_uniq=n,
        )
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=4)
    os.replace(tmp, dest)
    print(f"[extract] saved {dest}: {len(out)} prompts")
    return dest


# --------------------------------------------------------------------------
# T1: exact enumeration on real sub-instances
# --------------------------------------------------------------------------
def t1_enumeration(task, pooled, rng, cases):
    cfg = CONFIGS[task]
    temperature = cfg["sweeps"][0]
    res = []
    prompts = sorted(pooled)
    for (nY, K, M, d_sub) in cases:
        pt = prompts[len(res) % len(prompts)]
        rec = pooled[pt]
        if rec["n_uniq"] < nY:
            continue
        # pick the nY most frequent strings (real embeddings, real weights)
        tot = rec["Cnt"].sum(0)
        sel = np.argsort(-tot)[:nY]
        Phi = rec["Phi"][sel]
        if d_sub is not None and d_sub < Phi.shape[1]:
            Phi = Phi[:, :d_sub]
            nrm = np.linalg.norm(Phi, axis=1, keepdims=True)
            Phi = Phi / np.where(nrm > 0, nrm, 1.0)     # keep unit features
        q, rho, kappa = build_instance(Phi, tot[sel], rec["logps"][sel], rec["logps_t1"][sel], temperature)

        tgt = exact_target(Phi, kappa, K)
        if tgt is None:
            continue
        Z = math.exp(tgt["logZ"])
        # exact target weights f(S) for every K-subset
        L = (np.sqrt(kappa)[:, None] * Phi) @ (np.sqrt(kappa)[:, None] * Phi).T
        subsets = list(itertools.combinations(range(nY), K))
        fS = {S: float(np.linalg.det(L[np.ix_(S, S)])) for S in subsets}
        Zc = sum(fS.values())

        # exact Var(Zhat) by enumerating all omega in Y^M
        alpha = math.exp(log_alpha(M, K))
        Isub = list(itertools.combinations(range(M), K))
        EZ = EZ2 = 0.0
        for omega in itertools.product(range(nY), repeat=M):
            w = float(np.prod(q[list(omega)]))
            B = np.sqrt(rho[list(omega)] / M)[:, None] * Phi[list(omega)]
            Lh = B @ B.T
            Zh = alpha * sum(np.linalg.det(Lh[np.ix_(I, I)]) for I in Isub)
            EZ += w * Zh
            EZ2 += w * Zh * Zh
        var_exact = EZ2 - EZ * EZ

        # identity RHS, all c, exact
        ac = a_coeffs(M, K)
        chi2s = []
        for c in range(1, K + 1):
            fall = math.perm(K, c)
            s = 0.0
            for tup in itertools.permutations(range(nY), c):
                T = set(tup)
                Pc = sum(fS[S] for S in subsets if T.issubset(S)) / Zc
                s += (Pc / fall) ** 2 / float(np.prod(q[list(tup)]))
            chi2s.append(s - 1.0)
        rhs = float(np.dot(ac, chi2s))

        # exact leading term via the marginal-kernel formula (independent path)
        pi1 = tgt["P1"] / K
        m1 = float((pi1 ** 2 / q).sum())           # E_q[u_1^2]
        chi2_1_mk = m1 - 1.0
        r1 = float((pi1 / q).max())
        # bounds on this sub-instance vs the exact value of log(1 + Var/Z^2)
        log1p_exact = math.log1p(var_exact / (Zc ** 2))
        bounds = dict(
            bound_hybrid_m1_KL=bf.kl_hybrid_m1(M, K, chi2_1_mk),
            bound_simple_m1_KL=bf.kl_closed_m1(M, K, m1),
            bound_theta_m1=bf.kl_theta_m1(M, K, m1),
            cert_hybrid_KL=bf.kl_hybrid_r1(M, K, chi2_1_mk, r1),
            cert_simple_KL=bf.kl_closed_r1(M, K, r1),
        )
        # Hoeffding's bracket on Var(Zhat)/Z^2: (K^2/M) Var(u_1) <= . <= (K/M) Var(u_K).
        # The 1e-9 relative slack absorbs rounding in the enumeration only.
        rv_exact = var_exact / (Zc ** 2)
        lower_h = bf.lower_hoeffding(M, K, chi2_1_mk)
        upper_h = K / M * chi2s[K - 1]
        hoeff_hold = bool(lower_h <= rv_exact * (1 + 1e-9) + 1e-15
                          and rv_exact <= upper_h * (1 + 1e-9) + 1e-15)
        res.append(dict(
            nY=nY, K=K, M=M, D=Phi.shape[1],
            Z=Zc, Z_from_eK=Z, logZ_relerr=abs(Z - Zc) / Zc,
            EZ_over_Z=EZ / Zc,
            var_over_Z2_exact=var_exact / (Zc ** 2),
            identity_rhs=rhs,
            ratio_exact_over_identity=(var_exact / (Zc ** 2)) / rhs if rhs > 0 else float("nan"),
            chi2_c=chi2s, a_c=ac.tolist(),
            chi2_1_marginal_kernel=chi2_1_mk,
            chi2_1_enumerated=chi2s[0],
            sumP1=tgt["sumP1"],
            m1=m1, r1=r1, log1p_var_exact=log1p_exact, **bounds,
            bounds_hold=bool(all(log1p_exact <= b for b in bounds.values())),
            lower_hoeffding=lower_h, upper_hoeffding=upper_h,
            hoeffding_bracket_hold=hoeff_hold,
        ))
        r = res[-1]
        print(f"[T1] nY={nY} K={K} M={M} D={r['D']}: "
              f"exact Var/Z^2={r['var_over_Z2_exact']:.10e}  "
              f"identity={r['identity_rhs']:.10e}  "
              f"ratio={r['ratio_exact_over_identity']:.12f}  "
              f"E[Zhat]/Z={r['EZ_over_Z']:.12f}  "
              f"chi2_1 (marg.kernel vs enum)={chi2_1_mk:.8e} / {chi2s[0]:.8e}  "
              f"sum P1={r['sumP1']:.10f} (=K)", flush=True)
        print(f"[T1]   log(1+exact)={log1p_exact:.6g} <= bound on the KL: "
              f"m1 hybrid={bounds['bound_hybrid_m1_KL']:.6g} "
              f"m1 closed={bounds['bound_simple_m1_KL']:.6g} "
              f"theta={bounds['bound_theta_m1']:.6g} | "
              f"r1 (secondary) hybrid={bounds['cert_hybrid_KL']:.6g} "
              f"closed={bounds['cert_simple_KL']:.6g}  "
              f"holds={r['bounds_hold']}", flush=True)
        print(f"[T1]   Hoeffding: {lower_h:.6g} <= exact Var/Z^2={rv_exact:.6g} "
              f"<= {upper_h:.6g}  holds={hoeff_hold}", flush=True)
        if not r["bounds_hold"]:
            raise RuntimeError(f"[T1] a bound is below log(1 + exact Var/Z^2) "
                               f"on sub-instance {r}")
        if not hoeff_hold:
            raise RuntimeError(f"[T1] exact Var/Z^2 is outside Hoeffding's bracket "
                               f"on sub-instance {r}")
    return res


# --------------------------------------------------------------------------
# T2 / T3
# --------------------------------------------------------------------------
def analyze(task, max_prompts, R_mc, workers, seed=0):
    global _POOLED
    cfg = CONFIGS[task]
    with open(pooled_path(task), "rb") as f:
        _POOLED = pickle.load(f)
    prompts = sorted(_POOLED)[:max_prompts]
    print(f"[analyze] {task}: {len(prompts)} prompts, K={cfg['K']}, M_exp={cfg['M_exp']}, "
          f"R_mc={R_mc}, workers={workers}", flush=True)
    per_setting = {}
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for T in cfg["sweeps"]:
            jobs = [(task, pt, ip, T, R_mc, seed) for ip, pt in enumerate(prompts)]
            rows = [r for r in pool.map(analyze_prompt, jobs, chunksize=1) if r is not None]
            print(f"[analyze] {setting_key(task, T)}: {len(rows)} of {len(prompts)} prompts", flush=True)
            per_setting[setting_key(task, T)] = rows
    return per_setting


def analyze_prompt(job):
    """Bound row of one prompt at one T. Each prompt has its own stream, so results do not depend
    on the number of workers."""
    task, pt, ip, T, R_mc, seed = job
    cfg = CONFIGS[task]
    K, M_exp = cfg["K"], cfg["M_exp"]
    rng = np.random.default_rng([seed, ip])
    rec = _POOLED[pt]
    Phi, Cnt = rec["Phi"], rec["Cnt"]
    tot = Cnt.sum(0)
    # exact proposal mass covered by the distinct observed strings (raises on a
    # non-finite log q, before the weights are formed from it)
    log_cov = log_covered_mass(rec["logps"], Cnt, rec["pools"],
                               f"[{setting_key(task, T)}] prompt {ip} ({pt[:60]!r})")
    q, rho, kappa = build_instance(Phi, tot, rec["logps"], rec["logps_t1"], T)
    tgt = exact_target(Phi, kappa, K)
    if tgt is None:
        print(f"  [{setting_key(task, T)}] p{ip}: no target (rows need tiers), skipped", flush=True)
        return None
    logZ = tgt["logZ"]
    pi1 = tgt["P1"] / K
    m1 = float((pi1 ** 2 / q).sum())       # E_q[u_1^2]
    chi2_1 = m1 - 1.0                      # Var_q(u_1) (E_q[u_1] = 1)
    r1 = float((pi1 / q).max())            # sup u_1, secondary

    ac = a_coeffs(M_exp, K)
    lead = bf.leading(M_exp, K, chi2_1)                 # lower, secondary
    lower_h = bf.lower_hoeffding(M_exp, K, chi2_1)      # lower, primary
    # NA bounds on the c>=2 remainder (bound_formulas.py):
    #   primary   a_c ((K^c/(K)_c)^2 m1^c - 1)   (second moment)
    #   secondary a_c (K^c/(K)_c r1^c - 1)       (sup)
    tail_m1 = bf.na_tail_m1(M_exp, K, chi2_1)
    tail = bf.na_tail_r1(M_exp, K, chi2_1, r1)
    bound_hybrid_m1 = bf.kl_hybrid_m1(M_exp, K, chi2_1)
    bound_simple_m1 = bf.kl_closed_m1(M_exp, K, m1)
    bound_theta_m1 = bf.kl_theta_m1(M_exp, K, m1)
    cert_hybrid = bf.kl_hybrid_r1(M_exp, K, chi2_1, r1)
    cert_simple = bf.kl_closed_r1(M_exp, K, r1)

    mc = mc_relvar(Phi, rho, q, K, M_exp, logZ, R_mc, rng)

    # measured across-pool relvar at the full pool size 
    logs = [pool_logzhat(Phi, rho, Cnt[r], K) for r in range(Cnt.shape[0])]
    logs = [x for x in logs if x is not None]
    meas = relvar_from_logs(logs) if len(logs) >= 10 else None

    # T3: disjoint within-pool blocks -> true-instance relvar at each m,
    # plus the identity's prediction for the same m on this instance
    sub, sub_n, pred, pred_lo_h, pred_hi, pred_hi_m1 = {}, {}, {}, {}, {}, {}
    for m in cfg["M_grid"]:
        rv, nb = block_relvar(Phi, rho, Cnt, K, m)
        if rv is not None:
            sub[m] = rv
            sub_n[m] = nb
        pred_lo_h[m] = bf.lower_hoeffding(m, K, chi2_1)     # lower, primary
        pred[m] = bf.leading(m, K, chi2_1)                  # lower, secondary
        pred_hi_m1[m] = bf.relvar_bound_m1(m, K, chi2_1)    # upper, primary
        pred_hi[m] = bf.relvar_bound_r1(m, K, chi2_1, r1)   # upper, secondary

    # Good-Turing ESTIMATE of the missing mass, (#seen once)/(#draws). Kept so
    # earlier outputs stay comparable; the exact value is 1 - covered_mass below.
    singles = float((tot == 1).sum())
    miss = singles / float(tot.sum())

    row = dict(
        prompt_hash=abs(hash(pt)) % 10**8, n_uniq=int(rec["n_uniq"]),
        N_total=int(tot.sum()), missing_mass=miss,
        log10_covered_mass=log_cov / math.log(10), covered_mass=math.exp(log_cov),
        logZ=logZ, sumP1=tgt["sumP1"], chi2_1=chi2_1, m1=m1, r1=r1,
        lower_hoeffding=lower_h,
        a1=float(ac[0]), leading=lead,
        na_tail_m1=tail_m1,
        bound_hybrid_m1_KL=bound_hybrid_m1,
        bound_simple_m1_KL=bound_simple_m1,
        bound_theta_m1=bound_theta_m1,
        na_tail=tail,
        cert_hybrid_KL=cert_hybrid, cert_simple_KL=cert_simple,
        mc_relvar=mc["relvar"], mc_se=mc["se_relvar"],
        mc_mean_ratio=mc["mean_ratio"],
        measured_relvar=meas, sub_relvar=sub, sub_nblocks=sub_n,
        pred_lower_hoeffding_by_m=pred_lo_h,
        pred_leading_by_m=pred, pred_upper_m1_by_m=pred_hi_m1,
        pred_upper_by_m=pred_hi,
        iid=iid_diagnostics(Cnt, rec["logps"]),
    )
    if ip % 5 == 0:
        r = row
        print(f"  [{setting_key(task, T)}] p{ip}: nY={r['n_uniq']} "
              f"chi2_1={r['chi2_1']:.4g} m1={r['m1']:.4g} "
              f"r1={r['r1']:.3g} (secondary) "
              f"lead={r['leading']:.4g} MC={r['mc_relvar']:.4g}"
              f"+-{r['mc_se']:.2g} meas={r['measured_relvar']} "
              f"E[Zhat]/Z={r['mc_mean_ratio']:.4f} "
              f"KL bound m1 hybrid={r['bound_hybrid_m1_KL']:.4g} "
              f"(r1 hybrid={r['cert_hybrid_KL']:.4g})", flush=True)
    return row


def fmt(x):
    return f"{float(x):g}"


def setting_key(task, T):
    """Result key, e.g. 'gsm8k|T=1.05'."""
    return f"{task}|T={fmt(T)}"


def g(v, spec=".4g"):
    """Format a value that may be None."""
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else format(v, spec)


def summarize(per_setting):
    summ = {}
    for key, rows in per_setting.items():
        if not rows:
            continue
        def med(f, filt=lambda v: v is not None):
            vals = [f(r) for r in rows]
            vals = [v for v in vals if filt(v) and np.isfinite(v)]
            return float(np.median(vals)) if vals else None
        mcs = [r["mc_relvar"] for r in rows if np.isfinite(r["mc_relvar"])]
        leads = [r["leading"] for r in rows]
        ratio = [r["leading"] / r["mc_relvar"] for r in rows
                 if r["mc_relvar"] and np.isfinite(r["mc_relvar"]) and r["mc_relvar"] > 0]
        ratio_h = [r["lower_hoeffding"] / r["mc_relvar"] for r in rows
                   if r["mc_relvar"] and np.isfinite(r["mc_relvar"]) and r["mc_relvar"] > 0]
        Ms = sorted({m for r in rows for m in r["sub_relvar"]})
        sub_med = {m: float(np.median([r["sub_relvar"][m] for r in rows
                                       if m in r["sub_relvar"]]))
                   for m in Ms}
        allm = sorted({m for r in rows for m in r["pred_leading_by_m"]})
        pred_med = {m: float(np.median([r["pred_leading_by_m"][m] for r in rows
                                        if m in r["pred_leading_by_m"]]))
                    for m in allm}
        pred_lo_h_med = {m: float(np.median([r["pred_lower_hoeffding_by_m"][m] for r in rows
                                             if m in r["pred_lower_hoeffding_by_m"]]))
                         for m in allm}
        pred_hi_med = {m: float(np.median([r["pred_upper_by_m"][m] for r in rows
                                           if m in r["pred_upper_by_m"]]))
                       for m in allm}
        pred_hi_m1_med = {m: float(np.median([r["pred_upper_m1_by_m"][m] for r in rows
                                              if m in r["pred_upper_m1_by_m"]]))
                          for m in allm}
        # fraction of prompts whose measured relvar falls below the identity's
        # EXACT lower bound a_1(m) chi2_1 (impossible for an unbiased estimate)
        below = {m: float(np.mean([r["sub_relvar"][m] < r["pred_leading_by_m"][m]
                                   for r in rows if m in r["sub_relvar"]]))
                 for m in sub_med}
        # same for the (larger) Hoeffding lower bound (K^2/m) chi2_1
        below_h = {m: float(np.mean([r["sub_relvar"][m] < r["pred_lower_hoeffding_by_m"][m]
                                     for r in rows if m in r["sub_relvar"]]))
                   for m in sub_med}
        # ... and above the second-moment UPPER bound (impossible for the true
        # value; nonzero means block-estimate noise or a broken bound)
        above_m1 = {m: float(np.mean([r["sub_relvar"][m] > r["pred_upper_m1_by_m"][m]
                                      for r in rows if m in r["sub_relvar"]]))
                    for m in sub_med}
        # the two hybrids are not ordered (bound_formulas.py): how often is the
        # second-moment one the larger?
        frac_m1_above = float(np.mean([r["bound_hybrid_m1_KL"] > r["cert_hybrid_KL"]
                                       for r in rows]))
        iid_med = {k: float(np.median([r["iid"][k] for r in rows
                                       if r["iid"].get(k) is not None]))
                   for k in ("obs_collisions", "pred_collisions",
                             "obs_shared_pairwise", "pred_shared_pairwise",
                             "pred_shared_pairwise_emp",
                             "max_q", "sum_q_observed")}
        summ[key] = dict(
            n_prompts=len(rows),
            median_n_uniq=med(lambda r: r["n_uniq"]),
            median_missing_mass=med(lambda r: r["missing_mass"]),      # Good-Turing estimate
            median_log10_covered_mass=med(lambda r: r["log10_covered_mass"]),  # exact
            median_covered_mass=med(lambda r: r["covered_mass"]),              # exact
            median_sumP1_minus_K=med(lambda r: r["sumP1"]),
            median_chi2_1=med(lambda r: r["chi2_1"]),
            median_m1=med(lambda r: r["m1"]),
            median_r1=med(lambda r: r["r1"]),
            median_lower_hoeffding=med(lambda r: r["lower_hoeffding"]),
            median_leading=float(np.median(leads)),
            median_mc_relvar=float(np.median(mcs)) if mcs else None,
            median_mc_mean_ratio=med(lambda r: r["mc_mean_ratio"]),
            median_lower_hoeffding_over_mc=float(np.median(ratio_h)) if ratio_h else None,
            median_leading_over_mc=float(np.median(ratio)) if ratio else None,
            median_measured_relvar=med(lambda r: r["measured_relvar"]),
            median_bound_hybrid_m1_KL=med(lambda r: r["bound_hybrid_m1_KL"]),
            median_bound_simple_m1_KL=med(lambda r: r["bound_simple_m1_KL"]),
            median_bound_theta_m1=med(lambda r: r["bound_theta_m1"]),
            median_cert_hybrid_KL=med(lambda r: r["cert_hybrid_KL"]),
            median_cert_simple_KL=med(lambda r: r["cert_simple_KL"]),
            frac_hybrid_m1_above_r1=frac_m1_above,
            block_relvar_median=sub_med,
            block_nblocks=({m: int(np.median([r["sub_nblocks"][m] for r in rows
                                              if m in r["sub_nblocks"]]))
                            for m in sub_med} if sub_med else {}),
            predicted_lower_hoeffding_by_m=pred_lo_h_med,
            predicted_leading_by_m=pred_med,
            predicted_upper_m1_by_m=pred_hi_m1_med,
            predicted_upper_by_m=pred_hi_med,
            frac_below_hoeffding_lower_bound=below_h,
            frac_below_exact_lower_bound=below,
            frac_above_upper_m1_bound=above_m1,
            median_mc_se=med(lambda r: r["mc_se"]),
            iid_diagnostics_median=iid_med,
        )
        s = summ[key]
        print(f"\n=== {key} ===")
        print(f"  prompts={s['n_prompts']} median |Y|={g(s['median_n_uniq'], '.0f')} "
              f"Good-Turing missing mass (estimate)={g(s['median_missing_mass'], '.3f')}")
        print(f"  exact covered proposal mass sum_y q(y), median over prompts: "
              f"log10={g(s['median_log10_covered_mass'], '.4g')} "
              f"value={g(s['median_covered_mass'], '.4g')}")
        print(f"  sum P1 (should = K): {g(s['median_sumP1_minus_K'], '.8f')}")
        print(f"  E[Zhat]/Z (MC, should = 1): {g(s['median_mc_mean_ratio'], '.4f')}")
        print(f"  chi2(pi_1||q)={g(s['median_chi2_1'])}  "
              f"m1=E_q[u_1^2]={g(s['median_m1'])}  "
              f"r1=sup u_1 (secondary)={g(s['median_r1'])}")
        print(f"  lower bound (K^2/N) chi2_1 (Hoeffding) = {g(s['median_lower_hoeffding'])}"
              f"  (a_1 chi2_1, secondary: {g(s['median_leading'])})")
        print(f"  MC Var(Zhat)/Z^2 (= full RHS) = {g(s['median_mc_relvar'])}"
              f" +- {g(s['median_mc_se'], '.3g')}")
        print(f"  Hoeffding lower/MC = {g(s['median_lower_hoeffding_over_mc'])}  "
              f"(leading/MC = {g(s['median_leading_over_mc'])})")
        print(f"  measured across-pool relvar   = {g(s['median_measured_relvar'])}")
        print(f"  bound on the KL (m1): hybrid={g(s['median_bound_hybrid_m1_KL'])}  "
              f"closed form log(e^theta-1+a_0)={g(s['median_bound_simple_m1_KL'])}  "
              f"theta=K^2 m1/(M-K+1)={g(s['median_bound_theta_m1'])}")
        print(f"  bound on the KL (r1, secondary): hybrid={g(s['median_cert_hybrid_KL'])}  "
              f"closed form K^2 r1/(M-K+1)={g(s['median_cert_simple_KL'])}")
        print(f"  fraction of prompts with m1 hybrid > r1 hybrid: "
              f"{g(s['frac_hybrid_m1_above_r1'], '.3f')}")
        print("  i.i.d. diagnostics (LM log-probs vs pools): "
              f"within-pool collisions obs={g(iid_med.get('obs_collisions'), '.3g')} "
              f"pred={g(iid_med.get('pred_collisions'), '.3g')}; "
              f"cross-pool shared strings obs={g(iid_med.get('obs_shared_pairwise'), '.3g')} "
              f"pred(LM logps)={g(iid_med.get('pred_shared_pairwise'), '.3g')} "
              f"pred(empirical q)={g(iid_med.get('pred_shared_pairwise_emp'), '.3g')}")
        print("  true-instance relvar from disjoint blocks vs bracket "
              "[Hoeffding lower, m1 upper] (a1 lower, r1 upper: secondary):")
        for m in sub_med:
            print(f"    m={m:>4}: obs={g(sub_med[m])} "
                  f"(n_blocks={s['block_nblocks'][m]})  "
                  f"in [{g(pred_lo_h_med.get(m))}, "
                  f"{g(pred_hi_m1_med.get(m))}] (a1: {g(pred_med.get(m))}, "
                  f"r1: {g(pred_hi_med.get(m))})  "
                  f"obs*m={g(sub_med[m]*m, '.3g')}  "
                  f"frac prompts below Hoeffding lower bd={g(below_h[m], '.2f')} "
                  f"(a1: {g(below[m], '.2f')})  "
                  f"above m1 upper bd={g(above_m1[m], '.2f')}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both",
                    choices=["ambrosia", "fineweb", "gsm8k", "all"])
    ap.add_argument("--stage", default="all",
                    choices=["extract", "analyze", "t1", "all"])
    ap.add_argument("--max-prompts", type=int, default=100)
    ap.add_argument("--R-mc", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--Ts", nargs="+", type=float, required=True, help="reweighting temperatures")
    ap.add_argument("--cache", default=None, help="override the task's cache dir")
    ap.add_argument("--n-pools", type=int, default=None)
    ap.add_argument("--pool-files", nargs="+", default=None,
                    help="explicit embedding pkls, one per pool (overrides discovery)")
    ap.add_argument("--no-extra-pools", action="store_true",
                    help="fineweb: ignore cache-fineweb-pools/pool*")
    args = ap.parse_args()

    tasks = (["ambrosia", "fineweb", "gsm8k"] if args.task == "all"
             else [args.task])
    out = {}
    for task in tasks:
        cfg = CONFIGS[task]
        if args.cache:
            cfg["cache"] = args.cache
        if args.n_pools:
            cfg["n_pools"] = args.n_pools
        if args.pool_files:
            cfg["pool_files"] = [os.path.join(ROOT_DIR, p) for p in args.pool_files]
        if args.no_extra_pools:
            cfg.pop("extra_pool_glob", None)
        cfg["sweeps"] = [float(t) for t in args.Ts]
        out[f"{task}|meta"] = dict(
            sT=cfg["sT"], K=cfg["K"], N_pool=cfg["M_exp"], Ts=cfg["sweeps"],
            pool_files=[os.path.relpath(p, ROOT_DIR) for p in pool_files(task)],
            note="L weight rho = plm^(1/T) / q")
        print(f"[{task}] sT={cfg['sT']}, T in {cfg['sweeps']}", flush=True)
        print(f"[{task}] pools: {len(pool_files(task))} files; extract -> "
              f"{pooled_path(task)}", flush=True)
        if args.stage in ("extract", "all"):
            extract(task)
        if args.stage in ("t1", "all"):
            with open(pooled_path(task), "rb") as f:
                pooled = pickle.load(f)
            rng = np.random.default_rng(7)
            # (|Y|, K, M, feature-dim truncation or None for the full embedding)
            cases = [(6, 2, 4, None), (5, 3, 5, None), (8, 2, 4, None),
                     (6, 2, 5, None), (5, 3, 6, None)]
            out[f"{task}|T1_enumeration"] = t1_enumeration(task, pooled, rng, cases)
        if args.stage in ("analyze", "all"):
            ps = analyze(task, args.max_prompts, args.R_mc, args.workers)
            out.update({k: v for k, v in ps.items()})
            tmp = os.path.join(SCRIPT_DIR, f"bound_raw_{task}.json")
            with open(tmp, "w") as fh:
                json.dump(ps, fh, indent=2, default=float)
            print(f"[analyze] raw rows saved to {tmp}")
            out[f"{task}|summary"] = summarize(ps)

    dest = os.path.join(SCRIPT_DIR,
                        f"bound_empirical_{args.task}_{args.stage}.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
