"""Exact k-DPP sampling that stays accurate when row weights span many orders of magnitude.

The kernel is L = B B^T with rows B_i = exp(log_s_i) * phi_i, ||phi_i|| = 1. Forming
C = B^T B squares the scales, so eigenvalues of C below eps * lambda_max are rounding
noise, and a sampler that normalizes eigenvectors in the C metric then divides by the
square root of a value that can be negative. This module avoids C:

1. One-sided Jacobi SVD (LAPACK dgejsv) after pivoted QR of the column-graded matrix.
   This computes singular values and vectors of D-scaled matrices to high relative
   accuracy, however small the scales are (Demmel and Veselic, 1992).
2. Eigenvector selection with elementary symmetric polynomials in log space.
3. The usual projection sampler on orthonormal primal eigenvectors.

Scales that a double cannot hold in one factorization are handled by tiers. Both tier
paths are approximations; each StableKDPP records which path its kernel takes in .info
(kernel_info returns it without drawing):
  "flat"    every row is within TIER_NATS of the top scale: one factorization, exact.
  "drop"    the rows within TIER_NATS of the top span k dimensions, and the rows below are
            dropped. info["log10_drop_bound"] is a rigorous upper bound on the probability
            mass of the sets this leaves out: by Fischer's and Hadamard's inequalities,
            det(L_S) <= det(L_A) prod_{b in B} L_bb for S = A + B with B the dropped rows, so
            the left-out mass is at most sum_{i=1..k} D^i e_{k-i}(top) / e_k(top), with
            D = sum of the dropped rows' squared scales. The draw is not stopped; the bound
            is recorded so the size of the approximation can be read.
  "tiered"  the top rows span fewer than k dimensions: they are sampled first and the lower
            rows continue on their orthogonal complement. This is the k-DPP only in the limit
            of separated scales (example: with log scales 0, -299.99, -300.01 x 3 and k = 3,
            the tiered draw always contains both top rows; the k-DPP does so with
            probability 0.497). sample()
            raises RuntimeError unless, at every cut where the draw continues below the tier,
            the lowest log scale in the tier exceeds the highest log scale below the cut by
            more than TIER_NATS (info["min_gap_nats"] holds the first cut's gap).

Rows with one direction are merged first, see _merge_duplicates.

Failures stop with an error. A row whose log scale is NaN or +inf, or whose embedding has
a NaN or inf entry, raises ValueError (a row with log scale -inf or a zero embedding has
no mass and is left out, which is exact). sample() returns [] only when the rows with mass
span fewer than k dimensions (numpy.linalg.matrix_rank with tol=RANK_TOL), the case where no
size-k set has positive probability. A draw with fewer than k rows from rows that do span k
dimensions is a numerical failure and raises RuntimeError.
"""
import numpy as np
from scipy.linalg import lapack, qr

TIER_NATS = 300.0
RANK_TOL = 1e-10
DUP_DECIMALS = 9


def _unit_rows(log_s, Phi):
    """Move row norms into log_s; drop rows that are zero or carry no mass (log scale -inf).
    Raises ValueError on a NaN or +inf log scale or a non-finite embedding entry."""
    bad = np.flatnonzero(np.isnan(log_s) | (log_s == np.inf))
    if bad.size:
        raise ValueError(f"{bad.size} of {len(log_s)} rows have a NaN or +inf log scale "
                         f"(first: row {bad[0]}, log scale {log_s[bad[0]]!r})")
    bad = np.flatnonzero(~np.isfinite(Phi).all(axis=1))
    if bad.size:
        raise ValueError(f"{bad.size} of {len(Phi)} embedding rows have a NaN or inf entry (first: row {bad[0]})")
    norms = np.linalg.norm(Phi, axis=1)
    ok = (norms > 0) & np.isfinite(log_s)
    idx = np.flatnonzero(ok)
    return idx, log_s[ok] + np.log(norms[ok]), Phi[ok] / norms[ok, None]


def _merge_duplicates(log_s, Phi):
    """Rows with one direction cannot share a k-DPP set, so a group acts as one row with the
    group's total squared scale, and its member follows by squared scale. Keeping such a pair
    would add a singular value made of rounding error, scale * 1e-16, which competes with real
    ones when scales are graded. Returns merged log scales, merged rows, and each input row's
    group. Groups keep the order of first appearance, so pools without duplicates are unchanged."""
    _, first, inv = np.unique(np.round(Phi, DUP_DECIMALS), axis=0, return_index=True, return_inverse=True)
    rank = np.empty(len(first), int)
    rank[np.argsort(first)] = np.arange(len(first))
    group = rank[inv.ravel()]
    merged = np.full(len(first), -np.inf)
    np.logaddexp.at(merged, group, 2 * log_s)
    return merged / 2, Phi[np.sort(first)], group


def graded_svd(log_s, Phi):
    """Singular values and primal eigenvectors U of L = B B^T, in the input row order."""
    n = len(log_s)
    order = np.argsort(-log_s)
    A = Phi[order].T * np.exp(log_s[order] - log_s[order[0]])
    _, R, piv = qr(A, mode="economic", pivoting=True)
    sva, Up, _, work, _, info = lapack.dgejsv(R.T, joba=1, jobu=0, jobv=0)
    if info != 0:
        raise np.linalg.LinAlgError(f"dgejsv failed: info={info}")
    sv = sva * (work[1] / work[0])
    r = len(sv)
    U = np.zeros((n, r))
    U[order[piv]] = Up[:, :r]
    keep = sv > 0
    return np.log(sv[keep]) + log_s[order[0]], U[:, keep]


def _log_esp_table(ll, k):
    n = len(ll)
    T = np.full((n + 1, k + 1), -np.inf)
    T[:, 0] = 0.0
    for i in range(1, n + 1):
        T[i, 1:] = np.logaddexp(T[i - 1, 1:], ll[i - 1] + T[i - 1, :-1])
    return T


def _sample_projection(V):
    """Sample V.shape[1] rows from the projection DPP with orthonormal columns V."""
    V = V.copy()
    out = []
    for _ in range(V.shape[1]):
        p = (V ** 2).sum(1)
        p[out] = 0.0
        i = int(np.random.choice(len(p), p=p / p.sum()))
        out.append(i)
        j = int(np.abs(V[i]).argmax())
        V = V - np.outer(V[:, j] / V[i, j], V[i])
        V = np.delete(V, j, axis=1)
        if V.shape[1]:
            V, _ = np.linalg.qr(V)
    return out


def _sample_flat(log_sv, U, k):
    """One k-DPP draw given log singular values and primal eigenvectors."""
    ll = 2 * log_sv
    T = _log_esp_table(ll, k)
    chosen, rem = [], k
    for i in range(len(ll), 0, -1):
        if rem == 0:
            break
        if i == rem or np.random.random() < np.exp(ll[i - 1] + T[i - 1, rem - 1] - T[i, rem]):
            chosen.append(i - 1)
            rem -= 1
    return _sample_projection(U[:, chosen])


class StableKDPP:
    """k-DPP over rows exp(log_s_i) * Phi_i. sample() returns [] if no size-k set has
    positive probability, meaning the rows with mass span fewer than k dimensions, and raises
    RuntimeError if it draws fewer than k rows from rows that span k dimensions."""

    def __init__(self, log_s, Phi, k):
        self.k = k
        self.n_rows = len(log_s)
        self.idx, self.row_log_s, Phi = _unit_rows(np.asarray(log_s, float), np.asarray(Phi, float))
        self.log_s, self.Phi, self.group = self.row_log_s, Phi, np.arange(len(self.idx))
        if len(self.idx):
            self.log_s, self.Phi, self.group = _merge_duplicates(self.row_log_s, Phi)
        self.tiers = self._build_tiers()
        self.info = self._tier_info()

    def _tier_info(self):
        """Which tier path the kernel takes (module docstring) and the size of its approximation."""
        info = dict(path="none", n_rows=int(len(self.log_s)), n_dropped=0, log10_drop_bound=None,
                    min_gap_nats=None)
        if len(self.log_s) == 0:
            return info
        top = self.log_s >= self.log_s.max() - TIER_NATS
        if self.tiers is not None:
            info["n_dropped"] = int((~top).sum())
            if not info["n_dropped"]:
                info["path"] = "flat"
                return info
            info["path"] = "drop"
            _, ls, _ = self.tiers
            k, T = self.k, _log_esp_table(2 * ls, self.k)
            log_D = np.logaddexp.reduce(2 * self.log_s[~top])
            terms = [i * log_D + T[len(ls), k - i] for i in range(1, k + 1)]
            info["log10_drop_bound"] = float((np.logaddexp.reduce(terms) - T[len(ls), k]) / np.log(10))
            return info
        info["path"] = "tiered"
        if (~top).any():
            info["min_gap_nats"] = float(self.log_s[top].min() - self.log_s[~top].max())
        return info

    def _build_tiers(self):
        if len(self.idx) == 0:
            return None
        top = self.log_s >= self.log_s.max() - TIER_NATS
        if np.linalg.matrix_rank(self.Phi[top], tol=RANK_TOL) >= self.k:
            sub = np.flatnonzero(top)
            ls, U = graded_svd(self.log_s[sub], self.Phi[sub])
            return sub, ls, U
        return None

    def _member(self, g):
        """Input row for merged row g, drawn by squared scale within its group."""
        rows = np.flatnonzero(self.group == g)
        if len(rows) > 1:
            w = np.exp(2 * (self.row_log_s[rows] - self.row_log_s[rows].max()))
            rows = rows[[np.random.choice(len(rows), p=w / w.sum())]]
        return int(self.idx[rows[0]])

    def spans_k(self):
        """True when the rows with mass span k dimensions (rank with tol=RANK_TOL)."""
        return len(self.Phi) > 0 and np.linalg.matrix_rank(self.Phi, tol=RANK_TOL) >= self.k

    def sample(self):
        if self.tiers is not None:
            sub, ls, U = self.tiers
            picked = [sub[i] for i in _sample_flat(ls, U, self.k)]
        else:
            picked = self._sample_tiered()
        # both checks come after the draw, so the random stream is the same as before them
        if len(picked) != self.k:
            if self.spans_k():
                raise RuntimeError(
                    f"k-DPP draw returned {len(picked)} of k={self.k} rows, but the {len(self.Phi)} rows "
                    f"with mass span at least k dimensions (rank tol {RANK_TOL}): a numerical failure, "
                    f"not Zhat = 0. Fix the sampler for this kernel; do not treat the pool as degenerate")
            return []
        if self.tiers is None and self._gaps and min(self._gaps) <= TIER_NATS:
            raise RuntimeError(
                f"tiered k-DPP draw (k={self.k}, {len(self.log_s)} rows, log scales spanning "
                f"{self.log_s.max() - self.log_s.min():.1f} nats): the draw continued below a tier whose "
                f"lowest log scale is only {min(self._gaps):.2f} nats above the highest row below it "
                f"(needs > TIER_NATS = {TIER_NATS:g}), so it is not a draw from the k-DPP. The fix is an "
                f"exact sampler for this kernel, a decision for timv/Manuel (PR 'fail loudly'); do not "
                f"widen the cut silently")
        return [self._member(g) for g in picked]

    def _sample_tiered(self):
        idx, log_s, Phi, need, out = np.arange(len(self.log_s)), self.log_s, self.Phi, self.k, []
        self._gaps = []  # per cut where the draw continues below the tier: lowest tier scale - highest below
        while need > 0:
            if len(idx) == 0:
                return []
            top = np.flatnonzero(log_s >= log_s.max() - TIER_NATS)
            r = min(np.linalg.matrix_rank(Phi[top], tol=RANK_TOL), need)
            ls, U = graded_svd(log_s[top], Phi[top])
            picked = [top[i] for i in _sample_flat(ls, U, r)]
            out += [int(idx[i]) for i in picked]
            need -= r
            Q, _ = np.linalg.qr(Phi[picked].T)
            rest = np.setdiff1d(np.arange(len(idx)), top)
            if need > 0 and len(rest):
                self._gaps.append(float(log_s[top].min() - log_s[rest].max()))
            res = Phi[rest] - (Phi[rest] @ Q) @ Q.T
            norms = np.linalg.norm(res, axis=1)
            keep = norms > RANK_TOL
            idx, log_s, Phi = idx[rest[keep]], log_s[rest[keep]] + np.log(norms[keep]), res[keep] / norms[keep, None]
        return out

    def inclusion_probs(self):
        """P(row is in the set) for every input row, or None when the kernel needs tiers. Raises
        RuntimeError when the top tier spans k dimensions but its SVD keeps fewer than k positive
        singular values (a numerical failure)."""
        if self.tiers is None:
            return None
        sub, ls, U = self.tiers
        ll, k = 2 * ls, self.k
        n = len(ll)
        if n < k:
            raise RuntimeError(f"the top tier spans k={k} dimensions (rank tol {RANK_TOL}) but its SVD "
                               f"kept {n} positive singular values")
        F, G = _log_esp_table(ll, k), _log_esp_table(ll[::-1], k)
        loo = np.array([np.logaddexp.reduce(F[j, :k] + G[n - 1 - j, k - 1::-1]) for j in range(n)])
        merged = np.zeros(len(self.log_s))
        merged[sub] = (U ** 2 * np.exp(ll + loo - F[n, k])).sum(1)
        out = np.zeros(self.n_rows)
        out[self.idx] = merged[self.group] * np.exp(2 * (self.row_log_s - self.log_s[self.group]))
        return out


def kernel_info(log_s, Phi, k):
    """StableKDPP(log_s, Phi, k).info without drawing: the tier path and its approximation size."""
    return StableKDPP(log_s, Phi, k).info
