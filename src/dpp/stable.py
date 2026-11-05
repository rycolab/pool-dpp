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

Scales that a double cannot hold in one factorization are handled by tiers. A row more
than TIER_NATS below the top scale has relative probability below exp(-2 * TIER_NATS),
so it is dropped when the higher rows already span k dimensions. Otherwise the higher
rows are sampled first and the lower rows continue on their orthogonal complement,
which is the limit of the k-DPP as the scales separate.

Rows with one direction are merged first, see _merge_duplicates.
"""
import numpy as np
from scipy.linalg import lapack, qr

TIER_NATS = 300.0
RANK_TOL = 1e-10
DUP_DECIMALS = 9


def _unit_rows(log_s, Phi):
    """Move row norms into log_s; drop rows that are zero or carry no mass."""
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
    positive probability, meaning the rows span fewer than k dimensions."""

    def __init__(self, log_s, Phi, k):
        self.k = k
        self.n_rows = len(log_s)
        self.idx, self.row_log_s, Phi = _unit_rows(np.asarray(log_s, float), np.asarray(Phi, float))
        self.log_s, self.Phi, self.group = self.row_log_s, Phi, np.arange(len(self.idx))
        if len(self.idx):
            self.log_s, self.Phi, self.group = _merge_duplicates(self.row_log_s, Phi)
        self.tiers = self._build_tiers()

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

    def sample(self):
        if self.tiers is not None:
            sub, ls, U = self.tiers
            picked = [sub[i] for i in _sample_flat(ls, U, self.k)]
        else:
            picked = self._sample_tiered()
        return [self._member(g) for g in picked]

    def _sample_tiered(self):
        idx, log_s, Phi, need, out = np.arange(len(self.log_s)), self.log_s, self.Phi, self.k, []
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
            res = Phi[rest] - (Phi[rest] @ Q) @ Q.T
            norms = np.linalg.norm(res, axis=1)
            keep = norms > RANK_TOL
            idx, log_s, Phi = idx[rest[keep]], log_s[rest[keep]] + np.log(norms[keep]), res[keep] / norms[keep, None]
        return out

    def inclusion_probs(self):
        """P(row is in the set) for every input row, or None when the kernel needs tiers."""
        if self.tiers is None:
            return None
        sub, ls, U = self.tiers
        ll, k = 2 * ls, self.k
        n = len(ll)
        if n < k:
            return None
        F, G = _log_esp_table(ll, k), _log_esp_table(ll[::-1], k)
        loo = np.array([np.logaddexp.reduce(F[j, :k] + G[n - 1 - j, k - 1::-1]) for j in range(n)])
        merged = np.zeros(len(self.log_s))
        merged[sub] = (U ** 2 * np.exp(ll + loo - F[n, k])).sum(1)
        out = np.zeros(self.n_rows)
        out[self.idx] = merged[self.group] * np.exp(2 * (self.row_log_s - self.log_s[self.group]))
        return out
