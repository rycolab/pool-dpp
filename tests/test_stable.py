"""Checks for dpp.stable: brute force on small kernels, 150-digit ground truth on graded ones.
Run: python tests/test_stable.py"""
import itertools
import numpy as np
import mpmath as mp
from dpp.stable import StableKDPP, graded_svd, _log_esp_table


def brute(log_s, Phi, k):
    G = Phi @ Phi.T
    out = {}
    for S in itertools.combinations(range(len(log_s)), k):
        out[S] = np.linalg.det(G[np.ix_(S, S)]) * np.exp(2 * log_s[list(S)].sum())
    t = sum(out.values())
    return {S: v / t for S, v in out.items()}


def mp_marginals(log_s, Phi, k, dps=200):
    mp.mp.dps = dps
    n = len(log_s)
    G = Phi @ Phi.T
    s = [mp.exp(mp.mpf(float(x))) for x in log_s - log_s.max()]
    L = mp.matrix(n, n)
    for a in range(n):
        for b in range(n):
            L[a, b] = s[a] * s[b] * mp.mpf(float(G[a, b]))
    E, Q = mp.eigsy(L)
    e = [E[i] for i in range(n)]

    def esp(v, m):
        t = [mp.mpf(1)] + [mp.mpf(0)] * m
        for x in v:
            for q in range(m, 0, -1):
                t[q] += x * t[q - 1]
        return t[m]
    ek = esp(e, k)
    pj = [e[j] * esp(e[:j] + e[j + 1:], k - 1) / ek for j in range(n)]
    return np.array([float(sum(pj[j] * Q[a, j] ** 2 for j in range(n))) for a in range(n)])


def analytic_marginals(log_s, Phi, k):
    ls, U = graded_svd(log_s, Phi)
    ll = 2 * ls
    n = len(ll)
    te = _log_esp_table(ll, k)[n, k]
    pj = np.array([np.exp(ll[j] + _log_esp_table(np.delete(ll, j), k - 1)[n - 1, k - 1] - te) for j in range(n)])
    return (U ** 2 * pj).sum(1)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(0)
    np.random.seed(0)
    # 1. empirical law against brute force, moderate and steep scales
    for spread in (0.5, 5.0, 30.0):
        m, k, D = 8, 3, 5
        Phi = unit(rng.normal(size=(m, D)))
        ls = -spread * rng.random(m)
        P = brute(ls, Phi, k)
        d = StableKDPP(ls, Phi, k)
        c = {}
        R = 40000
        for _ in range(R):
            S = tuple(sorted(d.sample()))
            assert len(set(S)) == k
            c[S] = c.get(S, 0) + 1
        tv = .5 * sum(abs(P[S] - c.get(S, 0) / R) for S in P)
        print(f"law vs brute force, spread {spread}: TV {tv:.4f}")
        assert tv < 0.02
    # 2. marginals against 200-digit arithmetic, scales far below rounding
    for spread, m, D, k in ((20, 30, 12, 5), (60, 30, 12, 5), (200, 30, 12, 5), (60, 40, 40, 10)):
        Phi = unit(rng.normal(size=(m, D)))
        ls = -spread * rng.random(m)
        err = np.abs(analytic_marginals(ls, Phi, k) - mp_marginals(ls, Phi, k)).max()
        print(f"marginals vs 200 digits, row log-scale spread {spread}, n={m}, D={D}, K={k}: max error {err:.2e}")
        assert err < 1e-9
    # 3. scales wider than a double: tiers against the separated limit
    Phi = unit(rng.normal(size=(9, 6)))
    ls = np.array([0., 0., -2000., -2000., -2000., -2000., -5000., -5000., -5000.])
    d = StableKDPP(ls, Phi, 4)
    seen = [set(d.sample()) for _ in range(300)]
    assert all(len(s) == 4 and {0, 1} <= s and not (s & {6, 7, 8}) for s in seen)
    print("tiers: top tier always taken, lowest tier never needed")
    # 4. rows spanning too few dimensions: no valid set
    Phi = unit(np.array([[1., 0, 0], [1., 0, 0], [0, 1., 0], [0, 1., 0]]))
    assert StableKDPP(np.zeros(4), Phi, 3).sample() == []
    s = StableKDPP(np.zeros(4), Phi, 2).sample()
    assert len(s) == 2 and {i // 2 for i in s} == {0, 1}
    print("rank-deficient pool returns [] and never a repeated direction")
    # 5. inclusion_probs agrees with the leave-one-out computation
    Phi = unit(rng.normal(size=(30, 12)))
    ls = -60 * rng.random(30)
    err = np.abs(StableKDPP(ls, Phi, 5).inclusion_probs() - analytic_marginals(ls, Phi, 5)).max()
    print(f"inclusion_probs vs leave-one-out: max error {err:.2e}")
    assert err < 1e-10
    # 6. identical rows on graded scales: unmerged, a singular value of size scale * 1e-16 displaces real ones
    Phi = unit(rng.normal(size=(30, 25)))
    Phi[7] = Phi[0]
    ls = -4.5 * np.arange(30.)
    ls[7] = -5.0
    truth = mp_marginals(ls, Phi, 10)
    err_old = np.abs(analytic_marginals(ls, Phi, 10) - truth).max()
    d = StableKDPP(ls, Phi, 10)
    probs = d.inclusion_probs()
    err_new = np.abs(probs - truth).max()
    print(f"identical rows vs 200 digits: max error {err_new:.2e} merged, {err_old:.2e} unmerged")
    assert err_new < 1e-9 and err_old > 0.1
    assert abs(probs[7] / probs[0] - np.exp(-10.0)) < 1e-9
    draws = [d.sample() for _ in range(3000)]
    assert all(len(set(s)) == 10 and not {0, 7} <= set(s) for s in draws)
    print("identical rows never share a set")
    print("OK")


if __name__ == "__main__":
    main()
