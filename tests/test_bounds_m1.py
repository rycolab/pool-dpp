"""Checks for the bound on KL(P || pi_bar) in analysis/bound_formulas.py, on tiny
synthetic instances. No pools, models or GPU needed.

Run: pytest tests/test_bounds_m1.py      (python tests/test_bounds_m1.py prints the
numbers for each instance)

An instance is a ground set of a few strings with random unit embeddings phi,
positive weights rho, and a proposal q; the target K-DPP has L-ensemble
L = diag(sqrt(kappa)) Phi Phi^T diag(sqrt(kappa)), kappa = rho q. Var(Zhat)/Z^2
is computed exactly by enumerating every pool omega in Y^N (all |Y|^N of them,
with probability prod_i q(omega_i)), with
    Zhat(omega) = (1/(N)_K) sum_{I subset [N], |I| = K} prod_{i in I} rho(omega_i) det(Phi_I Phi_I^T),
the estimator bound_empirical.py evaluates as alpha(N,K) e_K(eig B B^T); the
test checks the two agree on a few pools. Var_{q^c}(u_c) is computed by
enumerating ordered c-tuples (u_c = 0 on tuples with a repeated string, since a
pool K-subset with a repeat has determinant 0).
"""
import itertools
import math
import os
import sys

import numpy as np
import pytest

ANALYSIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
sys.path.insert(0, ANALYSIS)

import bound_empirical as be  # noqa: E402
import bound_formulas as bf  # noqa: E402

# (|Y|, embedding dim D, K, N); |Y|^N pools are enumerated
CONFIGS = [(5, 3, 2, 4), (5, 4, 2, 6), (4, 4, 3, 6), (6, 4, 3, 5), (4, 5, 3, 7), (5, 5, 4, 6)]
SEEDS = [0, 1, 2]


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def make_instance(nY, D, seed, heavy=False):
    """Random instance. heavy=True plants one rare string (small q) with a large
    weight, so u_1 is heavy-tailed (r_1 well above m_1)."""
    rng = np.random.default_rng(seed)
    Phi = unit(rng.standard_normal((nY, D)))
    q = rng.dirichlet(np.ones(nY))
    rho = np.exp(rng.normal(0.0, 1.0, nY))
    if heavy:
        q[-1] = 0.01
        q[:-1] *= 0.99 / q[:-1].sum()
        rho[-1] = 200.0
    rho = rho / float(q @ rho)          # gauge E_q[rho] = 1 (the bounds do not depend on it)
    return Phi, q, rho


def uniform_instance(nY):
    """Orthonormal embeddings, uniform q and rho: P is uniform over K-subsets and
    u_1 = 1 everywhere, so m_1 = r_1 = 1."""
    return np.eye(nY), np.full(nY, 1.0 / nY), np.ones(nY)


def exact(Phi, q, rho, K, N):
    nY = len(q)
    kappa = rho * q
    A = np.sqrt(kappa)[:, None] * Phi
    L = A @ A.T
    subsets = list(itertools.combinations(range(nY), K))
    f = {S: float(np.linalg.det(L[np.ix_(S, S)])) for S in subsets}
    Z = sum(f.values())
    P1 = np.array([sum(v for S, v in f.items() if y in S) for y in range(nY)]) / Z
    u1 = P1 / (K * q)
    m1 = float(q @ u1 ** 2)
    var_u1 = m1 - float(q @ u1) ** 2
    r1 = float(u1.max())

    # H[y_1..y_K] = prod rho det G = f(S)/prod q(S) on distinct tuples, 0 otherwise
    H = np.zeros((nY,) * K)
    for S in subsets:
        val = f[S] / float(np.prod(q[list(S)]))
        for perm in itertools.permutations(S):
            H[perm] = val
    omegas = np.array(list(itertools.product(range(nY), repeat=N)))
    w = np.prod(q[omegas], axis=1)
    Zhat = np.zeros(len(omegas))
    for I in itertools.combinations(range(N), K):
        Zhat += H[tuple(omegas[:, list(I)].T)]
    Zhat /= math.perm(N, K)
    EZ = float(w @ Zhat)
    relvar = (float(w @ Zhat ** 2) - EZ ** 2) / Z ** 2

    var_uc = []
    for c in range(1, K + 1):
        s = 0.0
        for tup in itertools.permutations(range(nY), c):
            Pc = sum(v for S, v in f.items() if set(tup) <= set(S)) / Z
            qt = float(np.prod(q[list(tup)]))
            s += qt * (Pc / (math.perm(K, c) * qt)) ** 2
        var_uc.append(s - 1.0)

    return dict(Z=Z, P1=P1, m1=m1, var_u1=var_u1, r1=r1, EZ_over_Z=EZ / Z,
                relvar=relvar, var_uc=var_uc, omegas=omegas, Zhat=Zhat, kappa=kappa)


def bounds(K, N, e):
    chi2_1 = e["m1"] - 1.0          # what bound_empirical.py stores as chi2_1
    return dict(
        log1p_exact=math.log1p(e["relvar"]),
        hybrid_m1=bf.kl_hybrid_m1(N, K, chi2_1),
        closed_m1=bf.kl_closed_m1(N, K, e["m1"]),
        hybrid_r1=bf.kl_hybrid_r1(N, K, chi2_1, e["r1"]),
        closed_r1=bf.kl_closed_r1(N, K, e["r1"]),
        relvar_m1=bf.relvar_bound_m1(N, K, chi2_1),
        relvar_r1=bf.relvar_bound_r1(N, K, chi2_1, e["r1"]),
        leading=bf.leading(N, K, chi2_1),
    )


def check_instance(Phi, q, rho, K, N):
    e = exact(Phi, q, rho, K, N)
    b = bounds(K, N, e)

    # the enumeration is right: unbiased, and the Z-free identity holds exactly
    assert e["EZ_over_Z"] == pytest.approx(1.0, abs=1e-10)
    identity = float(np.dot(bf.a_coeffs(N, K), e["var_uc"]))
    assert e["relvar"] == pytest.approx(identity, rel=1e-9, abs=1e-13)

    # the repo's code paths agree with brute force: P(y in S) from the
    # leave-one-out ESP formula, and Zhat from alpha e_K(eig B B^T)
    tgt = be.exact_target(Phi, e["kappa"], K)
    np.testing.assert_allclose(tgt["P1"], e["P1"], rtol=1e-8, atol=1e-12)
    assert tgt["sumP1"] == pytest.approx(K, rel=1e-10)
    for j in np.linspace(0, len(e["omegas"]) - 1, 7).astype(int):
        cnt = np.bincount(e["omegas"][j], minlength=len(q)).astype(float)
        lz = be.pool_logzhat(Phi, rho, cnt, K)
        if lz is None:                  # fewer than K distinct strings: Zhat = 0
            assert e["Zhat"][j] == pytest.approx(0.0, abs=1e-12 * e["Z"])
        else:
            assert math.exp(lz) == pytest.approx(e["Zhat"][j], rel=1e-8)

    # m_1 = 1 + Var_q(u_1) and m_1 <= r_1
    assert e["m1"] == pytest.approx(1.0 + e["var_u1"], abs=1e-12)
    assert e["m1"] <= e["r1"] * (1 + 1e-12)

    # per-c bounds on Var(u_c) (c = 1 is exact for the second-moment route)
    for c, v in enumerate(e["var_uc"], start=1):
        assert v <= bf.alpha(K, c) ** 2 * e["m1"] ** c - 1.0 + 1e-10
        assert v <= bf.alpha(K, c) * e["r1"] ** c - 1.0 + 1e-10
    assert e["var_uc"][0] == pytest.approx(e["var_u1"], rel=1e-9, abs=1e-13)

    # bounds on Var(Zhat)/Z^2 and on log(1 + Var/Z^2)
    tol = 1e-12
    assert b["leading"] <= e["relvar"] + tol
    assert b["relvar_m1"] >= e["relvar"] - tol
    assert b["relvar_r1"] >= e["relvar"] - tol
    assert b["hybrid_m1"] >= b["log1p_exact"] - tol
    assert b["hybrid_r1"] >= b["log1p_exact"] - tol
    assert b["closed_m1"] >= b["hybrid_m1"] - tol
    assert b["closed_r1"] >= b["hybrid_r1"] - tol
    assert b["closed_m1"] <= b["closed_r1"] + tol
    return e, b


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("heavy", [False, True])
def test_bounds_against_exact_enumeration(cfg, seed, heavy):
    nY, D, K, N = cfg
    check_instance(*make_instance(nY, D, seed, heavy), K, N)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_m1_hybrid_below_r1_hybrid_when_u1_heavy_tailed(cfg):
    nY, D, K, N = cfg
    e, b = check_instance(*make_instance(nY, D, 0, heavy=True), K, N)
    assert e["r1"] > 2 * e["m1"]
    assert b["hybrid_m1"] < b["hybrid_r1"]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_m1_hybrid_can_exceed_r1_hybrid(cfg):
    """The hybrids are not ordered: with u_1 = 1 (m_1 = r_1 = 1) the
    second-moment tail has alpha_c^2 where the r_1 tail has alpha_c."""
    nY, _, K, N = cfg
    e, b = check_instance(*uniform_instance(nY), K, N)
    assert e["m1"] == pytest.approx(1.0) and e["r1"] == pytest.approx(1.0)
    assert b["hybrid_m1"] > b["hybrid_r1"]


def test_closed_forms_dominate_hybrids_on_grid():
    for K in (2, 3, 5, 10):
        for N in sorted({K, K + 1, 2 * K, K * K, 100, 1000, 10 ** 6}):
            if N < K:
                continue
            for var in (0.0, 1e-3, 0.3, 3.0, 30.0):
                m1 = 1.0 + var
                h_m1 = bf.kl_hybrid_m1(N, K, var)
                assert bf.kl_closed_m1(N, K, m1) >= h_m1 * (1 - 1e-12)
                for ratio in (1.0, 1.5, 10.0):
                    r1 = ratio * m1
                    assert bf.kl_closed_r1(N, K, r1) >= bf.kl_hybrid_r1(N, K, var, r1) * (1 - 1e-12)
                    assert bf.kl_closed_m1(N, K, m1) <= bf.kl_closed_r1(N, K, r1)


def _old_cert(N, K, chi2_1, r1):
    """The r_1 hybrid as the scripts computed it before bound_formulas.py."""
    denom = math.comb(N, K)
    ac = [math.comb(K, c) * math.comb(N - K, K - c) / denom for c in range(1, K + 1)]
    v = ac[0] * chi2_1
    for c in range(2, K + 1):
        if ac[c - 1] == 0.0:
            continue
        v += ac[c - 1] * max(K ** c / math.perm(K, c) * r1 ** c - 1.0, 0.0)
    return math.log1p(v)


def _old_required_M(K, chi2_1, r1, target, hybrid, M_hi=10 ** 9):
    """bound_required_n.py's search before this change."""
    def bound_at(M):
        ac = bf.a_coeffs(M, K)
        v = ac[0] * chi2_1
        if hybrid:
            for c in range(2, K + 1):
                if ac[c - 1] == 0.0:
                    continue
                v += ac[c - 1] * max(K ** c / math.perm(K, c) * r1 ** c - 1.0, 0.0)
        return math.log1p(v)
    lo, hi = 2 * K, M_hi
    if bound_at(hi) > target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if mid <= 2 * K or bound_at(mid) > target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def test_r1_hybrid_matches_previous_code():
    rng = np.random.default_rng(0)
    for K in (3, 10):
        for N in (K, 2 * K, 100, 200, 400, 2500, 10 ** 6):
            for _ in range(5):
                chi2_1 = float(rng.exponential(2.0))
                r1 = (1 + chi2_1) * float(1 + rng.exponential(5.0))
                assert bf.kl_hybrid_r1(N, K, chi2_1, r1) == pytest.approx(
                    _old_cert(N, K, chi2_1, r1), rel=1e-12)


CASES_REQ = [(3, 0.5, 4.0), (3, 5.0, 40.0), (10, 0.02, 1.5), (10, 1.0, 8.0), (10, 3.0, 30.0)]


@pytest.mark.parametrize("K,chi2_1,r1", CASES_REQ)
@pytest.mark.parametrize("target", [1.0, 0.1])
def test_required_N_matches_brute_force(K, chi2_1, r1, target):
    fns = {
        "lead": lambda N: math.log1p(bf.leading(N, K, chi2_1)),
        "m1": lambda N: bf.kl_hybrid_m1(N, K, chi2_1),
        "r1": lambda N: bf.kl_hybrid_r1(N, K, chi2_1, r1),
    }
    for name, fn in fns.items():
        ans = bf.required_N(fn, K, target, N_hi=10 ** 7)
        N_big = 2 * ans + K * K
        assert fn(N_big) <= target
        vals = [fn(N) for N in range(K, N_big + 1)]
        above = [N for N, v in zip(range(K, N_big + 1), vals) if v > target]
        brute = (max(above) + 1) if above else K
        assert ans == brute, (name, ans, brute)
        # the premise of the bisection: non-increasing from K^2 - 1 on
        tail = vals[max(K, K * K - 1) - K:]
        assert all(b <= a * (1 + 1e-12) for a, b in zip(tail, tail[1:]))
        # agrees with the old bisection when the crossing is above K^2 - 1
        if name != "m1" and ans > K * K - 1:
            assert _old_required_M(K, chi2_1, r1, target, name == "r1") == ans


def test_required_N_leading_at_K_when_nothing_is_ruled_out():
    # tiny chi2_1: the leading term never exceeds the target
    K, chi2_1 = 10, 1e-3
    assert bf.required_N(lambda N: math.log1p(bf.leading(N, K, chi2_1)), K, 0.1) == K


def test_input_checks_raise():
    with pytest.raises(ValueError):
        bf.kl_hybrid_r1(100, 10, 0.5, 0.5)          # r_1 < 1
    with pytest.raises(ValueError):
        bf.kl_hybrid_m1(100, 10, -0.1)              # Var_q(u_1) < 0
    with pytest.raises(ValueError):
        bf.kl_closed_m1(100, 10, 0.5)               # m_1 < 1
    with pytest.raises(ValueError):
        bf.a_coeffs(5, 10)                          # N < K
    with pytest.raises(ValueError):
        bf.required_N(lambda N: 1.0, 3, 0.1, N_hi=1000)   # target never reached
    assert bf.m1_of_row({"chi2_1": 0.25}) == 1.25
    assert bf.m1_of_row({"chi2_1": 0.25, "m1": 1.25}) == 1.25
    with pytest.raises(ValueError):
        bf.m1_of_row({"chi2_1": 0.25, "m1": 1.5})


if __name__ == "__main__":
    print(f"{'instance':>22} {'Var/Z^2':>10} {'log1p':>9} {'hyb m1':>9} {'cls m1':>9} "
          f"{'hyb r1':>9} {'cls r1':>9} {'m1':>7} {'r1':>8}")
    insts = [(f"{cfg} s{s}{' heavy' if h else ''}", make_instance(cfg[0], cfg[1], s, h), cfg)
             for cfg in CONFIGS for s in SEEDS[:1] for h in (False, True)]
    insts += [(f"{cfg} uniform", uniform_instance(cfg[0]), cfg) for cfg in CONFIGS[:2]]
    for name, inst, (_, _, K, N) in insts:
        e, b = check_instance(*inst, K, N)
        print(f"{name:>22} {e['relvar']:10.4g} {b['log1p_exact']:9.4g} {b['hybrid_m1']:9.4g} "
              f"{b['closed_m1']:9.4g} {b['hybrid_r1']:9.4g} {b['closed_r1']:9.4g} "
              f"{e['m1']:7.3f} {e['r1']:8.3f}")
