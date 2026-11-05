"""
Closed-form pieces of the bound on KL(P || pi_bar) (thm:zfree, cor:certificates)
=================================================================================
One copy of the formulas that bound_empirical.py, bound_required_n.py,
bound_tables.py, gsm8k_n2500.py, plot_ndial.py and the ess_*.py scripts used to
repeat. Notation: N = pool size, K = set size, q = proposal, P = target K-DPP.

    Var(Zhat)/Z^2 = sum_{c=1}^K a_c Var_{q^c}(u_c),
    a_c(N) = C(K,c) C(N-K,K-c) / C(N,K),
    u_1(y) = P(y in S) / (K q(y)),   E_q[u_1] = 1,
    KL(P || pi_bar) <= log(1 + Var(Zhat)/Z^2).

The c = 1 term is exact: Var_q(u_1) = chi2(pi_1 || q) with pi_1 = P(y in S)/K.
In the per-prompt rows this is the key ``chi2_1``. For c >= 2, negative
correlation gives the pointwise bound

    u_c(y_1..y_c) <= alpha_c prod_i u_1(y_i),   alpha_c = K^c / (K)_c,

and the two routes from it to Var(u_c) are:

  second moment (primary, the paper's bound). Square the pointwise bound and
      average over independent draws:
          Var(u_c) <= alpha_c^2 m_1^c - 1,   m_1 = E_q[u_1^2] = 1 + Var_q(u_1).
      Hybrid:  KL <= log(1 + a_1 Var_q(u_1) + sum_{c>=2} a_c (alpha_c^2 m_1^c - 1)).
      At c = 1, alpha_1^2 m_1 - 1 = Var_q(u_1), so this is the same as using
      the squared pointwise bound for every c: it depends on the instance only
      through Var_q(u_1). Since sum_{c=0}^K a_c = 1 (a_0 = C(N-K,K)/C(N,K),
      no overlap) and alpha_0 = 1, the hybrid equals the paper's Corollary 5.6,
          KL <= log sum_{c=0}^K a_c alpha_c^2 m_1^c,
      exactly (kl_hybrid_m1; tests/test_bounds_m1.py checks the equality).
      Closed form (the paper's Theorem 5.5), theta = K^2 m_1 / (N - K + 1):
          KL <= log(e^theta - 1 + a_0) <= theta.
      kl_closed_m1 returns the middle expression and kl_theta_m1 returns theta.
      The first commit of this change returned theta from kl_closed_m1 (key
      bound_simple_m1_KL); that key now holds log(e^theta - 1 + a_0) and theta
      has its own key, bound_theta_m1.

  sup (secondary, kept for the paper's footnote). Var(u_c) <= sup u_c - 1:
          Var(u_c) <= alpha_c r_1^c - 1,   r_1 = sup_y u_1(y).
      Hybrid:  KL <= log(1 + a_1 Var_q(u_1) + sum_{c>=2} a_c (alpha_c r_1^c - 1)).
      Closed form:  KL <= K^2 r_1 / (N - K + 1)   (no a_0 refinement).

Each closed form is at least its hybrid, at every N >= K. Proof:
a_c alpha_c^2 m_1^c <= theta^c / c! for c >= 1, so the Corollary 5.6 sum is at
most a_0 + e^theta - 1 <= e^theta; and the r_1 sum is at most
(1 + K r_1/(N-K+1))^K <= e^{K^2 r_1/(N-K+1)}. tests/test_bounds_m1.py checks
this on a grid.

Lower bounds on Var(Zhat)/Z^2 (and so on the reduced quantity, not on the KL):
  Hoeffding (primary). Var(u_c)/c is nondecreasing in c (Hoeffding 1948,
      Thm 5.2: u_c is the c-th conditional expectation of the U-statistic
      kernel of Zhat/Z), and sum_c c a_c = K^2/N (the expected overlap of two
      random K-subsets of N slots), so
          Var(Zhat)/Z^2 >= (K^2/N) Var_q(u_1)            (lower_hoeffding)
      at every N, which is >= a_1 Var_q(u_1).
  leading term (secondary). a_1(N) Var_q(u_1), the exact c = 1 term (leading).
The same theorem gives the upper bound Var(Zhat)/Z^2 <= (K/N) Var(u_K) (the
paper's Rao-Blackwell remark). Var(u_K) is not computed from u_1, so it appears
only in tests/test_bounds_m1.py and in bound_empirical.py's T1 enumeration.

m_1 <= r_1 always (E_q[u_1^2] <= sup u_1 E_q[u_1]), so the second-moment closed
form (<= theta) is never larger than the r_1 closed form. The HYBRIDS ARE NOT ORDERED: the
second-moment tail squares alpha_c, so when u_1 is nearly constant (r_1 close to
m_1) the second-moment hybrid is the larger of the two. It is the smaller one
only when r_1/m_1 exceeds a threshold that depends on K, N and Var_q(u_1)
(about 1.05 to 1.9 at the paper's K and N). bound_empirical.py and
bound_tables.py report, per setting, the fraction of prompts where the
second-moment hybrid is the larger; tests/test_bounds_m1.py has an instance
where it is.

Input checks (no clamping). The old scripts wrapped each c >= 2 tail term in
max(., 0). For valid inputs (r_1 >= 1, m_1 >= 1, alpha_c > 1 for c >= 2) no
term is negative, so that clamp never fired; it is replaced by explicit checks
that raise ValueError when Var_q(u_1) < 0 or r_1 < 1 beyond rounding (relative
tolerance INPUT_RTOL), since either one means P(y in S) does not sum to K.
A tail term a_c(...) is skipped only when a_c = 0 exactly (c overlaps are
impossible at this N), which is exact, not an approximation.
"""

import math
from fractions import Fraction

import numpy as np

# Rounding slack for the input checks: Var_q(u_1) >= -INPUT_RTOL and
# r_1 >= 1 - INPUT_RTOL. Only decides whether to raise; never changes a value.
INPUT_RTOL = 1e-9


def a_coeffs(N, K):
    """a_c(N) = C(K,c) C(N-K,K-c)/C(N,K) for c=1..K (0 when infeasible).

    a_c is the probability that two independent uniformly random K-subsets of
    the N pool slots share exactly c slots.
    """
    N, K = int(N), int(K)
    if N < K:
        raise ValueError(f"pool size N={N} is smaller than K={K}")
    denom = math.comb(N, K)
    return np.array([
        math.comb(K, c) * math.comb(N - K, K - c) / denom
        if (N - K) >= (K - c) else 0.0
        for c in range(1, K + 1)
    ])


def alpha(K, c):
    """alpha_c = K^c / (K)_c, the constant in u_c <= alpha_c prod_i u_1(y_i)."""
    return K ** c / math.perm(K, c)


def _check_var(chi2_1):
    chi2_1 = float(chi2_1)
    if not chi2_1 >= -INPUT_RTOL:      # also catches NaN
        raise ValueError(f"Var_q(u_1) = {chi2_1!r} < 0: P(y in S) does not sum "
                         "to K, or the input is corrupt")
    return chi2_1


def _check_r1(r1):
    r1 = float(r1)
    if not r1 >= 1.0 - INPUT_RTOL:
        raise ValueError(f"r_1 = sup u_1 = {r1!r} < 1, impossible when "
                         "E_q[u_1] = 1: P(y in S) does not sum to K")
    return r1


def leading(N, K, chi2_1):
    """Exact c = 1 term a_1(N) Var_q(u_1) (secondary lower bound on Var(Zhat)/Z^2)."""
    return float(a_coeffs(N, K)[0] * _check_var(chi2_1))


def lower_hoeffding(N, K, chi2_1):
    """Primary lower bound on Var(Zhat)/Z^2: (K^2/N) Var_q(u_1).

    From Hoeffding (1948, Thm 5.2), Var(u_c)/c is nondecreasing in c, and
    sum_c c a_c(N) = K^2/N. It is >= leading(N, K, chi2_1) at every N.
    """
    N, K = int(N), int(K)
    if N < K:
        raise ValueError(f"pool size N={N} is smaller than K={K}")
    return K * K / N * _check_var(chi2_1)


def required_N_lower_hoeffding(K, chi2_1, target):
    """Smallest N >= K with log(1 + (K^2/N) Var_q(u_1)) <= target, i.e.
    N >= K^2 Var_q(u_1) / (e^target - 1).

    The bound decreases in N at every N, so this is also the smallest N0 with
    the bound <= target at every N >= N0 (no floor effects, unlike the a_1
    version, whose a_1(N) is 0 for N < 2K - 1). The ceiling is checked against
    the predicate itself, so rounding in the division cannot shift the answer.
    """
    if not target > 0:
        raise ValueError(f"target must be > 0, got {target!r}")
    chi2_1 = _check_var(chi2_1)

    def ok(N):
        return math.log1p(lower_hoeffding(N, K, chi2_1)) <= target

    N = max(K, math.ceil(K * K * max(chi2_1, 0.0) / math.expm1(target)))
    while N > K and ok(N - 1):
        N -= 1
    while not ok(N):
        N += 1
    return N


def relvar_bound_m1(N, K, chi2_1):
    """Second-moment upper bound on Var(Zhat)/Z^2:
    a_1 Var_q(u_1) + sum_{c>=2} a_c (alpha_c^2 m_1^c - 1), m_1 = 1 + Var_q(u_1)."""
    chi2_1 = _check_var(chi2_1)
    m1 = 1.0 + chi2_1
    a = a_coeffs(N, K)
    v = float(a[0]) * chi2_1
    for c in range(2, K + 1):
        if a[c - 1] == 0.0:
            continue
        v += float(a[c - 1]) * (alpha(K, c) ** 2 * m1 ** c - 1.0)
    return v


def relvar_bound_r1(N, K, chi2_1, r1):
    """sup-route upper bound on Var(Zhat)/Z^2 (secondary):
    a_1 Var_q(u_1) + sum_{c>=2} a_c (alpha_c r_1^c - 1)."""
    chi2_1 = _check_var(chi2_1)
    r1 = _check_r1(r1)
    a = a_coeffs(N, K)
    v = float(a[0]) * chi2_1
    for c in range(2, K + 1):
        if a[c - 1] == 0.0:
            continue
        v += float(a[c - 1]) * (alpha(K, c) * r1 ** c - 1.0)
    return v


def na_tail_m1(N, K, chi2_1):
    """c >= 2 part of relvar_bound_m1."""
    return relvar_bound_m1(N, K, chi2_1) - leading(N, K, chi2_1)


def na_tail_r1(N, K, chi2_1, r1):
    """c >= 2 part of relvar_bound_r1 (does not depend on chi2_1)."""
    return relvar_bound_r1(N, K, chi2_1, r1) - leading(N, K, chi2_1)


def kl_hybrid_m1(N, K, chi2_1):
    """Primary bound: KL <= log(1 + relvar_bound_m1).

    Equal to the paper's Corollary 5.6, log sum_{c=0}^K a_c alpha_c^2 m_1^c
    (a_0 = C(N-K,K)/C(N,K), alpha_0 = 1), because sum_{c=0}^K a_c = 1.
    """
    return math.log1p(relvar_bound_m1(N, K, chi2_1))


def kl_hybrid_r1(N, K, chi2_1, r1):
    """Secondary bound: KL <= log(1 + relvar_bound_r1)."""
    return math.log1p(relvar_bound_r1(N, K, chi2_1, r1))


def kl_theta_m1(N, K, m1):
    """theta = K^2 m_1/(N-K+1), the outer bound in KL <= log(e^theta - 1 + a_0) <= theta."""
    if N < K:
        raise ValueError(f"pool size N={N} is smaller than K={K}")
    _check_var(float(m1) - 1.0)
    return K * K * float(m1) / (N - K + 1)


def _expm1_minus_x(x):
    """e^x - 1 - x for 0 <= x <= 1, by its series (no cancellation)."""
    total, term, k = 0.0, x * x / 2.0, 2
    while term > 0.0 and term > 1e-18 * total:
        total += term
        k += 1
        term *= x / k
    return total


def kl_closed_m1(N, K, m1):
    """Primary closed form (the paper's Theorem 5.5):
    KL <= log(e^theta - 1 + a_0),  theta = K^2 m_1/(N-K+1),  a_0 = C(N-K,K)/C(N,K).

    Evaluation, accurate to rounding at every N and m_1 (checked against
    60-digit arithmetic in tests/test_bounds_m1.py):
      theta <= 1: log1p(D) with D = [theta - (1 - a_0)] + [e^theta - 1 - theta].
        For large N, theta and 1 - a_0 are both about K^2/N while D is about
        K^2 (m_1 - 1)/N, so the first bracket is formed in exact rational
        arithmetic (m_1 is a float, hence rational; 1 - a_0 from exact
        integers) and rounded once; the second is a series. Both brackets are
        >= 0 (1 - a_0 = P(overlap >= 1) <= E[overlap] = K^2/N < theta).
      theta > 1: theta + log1p(-(1 - a_0) e^-theta), which does not overflow.
    The first commit of this change returned theta here; theta is now
    kl_theta_m1.
    """
    theta = kl_theta_m1(N, K, m1)
    N, K = int(N), int(K)
    denom = math.comb(N, K)
    one_minus_a0 = Fraction(denom - math.comb(N - K, K), denom)
    if theta > 1.0:
        return theta + math.log1p(-float(one_minus_a0) * math.exp(-theta))
    lin = Fraction(K * K) * Fraction(float(m1)) / (N - K + 1) - one_minus_a0
    return math.log1p(float(lin) + _expm1_minus_x(theta))


def kl_closed_r1(N, K, r1):
    """Secondary closed form: KL <= K^2 r_1/(N-K+1)."""
    if N < K:
        raise ValueError(f"pool size N={N} is smaller than K={K}")
    return K * K * _check_r1(r1) / (N - K + 1)


def m1_of_row(row, rtol=1e-9):
    """m_1 = E_q[u_1^2] for a per-prompt bound row.

    Rows store chi2_1 = E_q[u_1^2] - 1 (bound_empirical.py computes it as
    sum_y pi_1(y)^2/q(y) - 1), so m_1 = 1 + chi2_1 by definition. That makes
    bound_raw files written before the key ``m1`` existed usable unchanged.
    When the row also stores ``m1``, the two are checked to agree.
    """
    m1 = 1.0 + float(row["chi2_1"])
    if "m1" in row and not math.isclose(float(row["m1"]), m1, rel_tol=rtol):
        raise ValueError(f"row has m1={row['m1']!r} but 1 + chi2_1 = {m1!r}")
    return m1


def required_N(bound_fn, K, target, N_hi=10 ** 9):
    """Smallest N0 >= K such that bound_fn(N) <= target for EVERY N >= N0.

    bound_fn(N) is one of the bounds above at a fixed instance: a combination
    of a_1(N), ..., a_K(N) with non-negative, N-independent coefficients. For
    N >= K^2 - 1 every a_c (c >= 1) is non-increasing in N
    (a_c(N+1)/a_c(N) = (N+1-K)^2 / ((N+1)(N+1-2K+c)) <= 1 iff c(N+1) >= K^2),
    so the bound is non-increasing there and bisection finds the crossing.
    Below K^2 - 1 the a_c are not monotone (a_1(N) = 0 for N < 2K-1, then
    rises), so "the smallest N with bound <= target" can be a spurious small N;
    for the leading term alone it would be N = K for every prompt. Hence the
    "for every N >= N0" definition: when the bound is already <= target at
    K^2 - 1, walk down N by N while it stays <= target. A result of K means
    the bound rules out no pool size.

    The old code bisected on [2K, N_hi] as if the bound were monotone
    everywhere, so its smallest possible answer was 2K+1. Its answers agree
    with this function whenever the crossing is above K^2 - 1.

    Raises ValueError when the bound is still above target at N_hi (the old
    code returned None there and the caller dropped the prompt from the
    median).
    """
    N0 = max(K, K * K - 1)
    if N_hi < N0:
        raise ValueError(f"N_hi={N_hi} < max(K, K^2-1)={N0}")
    if bound_fn(N_hi) > target:
        raise ValueError(f"bound {bound_fn(N_hi):.4g} > target {target} even at "
                         f"N_hi={N_hi}; raise N_hi")
    if bound_fn(N0) > target:
        # monotone range: bound(lo - 1) > target >= bound(hi)
        lo, hi = N0 + 1, N_hi
        while lo < hi:
            mid = (lo + hi) // 2
            if bound_fn(mid) > target:
                lo = mid + 1
            else:
                hi = mid
        return lo
    N = N0
    while N > K and bound_fn(N - 1) <= target:
        N -= 1
    return N
