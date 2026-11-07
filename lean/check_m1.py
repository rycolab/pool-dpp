"""Exact check of the second-moment form of `thm:zfree` and `cor:kl-sum`.

For random small instances (base set of n strings with integer embeddings phi, positive
rational quality kappa, positive rational proposal q), everything is computed exactly with
Fractions, by enumerating every pool omega in [n]^N and every tuple of strings:

  u_c(y_1..y_c) = P({y_1..y_c} in S) / ((K)_c q(y_1)...q(y_c))   (0 on tuples with a repeat)
  rho = E_q[u_1^2] (the variable m1 below),  a_c = C(K,c) C(N-K,K-c) / C(N,K)
  Zhat(omega) = (1/(N)_K) sum_{|I|=K} prod_{i in I} w(y_i) det(G_I),  w = kappa/q

and the script checks, stopping with an error on the first violation:
  (1) E[Zhat] = Z and the identity Var(Zhat)/Z^2 = sum_c a_c Var(u_c)          (`thm:zfree`)
  (2) Var(u_c) <= (K^c/(K)_c)^2 rho^c - 1 for every c                        (eq:second-moment)
  (3) Var(Zhat)/Z^2 <= a_1 Var(u_1) + sum_{c>=2} a_c ((K^c/(K)_c)^2 rho^c - 1)  (`cor:kl-sum`)
  (4) 1 + Var(Zhat)/Z^2 <= a_0 + sum_{c=1}^K theta^c/c!,  theta = K^2 rho/(N-K+1),
      a_0 = C(N-K,K)/C(N,K)  (`thm:zfree`'s closed form keeping the -1; the right side is
      at most e^theta - 1 + a_0, so KL <= log(e^theta - 1 + a_0) <= theta)
  (5) rho = 1 + Var_q(u_1)                                     (paragraph after `thm:zfree`)
  (6) `cor:kl-sum` as one sum: 1 + its sum = sum_{c=0}^K a_c (K^c/(K)_c)^2 rho^c
  (7) Hoeffding bounds (eq:hoeffding-bounds): (K^2/N) Var(u_1) <= Var(Zhat)/Z^2 <= (K/N) Var(u_K)
All of (1)-(7) are exact rational comparisons. Run: python3 check_m1.py [instances] [seed]
"""
import itertools, math, random, sys
from fractions import Fraction as F


def det(A):
    """Exact determinant of a small matrix of Fractions (Gaussian elimination)."""
    A = [row[:] for row in A]
    n = len(A)
    d = F(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] != 0), None)
        if piv is None:
            return F(0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            d = -d
        d *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            for k in range(c, n):
                A[r][k] -= f * A[c][k]
    return d


def gram_det(phi, idx):
    G = [[sum(F(a) * b for a, b in zip(phi[i], phi[j])) for j in idx] for i in idx]
    return det(G)


def falling(n, k):
    return math.perm(n, k)


def check_instance(rng, K, n, N, D):
    phi = [[rng.randint(-3, 3) for _ in range(D)] for _ in range(n)]
    kappa = [F(rng.randint(1, 9), rng.randint(1, 9)) for _ in range(n)]
    raw = [F(rng.randint(1, 12)) for _ in range(n)]
    q = [x / sum(raw) for x in raw]
    w = [kappa[i] / q[i] for i in range(n)]

    # target K-DPP and its inclusion probabilities P(T in S) for every T with |T| <= K
    f = {S: (math.prod(kappa[i] for i in S) * gram_det(phi, S)) for S in itertools.combinations(range(n), K)}
    Z = sum(f.values())
    if Z == 0:
        return None
    incl = {}
    for c in range(1, K + 1):
        for T in itertools.combinations(range(n), c):
            incl[T] = sum(v for S, v in f.items() if set(T) <= set(S)) / Z

    def u(c, ys):
        if len(set(ys)) < c:
            return F(0)
        return incl[tuple(sorted(ys))] / (falling(K, c) * math.prod(q[y] for y in ys))

    def pw(ys):
        return math.prod(q[y] for y in ys)

    Eu2 = {}
    for c in range(1, K + 1):
        Eu, Esq = F(0), F(0)
        for ys in itertools.product(range(n), repeat=c):
            uc = u(c, ys)
            Eu += pw(ys) * uc
            Esq += pw(ys) * uc * uc
        assert Eu == 1, ("E[u_c] != 1", c, Eu)
        Eu2[c] = Esq
    var_u = {c: Eu2[c] - 1 for c in Eu2}
    m1 = Eu2[1]

    # exact Var(Zhat)/Z^2 by enumerating every pool
    hcache = {}
    def h(ys):
        key = tuple(sorted(ys))
        if key not in hcache:
            hcache[key] = F(0) if len(set(key)) < K else math.prod(w[i] for i in key) * gram_det(phi, key)
        return hcache[key]
    subsets = list(itertools.combinations(range(N), K))
    EZ, EZ2 = F(0), F(0)
    for om in itertools.product(range(n), repeat=N):
        z = sum(h([om[i] for i in I]) for I in subsets) / falling(N, K)
        p = pw(om)
        EZ += p * z
        EZ2 += p * z * z
    assert EZ == Z, ("E[Zhat] != Z", EZ, Z)
    V = EZ2 / Z**2 - 1

    a = {c: F(math.comb(K, c) * math.comb(N - K, K - c), math.comb(N, K)) for c in range(1, K + 1)}
    # (1) `thm:zfree` identity
    assert V == sum(a[c] * var_u[c] for c in a), ("identity", V)
    # (2) second-moment bound for every overlap
    for c in a:
        bc = F(K**c, falling(K, c)) ** 2 * m1**c - 1
        assert var_u[c] <= bc, ("eq:second-moment", c, var_u[c], bc)
    # (3) `cor:kl-sum`
    cor = a[1] * var_u[1] + sum(a[c] * (F(K**c, falling(K, c)) ** 2 * m1**c - 1) for c in a if c >= 2)
    assert V <= cor, ("corollary", V, cor)
    # (4) closed form, keeping the -1
    theta = F(K * K) * m1 / (N - K + 1)
    closed = sum(theta**c / math.factorial(c) for c in a)
    a0 = F(math.comb(N - K, K), math.comb(N, K))
    assert 1 + V <= a0 + closed, ("closed form", V, closed)
    # (6) `cor:kl-sum` as one sum over c = 0..K
    assert 1 + cor == a0 + sum(a[c] * F(K**c, falling(K, c)) ** 2 * m1**c for c in a), ("one sum", cor)
    # (7) Hoeffding bounds
    assert F(K * K, N) * var_u[1] <= V <= F(K, N) * var_u[K], ("hoeffding", V)
    # (5) rho = 1 + Var(u1)
    assert m1 == 1 + var_u[1], ("rho vs Var(u1)", m1, var_u[1])
    return float(V), float(cor), float(closed)


def main():
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = random.Random(seed)
    done, ratios_cor, ratios_closed = 0, [], []
    while done < n_inst:
        K = rng.choice([1, 2, 3])
        n = rng.choice([K + 1, K + 2, 5]) if K < 3 else rng.choice([4, 5])
        N = rng.randint(K, 5 if n <= 4 else 4)
        D = rng.choice([K, K + 1])
        res = check_instance(rng, K, n, N, D)
        if res is None:
            continue
        V, cor, closed = res
        done += 1
        if V > 0:
            ratios_cor.append(V / cor)
            ratios_closed.append(V / closed)
    print(f"{done} instances, all checks passed (exact arithmetic).")
    print(f"  Var/Z^2 as a fraction of `cor:kl-sum`'s sum: max {max(ratios_cor):.3f}, median {sorted(ratios_cor)[len(ratios_cor)//2]:.3f}")
    print(f"  Var/Z^2 as a fraction of the closed-form sum:  max {max(ratios_closed):.3f}, median {sorted(ratios_closed)[len(ratios_closed)//2]:.3f}")


if __name__ == "__main__":
    main()
