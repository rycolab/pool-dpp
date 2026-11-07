"""Brute-force check of `thm:var-kl-mixture` / `thm:zfree` bounds when Z-hat = 0 has positive probability.

Enumerates every pool omega in Y^N exactly (no Monte Carlo), using the paper's sketch
f-hat_omega(S) = alpha * sum_{I: image(I) = S} det(Lhat_I), Lhat = Bhat^T Bhat,
Bhat columns sqrt(kappa/q/N) phi(y_n), alpha = N^K / (N)_K.

Embeddings are drawn so that some pairs are collinear (the case the paper's former pool non-degeneracy assumption
ruled out), so
Z-hat = 0 on a set of pools of positive probability. For each instance we compute:
  pibar0(S) = E[pihat(S); Zhat > 0]  (the law with the Zhat = 0 pools removed)
and several complete output laws (redraw; fixed fallback S0 for every S0; uniform fallback),
then check chi2(p || pibar) <= Var(Zhat)/Z^2 and KL <= log(1 + Var/Z^2) <= Var/Z^2, with
Var under the plain i.i.d. pool law, plus KL <= K^2 rho/(N-K+1) (`thm:zfree`, rho = E_q[u1^2], the variable m1),
the exact rational inequality Var/Z^2 <= sum_{c=1}^K theta^c/c! behind it, and the r1 corollary.
All arithmetic is exact (Fractions) except the KL logarithms.
Stops with an error on the first violation.
"""
import itertools, math, sys
from fractions import Fraction as F
import random

TOL = 1e-9

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
            m = A[r][c] / A[c][c]
            for k in range(c, n):
                A[r][k] -= m * A[c][k]
    return d

def gram(vecs):
    return [[sum(a * b for a, b in zip(u, v)) for v in vecs] for u in vecs]

def run_instance(rng, M, K, N, D, n_collinear):
    phi = [[F(rng.randint(-3, 3)) for _ in range(D)] for _ in range(M)]
    for _ in range(n_collinear):
        i, j = rng.sample(range(M), 2)
        c = F(rng.randint(1, 4), rng.randint(1, 3)) * rng.choice([-1, 1])
        phi[j] = [c * x for x in phi[i]]
    kappa = [F(rng.randint(1, 8), 4) for _ in range(M)]
    qw = [rng.randint(1, 6) for _ in range(M)]
    q = [F(x, sum(qw)) for x in qw]
    w = [kappa[y] / q[y] for y in range(M)]
    sets = list(itertools.combinations(range(M), K))
    # det(L_S) = prod kappa * det(Gram_S), exactly
    f = {S: math.prod(kappa[y] for y in S) * det(gram([phi[y] for y in S])) for S in sets}
    Z = sum(f.values())
    if Z == 0:
        return None
    p = {S: f[S] / Z for S in sets}
    alpha = F(N ** K, math.perm(N, K))
    Ef = {S: F(0) for S in sets}
    pibar0 = {S: F(0) for S in sets}
    EZ = EZ2 = P0 = F(0)
    idx_sets = list(itertools.combinations(range(N), K))
    for omega in itertools.product(range(M), repeat=N):
        prob = math.prod(q[y] for y in omega)
        fh = {}
        for I in idx_sets:
            img = tuple(sorted(set(omega[i] for i in I)))
            if len(img) < K:
                continue  # repeated string: two equal columns, det is exactly 0
            # det(Lhat_I) = prod_{i in I} w(y_i)/N * det(Gram_I)
            d = math.prod(w[omega[i]] / N for i in I) * det(gram([phi[omega[i]] for i in I]))
            fh[img] = fh.get(img, F(0)) + alpha * d
        Zh = sum(fh.values(), F(0))
        EZ += prob * Zh
        EZ2 += prob * Zh ** 2
        for S, v in fh.items():
            Ef[S] += prob * v
        if Zh == 0:
            P0 += prob
        else:
            for S, v in fh.items():
                pibar0[S] += prob * v / Zh
    for S in sets:
        if Ef[S] != f[S]:
            sys.exit(f"unbiasedness fails at {S}: {Ef[S]} vs {f[S]}")
    if EZ != Z:
        sys.exit(f"E[Zhat] != Z: {EZ} vs {Z}")
    V = EZ2 - Z ** 2
    ratio = V / Z ** 2
    if sum(pibar0.values()) != 1 - P0:
        sys.exit("mass of pibar0 != P(Zhat > 0)")
    incl = [F(0)] * M
    for S in sets:
        for y in S:
            incl[y] += p[S]
    u1 = [incl[y] / (K * q[y]) for y in range(M)]
    r1 = max(u1)
    m1 = sum(q[y] * u1[y] ** 2 for y in range(M))  # second moment E_q[u1^2]
    if not (m1 <= r1):
        sys.exit(f"rho {m1} > r1 {r1}")
    theta = K ** 2 * m1 / (N - K + 1)       # `thm:zfree` (second-moment form)
    theta_r1 = K ** 2 * r1 / (N - K + 1)    # r1 corollary
    closed = sum(theta ** c / math.factorial(c) for c in range(1, K + 1))  # exact; <= e^theta - 1
    if ratio > closed:
        sys.exit(f"Var/Z^2 {ratio} > sum_c theta^c/c! {closed}")

    laws = {"redraw": {S: pibar0[S] / (1 - P0) for S in sets},
            "uniform": {S: pibar0[S] + P0 / len(sets) for S in sets}}
    for S0 in sets:
        laws[f"fixed{S0}"] = {S: pibar0[S] + (P0 if S == S0 else 0) for S in sets}
    worst = {}
    for name, pb in laws.items():
        if sum(pb.values()) != 1:
            sys.exit(f"{name}: law does not sum to 1")
        for S in sets:
            if pb[S] == 0 and p[S] > 0:
                sys.exit(f"{name}: pibar({S}) = 0 < p({S})")
        chi2 = sum((p[S] - pb[S]) ** 2 / pb[S] for S in sets if pb[S] > 0)  # exact
        kl = sum(float(p[S]) * math.log(float(p[S]) / float(pb[S])) for S in sets if p[S] > 0)
        if chi2 > ratio:
            sys.exit(f"{name}: chi2 {float(chi2)} > Var/Z^2 {float(ratio)}")
        if kl > math.log1p(float(ratio)) + TOL:
            sys.exit(f"{name}: KL {kl} > log(1+Var/Z^2) {math.log1p(float(ratio))}")
        if kl > float(theta) + TOL:
            sys.exit(f"{name}: KL {kl} > theta {float(theta)}")
        if kl > float(theta_r1) + TOL:
            sys.exit(f"{name}: KL {kl} > theta_r1 {float(theta_r1)}")
        worst[name] = (float(chi2 / ratio) if ratio > 0 else 0.0, kl)
    return dict(P0=float(P0), ratio=float(ratio), theta=float(theta), worst=worst)


def main():
    rng = random.Random(0)
    configs = [(4, 2, 3, 2, 1), (4, 2, 4, 2, 1), (5, 2, 3, 2, 2), (5, 3, 4, 3, 1),
               (5, 3, 5, 3, 2), (6, 2, 4, 3, 2), (4, 2, 2, 2, 1)]
    n_inst = n_pos = 0
    max_P0 = 0.0
    max_chi_frac = 0.0
    for cfg in configs:
        for _ in range(40):
            r = run_instance(rng, *cfg)
            if r is None:
                continue
            n_inst += 1
            if r["P0"] > 0:
                n_pos += 1
            max_P0 = max(max_P0, r["P0"])
            max_chi_frac = max(max_chi_frac, max(v[0] for v in r["worst"].values()))
        print(f"config M,K,N,D,collinear={cfg}: done", flush=True)
    print(f"instances checked: {n_inst}; with P(Zhat=0) > 0: {n_pos}; "
          f"max P(Zhat=0) = {max_P0:.3f}")
    print(f"largest chi2 / (Var/Z^2) over all laws and instances: {max_chi_frac:.3f} (bound is 1)")
    print("all bounds hold")


if __name__ == "__main__":
    main()
