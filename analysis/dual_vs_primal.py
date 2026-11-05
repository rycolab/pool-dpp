"""
C1 - Wall-clock cost of exact k-DPP sampling (tab:dual-primal)
==============================================================
Times, on synthetic pools over an (N, D) grid and one CPU thread:

  dual         form C = B^T B, eigendecompose it, one draw of the dual sampler
               of the paper's algorithm
  primal eigh  form L = B B^T and eigendecompose it, a lower bound on the cost
               of a primal sampler
  stable       dpp.stable.StableKDPP: Jacobi SVD of B and one draw, the sampler
               the experiments use
  greedy MAP   lm_dpp.estimation.greedy_map

Usage:
    OPENBLAS_NUM_THREADS=1 python analysis/dual_vs_primal.py
Output: analysis/dual_vs_primal.json
"""

import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from dpp.stable import StableKDPP, _log_esp_table  # noqa: E402
from lm_dpp.estimation import greedy_map  # noqa: E402

N_GRID = [25, 50, 100, 200, 400, 800, 2000, 10000]
D_GRID = [384, 1024]
SETTINGS = [("FineWeb (MiniLM)", "fineweb", 384, 400),
            ("Ambrosia (MiniLM)", "ambrosia", 384, 200),
            ("GSM8K (Qwen3-Emb)", "gsm8k", 1024, 100)]
K = 10
REPS = 5
BETA = 0.01


def orthonormalize_C(V, C):
    def inner(a, b):
        return a @ C @ b

    for a in range(V.shape[1]):
        for b in range(a):
            V[:, a] = V[:, a] - inner(V[:, a], V[:, b]) * V[:, b]
        V[:, a] = V[:, a] / np.sqrt(inner(V[:, a], V[:, a]))


def dual_draw(B, k):
    """One draw of the dual sampler: eigenvectors of C by elementary symmetric
    polynomials, then the projection loop with re-orthonormalization in the C metric.
    Only for timing: it fails when row weights span more than about 15 orders of magnitude."""
    C = B.T @ B
    vals, vecs = np.linalg.eigh(C)
    ll = np.log(np.maximum(vals, 1e-300))
    T = _log_esp_table(ll, k)
    index, rem = [], k
    for i in range(len(ll), 0, -1):
        if rem == 0:
            break
        if i == rem or np.random.random() < np.exp(ll[i - 1] + T[i - 1, rem - 1] - T[i, rem]):
            index.append(i - 1)
            rem -= 1
    V = vecs[:, index]
    orthonormalize_C(V, C)
    items = []
    for _ in range(k):
        p = np.maximum(np.sum((B @ V) ** 2, axis=1), 0)
        item = int(np.random.choice(len(p), p=p / p.sum()))
        items.append(item)
        S = B[item] @ V
        j = np.abs(S).argmax()
        V = V - (1 / S[j]) * np.outer(V[:, j], S)
        V = np.delete(V, j, axis=1)
        orthonormalize_C(V, C)
    return items


def reps_for(N):
    """Fewer repetitions at large N (the 10k eigh runs for minutes)."""
    return 1 if N >= 2000 else REPS


def make_pool(N, D, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(N, D))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    # anisotropic, like real sentence embeddings: push mass onto a few directions
    E += 0.7 * rng.normal(size=(1, D))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    lp = rng.normal(loc=-40, scale=10, size=N)
    return E, lp


def timed(f):
    t0 = time.perf_counter()
    f()
    return (time.perf_counter() - t0) * 1e3


def main():
    rows = []
    print(f"{'D':>6} {'N':>6} {'dual':>10} {'primal eigh':>12} {'stable':>10} {'greedy MAP':>11}  (ms)")
    for D in D_GRID:
        for N in N_GRID:
            E, lp = make_pool(N, D)
            log_s = BETA * lp
            B = np.exp(log_s - log_s.max())[:, None] * E
            r = reps_for(N)
            row = {"D": D, "N": N,
                   "dual_ms": float(np.median([timed(lambda: dual_draw(B, K)) for _ in range(r)])),
                   "primal_eigh_ms": float(np.median([timed(lambda: np.linalg.eigh(B @ B.T)) for _ in range(r)])),
                   "stable_ms": float(np.median([timed(lambda: StableKDPP(log_s, E, K).sample()) for _ in range(r)])),
                   "greedy_map_ms": float(np.median([timed(lambda: greedy_map(log_s, E, K)) for _ in range(r)]))}
            rows.append(row)
            print(f"{D:>6} {N:>6} {row['dual_ms']:>10.2f} {row['primal_eigh_ms']:>12.2f} "
                  f"{row['stable_ms']:>10.2f} {row['greedy_map_ms']:>11.2f}", flush=True)

    settings = {}
    for name, key, D, N in SETTINGS:
        r = min((x for x in rows if x["D"] == D), key=lambda x: abs(x["N"] - N))
        settings[key] = {"D": D, "N": N, "measured": r,
                         "dual_over_primal_eigh": r["dual_ms"] / r["primal_eigh_ms"]}
        print(f"{name:<20s} D={D:<5d} N={N:<5d} dual / primal eigh = "
              f"{settings[key]['dual_over_primal_eigh']:.1f}")

    dest = os.path.join(SCRIPT_DIR, "dual_vs_primal.json")
    with open(dest, "w") as f:
        json.dump({"grid": rows, "K": K, "settings": settings}, f, indent=2)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
