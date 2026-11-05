"""
Validation suite for lm_dpp.sbs_decoding (SBS and Conditional Poisson SBS).

What is checked, and against what ground truth:

1. The conditional Poisson samplers (batched rejection sampler and the
   sequential DP reference) match the exact CP(w, K) set distribution
   computed by enumeration, including under head truncation.
2. The exact CP inclusion-probability DP matches enumeration.
3. SBS end-to-end on a tiny enumerable model samples sets whose empirical
   distribution matches exact sampling-without-replacement from the
   (temperature-annealed) leaf distribution, at tau = 1 and tau != 1.
4. CPSBS end-to-end matches a brute-force enumeration of the full CPSBS
   Markov chain over beam sets (Meister et al. 2021, Eq. 8-11), at tau = 1
   and tau != 1.
5. On a real (tiny, randomly initialized) Hugging Face GPT-2, the cumulative
   log-probabilities returned by the decoder match from-scratch rescoring of
   the returned token sequences (validates KV-cache reordering, carry logic,
   and the dual raw/annealed bookkeeping), and the K returned sequences are
   always distinct (the without-replacement property).

Run:  pytest tests/test_sbs.py
"""

import math
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import torch

from lm_dpp.sbs_decoding import (
    TableBackend,
    cp_inclusion_reference,
    cp_sample_batch,
    cp_sample_reference,
    swor_beam_generate,
)

SEED = 20260727


def logsumexp(xs):
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


# ---------------------------------------------------------------------------
# Exact CP(w, K) set distribution by enumeration
# ---------------------------------------------------------------------------

def cp_exact_set_probs(logw, K):
    logw = np.asarray(logw, dtype=np.float64)
    sets = {}
    for comb in combinations(range(len(logw)), K):
        sets[frozenset(comb)] = sum(logw[list(comb)])
    Z = logsumexp(list(sets.values()))
    return {s: math.exp(v - Z) for s, v in sets.items()}


def tv(emp: Counter, exact: dict, n: int) -> float:
    keys = set(emp) | set(exact)
    return 0.5 * sum(abs(emp.get(k, 0) / n - exact.get(k, 0.0)) for k in keys)


def test_cp_sample_batch_exact():
    rng = np.random.default_rng(SEED)
    logw = rng.normal(0.0, 2.0, size=7)
    K = 3
    exact = cp_exact_set_probs(logw, K)

    n = 200_000
    gen = torch.Generator().manual_seed(SEED)
    lw = torch.tensor(logw).expand(n, -1).contiguous()
    idx = cp_sample_batch(lw, K, gen)
    emp = Counter(frozenset(row) for row in idx.tolist())
    d = tv(emp, exact, n)
    print(f"  cp_sample_batch TV = {d:.4f} (n={n})")
    assert d < 0.01, d


def test_cp_sample_reference_exact():
    rng = np.random.default_rng(SEED + 1)
    logw = rng.normal(0.0, 2.0, size=7)
    K = 3
    exact = cp_exact_set_probs(logw, K)

    n = 50_000
    emp = Counter(frozenset(cp_sample_reference(logw, K, rng)) for _ in range(n))
    d = tv(emp, exact, n)
    print(f"  cp_sample_reference TV = {d:.4f} (n={n})")
    assert d < 0.02, d


def test_cp_inclusion_reference():
    rng = np.random.default_rng(SEED + 2)
    logw = rng.normal(0.0, 1.5, size=9)
    K = 4
    exact = cp_exact_set_probs(logw, K)
    inc = np.zeros(9)
    for s, p in exact.items():
        for i in s:
            inc[i] += p
    got = cp_inclusion_reference(logw, K)
    err = np.abs(got - inc).max()
    print(f"  cp_inclusion_reference max err = {err:.2e}")
    assert err < 1e-10, err


def test_cp_head_truncation():
    """Truncated sampling must match the full CP distribution when the tail
    bound holds (geometrically decaying weights, head << N)."""
    N, K = 400, 4
    logw = -0.25 * np.arange(N)  # w_i = exp(-i/4): heavy head, long thin tail
    inc_exact = cp_inclusion_reference(logw, K)

    n = 100_000
    gen = torch.Generator().manual_seed(SEED + 3)
    lw = torch.tensor(logw).expand(n, -1).contiguous()
    idx = cp_sample_batch(lw, K, gen, head_size=64)
    inc_emp = np.bincount(idx.flatten().numpy(), minlength=N) / n
    err = np.abs(inc_emp - inc_exact).max()
    print(f"  truncated-head inclusion max err = {err:.4f} (binom noise ~ {3 * math.sqrt(0.5 / n):.4f})")
    assert err < 0.01, err


# ---------------------------------------------------------------------------
# Tiny enumerable model
# ---------------------------------------------------------------------------

class TinyModel:
    """Fixed random locally normalized model over V tokens; token `eos` ends a
    sequence; sequences are truncated at max_len. Histories after EOS get
    uniform logits (never used by a correct decoder)."""

    def __init__(self, V, seed, spread=1.5):
        self.V = V
        self.seed = seed
        self.spread = spread
        self._cache = {}

    def logits(self, hist):
        hist = tuple(hist)
        if hist not in self._cache:
            h = abs(hash((self.seed, hist))) % (2**32)
            rng = np.random.default_rng(h)
            self._cache[hist] = rng.normal(0.0, self.spread, size=self.V)
        return self._cache[hist]

    def logp_local(self, hist, tau):
        z = self.logits(hist) / tau
        return z - logsumexp(list(z))

    def leaves(self, max_len, eos, tau):
        """All leaves (finished by eos, or truncated at max_len) with their
        log-probability under the annealed model (truncated leaf = prefix mass)."""
        out = {}

        def rec(hist, lp):
            if hist and hist[-1] == eos:
                out[hist] = lp
                return
            if len(hist) == max_len:
                out[hist] = lp
                return
            lps = self.logp_local(hist, tau)
            for v in range(self.V):
                rec(hist + (v,), lp + lps[v])

        rec((), 0.0)
        return out


def swor_set_probs(leaf_lp: dict, K: int):
    """Exact sampling-without-replacement (Gumbel top-K / Plackett-Luce) set
    probabilities: sum over orderings of prod p_i / (1 - sum_prev)."""
    from itertools import permutations
    items = list(leaf_lp)
    p = {i: math.exp(lp) for i, lp in leaf_lp.items()}
    out = {}
    for comb in combinations(items, K):
        tot = 0.0
        for perm in permutations(comb):
            q, rem = 1.0, 1.0
            for i in perm:
                q *= p[i] / rem
                rem -= p[i]
            tot += q
        out[frozenset(comb)] = tot
    return out


def run_decoder_many(model, *, method, K, max_len, tau, eos, n, batch=2000, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    counts = Counter()
    done = 0
    while done < n:
        G = min(batch, n - done)
        backend = TableBackend(model.logits, model.V)
        groups = swor_beam_generate(backend, method=method, K=K, G=G,
                                    max_new_tokens=max_len, temperature=tau,
                                    eos_ids={eos}, generator=gen)
        for g in groups:
            counts[frozenset(tuple(s.tokens) for s in g)] += 1
        done += G
    return counts


def test_sbs_swor_distribution():
    for tau in (1.0, 0.6):
        model = TinyModel(V=3, seed=11)
        eos, max_len, K = 2, 2, 2
        leaf_lp = model.leaves(max_len, eos, tau)
        assert abs(logsumexp(list(leaf_lp.values()))) < 1e-9
        exact = swor_set_probs(leaf_lp, K)

        n = 120_000
        emp = run_decoder_many(model, method="sbs", K=K, max_len=max_len, tau=tau,
                               eos=eos, n=n)
        d = tv(emp, exact, n)
        print(f"  SBS set-distribution TV (tau={tau}) = {d:.4f} over {len(exact)} sets")
        assert d < 0.012, (tau, d)


def test_sbs_inclusion_bigger_tree():
    """Larger tree: check per-leaf inclusion frequencies instead of full sets."""
    tau = 1.0
    model = TinyModel(V=4, seed=13)
    eos, max_len, K = 3, 3, 3
    leaf_lp = model.leaves(max_len, eos, tau)
    exact_sets = swor_set_probs(leaf_lp, K)
    inc = defaultdict(float)
    for s, p in exact_sets.items():
        for leaf in s:
            inc[leaf] += p

    n = 60_000
    emp = run_decoder_many(model, method="sbs", K=K, max_len=max_len, tau=tau,
                           eos=eos, n=n, seed=SEED + 7)
    inc_emp = defaultdict(float)
    for s, c in emp.items():
        for leaf in s:
            inc_emp[leaf] += c / n
    err = max(abs(inc_emp[leaf] - inc[leaf]) for leaf in inc)
    print(f"  SBS inclusion max err = {err:.4f} over {len(inc)} leaves")
    assert err < 0.012, err


# ---------------------------------------------------------------------------
# CPSBS brute force
# ---------------------------------------------------------------------------

def cpsbs_bruteforce(model: TinyModel, K, max_len, tau, eos):
    """Exact CPSBS final-beam distribution by enumerating all beam-set chains.
    Hypotheses are (tokens, finished); weights are Hajek odds of the annealed
    cumulative prefix log-probability, matching the decoder."""

    def cum_tmp(toks):
        lp = 0.0
        for i, v in enumerate(toks):
            lp += model.logp_local(tuple(toks[:i]), tau)[v]
        return lp

    def odds(lp):
        lp = min(lp, -1e-12)
        return lp - math.log(-math.expm1(lp))

    def cp_sets(cands):
        """cands: list of (hyp, logw) -> {frozenset(hyps): prob}"""
        raw = {}
        for comb in combinations(range(len(cands)), K):
            raw[comb] = sum(cands[i][1] for i in comb)
        Z = logsumexp(list(raw.values()))
        return {frozenset(cands[i][0] for i in comb): math.exp(v - Z)
                for comb, v in raw.items()}

    out = defaultdict(float)

    def rec(beam, prob, t):
        if prob < 1e-14:
            return
        if all(fin for _, fin in beam) or t == max_len:
            out[frozenset(toks for toks, _ in beam)] += prob
            return
        cands = []
        for toks, fin in beam:
            if fin:
                cands.append(((toks, True), odds(cum_tmp(toks))))
            else:
                lps = model.logp_local(toks, tau)
                base = cum_tmp(toks)
                for v in range(model.V):
                    nt = toks + (v,)
                    cands.append(((nt, v == eos), odds(base + lps[v])))
        for sel, p in cp_sets(cands).items():
            rec(sel, prob * p, t + 1)

    # step 0: CP over the V first tokens
    root = [(((v,), v == eos), odds(model.logp_local((), tau)[v])) for v in range(model.V)]
    for sel, p in cp_sets(root).items():
        rec(sel, p, 1)
    return dict(out)


def test_cpsbs_matches_bruteforce():
    for tau in (1.0, 0.6):
        model = TinyModel(V=3, seed=17)
        eos, max_len, K = 2, 3, 2
        exact = cpsbs_bruteforce(model, K, max_len, tau, eos)
        assert abs(sum(exact.values()) - 1.0) < 1e-9

        n = 120_000
        emp = run_decoder_many(model, method="cps", K=K, max_len=max_len, tau=tau,
                               eos=eos, n=n, seed=SEED + 11)
        d = tv(emp, exact, n)
        print(f"  CPSBS set-distribution TV (tau={tau}) = {d:.4f} over {len(exact)} sets")
        assert d < 0.012, (tau, d)


# ---------------------------------------------------------------------------
# Real HF model: bookkeeping + SWOR property
# ---------------------------------------------------------------------------

def _tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(SEED)
    cfg = GPT2Config(vocab_size=97, n_positions=64, n_embd=64, n_layer=2, n_head=2,
                     bos_token_id=0, eos_token_id=5)
    return GPT2LMHeadModel(cfg).eval()


def test_hf_backend_bookkeeping_and_swor():
    from lm_dpp.sbs_decoding import HFBackend

    model = _tiny_gpt2()
    prompt = torch.tensor([3, 11, 42, 7])
    tau = 0.8
    K, G, max_new = 6, 3, 12
    eos = {5}

    for method in ("sbs", "cps"):
        gen = torch.Generator().manual_seed(SEED + 100)
        backend = HFBackend(model, prompt)
        groups = swor_beam_generate(backend, method=method, K=K, G=G,
                                    max_new_tokens=max_new, temperature=tau,
                                    eos_ids=eos, generator=gen)
        assert len(groups) == G
        for g in groups:
            assert len(g) == K
            seqs = [tuple(s.tokens) for s in g]
            assert len(set(seqs)) == K, f"duplicate sequences under {method} (SWOR violated)"
            for s in g:
                assert len(s.tokens) <= max_new
                if s.finished:
                    assert s.tokens[-1] == 5
                # rescore from scratch
                full = torch.cat([prompt, torch.tensor(s.tokens, dtype=torch.long)])
                with torch.no_grad():
                    logits = model(full.unsqueeze(0)).logits[0].double()
                lp_raw = torch.log_softmax(logits, -1)
                lp_tmp = torch.log_softmax(logits / tau, -1)
                P = len(prompt)
                tgt = full[P:]
                pos = torch.arange(P - 1, P - 1 + len(tgt))
                want_raw = lp_raw[pos, tgt].sum().item()
                want_tmp = lp_tmp[pos, tgt].sum().item()
                assert abs(want_raw - s.logp_t1) < 1e-3, (method, want_raw, s.logp_t1)
                assert abs(want_tmp - s.logp) < 1e-3, (method, want_tmp, s.logp)
    print("  HF backend bookkeeping + SWOR property OK (sbs, cps)")


def test_hf_backend_no_eos_truncation():
    """All sequences must be exactly max_new_tokens long when EOS never fires."""
    from lm_dpp.sbs_decoding import HFBackend

    model = _tiny_gpt2()
    prompt = torch.tensor([3, 11])
    gen = torch.Generator().manual_seed(SEED + 5)
    backend = HFBackend(model, prompt)
    groups = swor_beam_generate(backend, method="sbs", K=4, G=2, max_new_tokens=5,
                                temperature=1.0, eos_ids=set(), generator=gen)
    for g in groups:
        for s in g:
            assert len(s.tokens) == 5 and not s.finished
    print("  truncation-without-EOS OK")


# ---------------------------------------------------------------------------
# Determinantal Beam Search
# ---------------------------------------------------------------------------

def test_greedy_map_logdet_vs_naive():
    """Chen et al. (2018) incremental greedy MAP must match the naive greedy
    (recompute log det from scratch for every candidate at every step)."""
    from lm_dpp.det_beam import greedy_map_logdet
    rng = np.random.default_rng(SEED + 21)
    for trial in range(20):
        N, k = 30, 8
        B = rng.normal(size=(N, N // 2))
        L = B @ B.T + np.diag(rng.uniform(0.01, 1.0, N))
        got = greedy_map_logdet(L, k)

        sel = []
        for _ in range(k):
            best, best_v = None, -np.inf
            for i in range(N):
                if i in sel:
                    continue
                s, v = np.linalg.slogdet(L[np.ix_(sel + [i], sel + [i])])
                if s > 0 and v > best_v:
                    best, best_v = i, v
            sel.append(best)
        assert got == sel, (trial, got, sel)
    print("  greedy MAP (incremental Cholesky) == naive greedy on 20 random PSD matrices")


def _beam_search_reference(model, K, max_len, tau, eos):
    """Standard beam search with the same candidate restriction as DetBS
    (top-K children per active hypothesis, finished carried as-is)."""
    beams = [((), 0.0, False)]
    lps0 = model.logp_local((), tau)
    order = np.argsort(-lps0)[:K]
    beams = [((int(v),), lps0[int(v)], int(v) == eos) for v in order]
    for _ in range(1, max_len):
        if all(f for _, _, f in beams):
            break
        cands = []
        for toks, lp, fin in beams:
            if fin:
                cands.append((toks, lp, True))
            else:
                lps = model.logp_local(toks, tau)
                for v in np.argsort(-lps)[:K]:
                    cands.append((toks + (int(v),), lp + lps[int(v)], int(v) == eos))
        cands.sort(key=lambda c: -c[1])
        beams = cands[:K]
    return frozenset(c[0] for c in beams)


def test_detbs_w0_equals_beam_search():
    from lm_dpp.det_beam import detbs_generate

    for seed, tau in ((23, 1.0), (29, 0.7), (31, 1.3)):
        model = TinyModel(V=5, seed=seed)
        eos, max_len, K = 4, 6, 3
        want = _beam_search_reference(model, K, max_len, tau, eos)
        backend = TableBackend(model.logits, model.V)
        groups = detbs_generate(backend, K=K, weights=[0.0], max_new_tokens=max_len,
                                temperature=tau, eos_ids={eos})
        got = frozenset(tuple(s.tokens) for s in groups[0])
        assert got == want, (seed, tau, sorted(got), sorted(want))
    print("  DetBS w=0 == beam search on 3 tiny models")


def test_detbs_bookkeeping_and_determinism():
    from lm_dpp.det_beam import detbs_generate
    from lm_dpp.sbs_decoding import HFBackend

    model = _tiny_gpt2()
    prompt = torch.tensor([3, 11, 42, 7])
    tau, K, max_new = 1.0, 5, 10
    weights = [0.0, 0.05, 5.0]

    runs = []
    for _ in range(2):
        backend = HFBackend(model, prompt)
        runs.append(detbs_generate(backend, K=K, weights=weights,
                                   max_new_tokens=max_new, temperature=tau, eos_ids={5}))
    a, b = runs
    assert [[s.tokens for s in g] for g in a] == [[s.tokens for s in g] for g in b], \
        "DetBS must be deterministic"

    for wi, g in enumerate(a):
        seqs = [tuple(s.tokens) for s in g]
        assert len(set(seqs)) == K, f"duplicate sequences at w={weights[wi]}"
        for s in g:
            full = torch.cat([prompt, torch.tensor(s.tokens, dtype=torch.long)])
            with torch.no_grad():
                logits = model(full.unsqueeze(0)).logits[0].double()
            lp_raw = torch.log_softmax(logits, -1)
            P = len(prompt)
            tgt = full[P:]
            pos = torch.arange(P - 1, P - 1 + len(tgt))
            want_raw = lp_raw[pos, tgt].sum().item()
            assert abs(want_raw - s.logp_t1) < 1e-3, (weights[wi], want_raw, s.logp_t1)
    assert ([tuple(s.tokens) for s in a[0]] != [tuple(s.tokens) for s in a[2]]), \
        "w=5 should change the selected set vs w=0"
    print("  DetBS determinism + bookkeeping + SWOR-style distinctness OK")


ALL = [
    test_cp_sample_batch_exact,
    test_cp_sample_reference_exact,
    test_cp_inclusion_reference,
    test_cp_head_truncation,
    test_sbs_swor_distribution,
    test_sbs_inclusion_bigger_tree,
    test_cpsbs_matches_bruteforce,
    test_hf_backend_bookkeeping_and_swor,
    test_hf_backend_no_eos_truncation,
    test_greedy_map_logdet_vs_naive,
    test_detbs_w0_equals_beam_search,
    test_detbs_bookkeeping_and_determinism,
]

if __name__ == "__main__":
    for fn in ALL:
        print(f"[{fn.__name__}]")
        fn()
    print("\nAll SBS/CPSBS validation tests passed.")
