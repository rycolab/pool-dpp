"""
Stochastic Beam Search (SBS; Kool et al., 2019) and Conditional Poisson
Stochastic Beam Search (CPSBS; Meister et al., 2021) for Hugging Face causal
LMs.

Both draw a set of K sequences without replacement from a sequence model.
Notation below: V = vocab size, K = beam / sample-set size, G = number of
independent sample sets decoded in one batch (all for the same prompt and
temperature).

- SBS perturbs the annealed cumulative prefix log-probability phi of every
  candidate extension with Gumbel noise conditioned so that the maximum
  perturbed score among a node's children equals the node's own score
  (Kool et al., 2019), then keeps the top-K candidates per step. The K leaves
  held at termination are an exact sample without replacement from the
  annealed sequence distribution. Pruning finished leaves that fall out of
  the top-K is safe: by the max-conditioning, the best leaf under any
  surviving candidate equals that candidate's score.

- CPSBS replaces the top-K with a draw from the conditional Poisson (CP)
  sampling design over the candidates, with Hajek weights
  w_n = p(y_<=t) / (1 - p(y_<=t)) on the annealed prefix probability
  (Meister et al., 2021, Sec. 3 "Setting w_n" and Sec. 5.2).

Finished hypotheses (EOS produced) are carried "as is" as candidates in later
rounds with frozen score/weight, competing with the expansions of the active
beams (Meister et al., 2021, footnote 2). A finished hypothesis not selected
at some round is dropped for good.

Temperature enters exactly as in the reference fairseq implementations
(wouterkool/stochastic-beam-search, rycolab/cpsbs): local conditionals are
annealed and renormalized per step, log p_tau(. | y_<t) = log_softmax(z / tau),
and the *cumulative* annealed log-probability is what gets perturbed (SBS) or
Hajek-transformed (CPSBS). The raw (T=1) cumulative log-probability is tracked
alongside for reporting, matching the vLLM sampler's dual-logprob bookkeeping.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Gumbel machinery (Kool et al., 2019, Appendix B)
# ---------------------------------------------------------------------------

def gumbel_like(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    u = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    tiny = torch.finfo(x.dtype).tiny
    return -torch.log(-torch.log(u.clamp(min=tiny, max=1.0 - 1e-16)))


def _shift_gumbel_maximum(g_phi: torch.Tensor, T: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Shift gumbels g_phi (..., N) with observed max Z (...,) to have max T (...,)."""
    u = T.unsqueeze(-1) - g_phi + torch.log1p(-torch.exp((g_phi - Z.unsqueeze(-1)).clamp(max=0.0)))
    return T.unsqueeze(-1) - F.relu(u) - torch.log1p(torch.exp(-u.abs()))


def gumbel_with_maximum(phi: torch.Tensor, T: torch.Tensor,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Sample gumbels with location phi (..., N) conditioned on their maximum
    along the last dim being exactly T (...,). Rows of phi that are all -inf
    yield all -inf (they represent no candidates and must be masked by the
    caller anyway).
    """
    g_phi = phi + gumbel_like(phi, generator)
    Z = g_phi.max(dim=-1).values
    g = _shift_gumbel_maximum(g_phi, T, Z)
    return torch.where(torch.isfinite(Z).unsqueeze(-1), g, NEG_INF)


# ---------------------------------------------------------------------------
# Conditional Poisson sampling
# ---------------------------------------------------------------------------

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x < 0, numerically stable."""
    x = x.clamp(max=-1e-12)
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def hajek_log_odds(logp: torch.Tensor) -> torch.Tensor:
    """log(p / (1 - p)) from log p; -inf maps to -inf."""
    return logp - log1mexp(logp)


def cp_sample_reference(logw: np.ndarray, K: int, rng: np.random.Generator) -> list[int]:
    """
    Exact CP(w, K) sampler via the O(NK) dynamic program in log space
    (Kulesza & Taskar 2012, Alg. 7; same DP as rycolab/cpsbs cps_dp.pyx).
    Used for validation and as a fallback; slow but exact.
    """
    logw = np.asarray(logw, dtype=np.float64)
    N = len(logw)
    assert np.isfinite(logw).sum() >= K, "need at least K items with positive weight"
    # W[r, i] = log e_r(w_1..w_i), elementary symmetric polynomial
    W = np.full((K + 1, N + 1), -np.inf)
    W[0, :] = 0.0
    for r in range(1, K + 1):
        with np.errstate(invalid="ignore"):
            W[r, 1:] = np.logaddexp.accumulate(W[r - 1, :-1] + logw)
    Y = []
    k = K
    for i in range(N, 0, -1):
        if k == 0:
            break
        t = logw[i - 1] + W[k - 1, i - 1] - W[k, i]
        if np.log(rng.uniform()) < t:
            Y.append(i - 1)
            k -= 1
    assert k == 0
    return Y


def cp_inclusion_reference(logw: np.ndarray, K: int) -> np.ndarray:
    """Exact CP(w, K) inclusion probabilities via the forward/backward DP."""
    logw = np.asarray(logw, dtype=np.float64)
    N = len(logw)
    W = np.full((K + 1, N + 1), -np.inf)
    W[0, :] = 0.0
    for r in range(1, K + 1):
        with np.errstate(invalid="ignore"):
            W[r, 1:] = np.logaddexp.accumulate(W[r - 1, :-1] + logw)
    # B[r, i] = log e_r(w_{i+1}..w_N) (suffix polynomials)
    B = np.full((K + 1, N + 1), -np.inf)
    B[0, :] = 0.0
    for r in range(1, K + 1):
        for i in range(N - 1, -1, -1):
            B[r, i] = np.logaddexp(B[r, i + 1], logw[i] + B[r - 1, i + 1])
    Z = W[K, N]
    inc = np.full(N, -np.inf)
    for i in range(N):
        terms = [W[r, i] + B[K - 1 - r, i + 1] for r in range(K)]
        inc[i] = logw[i] + np.logaddexp.reduce(terms) - Z
    return np.exp(inc)


def cp_sample_batch(logw: torch.Tensor, K: int, generator: torch.Generator | None = None,
                    head_size: int = 8192, max_rounds: int = 400, replicates: int = 16,
                    tail_tol: float = 1e-7) -> torch.Tensor:
    """
    Batched exact sampler for the conditional Poisson design CP(w, K).

    logw: (G, N) float64 log-weights, -inf allowed. Returns (G, K) int64
    indices (unordered sets).

    Method: independent Bernoulli draws with pi_i = sigmoid(logw_i + c),
    conditioned on the draw size being exactly K via rejection (the
    conditional law is CP(w, K) for any c; c is chosen by bisection so that
    E[size] = K, which maximizes the acceptance rate, >= ~1/sqrt(2 pi K)).

    For efficiency the problem is restricted to the top-`head_size` items by
    weight: this equals CP(w_full, K) conditioned on no discarded item being
    selected, so the total variation error is at most P(some discarded item
    selected) <= sum_tail sigmoid(logw + c) <= exp(c) * sum_tail w. That bound
    is checked against tail_tol and the head is grown (up to the full set) if
    it is exceeded. Groups that do not accept within max_rounds x replicates
    draws fall back to the exact sequential DP on CPU.
    """
    G, N = logw.shape
    device = logw.device
    M = min(N, head_size)
    vals, head_idx = torch.topk(logw, M, dim=-1)

    n_finite = torch.isfinite(vals).sum(-1)
    if (n_finite < K).any():
        raise ValueError("fewer than K candidates with positive weight")

    out = torch.full((G, K), -1, dtype=torch.long, device=device)

    # Degenerate groups: exactly K finite weights -> the set is deterministic.
    exact_mask = n_finite == K
    if exact_mask.any():
        for g in torch.nonzero(exact_mask).flatten().tolist():
            out[g] = head_idx[g, : K]

    todo = torch.nonzero(~exact_mask).flatten()
    if todo.numel() == 0:
        return out

    v = vals[todo]  # (T, M)
    T_ = v.shape[0]

    # Bisection for c: sum_i sigmoid(v_i + c) == K.
    S = torch.logsumexp(v, dim=-1)
    lo = float(np.log(K)) - S - 30.0
    hi = -v[:, K] + 40.0  # top-(K+1) items alone then exceed K
    for _ in range(90):
        mid = (lo + hi) / 2
        s = torch.sigmoid(v + mid.unsqueeze(-1)).sum(-1)
        too_small = s < K
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)
    c = (lo + hi) / 2
    pi = torch.sigmoid(v + c.unsqueeze(-1))  # (T, M)

    # Tail bound: P(any discarded item would be selected) <= exp(c) * tail_mass.
    if M < N:
        full_mass = torch.logsumexp(logw[todo], dim=-1)
        head_mass = torch.logsumexp(v, dim=-1)
        diff = (head_mass - full_mass).clamp(max=0.0)
        log_tail = torch.where(diff < -1e-15, full_mass + log1mexp(diff), NEG_INF)
        bound = torch.exp(c + log_tail)
        if (bound > tail_tol).any():
            if M < N:
                return cp_sample_batch(logw, K, generator, head_size=min(N, head_size * 8),
                                       max_rounds=max_rounds, replicates=replicates,
                                       tail_tol=tail_tol)

    pending = torch.ones(T_, dtype=torch.bool, device=device)
    sel = torch.zeros(T_, M, dtype=torch.bool, device=device)
    for _ in range(max_rounds):
        idx_p = torch.nonzero(pending).flatten()
        if idx_p.numel() == 0:
            break
        pi_p = pi[idx_p].unsqueeze(1)  # (P, 1, M)
        u = torch.rand((idx_p.numel(), replicates, M), device=device, dtype=pi.dtype,
                       generator=generator)
        X = u < pi_p  # (P, R, M)
        cnt = X.sum(-1)  # (P, R)
        hit = cnt == K
        has = hit.any(-1)
        if has.any():
            first = torch.argmax(hit.long(), dim=-1)  # first accepting replicate
            rows = torch.nonzero(has).flatten()
            sel[idx_p[rows]] = X[rows, first[rows]]
            pending[idx_p[rows]] = False

    # Rare stragglers: exact sequential DP on CPU.
    for t in torch.nonzero(pending).flatten().tolist():
        rng = np.random.default_rng(
            int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=device).item()))
        picked = cp_sample_reference(v[t].cpu().numpy(), K, rng)
        sel[t, picked] = True

    sel_idx = torch.nonzero(sel)  # (T*K, 2) ordered by group
    assert sel_idx.shape[0] == T_ * K
    picked_cols = sel_idx[:, 1].view(T_, K)
    out[todo] = torch.gather(head_idx[todo], 1, picked_cols)
    return out


# ---------------------------------------------------------------------------
# Decoding backends
# ---------------------------------------------------------------------------

class HFBackend:
    """
    Incremental decoding backend for a Hugging Face causal LM.

    The prompt is prefTled once with batch 1; the KV cache is then expanded to
    `rows` identical rows. Each step() reorders the cache rows by `parents`
    (beam reordering / dropping completed groups) and appends one token per
    row.
    """

    def __init__(self, model, prompt_ids: torch.Tensor):
        self.model = model
        self.device = model.device
        self.prompt_ids = prompt_ids.to(self.device).view(1, -1)
        self.cache = None
        self.pos = None

    @torch.inference_mode()
    def prefill(self) -> torch.Tensor:
        from transformers import DynamicCache
        self.cache = DynamicCache()
        try:
            out = self.model(input_ids=self.prompt_ids, past_key_values=self.cache,
                             use_cache=True, logits_to_keep=1)
        except TypeError:
            out = self.model(input_ids=self.prompt_ids, past_key_values=self.cache, use_cache=True)
        self.pos = self.prompt_ids.shape[1]
        return out.logits[:, -1, :].float()

    def expand(self, rows: int) -> None:
        self.cache.batch_repeat_interleave(rows)

    @torch.inference_mode()
    def step(self, tokens: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
        rows = tokens.shape[0]
        if parents.shape[0] != self._cache_rows() or not torch.equal(
                parents, torch.arange(parents.shape[0], device=parents.device)):
            self.cache.batch_select_indices(parents)
        cache_position = torch.arange(self.pos, self.pos + 1, device=self.device)
        out = self.model(input_ids=tokens.view(rows, 1), past_key_values=self.cache,
                         use_cache=True, cache_position=cache_position)
        self.pos += 1
        return out.logits[:, -1, :].float()

    def _cache_rows(self) -> int:
        layer = self.cache.layers[0] if hasattr(self.cache, "layers") else None
        if layer is not None and getattr(layer, "keys", None) is not None:
            return layer.keys.shape[0]
        return self.cache.key_cache[0].shape[0]


class TableBackend:
    """
    Test backend: an explicit locally normalized model given by a callable
    logits_fn(tuple_of_generated_tokens) -> np.ndarray (V,) of logits.
    No prompt, no cache; histories are tracked exactly.
    """

    def __init__(self, logits_fn, V: int, device: str = "cpu"):
        self.logits_fn = logits_fn
        self.V = V
        self.device = device
        self.histories: list[tuple[int, ...]] = []

    def prefill(self) -> torch.Tensor:
        self.histories = [()]
        return torch.tensor(np.asarray(self.logits_fn(()), dtype=np.float32),
                            device=self.device).view(1, -1)

    def expand(self, rows: int) -> None:
        self.histories = [self.histories[0]] * rows

    def step(self, tokens: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
        toks = tokens.tolist()
        pars = parents.tolist()
        self.histories = [self.histories[p] + (t,) for p, t in zip(pars, toks)]
        logits = np.stack([np.asarray(self.logits_fn(h), dtype=np.float32) for h in self.histories])
        return torch.tensor(logits, device=self.device)


# ---------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------

@dataclass
class SworSequence:
    tokens: list[int]
    logp: float          # cumulative log-prob at the sampling temperature
    logp_t1: float       # cumulative log-prob at T=1
    finished: bool       # ended with EOS (False = truncated at max_new_tokens)
    score: float = 0.0   # SBS: final Gumbel score; CPSBS: annealed cumulative log-prob


@torch.inference_mode()
def swor_beam_generate(backend, *, method: str, K: int, G: int, max_new_tokens: int,
                       temperature: float = 1.0, eos_ids: set[int] | None = None,
                       generator: torch.Generator | None = None,
                       cp_head_size: int = 8192) -> list[list[SworSequence]]:
    """
    Decode G independent SWOR sample sets of K sequences each.

    method: "sbs" (Gumbel top-K) or "cps" (conditional Poisson).
    Returns a list of G lists of K SworSequence.
    """
    assert method in ("sbs", "cps")
    eos_ids = eos_ids or set()
    tau = float(temperature)

    logits0 = backend.prefill()  # (1, V)
    V = logits0.shape[-1]
    assert K <= V
    device = logits0.device
    eos_list = sorted(eos_ids)
    eos_t = torch.tensor(eos_list, dtype=torch.long, device=device) if eos_list else None
    junk_token = eos_list[0] if eos_list else 0

    lp_raw0 = F.log_softmax(logits0.double(), dim=-1)          # (1, V)
    lp_tmp0 = F.log_softmax(logits0.double() / tau, dim=-1)

    # ---- step 0: pick K first tokens per group from the root ----
    if method == "sbs":
        pert = lp_tmp0.expand(G, V) + gumbel_like(lp_tmp0.expand(G, V).contiguous(), generator)
        score, tok0 = torch.topk(pert, K, dim=-1)              # (G, K)
    else:
        logw = hajek_log_odds(lp_tmp0).expand(G, V).contiguous()
        tok0 = cp_sample_batch(logw, K, generator, head_size=cp_head_size)
        score = torch.zeros(G, K, dtype=torch.float64, device=device)

    cum_raw = lp_raw0.expand(G, V).gather(1, tok0).clone()     # (G, K) float64
    cum_tmp = lp_tmp0.expand(G, V).gather(1, tok0).clone()
    finished = torch.zeros(G, K, dtype=torch.bool, device=device)
    if eos_t is not None:
        finished = (tok0.unsqueeze(-1) == eos_t).any(-1)
    if method == "cps":
        score = cum_tmp.clone()

    toks: list[list[list[int]]] = [[[int(tok0[g, k])] for k in range(K)] for g in range(G)]

    backend.expand(G * K)
    pending_tokens = tok0.reshape(-1)
    parents = torch.arange(G * K, device=device)

    # groups still being decoded, in current batch order
    live = list(range(G))
    results: dict[int, list[SworSequence]] = {}

    def finalize(gi: int, b: int, cr, ct, fin, sc):
        results[gi] = [
            SworSequence(tokens=toks[gi][k], logp=float(ct[b, k]), logp_t1=float(cr[b, k]),
                         finished=bool(fin[b, k]), score=float(sc[b, k]))
            for k in range(K)
        ]

    # step 0 above already generated one token per slot
    for step in range(1, max_new_tokens):
        # retire completed groups (all K slots finished)
        done_rows = finished.all(-1)
        if done_rows.any():
            keep = torch.nonzero(~done_rows).flatten()
            for b in torch.nonzero(done_rows).flatten().tolist():
                finalize(live[b], b, cum_raw, cum_tmp, finished, score)
            live = [live[b] for b in keep.tolist()]
            if not live:
                return [results[g] for g in range(G)]
            row_keep = (keep.unsqueeze(-1) * K + torch.arange(K, device=device)).reshape(-1)
            parents = parents[row_keep]
            pending_tokens = pending_tokens[row_keep]
            cum_raw, cum_tmp = cum_raw[keep], cum_tmp[keep]
            finished, score = finished[keep], score[keep]

        Gc = len(live)
        logits = backend.step(pending_tokens, parents)          # (Gc*K, V)
        lp_raw = F.log_softmax(logits.double(), dim=-1).view(Gc, K, V)
        lp_tmp = F.log_softmax(logits.double() / tau, dim=-1).view(Gc, K, V)

        active = ~finished                                      # (Gc, K)
        phi = cum_tmp.unsqueeze(-1) + lp_tmp                    # (Gc, K, V)
        phi = torch.where(active.unsqueeze(-1), phi, NEG_INF)

        if method == "sbs":
            pert = gumbel_with_maximum(phi.view(Gc * K, V), score.view(Gc * K), generator)
            pert = pert.view(Gc, K, V)
            carry = torch.where(finished, score, NEG_INF).unsqueeze(-1)   # (Gc, K, 1)
            cand = torch.cat([pert, carry], dim=-1).view(Gc, K * (V + 1))
            new_score, flat = torch.topk(cand, K, dim=-1)                  # (Gc, K)
        else:
            logw_exp = hajek_log_odds(phi.clamp(max=-1e-9))
            carry_w = torch.where(finished, hajek_log_odds(cum_tmp.clamp(max=-1e-9)), NEG_INF)
            cand_w = torch.cat([logw_exp, carry_w.unsqueeze(-1)], dim=-1).view(Gc, K * (V + 1))
            flat = cp_sample_batch(cand_w, K, generator, head_size=cp_head_size)
            new_score = None

        par = flat // (V + 1)                                    # (Gc, K) parent slot
        col = flat % (V + 1)
        is_carry = col == V

        tok_new = torch.where(is_carry, junk_token, col)
        g_lpr = lp_raw.gather(1, par.unsqueeze(-1).expand(-1, -1, V)).gather(2, tok_new.unsqueeze(-1)).squeeze(-1)
        g_lpt = lp_tmp.gather(1, par.unsqueeze(-1).expand(-1, -1, V)).gather(2, tok_new.unsqueeze(-1)).squeeze(-1)

        pcum_raw = cum_raw.gather(1, par)
        pcum_tmp = cum_tmp.gather(1, par)
        cum_raw = torch.where(is_carry, pcum_raw, pcum_raw + g_lpr)
        cum_tmp = torch.where(is_carry, pcum_tmp, pcum_tmp + g_lpt)

        new_finished = is_carry.clone()
        if eos_t is not None:
            new_finished |= (col.unsqueeze(-1) == eos_t).any(-1)
        finished = new_finished

        if method == "sbs":
            score = new_score
        else:
            score = cum_tmp.clone()

        # token histories (small python loop; K*G appends per step)
        par_l, col_l, carry_l = par.tolist(), col.tolist(), is_carry.tolist()
        for b in range(Gc):
            gi = live[b]
            toks[gi] = [
                toks[gi][par_l[b][k]] + ([] if carry_l[b][k] else [col_l[b][k]])
                for k in range(K)
            ]

        # rows for the next forward: global row index of each new slot's parent
        parents = (torch.arange(Gc, device=device).unsqueeze(-1) * K + par).reshape(-1)
        pending_tokens = tok_new.reshape(-1)

    # max_new_tokens exhausted: truncate whatever is still active
    for b in range(len(live)):
        finalize(live[b], b, cum_raw, cum_tmp, finished, score)
    return [results[g] for g in range(G)]
