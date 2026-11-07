"""
Determinantal Beam Search (DetBS; Meister, Forster, Cotterell, 2021).

Beam search reformulated as iterative subdeterminant maximization: at each
step, the beam Y_t is the (approximate) argmax over size-K subsets of the
candidate set B_t of

    log det( D_Y + w * K_Y )

where D is diagonal with D_ii = the candidate's probability renormalized over
B_t (as in the reference implementation's compute_log_L), K is a unit-diagonal
PSD similarity Gram matrix between candidate strings, and w >= 0 is the
diversity weight. w = 0 recovers standard beam search; the argmax is
approximated with the greedy MAP algorithm of Chen et al. (2018). Candidates
are the top-K extensions of each active hypothesis (the reference's
sub_beam = beam default) plus finished hypotheses carried as-is; decoding is
deterministic given the model.

Similarity kernel: the paper's case study uses the gapped string subsequence
kernel of Lodhi et al. (2002) with subsequence length n = 2 and gap decay
lambda = 0.1, which is O(|s|.|t|) per candidate pair per step: practical for
~30-token MT hypotheses, not for the 10^2-10^3-token chains decoded here. At
lambda = 0.1 gapped matches are suppressed by lambda^gap, so we use the
gapless limit: a cosine Gram over hashed n-gram count features, averaged over
n = 1..NGRAM_N (the reference likewise averages the per-order normalized
kernels). This keeps K unit-diagonal and PSD, is incremental across steps,
and for fixed n the Lodhi per-match weight lambda^{2n} cancels under
normalization.

Multiple diversity weights are decoded in one batch (one "group" per w),
sharing each model forward pass.
"""

import math
import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

NEG_INF = float("-inf")
NGRAM_N = 2
HASH_DIM = 16384


def greedy_map_logdet(L: np.ndarray, k: int) -> list[int]:
    """
    Greedy MAP for log det(L_Y), |Y| = k (Chen et al., 2018, incremental
    Cholesky). L must be PSD. Selection is invariant to positive rescaling of
    L, so callers may rescale for conditioning.
    """
    N = L.shape[0]
    assert k <= N
    d2 = np.array(np.diag(L), dtype=np.float64)
    cis = np.zeros((k, N))
    sel: list[int] = []
    for step in range(k):
        j = int(np.argmax(d2))
        if not d2[j] > 0:
            # was clamped to 1e-300, which kept selecting from a set with zero determinant
            raise RuntimeError(f"greedy MAP: largest residual {d2[j]!r} at step {step} of k={k} over {N} "
                               f"candidates, so L has rank < k on them; DetBS needs k candidates with "
                               f"independent rows (check for weights D that underflow to 0)")
        sel.append(j)
        if step == k - 1:
            break
        dj = math.sqrt(d2[j])
        eis = (L[j] - cis[:step].T @ cis[:step, j]) / dj
        cis[step] = eis
        d2 = d2 - eis ** 2
        d2[sel] = NEG_INF
    return sel


def _ngram_bucket(n: int, toks: tuple[int, ...]) -> int:
    return zlib.crc32(repr((n, toks)).encode()) % HASH_DIM


@dataclass
class DetSequence:
    tokens: list[int]
    logp: float          # cumulative log-prob at the decoding temperature
    logp_t1: float       # cumulative log-prob at T=1
    finished: bool


@torch.inference_mode()
def detbs_generate(backend, *, K: int, weights: list[float], max_new_tokens: int,
                   temperature: float = 1.0, eos_ids: set[int] | None = None,
                   ngram_n: int = NGRAM_N) -> list[list[DetSequence]]:
    """
    Determinantal beam search. Decodes one beam of K sequences per diversity
    weight in `weights` (all in one batch). Returns [per weight][K sequences].
    """
    eos_ids = eos_ids or set()
    tau = float(temperature)
    G = len(weights)

    logits0 = backend.prefill()  # (1, V)
    V = logits0.shape[-1]
    device = logits0.device
    eos_list = sorted(eos_ids)
    eos_t = torch.tensor(eos_list, dtype=torch.long, device=device) if eos_list else None
    junk_token = eos_list[0] if eos_list else 0

    lp_raw0 = F.log_softmax(logits0.double(), dim=-1)[0]
    lp_tmp0 = F.log_softmax(logits0.double() / tau, dim=-1)[0]

    # step 0: identical for every w -- L is diagonal over a single parent's
    # top-K extensions, so the argmax is the top-K tokens.
    top0 = torch.topk(lp_tmp0, K).indices                      # (K,)
    tok0 = top0.unsqueeze(0).expand(G, K).contiguous()

    cum_raw = lp_raw0[top0].unsqueeze(0).expand(G, K).clone()  # (G, K) float64
    cum_tmp = lp_tmp0[top0].unsqueeze(0).expand(G, K).clone()
    finished = torch.zeros(G, K, dtype=torch.bool, device=device)
    if eos_t is not None:
        finished = (tok0.unsqueeze(-1) == eos_t).any(-1)

    toks: list[list[list[int]]] = [[[int(tok0[g, k])] for k in range(K)] for g in range(G)]

    # hashed n-gram count features per slot, one block per order n
    feats = torch.zeros(G, K, ngram_n, HASH_DIM, device=device)
    for g in range(G):
        for k in range(K):
            feats[g, k, 0, _ngram_bucket(1, (int(tok0[g, k]),))] += 1.0

    backend.expand(G * K)
    pending_tokens = tok0.reshape(-1)
    parents = torch.arange(G * K, device=device)

    live = list(range(G))
    results: dict[int, list[DetSequence]] = {}

    def finalize(gi: int, b: int, cr, ct, fin):
        results[gi] = [
            DetSequence(tokens=toks[gi][k], logp=float(ct[b, k]), logp_t1=float(cr[b, k]),
                        finished=bool(fin[b, k]))
            for k in range(K)
        ]

    for step in range(1, max_new_tokens):
        done_rows = finished.all(-1)
        if done_rows.any():
            keep = torch.nonzero(~done_rows).flatten()
            for b in torch.nonzero(done_rows).flatten().tolist():
                finalize(live[b], b, cum_raw, cum_tmp, finished)
            live = [live[b] for b in keep.tolist()]
            if not live:
                return [results[g] for g in range(G)]
            row_keep = (keep.unsqueeze(-1) * K + torch.arange(K, device=device)).reshape(-1)
            parents = parents[row_keep]
            pending_tokens = pending_tokens[row_keep]
            cum_raw, cum_tmp = cum_raw[keep], cum_tmp[keep]
            finished, feats = finished[keep], feats[keep]

        Gc = len(live)
        logits = backend.step(pending_tokens, parents)          # (Gc*K, V)
        lp_raw = F.log_softmax(logits.double(), dim=-1).view(Gc, K, V)
        lp_tmp = F.log_softmax(logits.double() / tau, dim=-1).view(Gc, K, V)

        # top-K extensions per active slot (sub_beam = K)
        ext_lp, ext_tok = torch.topk(lp_tmp, K, dim=-1)         # (Gc, K, K)
        ext_lp_raw = lp_raw.gather(2, ext_tok)

        new_par = torch.zeros(Gc, K, dtype=torch.long, device=device)
        new_tok = torch.full((Gc, K), -1, dtype=torch.long, device=device)

        fin_l = finished.tolist()
        for b in range(Gc):
            gi = live[b]
            w = weights[gi]
            # candidate list: (parent slot, token or None, cum_tmp, cum_raw)
            cands: list[tuple[int, int | None, float, float]] = []
            for k in range(K):
                if fin_l[b][k]:
                    cands.append((k, None, float(cum_tmp[b, k]), float(cum_raw[b, k])))
                else:
                    base_t, base_r = float(cum_tmp[b, k]), float(cum_raw[b, k])
                    for m in range(K):
                        cands.append((k, int(ext_tok[b, k, m]),
                                      base_t + float(ext_lp[b, k, m]),
                                      base_r + float(ext_lp_raw[b, k, m])))

            scores = np.array([c[2] for c in cands])
            m0 = np.max(scores)
            D = np.exp(scores - (m0 + math.log(np.exp(scores - m0).sum())))

            cf = None
            if w > 0.0:
                # candidate features = parent features + delta of the new n-gram(s)
                cf = feats[b, [c[0] for c in cands]].clone()    # (Nc, n, HASH_DIM)
                for ci, (k, t, _, _) in enumerate(cands):
                    if t is None:
                        continue
                    hist = toks[gi][k]
                    for n in range(1, ngram_n + 1):
                        if len(hist) >= n - 1:
                            gram = tuple(hist[len(hist) - (n - 1):] + [t])
                            cf[ci, n - 1, _ngram_bucket(n, gram)] += 1.0
                cfn = cf / cf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                Kmat = torch.einsum("ind,jnd->nij", cfn, cfn).mean(0).double().cpu().numpy()
                L = w * Kmat
                np.fill_diagonal(L, w + D)
            else:
                L = np.diag(D)
            sel = greedy_map_logdet(L / max(D.max(), 1e-300), K)

            # gather the new slot states before mutating anything
            sel_cands = [cands[ci] for ci in sel]
            new_toks_b = [toks[gi][k] if t is None else toks[gi][k] + [t]
                          for (k, t, _, _) in sel_cands]
            if cf is not None:
                new_feats = cf[sel].clone()
            else:
                new_feats = feats[b, [k for (k, _, _, _) in sel_cands]].clone()
                for slot, (k, t, _, _) in enumerate(sel_cands):
                    if t is None:
                        continue
                    hist = toks[gi][k]
                    for n in range(1, ngram_n + 1):
                        if len(hist) >= n - 1:
                            gram = tuple(hist[len(hist) - (n - 1):] + [t])
                            new_feats[slot, n - 1, _ngram_bucket(n, gram)] += 1.0

            for slot, (k, t, ct, cr) in enumerate(sel_cands):
                new_par[b, slot] = k
                new_tok[b, slot] = -1 if t is None else t
                cum_tmp[b, slot], cum_raw[b, slot] = ct, cr
            feats[b] = new_feats
            toks[gi] = new_toks_b

        is_carry = new_tok < 0
        tok_step = torch.where(is_carry, junk_token, new_tok)
        new_finished = is_carry.clone()
        if eos_t is not None:
            new_finished |= (new_tok.unsqueeze(-1) == eos_t).any(-1)
        finished = new_finished

        parents = (torch.arange(Gc, device=device).unsqueeze(-1) * K + new_par).reshape(-1)
        pending_tokens = tok_step.reshape(-1)

    for b in range(len(live)):
        finalize(live[b], b, cum_raw, cum_tmp, finished)
    return [results[g] for g in range(G)]
