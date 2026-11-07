"""
Determinantal Beam Search — HuggingFace custom_generate entry point.

Paper: Meister, Forster & Cotterell, "Determinantal Beam Search" (ACL 2021).
Original fairseq code: github.com/rycolab/swor  (branch: diverse_decoding)

Algorithm:
  At each beam step, replace the standard top-k selection with selecting
  the size-k subset Y* that approximately maximises

      log det( D_{Y*} + w · K_{Y*} )

  where D is a diagonal matrix of (normalised) sequence log-probs and K is
  the normalised string-subsequence-kernel Gram matrix (Lodhi et al. 2002).
  The argmax is approximated by the fast greedy Cholesky algorithm of
  Chen et al. (2018), run in log-space for numerical stability.

Key performance feature: the string kernel is computed incrementally.
Each step, sequences grow by one token; the DP tables (S-matrices) from
the previous step are reused, reducing per-pair kernel cost from O(p·L²)
to O(p·L) once the cache is warm (L = current sequence length).

Usage:
    model.generate(**inputs,
                   custom_generate="<hf_repo>/det_beam_search",
                   beam_size=5, kernel_weight=0.1,
                   trust_remote_code=True)

Or locally:
    from generate import generate
    seqs = generate(model, input_ids=..., beam_size=5, kernel_weight=0.1)
"""

import numpy as np
import torch


def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp (no scipy dependency)."""
    a_max = float(np.max(a))
    return a_max + float(np.log(np.sum(np.exp(a - a_max))))


# ── Log-space arithmetic ────────────────────────────────────────────────────
# Ported from utils.py (log_add, log_minus, log1pexp, log1mexp, bit_logsumexp)

_NEG_INF = float("-inf")


def _log1pexp(x: float) -> float:
    if x <= -37:
        return float(np.exp(x))
    if x <= 18:
        return float(np.log1p(np.exp(x)))
    if x <= 33.3:
        return x + float(np.exp(-x))
    return x


def _log1mexp(x: float) -> float:
    a = abs(x)
    if 0 < a <= 0.693:
        return float(np.log(-np.expm1(-a)))
    return float(np.log1p(-np.exp(-a)))


def _log_add(x: float, y: float) -> float:
    if x == _NEG_INF:
        return y
    if y == _NEG_INF:
        return x
    if y <= x:
        return x + _log1pexp(y - x)
    return y + _log1pexp(x - y)


def _log_minus(x: float, y: float) -> float:
    if x == y:
        return _NEG_INF
    if y > x:
        return float("nan")
    return x + _log1mexp(y - x)


def _bit_logsumexp(c1, c2, s1, s2):
    """Signed log inner product <c1, c2> with sign tracking."""
    sign, val = 1, _NEG_INF
    for x, y, sx, sy in zip(c1, c2, s1, s2):
        cur = sx * sy
        fn = _log_add if sign == cur else _log_minus
        if val > x + y:
            val = fn(val, x + y)
        else:
            sign = cur
            val = fn(x + y, val)
    return sign, val


# ── Greedy MAP inference (Chen et al. 2018, log-space; Appendix A) ──────────

def _greedy_map(log_L: np.ndarray, k: int) -> list:
    """
    Approximately maximise log det(L_Y) over |Y|=k by greedy Cholesky updates.
    Returns list of k selected indices.
    """
    N = log_L.shape[0]
    c = [[] for _ in range(N)]
    sb = [[] for _ in range(N)]
    ld = [float(log_L[i, i]) for i in range(N)]

    j = int(np.argmax(ld))
    Y = {j}

    while len(Y) < k:
        for i in range(N):
            if i in Y:
                continue
            sign, inner = _bit_logsumexp(c[i], c[j], sb[i], sb[j])
            fn = _log_add if sign == -1 else _log_minus
            lji = float(log_L[j, i])
            if lji > inner:
                s_new = 1
                e = fn(lji, inner) - 0.5 * ld[j]
            else:
                s_new = -sign
                e = fn(inner, lji) - 0.5 * ld[j]
            c[i].append(e)
            sb[i].append(s_new)
            ld[i] = _log_minus(ld[i], 2 * e)

        masked = [_NEG_INF if i in Y else ld[i] for i in range(N)]
        j = int(np.argmax(masked))
        Y.add(j)

    return list(Y)


# ── String Subsequence Kernel — incremental DP ──────────────────────────────
#
# Cache format across beam steps:
#   {(str(s), str(t), p, decay): (S_dict, K_dict)}
# where
#   K_dict = {l: float}         — kernel values per subsequence length 1..p
#   S_dict = {l: ndarray}       — intermediate DP recurrence matrices needed
#                                  to extend K by one token next step
#
# Both orientations (s,t) and (t,s) are stored; S(t,s) = S(s,t).T (provable
# from the symmetric recurrence).  This lets all future lookups hit directly.
#
# At each beam step, candidates are extensions of the previous beam by one
# token, so we need K(prev_i, prev_j) → K(cand_i, cand_j).  The three
# incremental helpers below handle the three length-difference cases:
#   both grew   (Δlen=0): _incremental_both
#   s grew only (Δlen=+1): _incremental_extend_s
#   t grew only (Δlen=-1): _incremental_extend_t
# When no usable previous entry exists, we fall back to a full O(p·|s|·|t|) DP.


def _kernel_dp_full(s: list, t: list, p: int, decay: float):
    """
    Full (non-incremental) string subsequence kernel DP.
    Rousu & Shawe-Taylor 2005 formulation.
    Returns (K dict {l: float}, S dict {l: ndarray}).
    """
    d2 = decay * decay
    K, k_, S = {}, {}, {}

    K[1] = 0.0
    k_[1] = np.zeros((len(s) + 1, len(t) + 1))
    for i in range(1, len(s) + 1):
        for j in range(1, len(t) + 1):
            if s[i - 1] == t[j - 1]:
                k_[1][i, j] = d2
                K[1] += d2

    for l in range(2, p + 1):
        K[l] = 0.0
        S[l] = np.zeros((len(s) + 1, len(t) + 1))
        k_[l] = np.zeros((len(s) + 1, len(t) + 1))
        for i in range(1, len(s) + 1):
            for j in range(1, len(t) + 1):
                S[l][i, j] = (
                    k_[l - 1][i, j]
                    + decay * S[l][i - 1, j]
                    + decay * S[l][i, j - 1]
                    - d2 * S[l][i - 1, j - 1]
                )
                if s[i - 1] == t[j - 1]:
                    k_[l][i, j] = d2 * S[l][i - 1, j - 1]
                    K[l] += k_[l][i, j]

    return K, S


def _incremental_both(s: list, t: list, p: int, decay: float,
                       d2: float, prev_S: dict, prev_K: dict):
    """
    Incremental DP: both s and t each grew by exactly one token since prev step.
    Only computes the new last row and last column; reuses prev_S for the interior.
    Cost: O(p · (|s| + |t|)) instead of O(p · |s| · |t|).
    """
    K, k_, S = {}, {}, {}

    k_[1] = np.zeros((len(s) + 1, len(t) + 1))
    K[1] = prev_K[1]
    # New last column (rows 1..len(s)-1)
    for i in range(1, len(s)):
        j = len(t)
        if s[i - 1] == t[j - 1]:
            k_[1][i, j] = d2
            K[1] += d2
    # New last row (all columns)
    for j in range(1, len(t) + 1):
        i = len(s)
        if s[i - 1] == t[j - 1]:
            k_[1][i, j] = d2
            K[1] += d2

    for l in range(2, p + 1):
        K[l] = prev_K[l]
        S[l] = np.zeros((len(s) + 1, len(t) + 1))
        S[l][:-1, :-1] = prev_S[l]      # fill interior from previous step
        k_[l] = np.zeros((len(s) + 1, len(t) + 1))

        # New last column (rows 1..len(s)-1, sequential for recurrence)
        for i in range(1, len(s)):
            j = len(t)
            S[l][i, j] = (
                k_[l - 1][i, j]
                + decay * S[l][i - 1, j]
                + decay * S[l][i, j - 1]
                - d2 * S[l][i - 1, j - 1]
            )
            if s[i - 1] == t[j - 1]:
                k_[l][i, j] = d2 * S[l][i - 1, j - 1]
                K[l] += k_[l][i, j]

        # New last row (all columns, sequential)
        for j in range(1, len(t) + 1):
            i = len(s)
            S[l][i, j] = (
                k_[l - 1][i, j]
                + decay * S[l][i - 1, j]
                + decay * S[l][i, j - 1]
                - d2 * S[l][i - 1, j - 1]
            )
            if s[i - 1] == t[j - 1]:
                k_[l][i, j] = d2 * S[l][i - 1, j - 1]
                K[l] += k_[l][i, j]

    return K, S


def _incremental_extend_s(s: list, t: list, p: int, decay: float,
                           d2: float, prev_S: dict, prev_K: dict):
    """Incremental DP: s grew by one, t unchanged. Cost: O(p · |t|)."""
    K, k_, S = {}, {}, {}

    k_[1] = np.zeros((len(s) + 1, len(t) + 1))
    K[1] = prev_K[1]
    for j in range(1, len(t) + 1):
        i = len(s)
        if s[i - 1] == t[j - 1]:
            k_[1][i, j] = d2
            K[1] += d2

    for l in range(2, p + 1):
        K[l] = prev_K[l]
        S[l] = np.zeros((len(s) + 1, len(t) + 1))
        S[l][:-1, :] = prev_S[l]        # fill old rows
        k_[l] = np.zeros((len(s) + 1, len(t) + 1))

        for j in range(1, len(t) + 1):
            i = len(s)
            S[l][i, j] = (
                k_[l - 1][i, j]
                + decay * S[l][i - 1, j]
                + decay * S[l][i, j - 1]
                - d2 * S[l][i - 1, j - 1]
            )
            if s[i - 1] == t[j - 1]:
                k_[l][i, j] = d2 * S[l][i - 1, j - 1]
                K[l] += k_[l][i, j]

    return K, S


def _incremental_extend_t(s: list, t: list, p: int, decay: float,
                           d2: float, prev_S: dict, prev_K: dict):
    """Incremental DP: t grew by one, s unchanged. Cost: O(p · |s|)."""
    K, k_, S = {}, {}, {}

    k_[1] = np.zeros((len(s) + 1, len(t) + 1))
    K[1] = prev_K[1]
    for i in range(1, len(s) + 1):
        j = len(t)
        if s[i - 1] == t[j - 1]:
            k_[1][i, j] = d2
            K[1] += d2

    for l in range(2, p + 1):
        K[l] = prev_K[l]
        S[l] = np.zeros((len(s) + 1, len(t) + 1))
        S[l][:, :-1] = prev_S[l]        # fill old columns
        k_[l] = np.zeros((len(s) + 1, len(t) + 1))

        for i in range(1, len(s) + 1):
            j = len(t)
            S[l][i, j] = (
                k_[l - 1][i, j]
                + decay * S[l][i - 1, j]
                + decay * S[l][i, j - 1]
                - d2 * S[l][i - 1, j - 1]
            )
            if s[i - 1] == t[j - 1]:
                k_[l][i, j] = d2 * S[l][i - 1, j - 1]
                K[l] += k_[l][i, j]

    return K, S


def _kernel_dp_cached(s: list, t: list, p: int, decay: float,
                       prev_cache: dict, curr_cache: dict) -> dict:
    """
    Compute K(s, t) — the string subsequence kernel for all lengths 1..p —
    with cross-step incremental caching.

    Cache lookup order:
    1. curr_cache (already computed this step)
    2. prev_cache exact match (EOS beam carried forward unchanged)
    3. Incremental update from prev_cache using parent sequences
    4. Full DP fallback

    After computing, both (s,t) and (t,s) orientations are stored in curr_cache
    (the transposed S-matrix is stored for the symmetric key, since S(t,s)=S(s,t)ᵀ).

    Returns K dict {1: float, ..., p: float}.
    """
    key    = (str(s), str(t), p, decay)
    key_ts = (str(t), str(s), p, decay)

    # 1. Already computed this step
    if key in curr_cache:
        return curr_cache[key][1]
    if key_ts in curr_cache:
        return curr_cache[key_ts][1]

    # 2. Sequence pair unchanged since last step (e.g. both beams were EOS)
    if key in prev_cache:
        curr_cache[key] = prev_cache[key]
        return prev_cache[key][1]
    if key_ts in prev_cache:
        curr_cache[key] = prev_cache[key_ts]
        return prev_cache[key_ts][1]

    # 3. Incremental update
    ls, lt = len(s), len(t)
    K = S = None
    d2 = decay * decay

    if ls >= 1 and lt >= 1:
        if ls == lt:
            pk = (str(s[:-1]), str(t[:-1]), p, decay)
            if pk in prev_cache:
                prev_S, prev_K = prev_cache[pk]
                K, S = _incremental_both(s, t, p, decay, d2, prev_S, prev_K)
        elif ls == lt + 1:
            pk = (str(s[:-1]), str(t), p, decay)
            if pk in prev_cache:
                prev_S, prev_K = prev_cache[pk]
                K, S = _incremental_extend_s(s, t, p, decay, d2, prev_S, prev_K)
        elif lt == ls + 1:
            pk = (str(s), str(t[:-1]), p, decay)
            if pk in prev_cache:
                prev_S, prev_K = prev_cache[pk]
                K, S = _incremental_extend_t(s, t, p, decay, d2, prev_S, prev_K)

    # 4. Full DP fallback (first two steps, dead-vs-live length mismatch, etc.)
    if K is None:
        K, S = _kernel_dp_full(s, t, p, decay)

    # Store both orientations; S(t,s) = S(s,t)ᵀ
    curr_cache[key] = (S, K)
    if key_ts not in curr_cache:
        curr_cache[key_ts] = ({l: S[l].T for l in S}, K)

    return K


def _norm_kernel_cached(s: list, t: list, p: int, decay: float,
                         prev_cache: dict, curr_cache: dict) -> float:
    """
    Lodhi-normalised string kernel averaged over lengths 1..p, with caching.
    All three kernel calls (s,t), (s,s), (t,t) benefit from incremental DP.
    """
    K_st = _kernel_dp_cached(s, t, p, decay, prev_cache, curr_cache)
    K_ss = _kernel_dp_cached(s, s, p, decay, prev_cache, curr_cache)
    K_tt = _kernel_dp_cached(t, t, p, decay, prev_cache, curr_cache)
    vals = []
    for i in range(1, p + 1):
        kst = K_st[i]
        if kst == 0.0:
            vals.append(1.0 if s == t else 0.0)
        else:
            denom = (K_ss[i] * K_tt[i]) ** 0.5
            vals.append(kst / denom if denom > 0 else 0.0)
    return float(np.mean(vals))


# ── L-matrix construction (D + w·K in log space) ────────────────────────────

def _build_log_L(
    gen_seqs: list,
    log_scores: np.ndarray,
    kernel_n: int,
    kernel_decay: float,
    weight: float,
    prev_cache: dict,
    curr_cache: dict,
) -> np.ndarray:
    """
    Build log(L) = log(diag(p_normalised) + w · K).

    All pairwise and self-kernels are computed with incremental caching.
    curr_cache is updated in-place and becomes the prev_cache for the next step.
    """
    N = len(gen_seqs)
    K = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            v = _norm_kernel_cached(
                gen_seqs[i], gen_seqs[j], kernel_n, kernel_decay,
                prev_cache, curr_cache,
            )
            K[i, j] = K[j, i] = v
        # Self-kernel K(s,s) is implicitly computed inside _norm_kernel_cached
        # (needed for normalisation AND for next step's incremental update)

    lp = log_scores - _logsumexp(log_scores)

    log_L = np.full((N, N), _NEG_INF)
    for i in range(N):
        for j in range(N):
            if i == j:
                log_L[i, i] = _log_add(np.log(weight), float(lp[i]))
            elif K[i, j] > 0:
                log_L[i, j] = np.log(weight * K[i, j])
    return log_L


# ── Beam-search loop ─────────────────────────────────────────────────────────

def _det_beam_search(
    model,
    input_ids: torch.Tensor,
    eos_ids: torch.Tensor,
    pad_id: int,
    beam_size: int,
    sub_beam_size: int,
    max_new_tokens: int,
    kernel_n: int,
    kernel_decay: float,
    kernel_weight: float,
    temperature: float,
):
    """
    Core DetBS loop with incremental kernel caching between steps.

    State between steps:
      current_ids  [B, L]   — full token sequences on device
      gen_tokens   B lists  — generated tokens per beam (no prompt)
      cum_logp     [B]      — cumulative log-prob per beam
      alive        [B]      — True until beam emits an EOS token
      kernel_cache  dict    — S/K matrices from previous step (for incremental DP)
    """
    device  = model.device
    eos_set = set(eos_ids.tolist())

    current_ids  = input_ids   # [1, prompt_len]
    gen_tokens   = [[]]
    cum_logp     = np.array([0.0])
    alive        = np.array([True])
    kernel_cache: dict = {}    # becomes prev_cache each step

    for _step in range(max_new_tokens):
        B = current_ids.shape[0]

        # ── Forward pass (full sequence; no KV-cache reordering needed) ─────
        with torch.no_grad():
            attn   = torch.ones(B, current_ids.shape[1], device=device, dtype=torch.long)
            logits = model(input_ids=current_ids, attention_mask=attn,
                           return_dict=True).logits[:, -1]             # [B, V]
        log_ps = (logits / temperature).log_softmax(-1).float().cpu().numpy()  # [B, V]

        # ── Expand candidates ────────────────────────────────────────────────
        cand_ids        = []
        cand_gen        = []
        cand_logp       = []
        cand_alive_flag = []

        for b in range(B):
            base = current_ids[b]
            if not alive[b]:
                # EOS beam: carry forward (append pad to keep tensor shapes uniform)
                cand_ids.append(
                    torch.cat([base, torch.tensor([pad_id], device=device)])
                )
                cand_gen.append(gen_tokens[b])
                cand_logp.append(float(cum_logp[b]))
                cand_alive_flag.append(False)
            else:
                top_idx = np.argpartition(log_ps[b], -sub_beam_size)[-sub_beam_size:]
                for tok in top_idx.tolist():
                    cand_ids.append(
                        torch.cat([base, torch.tensor([tok], device=device)])
                    )
                    cand_gen.append(gen_tokens[b] + [tok])
                    cand_logp.append(float(cum_logp[b]) + float(log_ps[b, tok]))
                    cand_alive_flag.append(tok not in eos_set)

        C          = len(cand_logp)
        k_sel      = min(beam_size, C)
        scores_arr = np.array(cand_logp)

        # ── DPP or top-k selection ───────────────────────────────────────────
        if kernel_weight > 0.0 and C > k_sel and np.any(alive):
            prev_cache = kernel_cache
            curr_cache: dict = {}
            log_L  = _build_log_L(
                cand_gen, scores_arr,
                kernel_n, kernel_decay, kernel_weight,
                prev_cache, curr_cache,
            )
            selected     = _greedy_map(log_L, k_sel)
            kernel_cache = curr_cache        # carry forward for next step
        else:
            selected     = list(np.argpartition(scores_arr, -k_sel)[-k_sel:])
            kernel_cache = {}                # no cache when DPP not used

        # ── Update beam state ────────────────────────────────────────────────
        current_ids = torch.stack([cand_ids[i]       for i in selected], dim=0)
        gen_tokens  = [cand_gen[i]                   for i in selected]
        cum_logp    = np.array([cand_logp[i]         for i in selected])
        alive       = np.array([cand_alive_flag[i]   for i in selected])

        if not alive.any():
            break

    return current_ids, cum_logp


# ── Public entry point ────────────────────────────────────────────────────────

def generate(model, **kwargs):
    """
    Determinantal Beam Search — custom_generate entry point.

    Produces a diverse set of `beam_size` sequences from a single prompt by
    replacing the top-k selection at each beam step with a k-DPP MAP query
    parameterised by a string-subsequence-kernel similarity matrix.

    Standard HF generate() kwargs are accepted. DetBS-specific kwargs:

        beam_size     (int,   default 5):
            Number of diverse output sequences.
        sub_beam_size (int,   default beam_size):
            Number of token extensions considered per beam per step before
            DPP selection. sub_beam_size ≥ beam_size required.
        kernel_n      (int,   default 2):
            Subsequence length p for the string kernel.  n=2 recommended.
        kernel_decay  (float, default 0.5):
            Gap-decay factor λ ∈ (0, 1].  Paper uses λ=0.1 (De→En),
            λ=0.3 (En→Fr).
        kernel_weight (float, default 0.1):
            Diversity weight w ≥ 0.  w=0 recovers standard beam search.
        temperature   (float, default 1.0):
            Softmax temperature applied to token log-probs.

    Returns:
        Tensor[beam_size, seq_len]  when return_dict_in_generate=False
        {"sequences": Tensor, "scores": ndarray}  otherwise
    """
    gc = model.generation_config

    max_new_tokens = kwargs.get("max_new_tokens",
                                getattr(gc, "max_new_tokens", None)) or 128

    eos_ids = kwargs.get("eos_token_ids", getattr(gc, "eos_token_id", None))
    if eos_ids is None:
        raise ValueError(
            "No EOS token id found. Pass eos_token_ids=... to generate()."
        )
    eos_ids = torch.as_tensor(eos_ids, device=model.device)
    if eos_ids.ndim == 0:
        eos_ids = eos_ids.unsqueeze(0)

    pad_id        = int(kwargs.get("pad_token_id",
                        getattr(gc, "pad_token_id", None) or int(eos_ids[0])))
    beam_size     = int(kwargs.get("beam_size",      5))
    sub_beam_size = int(kwargs.get("sub_beam_size",  beam_size))
    kernel_n      = int(kwargs.get("kernel_n",       2))
    kernel_decay  = float(kwargs.get("kernel_decay", 0.5))
    kernel_weight = float(kwargs.get("kernel_weight", 0.1))
    temperature   = float(kwargs.get("temperature",  1.0))
    return_dict   = bool(kwargs.get("return_dict_in_generate", False))

    input_ids = kwargs.get("input_ids")
    if input_ids is None:
        raise ValueError("input_ids must be provided.")
    if input_ids.shape[0] != 1:
        raise ValueError(
            "DetBS takes a single prompt (batch_size=1) and returns "
            "beam_size diverse sequences."
        )

    seqs, scores = _det_beam_search(
        model         = model,
        input_ids     = input_ids,
        eos_ids       = eos_ids,
        pad_id        = pad_id,
        beam_size     = beam_size,
        sub_beam_size = sub_beam_size,
        max_new_tokens= max_new_tokens,
        kernel_n      = kernel_n,
        kernel_decay  = kernel_decay,
        kernel_weight = kernel_weight,
        temperature   = temperature,
    )

    seqs = seqs.cpu()
    if return_dict:
        return {"sequences": seqs, "scores": scores}
    return seqs
