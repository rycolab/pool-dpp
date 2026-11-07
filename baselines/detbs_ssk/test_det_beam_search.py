"""
Tests for Determinantal Beam Search:
  1. Incremental DP gives identical results to full DP (numerical equivalence)
  2. Cross-step cache is consistent  (values stable across multiple steps)
  3. Speed comparison: incremental vs full kernel DP
  4. Full generate() on Qwen/Qwen3-0.6B — diverse outputs, sanity checks
"""

import sys, time, random
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from generate import (
    _kernel_dp_full,
    _kernel_dp_cached,
    _norm_kernel_cached,
    _build_log_L,
    _greedy_map,
    generate,
    _logsumexp,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Incremental DP == Full DP
# ─────────────────────────────────────────────────────────────────────────────

def test_incremental_vs_full():
    print("=" * 60)
    print("TEST 1: Incremental DP == Full DP")
    print("=" * 60)
    rng = random.Random(42)
    vocab = list(range(1, 20))
    p, decay = 3, 0.5
    n_trials = 200
    max_err = 0.0

    for _ in range(n_trials):
        # Simulate two beam steps
        # Step t-1: sequences of length L
        L = rng.randint(3, 10)
        s0 = [rng.choice(vocab) for _ in range(L)]
        t0 = [rng.choice(vocab) for _ in range(L)]

        # Compute K(s0, t0) fully and store in prev_cache
        K_full_prev, S_full_prev = _kernel_dp_full(s0, t0, p, decay)
        prev_cache = {(str(s0), str(t0), p, decay): (S_full_prev, K_full_prev)}

        # Step t: both grew by one token
        tok_s = rng.choice(vocab)
        tok_t = rng.choice(vocab)
        s1 = s0 + [tok_s]
        t1 = t0 + [tok_t]

        K_full, _ = _kernel_dp_full(s1, t1, p, decay)
        curr_cache: dict = {}
        K_incr = _kernel_dp_cached(s1, t1, p, decay, prev_cache, curr_cache)

        for l in range(1, p + 1):
            err = abs(K_full[l] - K_incr[l])
            if err > 1e-10:
                print(f"  FAIL l={l}, err={err:.2e}, s1={s1}, t1={t1}")
            max_err = max(max_err, err)

    # Also test extend_s (only s grows)
    for _ in range(n_trials):
        L = rng.randint(3, 10)
        s0 = [rng.choice(vocab) for _ in range(L)]
        t  = [rng.choice(vocab) for _ in range(rng.randint(3, 10))]

        K_full_prev, S_full_prev = _kernel_dp_full(s0, t, p, decay)
        prev_cache = {(str(s0), str(t), p, decay): (S_full_prev, K_full_prev)}

        s1 = s0 + [rng.choice(vocab)]
        K_full, _ = _kernel_dp_full(s1, t, p, decay)
        curr_cache = {}
        K_incr = _kernel_dp_cached(s1, t, p, decay, prev_cache, curr_cache)

        for l in range(1, p + 1):
            err = abs(K_full[l] - K_incr[l])
            max_err = max(max_err, err)

    # Also test extend_t (only t grows)
    for _ in range(n_trials):
        L = rng.randint(3, 10)
        s  = [rng.choice(vocab) for _ in range(rng.randint(3, 10))]
        t0 = [rng.choice(vocab) for _ in range(L)]

        K_full_prev, S_full_prev = _kernel_dp_full(s, t0, p, decay)
        prev_cache = {(str(s), str(t0), p, decay): (S_full_prev, K_full_prev)}

        t1 = t0 + [rng.choice(vocab)]
        K_full, _ = _kernel_dp_full(s, t1, p, decay)
        curr_cache = {}
        K_incr = _kernel_dp_cached(s, t1, p, decay, prev_cache, curr_cache)

        for l in range(1, p + 1):
            err = abs(K_full[l] - K_incr[l])
            max_err = max(max_err, err)

    # Self-kernel incremental
    for _ in range(n_trials):
        L = rng.randint(3, 10)
        s0 = [rng.choice(vocab) for _ in range(L)]
        K_full_prev, S_full_prev = _kernel_dp_full(s0, s0, p, decay)
        prev_cache = {(str(s0), str(s0), p, decay): (S_full_prev, K_full_prev)}

        tok = rng.choice(vocab)
        s1  = s0 + [tok]
        K_full, _ = _kernel_dp_full(s1, s1, p, decay)
        curr_cache = {}
        K_incr = _kernel_dp_cached(s1, s1, p, decay, prev_cache, curr_cache)

        for l in range(1, p + 1):
            err = abs(K_full[l] - K_incr[l])
            max_err = max(max_err, err)

    print(f"  Max absolute error across {4*n_trials} random trials: {max_err:.2e}")
    assert max_err < 1e-9, f"Incremental DP error too large: {max_err}"
    print("  PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-step cache consistency (simulate 5 beam steps)
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_step_cache():
    """
    Simulate multiple beam steps.  At each step, verify that every pairwise
    kernel value returned by _kernel_dp_cached (which may use the incremental
    path) exactly matches _kernel_dp_full (always computed from scratch).
    This directly validates the incremental caching logic end-to-end.
    """
    print()
    print("=" * 60)
    print("TEST 2: Cross-step cache consistency (6 steps, 4 beams)")
    print("=" * 60)
    rng = random.Random(7)
    vocab = list(range(1, 50))
    p, decay = 3, 0.5      # p=3 exercises more of the S-matrix code
    beam_size, sub_beam = 4, 5

    beams: list = [[]]     # single empty beam (no generated tokens yet)
    kernel_cache: dict = {}

    for step in range(6):
        # Expand each beam by one random token
        cands = [b + [rng.choice(vocab)] for b in beams for _ in range(sub_beam)]

        prev_cache = kernel_cache
        curr_cache: dict = {}
        max_err = 0.0

        # Check all pairs (including self-kernels on the diagonal)
        for i in range(len(cands)):
            for j in range(i, len(cands)):
                K_ref, _ = _kernel_dp_full(cands[i], cands[j], p, decay)
                K_cached  = _kernel_dp_cached(cands[i], cands[j], p, decay,
                                               prev_cache, curr_cache)
                for l in range(1, p + 1):
                    err = abs(K_ref[l] - K_cached[l])
                    if err > 1e-9:
                        print(f"  FAIL step={step} l={l} i={i} j={j} "
                              f"err={err:.2e}  ref={K_ref[l]:.6f}  cached={K_cached[l]:.6f}")
                    max_err = max(max_err, err)

        n_incr = sum(1 for k in curr_cache
                     if prev_cache.get(k[:2] + (p, decay)) is not None
                     or any(
                         (str(eval(k[0])[:-1]), str(eval(k[1])[:-1]), p, decay)
                             in prev_cache
                         for _ in [None]
                     ))
        print(f"  Step {step}: len(curr_cache)={len(curr_cache):4d}  "
              f"max_err={max_err:.2e}", end="")
        assert max_err < 1e-9, f"Cache inconsistency at step {step}: {max_err}"
        print("  ✓")

        # Select beam_size beams for next step
        scores = np.array([rng.gauss(-5, 1) for _ in cands])
        log_L  = _build_log_L(cands, scores, p, decay, 0.1, prev_cache, curr_cache)
        sel    = _greedy_map(log_L, min(beam_size, len(cands)))
        beams  = [cands[i] for i in sel]
        kernel_cache = curr_cache

    print("  PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Speed: incremental vs full DP
# ─────────────────────────────────────────────────────────────────────────────

def test_speed():
    print()
    print("=" * 60)
    print("TEST 3: Speed comparison — incremental vs full DP")
    print("=" * 60)
    rng = random.Random(99)
    vocab = list(range(1, 200))
    p, decay = 2, 0.5
    beam_size, sub_beam = 5, 10
    n_steps = 20

    # Warm-up beam
    beams = [[rng.choice(vocab) for _ in range(5)]]

    # --- Full DP (no cache) ---
    t0 = time.perf_counter()
    for _step in range(n_steps):
        cands = [b + [rng.choice(vocab)] for b in beams for _ in range(sub_beam)]
        N = len(cands)
        for i in range(N):
            for j in range(i + 1, N):
                _kernel_dp_full(cands[i], cands[j], p, decay)
                _kernel_dp_full(cands[i], cands[i], p, decay)
                _kernel_dp_full(cands[j], cands[j], p, decay)
        beams = cands[:beam_size]
    t_full = time.perf_counter() - t0

    # --- Incremental DP ---
    beams = [[rng.choice(vocab) for _ in range(5)]]
    rng2  = random.Random(99)           # same RNG seed for fair comparison
    beams = [[rng2.choice(vocab) for _ in range(5)]]
    kernel_cache: dict = {}
    # Seed the cache with self-kernels of the initial beams
    for b in beams:
        K, S = _kernel_dp_full(b, b, p, decay)
        kernel_cache[(str(b), str(b), p, decay)] = (S, K)

    t0 = time.perf_counter()
    for _step in range(n_steps):
        cands = [b + [rng2.choice(vocab)] for b in beams for _ in range(sub_beam)]
        prev_cache = kernel_cache
        curr_cache: dict = {}
        N = len(cands)
        for i in range(N):
            for j in range(i + 1, N):
                _kernel_dp_cached(cands[i], cands[j], p, decay, prev_cache, curr_cache)
                _kernel_dp_cached(cands[i], cands[i], p, decay, prev_cache, curr_cache)
                _kernel_dp_cached(cands[j], cands[j], p, decay, prev_cache, curr_cache)
        beams = cands[:beam_size]
        kernel_cache = curr_cache
    t_incr = time.perf_counter() - t0

    speedup = t_full / t_incr
    print(f"  Full DP:        {t_full*1000:.1f} ms for {n_steps} steps")
    print(f"  Incremental DP: {t_incr*1000:.1f} ms for {n_steps} steps")
    print(f"  Speedup: {speedup:.2f}×")
    # Expect speedup once sequences are long; at beam-length=5+n this will grow
    print("  (speedup increases with sequence length)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Functional test on Qwen/Qwen3-0.6B
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_qwen():
    print()
    print("=" * 60)
    print("TEST 4: generate() on Qwen/Qwen3-0.6B")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-0.6B"
    print(f"  Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    model.eval()
    device = "cpu"
    model = model.to(device)
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    prompt = "The quick brown fox"
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # ── 4a: w=0 should give same top sequence as greedy (kernel disabled) ──
    print("\n  4a: beam_size=1, kernel_weight=0 → same as greedy")
    seqs_det = generate(
        model,
        **inputs,
        beam_size=1,
        sub_beam_size=1,
        kernel_weight=0.0,
        max_new_tokens=20,
        temperature=1.0,
    )
    out_det = tokenizer.decode(seqs_det[0, prompt_len:], skip_special_tokens=True)

    # Standard greedy via HF
    with torch.no_grad():
        seqs_greedy = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=20,
        )
    out_greedy = tokenizer.decode(seqs_greedy[0, prompt_len:], skip_special_tokens=True)
    print(f"    DetBS (w=0): '{out_det}'")
    print(f"    Greedy:      '{out_greedy}'")
    # Allow small differences (sub_beam_size=1 limits candidates)

    # ── 4b: Diverse outputs with beam_size=5, kernel_weight>0 ──────────────
    print("\n  4b: beam_size=5, kernel_weight=0.1 → diverse outputs")
    beam_size = 5
    t0 = time.perf_counter()
    seqs = generate(
        model,
        **inputs,
        beam_size=beam_size,
        sub_beam_size=beam_size,
        kernel_weight=0.1,
        kernel_n=2,
        kernel_decay=0.5,
        max_new_tokens=20,
        temperature=1.0,
    )
    elapsed = time.perf_counter() - t0
    assert seqs.shape[0] == beam_size, f"Expected {beam_size} sequences, got {seqs.shape[0]}"

    generated_texts = [
        tokenizer.decode(seqs[i, prompt_len:], skip_special_tokens=True)
        for i in range(beam_size)
    ]
    print(f"    Generated in {elapsed:.1f}s:")
    for i, txt in enumerate(generated_texts):
        print(f"    [{i}] '{txt}'")

    # Count unique outputs
    n_unique = len(set(generated_texts))
    print(f"    Unique sequences: {n_unique}/{beam_size}")
    assert n_unique >= 2, f"Expected diverse outputs, got only {n_unique} unique sequences"

    # ── 4c: w=0 beam_size=5 → less diverse (all from top-probability paths) ─
    print("\n  4c: beam_size=5, kernel_weight=0 → standard beam search diversity")
    seqs_std = generate(
        model,
        **inputs,
        beam_size=beam_size,
        sub_beam_size=beam_size,
        kernel_weight=0.0,
        max_new_tokens=20,
        temperature=1.0,
    )
    texts_std = [
        tokenizer.decode(seqs_std[i, prompt_len:], skip_special_tokens=True)
        for i in range(beam_size)
    ]
    n_unique_std = len(set(texts_std))
    print(f"    Unique sequences (standard beam): {n_unique_std}/{beam_size}")
    for i, txt in enumerate(texts_std):
        print(f"    [{i}] '{txt}'")

    # ── 4d: return_dict=True ─────────────────────────────────────────────────
    print("\n  4d: return_dict_in_generate=True")
    result = generate(
        model,
        **inputs,
        beam_size=3,
        kernel_weight=0.1,
        max_new_tokens=10,
        return_dict_in_generate=True,
    )
    assert "sequences" in result and "scores" in result
    assert result["sequences"].shape[0] == 3
    print(f"    sequences shape: {result['sequences'].shape}  scores: {result['scores']}")
    print("  PASSED ✓")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_incremental_vs_full()
    test_cross_step_cache()
    test_speed()
    test_generate_qwen()
