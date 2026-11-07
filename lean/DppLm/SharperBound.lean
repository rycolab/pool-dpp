import DppLm.PoolDPP

/-!
# `cor:kl-sum` (Sharper bound)

Paper statement (`cor:kl-sum`): under the conditions of `thm:zfree`, if `ρ < ∞`, then
  `KL(p ‖ π̄) ≤ log ∑_{c=0}^{K} a_c (K^c/(K)_c)² ρ^c`
  `          = log(1 + a₁ Var_q(u₁) + ∑_{c=2}^{K} a_c ((K^c/(K)_c)² ρ^c - 1))`,
with `a_c = C(K,c) C(N-K,K-c)/C(N,K)` (`overlapCoeff`) and `ρ = E_q[u₁²]` (`secondMomentU1`).
Since `∑_c a_c = 1` (`sum_overlapCoeff`), the sum is the expectation of `(K^c/(K)_c)² ρ^c`
over the hypergeometric overlap `c`.

## What is proved here

* `poolDPP_sharper_fallback_bound` (any fallback) and `poolDPP_sharper_redraw_bound` (redraw
  the pool): the whole corollary for the paper's sketch (both lines of `eq:kl-sum` and the
  comparison with `thm:zfree`), with the hypotheses of the `thm:zfree` versions in
  `PoolDPP.lean`. `thm:var-kl-mixture` gives `KL ≤ log(E[Ẑ²]/Z²)`, and
  `poolDPP_secondMoment_le_sharper` (`PoolDPP.lean`) bounds `E[Ẑ²]/Z²` by the sum.
* `sharperBound_eq`, used for the second line: the `c = 0` term is `a₀`, the `c = 1` term is
  `a₁ ρ = a₁ (1 + Var_q(u₁))`, and `∑_c a_c = 1`.

The second-moment bound is proved in `FiniteSample.lean`, where `thm:zfree` also uses it:
`secondMoment_le_sharper` applies `E[u_c²] ≤ (K^c/(K)_c)² ρ^c` (`tsum_inclRatio_sq_le`, the
paper's `eq:second-moment`) to every term of `E[Ẑ²] = Z² ∑_{c=0}^{K} a_c E[u_c²]`
(`zfree_identity`), and `sharperBound_le_exp` shows that the sum is at most `thm:zfree`'s
`e^θ - 1 + a₀`, `θ = K² ρ/(N - K + 1)`, so the corollary is never weaker than the theorem.
-/

open Finset MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace DppLm

section General

variable {Y : Type*} [DecidableEq Y] [Countable Y] [MeasurableSpace Y]
  [MeasurableSingletonClass Y]
variable {K N : ℕ} {f : Finset Y → ℝ≥0∞}
variable (p : PMF Y)

/-- **`cor:kl-sum`, the second line of `eq:kl-sum`.** When `m₁ = ρ = E_q[u₁²]`,
`∑_{c=0}^{K} a_c (K^c/(K)_c)² m₁^c = 1 + a₁ Var_q(u₁) + ∑_{c=2}^{K} a_c ((K^c/(K)_c)² m₁^c - 1)`,
with `Var_q(u₁)` the variance of `u₁` under one draw from `q`. -/
theorem sharperBound_eq (hs : PoolSetting p K f) (hK : 1 ≤ K) (hKN : K ≤ N)
    (m₁ : ℝ≥0) (hm1 : secondMomentU1 p K f = m₁) :
    ∑ c ∈ range (K + 1), (overlapCoeff K N c).toReal *
        (((K : ℝ) ^ c / (K.descFactorial c : ℝ)) ^ 2 * (m₁ : ℝ) ^ c) =
      1 + (overlapCoeff K N 1).toReal *
          variance (fun y => (inclRatio p K f y).toReal) (poolLaw p 1) +
        ∑ c ∈ Icc 2 K, (overlapCoeff K N c).toReal *
          (((K : ℝ) ^ c / (K.descFactorial c : ℝ)) ^ 2 * (m₁ : ℝ) ^ c - 1) := by
  obtain ⟨-, -, -, hEu, ha, -⟩ := zfree_identity (N := N) p hs hKN
  set g : ℕ → ℝ := fun c => ((K : ℝ) ^ c / (K.descFactorial c : ℝ)) ^ 2 * (m₁ : ℝ) ^ c
  set a : ℕ → ℝ := fun c => (overlapCoeff K N c).toReal
  have hC0 : ((N.choose K : ℕ) : ℝ≥0∞) ≠ 0 := by exact_mod_cast (Nat.choose_pos hKN).ne'
  have ha_top : ∀ m, overlapCoeff K N m ≠ ∞ := fun m =>
    ENNReal.div_ne_top (ENNReal.natCast_ne_top _) hC0
  have hsum_a : ∑ c ∈ range (K + 1), a c = 1 := by
    simp only [a]
    rw [← ENNReal.toReal_sum (fun m _ => ha_top m), ha, ENNReal.toReal_one]
  -- `Var_q(u₁) = E[u₁²] - E[u₁]² = m₁ - 1`
  have hvar : variance (fun y => (inclRatio p K f y).toReal) (poolLaw p 1) = (m₁ : ℝ) - 1 := by
    have h2 : ∫⁻ y, inclRatio p K f y ^ 2 ∂poolLaw p 1 = m₁ := by
      rw [lintegral_poolLaw]; exact hm1
    rw [variance_toReal_eq (G := inclRatio p K f) Measurable.of_discrete.aemeasurable
      (by rw [h2]; exact ENNReal.coe_ne_top), h2, hEu 1 hK]
    simp
  have hg0 : g 0 = 1 := by simp [g]
  have hg1 : g 1 = m₁ := by
    have hK0 : (K : ℝ) ≠ 0 := by exact_mod_cast (by omega : K ≠ 0)
    simp [g, div_self hK0]
  -- `range (K + 1)` is `{0}`, `{1}`, and `Icc 2 K`
  have hsplit : ∀ h : ℕ → ℝ, ∑ c ∈ range (K + 1), h c = h 0 + (h 1 + ∑ c ∈ Icc 2 K, h c) := by
    intro h
    rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot (by omega),
      Finset.sum_eq_sum_Ico_succ_bot (by omega)]
    rfl
  calc ∑ c ∈ range (K + 1), a c * g c
      = ∑ c ∈ range (K + 1), a c + ∑ c ∈ range (K + 1), a c * (g c - 1) := by
        rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun c _ => by ring
    _ = 1 + (a 0 * (g 0 - 1) + (a 1 * (g 1 - 1) + ∑ c ∈ Icc 2 K, a c * (g c - 1))) := by
        rw [hsum_a, hsplit (fun c => a c * (g c - 1))]
    _ = _ := by rw [hg0, hg1, hvar]; ring

end General

section PoolDPP

variable {Y : Type*} [DecidableEq Y] [Countable Y] [MeasurableSpace Y]
  [MeasurableSingletonClass Y] {D : ℕ}

/-- The sum facts shared by the two `cor:kl-sum` theorems, with `ρ = E_q[u₁²] < ∞` (`m₁` in
the statement): `log(1 + Var(Ẑ)/Z²)`, the bound of `thm:var-kl-mixture`, is at most the log of
the sum; the sum has the second form of `eq:kl-sum` (`sharperBound_eq`); and its log is at most
`thm:zfree`'s `log(e^θ - 1 + a₀)` (`sharperBound_le_exp`). -/
lemma poolDPP_sharper_sum (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
    (q : PMF Y) (hq : ∀ y, q y = 0 → κ y = 0) {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N)
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞)
    (hm1 : secondMomentU1 q K (kdppWeight κ φ K) ≠ ∞) :
    let m₁ : ℝ := (secondMomentU1 q K (kdppWeight κ φ K)).toReal
    let a : ℕ → ℝ := fun c => (overlapCoeff K N c).toReal
    let g : ℕ → ℝ := fun c => ((K : ℝ) ^ c / (K.descFactorial c : ℝ)) ^ 2 * m₁ ^ c
    let varU₁ : ℝ :=
      variance (fun y => (inclRatio q K (kdppWeight κ φ K) y).toReal) (poolLaw q 1)
    let θ : ℝ := (K : ℝ) ^ 2 * m₁ / ((N : ℝ) - K + 1)
    Real.log (1 + variance (fun ω => (sketchZ (poolDPPFhat κ φ q K N) ω).toReal) (poolLaw q N) /
        ((∑' S, kdppWeight κ φ K S).toReal) ^ 2) ≤
      Real.log (∑ c ∈ range (K + 1), a c * g c) ∧
    ∑ c ∈ range (K + 1), a c * g c = 1 + a 1 * varU₁ + ∑ c ∈ Icc 2 K, a c * (g c - 1) ∧
    Real.log (∑ c ∈ range (K + 1), a c * g c) ≤ Real.log (Real.exp θ - 1 + a 0) := by
  intro m₁ a g varU₁ θ
  have hM := ENNReal.coe_toNNReal hm1
  have hb := poolDPP_wellBehaved κ φ hκ q hq hK hKN hZ0 hZtop hm1 (N := N)
  have h8 := hb.one_add_variance_div hZ0 hZtop
  have hVnn := div_nonneg
    (variance_nonneg (fun ω => (sketchZ (poolDPPFhat κ φ q K N) ω).toReal) (poolLaw q N))
    (sq_nonneg (∑' S, kdppWeight κ φ K S).toReal)
  have hratio := (poolDPP_secondMoment_le_sharper κ φ hκ q hq hK hKN hZ0 hZtop _ hM.ge).2
  have hA : 0 < (∫⁻ ω, sketchZ (poolDPPFhat κ φ q K N) ω ^ 2 ∂poolLaw q N).toReal /
      ((∑' S, kdppWeight κ φ K S).toReal) ^ 2 := by
    rw [← h8]; linarith
  have hpos : 0 < ∑ c ∈ range (K + 1), a c * g c := hA.trans_le hratio
  refine ⟨?_, sharperBound_eq q (kdpp_poolSetting κ φ hκ q hq hZ0 hZtop) hK hKN _ hM.symm,
    Real.log_le_log hpos (sharperBound_le_exp hKN ENNReal.toReal_nonneg)⟩
  rw [h8]
  exact Real.log_le_log hA hratio

/-- **`cor:kl-sum` for PoolDPP with any fallback rule.** Hypotheses and sampler as in
`poolDPP_fallback_bound` (`thm:zfree`): the pool is i.i.d. from `q`, the sampler draws from
`π̂_ω` when `Ẑ ≠ 0` and from any distribution `out ω` when `Ẑ = 0`, and
`ρ = E_q[u₁²] < ∞` (`secondMomentU1`; `m₁` in the statement). Then `π̄` sums to `1`, and the
three statements of the corollary hold, with `a_c` = `overlapCoeff K N c`:
* `KL(p ‖ π̄) ≤ log ∑_{c=0}^{K} a_c (K^c/(K)_c)² ρ^c` (first line of `eq:kl-sum`);
* the sum equals `1 + a₁ Var_q(u₁) + ∑_{c=2}^{K} a_c ((K^c/(K)_c)² ρ^c - 1)` (second line),
  with `Var_q(u₁)` the variance of `u₁` under one draw from `q`;
* its log is at most `log(e^θ - 1 + a₀)`, the bound of `thm:zfree`, `θ = K² ρ/(N - K + 1)`.
The KL takes values in `[0, ∞]` (`kl`), so the bound also shows `π̄(S) = 0 → p(S) = 0` and that
the KL series converges. -/
theorem poolDPP_sharper_fallback_bound (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
    (q : PMF Y) (hq : ∀ y, q y = 0 → κ y = 0) {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N)
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞)
    (hm1 : secondMomentU1 q K (kdppWeight κ φ K) ≠ ∞)
    (out : (Fin N → Y) → Finset Y → ℝ≥0∞) (hout_sum : ∀ ω, ∑' S, out ω S = 1)
    (hout_eq : ∀ ω, sketchZ (poolDPPFhat κ φ q K N) ω ≠ 0 →
      sketchZ (poolDPPFhat κ φ q K N) ω ≠ ∞ →
      ∀ S, out ω S = poolDPPFhat κ φ q K N ω S / sketchZ (poolDPPFhat κ φ q K N) ω) :
    let p : Finset Y → ℝ := fun S => (normalize (kdppWeight κ φ K) S).toReal
    let πbar : Finset Y → ℝ := fun S => (∫⁻ ω, out ω S ∂poolLaw q N).toReal
    let m₁ : ℝ := (secondMomentU1 q K (kdppWeight κ φ K)).toReal
    let a : ℕ → ℝ := fun c => (overlapCoeff K N c).toReal
    let g : ℕ → ℝ := fun c => ((K : ℝ) ^ c / (K.descFactorial c : ℝ)) ^ 2 * m₁ ^ c
    let varU₁ : ℝ :=
      variance (fun y => (inclRatio q K (kdppWeight κ φ K) y).toReal) (poolLaw q 1)
    let θ : ℝ := (K : ℝ) ^ 2 * m₁ / ((N : ℝ) - K + 1)
    HasSum πbar 1 ∧
    kl p πbar ≤ ENNReal.ofReal (Real.log (∑ c ∈ range (K + 1), a c * g c)) ∧
    ∑ c ∈ range (K + 1), a c * g c = 1 + a 1 * varU₁ + ∑ c ∈ Icc 2 K, a c * (g c - 1) ∧
    Real.log (∑ c ∈ range (K + 1), a c * g c) ≤ Real.log (Real.exp θ - 1 + a 0) := by
  intro p πbar m₁ a g varU₁ θ
  have hb := poolDPP_wellBehaved κ φ hκ q hq hK hKN hZ0 hZtop hm1 (N := N)
  obtain ⟨h1, -, h6, -⟩ := fallback_bound hZ0 hZtop hb out
    (fun S => Measurable.of_discrete.aemeasurable) hout_sum hout_eq
  obtain ⟨hlog, heq, hexp⟩ := poolDPP_sharper_sum κ φ hκ q hq hK hKN hZ0 hZtop hm1 (N := N)
  exact ⟨h1, h6.trans (ENNReal.ofReal_le_ofReal hlog), heq, hexp⟩

/-- **`cor:kl-sum` for PoolDPP, redrawing until `Ẑ ≠ 0`.** Same conclusions as
`poolDPP_sharper_fallback_bound`, for the sampler that redraws the pool until `Ẑ ≠ 0`, whose
output law is `π̄(S) = π̄₀(S)/P(Ẑ ≠ 0)` (`redraw_bound`). -/
theorem poolDPP_sharper_redraw_bound (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
    (q : PMF Y) (hq : ∀ y, q y = 0 → κ y = 0) {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N)
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞)
    (hm1 : secondMomentU1 q K (kdppWeight κ φ K) ≠ ∞) :
    let fhat := poolDPPFhat κ φ q K N
    let p : Finset Y → ℝ := fun S => (normalize (kdppWeight κ φ K) S).toReal
    let πbar : Finset Y → ℝ := fun S =>
      (marginalLaw (poolLaw q N) fhat S / poolLaw q N {ω | sketchZ fhat ω ≠ 0}).toReal
    let m₁ : ℝ := (secondMomentU1 q K (kdppWeight κ φ K)).toReal
    let a : ℕ → ℝ := fun c => (overlapCoeff K N c).toReal
    let g : ℕ → ℝ := fun c => ((K : ℝ) ^ c / (K.descFactorial c : ℝ)) ^ 2 * m₁ ^ c
    let varU₁ : ℝ :=
      variance (fun y => (inclRatio q K (kdppWeight κ φ K) y).toReal) (poolLaw q 1)
    let θ : ℝ := (K : ℝ) ^ 2 * m₁ / ((N : ℝ) - K + 1)
    HasSum πbar 1 ∧
    kl p πbar ≤ ENNReal.ofReal (Real.log (∑ c ∈ range (K + 1), a c * g c)) ∧
    ∑ c ∈ range (K + 1), a c * g c = 1 + a 1 * varU₁ + ∑ c ∈ Icc 2 K, a c * (g c - 1) ∧
    Real.log (∑ c ∈ range (K + 1), a c * g c) ≤ Real.log (Real.exp θ - 1 + a 0) := by
  intro fhat p πbar m₁ a g varU₁ θ
  have hb := poolDPP_wellBehaved κ φ hκ q hq hK hKN hZ0 hZtop hm1 (N := N)
  obtain ⟨h1, -, h6, -⟩ := redraw_bound hZ0 hZtop hb
  obtain ⟨hlog, heq, hexp⟩ := poolDPP_sharper_sum κ φ hκ q hq hK hKN hZ0 hZtop hm1 (N := N)
  exact ⟨h1, h6.trans (ENNReal.ofReal_le_ofReal hlog), heq, hexp⟩

end PoolDPP

end DppLm
