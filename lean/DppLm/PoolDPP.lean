import DppLm.VarianceBias
import DppLm.FiniteSample
import DppLm.Determinant
import DppLm.NegCorr

/-!
# `thm:zfree` for the paper's sketch

The paper's objects: the quality–diversity kernel `L`, the `K`-DPP target
`f(S) = det(L_S) 1{|S| = K}` (`kdppWeight`), and the sketch `f̂ = α ∑ det(L̂_I)` with
`L̂ = B̂ᵀB̂` (`poolDPPFhat`, equal to the general sketch `poolFhat` almost surely by
`paperFhat_eq`). The pool is drawn i.i.d. from `q`, and a run with `Ẑ = 0` uses any fallback;
the variance is under this plain i.i.d. pool law, the law under which `thm:zfree` computes it.

* `kdpp_poolSetting` puts the paper's `K`-DPP into the setting of `FiniteSample.lean`.
* `poolDPP_unbiased`, `poolDPP_wellBehaved`: the two sentences of `prop:PoolDPP-well-behaved`
  for the paper's sketch (unbiased; well-behaved when `ρ < ∞`).
* `poolDPP_variance_eq` identifies the sketch's variance with the one in `zfree_variance`.
* `poolDPP_secondMoment_le_sharper`, `poolDPP_secondMoment_le_exp`: `E[Ẑ²] < ∞`, and
  `E[Ẑ²]/Z²` is at most the sum of `cor:kl-sum` and at most `e^θ - 1 + a₀`.
* `poolDPP_fallback_bound`, `poolDPP_redraw_bound`: the KL bound `KL ≤ log(e^θ - 1 + a₀) ≤ θ` of
  `thm:zfree` (`θ = K² ρ/(N - K + 1)`, `ρ = E_q[u₁²]`, `a₀ = C(N-K,K)/C(N,K)`), for any
  fallback and for redrawing. Like the paper, they assume `ρ < ∞` and use `ρ` itself.

The theorems take negative correlation from `kdpp_negCorr` (`NegCorr.lean`), so their
hypotheses are exactly the paper's: `κ ≥ 0`, `q(y) = 0 → κ(y) = 0`, `K ≥ 1`, `N ≥ K`,
`0 < Z < ∞`, and `ρ < ∞`. No assumption on the embeddings is needed.
-/

open Finset MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace DppLm

section PoolDPP

variable {Y : Type*} [DecidableEq Y] [Countable Y] [MeasurableSpace Y]
  [MeasurableSingletonClass Y] {D : ℕ}

omit [DecidableEq Y] in
/-- Almost every pool consists of strings with positive proposal mass. -/
lemma ae_poolLaw_pos (p : PMF Y) (N : ℕ) : ∀ᵐ ω ∂poolLaw p N, ∀ i, p (ω i) ≠ 0 := by
  rw [ae_iff]
  rw [← lintegral_indicator_one (Set.Countable.measurableSet (Set.to_countable _)),
    lintegral_poolLaw]
  refine tsum_eq_zero_iff.2 fun ω => ?_
  by_cases hω : ω ∈ {ω : Fin N → Y | ¬∀ i, p (ω i) ≠ 0}
  · simp only [Set.mem_ofPred_eq, not_forall, not_not] at hω
    obtain ⟨i, hi⟩ := hω
    rw [show tupleWeight (⇑p) ω = 0 from Finset.prod_eq_zero (Finset.mem_univ i) hi, zero_mul]
  · rw [Set.indicator_of_notMem hω, mul_zero]

omit [Countable Y] [MeasurableSpace Y] [MeasurableSingletonClass Y] in
/-- The paper's `K`-DPP target `f(S) = det(L_S) 1{|S| = K}` satisfies the standing hypotheses
of `thm:zfree` (`PoolSetting`). -/
lemma kdpp_poolSetting (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
    (q : PMF Y) (hq : ∀ y, q y = 0 → κ y = 0) {K : ℕ}
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞) :
    PoolSetting q K (kdppWeight κ φ K) :=
  { q_sum := PMF.tsum_coe q
    f_card := fun S hS => by
      unfold kdppWeight at hS
      by_contra h
      exact hS (by simp [h])
    f_supp := fun S hS => kdppWeight_supp hκ hq S hS
    Z_ne_zero := hZ0
    Z_ne_top := hZtop }

/-- The paper's sketch `f̂_ω(S) = α ∑_I det(L̂_I)`, as an `ℝ≥0∞`-valued weight. -/
noncomputable def poolDPPFhat (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (q : PMF Y) (K N : ℕ) :
    (Fin N → Y) → Finset Y → ℝ≥0∞ :=
  fun ω S => ENNReal.ofReal (paperFhat κ φ (fun y => (q y).toReal) K N ω S)

/-- Under the plain i.i.d. pool law, the paper's sketch equals `poolFhat` almost surely. -/
lemma poolDPPFhat_ae (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y) (q : PMF Y)
    {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N) :
    (∀ᵐ ω ∂poolLaw q N, ∀ S, poolDPPFhat κ φ q K N ω S = poolFhat q K (kdppWeight κ φ K) ω S) ∧
    (∀ᵐ ω ∂poolLaw q N, sketchZ (poolDPPFhat κ φ q K N) ω = poolZ q K (kdppWeight κ φ K) ω) := by
  have hae : ∀ᵐ ω ∂poolLaw q N,
      ∀ S, poolDPPFhat κ φ q K N ω S = poolFhat q K (kdppWeight κ φ K) ω S := by
    filter_upwards [ae_poolLaw_pos q N] with ω hω
    exact fun S => paperFhat_eq hκ q hK hKN ω hω S
  refine ⟨hae, ?_⟩
  filter_upwards [hae] with ω hω
  rw [← sketchZ_poolFhat]
  unfold sketchZ
  exact tsum_congr hω

/-- `Var(Ẑ)` of the paper's sketch under the i.i.d. pool law, in `[0, ∞]`, is the variance of
the `U`-statistic `poolZ`, the quantity in `zfree_variance` (the identity of `thm:zfree`). -/
theorem poolDPP_variance_eq (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y) (q : PMF Y)
    {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N) :
    evariance (fun ω => (sketchZ (poolDPPFhat κ φ q K N) ω).toReal) (poolLaw q N) =
      evariance (fun ω => (poolZ q K (kdppWeight κ φ K) ω).toReal) (poolLaw q N) := by
  apply evariance_congr
  filter_upwards [(poolDPPFhat_ae κ φ hκ q hK hKN).2] with ω hω
  rw [hω]

/-- `E[Ẑ²] < ∞` and `E[Ẑ²]/Z² ≤ ∑_{c=0}^{K} a_c (K^c/(K)_c)² m₁^c` for the paper's sketch under
the i.i.d. pool law, for any `m₁ ≥ ρ = E_q[u₁²]`: `secondMoment_le_sharper` for the `K`-DPP.
This sum is the bound of `cor:kl-sum` before the log. -/
lemma poolDPP_secondMoment_le_sharper (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
    (q : PMF Y) (hq : ∀ y, q y = 0 → κ y = 0) {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N)
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞)
    (m₁ : ℝ≥0) (hm1 : secondMomentU1 q K (kdppWeight κ φ K) ≤ m₁) :
    ∫⁻ ω, sketchZ (poolDPPFhat κ φ q K N) ω ^ 2 ∂poolLaw q N ≠ ∞ ∧
    (∫⁻ ω, sketchZ (poolDPPFhat κ φ q K N) ω ^ 2 ∂poolLaw q N).toReal /
        ((∑' S, kdppWeight κ φ K S).toReal) ^ 2 ≤
      ∑ c ∈ range (K + 1), (overlapCoeff K N c).toReal *
        (((K : ℝ) ^ c / (K.descFactorial c : ℝ)) ^ 2 * (m₁ : ℝ) ^ c) := by
  have hs := kdpp_poolSetting κ φ hκ q hq hZ0 hZtop (K := K)
  have hsq : ∫⁻ ω, sketchZ (poolDPPFhat κ φ q K N) ω ^ 2 ∂poolLaw q N =
      ∫⁻ ω, poolZ q K (kdppWeight κ φ K) ω ^ 2 ∂poolLaw q N := by
    apply lintegral_congr_ae
    filter_upwards [(poolDPPFhat_ae κ φ hκ q hK hKN).2] with ω hω
    rw [hω]
  rw [hsq]
  exact secondMoment_le_sharper q hs hK hKN (kdpp_negCorr κ φ K hZ0 hZtop) m₁ hm1

/-- `E[Ẑ²] < ∞` and `E[Ẑ²]/Z² ≤ e^θ - 1 + a₀` for the paper's sketch under the i.i.d. pool law,
with `θ = K² m₁/(N - K + 1)`, `m₁ ≥ ρ = E_q[u₁²]` (`secondMomentU1`), and `a₀ = C(N-K,K)/C(N,K)`.
A step of the proof of `thm:zfree`: `poolDPP_secondMoment_le_sharper`, then
`sharperBound_le_exp`. -/
lemma poolDPP_secondMoment_le_exp (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
    (q : PMF Y) (hq : ∀ y, q y = 0 → κ y = 0) {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N)
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞)
    (m₁ : ℝ≥0) (hm1 : secondMomentU1 q K (kdppWeight κ φ K) ≤ m₁) :
    ∫⁻ ω, sketchZ (poolDPPFhat κ φ q K N) ω ^ 2 ∂poolLaw q N ≠ ∞ ∧
    (∫⁻ ω, sketchZ (poolDPPFhat κ φ q K N) ω ^ 2 ∂poolLaw q N).toReal /
        ((∑' S, kdppWeight κ φ K S).toReal) ^ 2 ≤
      Real.exp ((K : ℝ) ^ 2 * m₁ / ((N : ℝ) - K + 1)) - 1 + (overlapCoeff K N 0).toReal := by
  obtain ⟨hA_top, hratio⟩ :=
    poolDPP_secondMoment_le_sharper κ φ hκ q hq hK hKN hZ0 hZtop m₁ hm1
  exact ⟨hA_top, hratio.trans (sharperBound_le_exp hKN m₁.2)⟩

/-- **`prop:PoolDPP-well-behaved`, first sentence.** The paper's sketch is unbiased under the
plain i.i.d. pool law: `E[f̂_ω(S)] = det(L_S) 1{|S| = K}` for every `S`. -/
theorem poolDPP_unbiased (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
    (q : PMF Y) (hq : ∀ y, q y = 0 → κ y = 0) {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N)
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞) (S : Finset Y) :
    ∫⁻ ω, poolDPPFhat κ φ q K N ω S ∂poolLaw q N = kdppWeight κ φ K S := by
  have hs := kdpp_poolSetting κ φ hκ q hq hZ0 hZtop (K := K)
  obtain ⟨hunb, -, -, -, -, -⟩ := zfree_identity q hs hKN
  obtain ⟨hae, -⟩ := poolDPPFhat_ae κ φ hκ q hK hKN
  rw [← hunb S]
  apply lintegral_congr_ae
  filter_upwards [hae] with ω hω
  exact hω S

/-- **`prop:PoolDPP-well-behaved`, second sentence.** If `ρ = E_q[u₁²] < ∞`
(`secondMomentU1`), the paper's sketch is well-behaved under the plain i.i.d. pool law:
unbiased with `E[Ẑ²] < ∞`. No `Ẑ > 0` is needed. -/
theorem poolDPP_wellBehaved (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
    (q : PMF Y) (hq : ∀ y, q y = 0 → κ y = 0) {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N)
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞)
    (hm1 : secondMomentU1 q K (kdppWeight κ φ K) ≠ ∞) :
    WellBehaved (poolLaw q N) (kdppWeight κ φ K) (poolDPPFhat κ φ q K N) :=
  ⟨fun _ => Measurable.of_discrete.aemeasurable,
    poolDPP_unbiased κ φ hκ q hq hK hKN hZ0 hZtop,
    (poolDPP_secondMoment_le_exp κ φ hκ q hq hK hKN hZ0 hZtop _
      (ENNReal.coe_toNNReal hm1).ge).1⟩

/-- **`thm:zfree` for PoolDPP with any fallback rule.** The pool is drawn i.i.d. from `q`.
When `Ẑ ≠ 0` the sampler draws from `π̂_ω`; when `Ẑ = 0` it draws from any distribution
`out ω`. (For this sketch `Ẑ < ∞` always; the hypothesis allows any `out ω` at `Ẑ = ∞`.)
Then `π̄` sums to `1` and `KL(p ‖ π̄) ≤ log(e^θ - 1 + a₀) ≤ θ`, with `θ = K² ρ/(N - K + 1)`,
`ρ = E_q[u₁²]` (`secondMomentU1`; `m₁` in the statement), and `a₀ = C(N-K,K)/C(N,K)`. The KL
takes values in `[0, ∞]` (`kl`), so the bound also shows `π̄(S) = 0 → p(S) = 0` and that the KL
series converges. -/
theorem poolDPP_fallback_bound (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
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
    let θ : ℝ := (K : ℝ) ^ 2 * m₁ / ((N : ℝ) - K + 1)
    let a₀ : ℝ := (overlapCoeff K N 0).toReal
    HasSum πbar 1 ∧ kl p πbar ≤ ENNReal.ofReal (Real.log (Real.exp θ - 1 + a₀)) ∧
    Real.log (Real.exp θ - 1 + a₀) ≤ θ := by
  intro p πbar m₁ θ a₀
  have hb := poolDPP_wellBehaved κ φ hκ q hq hK hKN hZ0 hZtop hm1 (N := N)
  have hratio := (poolDPP_secondMoment_le_exp κ φ hκ q hq hK hKN hZ0 hZtop _
    (ENNReal.coe_toNNReal hm1).ge).2
  have h8 := hb.one_add_variance_div hZ0 hZtop
  obtain ⟨h1, -, h6, -⟩ := fallback_bound hZ0 hZtop hb out
    (fun S => Measurable.of_discrete.aemeasurable) hout_sum hout_eq
  have hVnn := div_nonneg
    (variance_nonneg (fun ω => (sketchZ (poolDPPFhat κ φ q K N) ω).toReal) (poolLaw q N))
    (sq_nonneg (∑' S, kdppWeight κ φ K S).toReal)
  have hθ0 : 0 ≤ θ := by
    have : (K : ℝ) ≤ N := by exact_mod_cast hKN
    exact div_nonneg (mul_nonneg (sq_nonneg _) ENNReal.toReal_nonneg) (by linarith)
  refine ⟨h1, h6.trans (ENNReal.ofReal_le_ofReal ?_),
    log_exp_sub_one_add_le hθ0 ENNReal.toReal_nonneg (overlapCoeff_zero_toReal_le_one hKN)⟩
  rw [h8]
  refine Real.log_le_log ?_ hratio
  rw [← h8]
  linarith

/-- **`thm:zfree` for PoolDPP, redrawing until `Ẑ ≠ 0`.** Same conclusions as
`poolDPP_fallback_bound`, for the sampler that redraws the pool until `Ẑ ≠ 0`, whose output
law is `π̄(S) = π̄₀(S)/P(Ẑ ≠ 0)` (`redraw_bound`). -/
theorem poolDPP_redraw_bound (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (hκ : ∀ y, 0 ≤ κ y)
    (q : PMF Y) (hq : ∀ y, q y = 0 → κ y = 0) {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N)
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞)
    (hm1 : secondMomentU1 q K (kdppWeight κ φ K) ≠ ∞) :
    let fhat := poolDPPFhat κ φ q K N
    let p : Finset Y → ℝ := fun S => (normalize (kdppWeight κ φ K) S).toReal
    let πbar : Finset Y → ℝ := fun S =>
      (marginalLaw (poolLaw q N) fhat S / poolLaw q N {ω | sketchZ fhat ω ≠ 0}).toReal
    let m₁ : ℝ := (secondMomentU1 q K (kdppWeight κ φ K)).toReal
    let θ : ℝ := (K : ℝ) ^ 2 * m₁ / ((N : ℝ) - K + 1)
    let a₀ : ℝ := (overlapCoeff K N 0).toReal
    HasSum πbar 1 ∧ kl p πbar ≤ ENNReal.ofReal (Real.log (Real.exp θ - 1 + a₀)) ∧
    Real.log (Real.exp θ - 1 + a₀) ≤ θ := by
  intro fhat p πbar m₁ θ a₀
  have hb := poolDPP_wellBehaved κ φ hκ q hq hK hKN hZ0 hZtop hm1 (N := N)
  have hratio := (poolDPP_secondMoment_le_exp κ φ hκ q hq hK hKN hZ0 hZtop _
    (ENNReal.coe_toNNReal hm1).ge).2
  have h8 := hb.one_add_variance_div hZ0 hZtop
  obtain ⟨h1, -, h6, -⟩ := redraw_bound hZ0 hZtop hb
  have hVnn := div_nonneg
    (variance_nonneg (fun ω => (sketchZ (poolDPPFhat κ φ q K N) ω).toReal) (poolLaw q N))
    (sq_nonneg (∑' S, kdppWeight κ φ K S).toReal)
  have hθ0 : 0 ≤ θ := by
    have : (K : ℝ) ≤ N := by exact_mod_cast hKN
    exact div_nonneg (mul_nonneg (sq_nonneg _) ENNReal.toReal_nonneg) (by linarith)
  refine ⟨h1, h6.trans (ENNReal.ofReal_le_ofReal ?_),
    log_exp_sub_one_add_le hθ0 ENNReal.toReal_nonneg (overlapCoeff_zero_toReal_le_one hKN)⟩
  rw [h8]
  refine Real.log_le_log ?_ hratio
  rw [← h8]
  linarith

end PoolDPP

end DppLm
