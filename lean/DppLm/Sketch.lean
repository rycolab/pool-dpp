import Mathlib

/-!
# Sketches: definitions and inequalities

A sketch is a random unnormalized weight `f̂_ω` on a countable set `S` of outcomes. This file
defines what `thm:var-kl-mixture` (`VarianceBias.lean`) is stated with (`sketchZ`, `marginalLaw`,
`normalize`, `chiSq`, `kl`) and proves the inequalities its proof uses (Cauchy–Schwarz for
lower integrals, and two pointwise bounds on `p log(p/q)`).

## Setting

* `S` is a countable domain (in the paper, size-`K` sets of strings).
* `f : S → ℝ≥0∞` is the unnormalized target, `Z = ∑ f`, `p = f / Z`. We assume `0 < Z < ∞`.
* `μ` is any probability measure on a measurable space `Ω` (the sketch's internal randomness).
* `fhat : Ω → S → ℝ≥0∞` is the random unnormalized weight, `Ẑ(ω) = ∑ₛ fhat ω s` (`sketchZ`),
  `π̂_ω = fhat ω / Ẑ(ω)`, and `marginalLaw μ fhat s = E_ω[π̂_ω(s)]`. Since `0 / 0 = 0` in
  `ℝ≥0∞`, this is `π̄₀(s) = E_ω[π̂_ω(s); Ẑ ≠ 0]`, the part of the output law that does not
  come from the fallback.

## Divergences

Both divergences take values in `[0, ∞]`, so a finite bound on either one cannot hold vacuously.

* `kl p q` is Mathlib's `InformationTheory.klDiv` of the measures `discreteMeasure p` and
  `discreteMeasure q`, which put mass `p(s)` and `q(s)` on each point `s`. It is `∞` unless
  `p ≪ q` and `log(p/q)` is `p`-integrable.
* `chiSq p q` is `∑ₛ (p(s) - q(s))² / q(s)` summed in `[0, ∞]`, where `x / 0 = ∞` for `x > 0`.

For distributions `p` and `q`, each divergence is finite exactly when `q(s) = 0 → p(s) = 0` and
the usual real series converges (`kl_ne_top_iff`, `chiSq_ne_top_iff`), and it then equals that
series (`kl_eq_ofReal`, `chiSq_eq_ofReal`). The proofs bound the real series and convert at the
end. (In the real numbers Lean has `x / 0 = 0`, `Real.log 0 = 0`, and `0` for the sum of a
divergent series, which is why the real series alone would not do.)

## Proof route

The `χ²` bound is the paper's argument, with conditional Jensen replaced by the equivalent
Cauchy–Schwarz step `f(s)² = E[π̂(s) Ẑ]² ≤ E[π̂(s)] · E[π̂(s) Ẑ²]`, summed over `s`.
For `KL ≤ log(1 + χ²)` we avoid Jensen over an infinite sum: with `c = ∑ p²/π̄ = 1 + χ²`,
`log x ≤ x - 1` at `x = p/(c π̄)` gives `∑ p log(p/π̄) - log c ≤ ∑ (p²/(c π̄) - p) = 0`.
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace DppLm

section Defs

variable {Ω S : Type*} [MeasurableSpace Ω]

/-- Sketched normalizer `Ẑ(ω) = ∑ₛ fhat ω s`. -/
noncomputable def sketchZ (fhat : Ω → S → ℝ≥0∞) (ω : Ω) : ℝ≥0∞ := ∑' s, fhat ω s

/-- Marginal output law `π̄(s) = E_ω[fhat ω s / Ẑ(ω)]` (as an extended nonnegative real). -/
noncomputable def marginalLaw (μ : Measure Ω) (fhat : Ω → S → ℝ≥0∞) (s : S) : ℝ≥0∞ :=
  ∫⁻ ω, fhat ω s / sketchZ fhat ω ∂μ

/-- Target distribution `p(s) = f(s) / ∑ₜ f(t)` (as an extended nonnegative real). -/
noncomputable def normalize (f : S → ℝ≥0∞) (s : S) : ℝ≥0∞ := f s / ∑' t, f t

/-- The measure on a countable set `S` with mass `p(s)` at each point `s` (the density `p`
with respect to counting measure; negative values count as `0`). -/
noncomputable def discreteMeasure [MeasurableSpace S] (p : S → ℝ) : Measure S :=
  Measure.count.withDensity fun s => ENNReal.ofReal (p s)

/-- `χ²(p ‖ q) = ∑ₛ (p(s) - q(s))² / q(s)`, summed in `[0, ∞]`: a term with
`q(s) = 0 ≠ p(s)` is `∞`, and one with `q(s) = p(s) = 0` is `0`. -/
noncomputable def chiSq (p q : S → ℝ) : ℝ≥0∞ :=
  ∑' s, ENNReal.ofReal ((p s - q s) ^ 2) / ENNReal.ofReal (q s)

/-- `KL(p ‖ q)`: Mathlib's `InformationTheory.klDiv` of `discreteMeasure p` and
`discreteMeasure q`. For probability distributions it is `∑ₛ p(s) log(p(s)/q(s))` when
`q(s) = 0 → p(s) = 0` and the series converges absolutely, and `∞` otherwise
(`kl_eq_ofReal`, `kl_ne_top_iff`). -/
noncomputable def kl [MeasurableSpace S] (p q : S → ℝ) : ℝ≥0∞ :=
  InformationTheory.klDiv (discreteMeasure p) (discreteMeasure q)

end Defs

section Divergences

variable {S : Type*}

lemma chiSq_term_eq {p q : ℝ} (hq : 0 ≤ q) (habs : q = 0 → p = 0) :
    ENNReal.ofReal ((p - q) ^ 2) / ENNReal.ofReal q = ENNReal.ofReal ((p - q) ^ 2 / q) := by
  rcases hq.eq_or_lt with h | h
  · rw [← h, habs h.symm]; simp
  · rw [ENNReal.ofReal_div_of_pos h]

/-- `chiSq` is the usual real series when `q(s) = 0 → p(s) = 0` and the series converges. -/
lemma chiSq_eq_ofReal {p q : S → ℝ} (hq : ∀ s, 0 ≤ q s) (habs : ∀ s, q s = 0 → p s = 0)
    (hsum : Summable fun s => (p s - q s) ^ 2 / q s) :
    chiSq p q = ENNReal.ofReal (∑' s, (p s - q s) ^ 2 / q s) := by
  rw [ENNReal.ofReal_tsum_of_nonneg (fun s => div_nonneg (sq_nonneg _) (hq s)) hsum]
  exact tsum_congr fun s => chiSq_term_eq (hq s) (habs s)

/-- `chiSq p q < ∞` iff `q(s) = 0 → p(s) = 0` and the series `∑ₛ (p(s) - q(s))²/q(s)`
converges. -/
lemma chiSq_ne_top_iff {p q : S → ℝ} (hq : ∀ s, 0 ≤ q s) :
    chiSq p q ≠ ∞ ↔
      (∀ s, q s = 0 → p s = 0) ∧ Summable fun s => (p s - q s) ^ 2 / q s := by
  constructor
  · intro h
    have habs : ∀ s, q s = 0 → p s = 0 := by
      intro s hs
      by_contra hp
      apply h
      refine ENNReal.eq_top_of_forall_nnreal_le fun r => ?_
      refine le_trans ?_ (le_tsum _ s)
      have hp2 : ENNReal.ofReal ((p s - q s) ^ 2) ≠ 0 := by
        rw [hs, sub_zero]
        exact (ENNReal.ofReal_pos.2 (sq_pos_of_ne_zero hp)).ne'
      rw [hs, ENNReal.ofReal_zero, ENNReal.div_zero (by rwa [hs] at hp2)]
      exact le_top
    refine ⟨habs, ?_⟩
    refine (ENNReal.summable_toReal h).congr fun s => ?_
    rw [chiSq_term_eq (hq s) (habs s), ENNReal.toReal_ofReal (div_nonneg (sq_nonneg _) (hq s))]
  · rintro ⟨habs, hsum⟩
    rw [chiSq_eq_ofReal hq habs hsum]
    exact ENNReal.ofReal_ne_top

variable [Countable S] [MeasurableSpace S] [MeasurableSingletonClass S]

omit [Countable S] in
lemma discreteMeasure_singleton (p : S → ℝ) (s : S) :
    discreteMeasure p {s} = ENNReal.ofReal (p s) := by
  rw [discreteMeasure, withDensity_apply _ (measurableSet_singleton s), Measure.restrict_singleton,
    Measure.count_singleton, one_smul, lintegral_dirac]

omit [Countable S] in
lemma discreteMeasure_univ {p : S → ℝ} (hp : ∀ s, 0 ≤ p s) (hp1 : HasSum p 1) :
    discreteMeasure p Set.univ = 1 := by
  rw [discreteMeasure, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
    lintegral_count, ← ENNReal.ofReal_tsum_of_nonneg hp hp1.summable, hp1.tsum_eq,
    ENNReal.ofReal_one]

/-- A function is integrable for `discreteMeasure p` iff `∑ₛ p(s) |h(s)|` converges. -/
lemma integrable_discreteMeasure_iff {p : S → ℝ} (hp : ∀ s, 0 ≤ p s) {h : S → ℝ} :
    Integrable h (discreteMeasure p) ↔ Summable fun s => p s * |h s| := by
  have hnn : ∀ s, 0 ≤ p s * |h s| := fun s => mul_nonneg (hp s) (abs_nonneg _)
  have hlint : ∫⁻ s, ‖h s‖ₑ ∂discreteMeasure p = ∑' s, ENNReal.ofReal (p s * |h s|) := by
    rw [lintegral_countable']
    refine tsum_congr fun s => ?_
    rw [discreteMeasure_singleton, Real.enorm_eq_ofReal_abs, ← ENNReal.ofReal_mul (abs_nonneg _),
      mul_comm]
  constructor
  · intro hi
    have hfin : ∑' s, ENNReal.ofReal (p s * |h s|) ≠ ∞ := hlint ▸ hi.2.ne
    refine (ENNReal.summable_toReal hfin).congr fun s => ?_
    rw [ENNReal.toReal_ofReal (hnn s)]
  · intro hs
    refine ⟨(measurable_of_countable h).aestronglyMeasurable, ?_⟩
    rw [hasFiniteIntegral_iff_enorm, hlint, ← ENNReal.ofReal_tsum_of_nonneg hnn hs]
    exact ENNReal.ofReal_lt_top

/-- When `q(s) = 0 → p(s) = 0`, `discreteMeasure p ≪ discreteMeasure q`, with density `p/q`, so
the log-likelihood ratio is `log(p/q)` almost everywhere. -/
lemma llr_discreteMeasure {p q : S → ℝ} (hp : ∀ s, 0 ≤ p s) (hq : ∀ s, 0 ≤ q s)
    (hq1 : HasSum q 1) (habs : ∀ s, q s = 0 → p s = 0) :
    discreteMeasure p ≪ discreteMeasure q ∧
    llr (discreteMeasure p) (discreteMeasure q) =ᵐ[discreteMeasure p]
      fun s => Real.log (p s / q s) := by
  set g : S → ℝ≥0∞ := fun s => ENNReal.ofReal (p s) / ENNReal.ofReal (q s)
  have : IsFiniteMeasure (discreteMeasure q) :=
    ⟨by rw [discreteMeasure_univ hq hq1]; exact ENNReal.one_lt_top⟩
  have heq : discreteMeasure p = (discreteMeasure q).withDensity g := by
    rw [Measure.ext_iff_singleton]
    intro s
    rw [discreteMeasure_singleton, withDensity_apply _ (measurableSet_singleton s),
      Measure.restrict_singleton, discreteMeasure_singleton, lintegral_smul_measure,
      lintegral_dirac, smul_eq_mul, mul_comm]
    rcases (hq s).eq_or_lt with h | h
    · simp [g, ← h, habs s h.symm]
    · rw [ENNReal.div_mul_cancel (ENNReal.ofReal_pos.2 h).ne' ENNReal.ofReal_ne_top]
  have hac : discreteMeasure p ≪ discreteMeasure q :=
    heq ▸ withDensity_absolutelyContinuous _ _
  refine ⟨hac, ?_⟩
  have hrn : (discreteMeasure p).rnDeriv (discreteMeasure q) =ᵐ[discreteMeasure q] g := by
    rw [heq]
    exact Measure.rnDeriv_withDensity _ (measurable_of_countable g)
  filter_upwards [hac hrn] with s hs
  rw [llr, hs]
  simp only [g, ENNReal.toReal_div, ENNReal.toReal_ofReal (hp s), ENNReal.toReal_ofReal (hq s)]

/-- `kl` is the usual sum `∑ₛ p(s) log(p(s)/q(s))` for distributions with
`q(s) = 0 → p(s) = 0` whose series converges. -/
lemma kl_eq_ofReal {p q : S → ℝ} (hp : ∀ s, 0 ≤ p s) (hq : ∀ s, 0 ≤ q s)
    (hp1 : HasSum p 1) (hq1 : HasSum q 1) (habs : ∀ s, q s = 0 → p s = 0)
    (hsum : Summable fun s => p s * Real.log (p s / q s)) :
    kl p q = ENNReal.ofReal (∑' s, p s * Real.log (p s / q s)) := by
  obtain ⟨hac, hllr⟩ := llr_discreteMeasure hp hq hq1 habs
  have hint' : Integrable (fun s => Real.log (p s / q s)) (discreteMeasure p) := by
    rw [integrable_discreteMeasure_iff hp]
    refine hsum.abs.congr fun s => ?_
    rw [abs_mul, abs_of_nonneg (hp s)]
  have hint : Integrable (llr (discreteMeasure p) (discreteMeasure q)) (discreteMeasure p) :=
    hint'.congr hllr.symm
  rw [kl, InformationTheory.klDiv_of_ac_of_integrable hac hint, integral_congr_ae hllr,
    integral_countable hint', measureReal_def, measureReal_def, discreteMeasure_univ hp hp1,
    discreteMeasure_univ hq hq1]
  simp only [ENNReal.toReal_one, add_sub_cancel_right, measureReal_def, discreteMeasure_singleton,
    ENNReal.toReal_ofReal (hp _), smul_eq_mul]

/-- For distributions `p` and `q` on a countable set, `kl p q < ∞` iff `q(s) = 0 → p(s) = 0` and
the series `∑ₛ p(s) log(p(s)/q(s))` converges. -/
lemma kl_ne_top_iff {p q : S → ℝ} (hp : ∀ s, 0 ≤ p s) (hq : ∀ s, 0 ≤ q s)
    (hp1 : HasSum p 1) (hq1 : HasSum q 1) :
    kl p q ≠ ∞ ↔
      (∀ s, q s = 0 → p s = 0) ∧ Summable fun s => p s * Real.log (p s / q s) := by
  constructor
  · intro h
    obtain ⟨hac, hint⟩ := InformationTheory.klDiv_ne_top_iff.1 h
    have habs : ∀ s, q s = 0 → p s = 0 := by
      intro s hs
      have h0 : discreteMeasure q {s} = 0 := by rw [discreteMeasure_singleton, hs]; simp
      have := hac h0
      rw [discreteMeasure_singleton, ENNReal.ofReal_eq_zero] at this
      exact le_antisymm this (hp s)
    refine ⟨habs, ?_⟩
    obtain ⟨-, hllr⟩ := llr_discreteMeasure hp hq hq1 habs
    have hint' := (integrable_discreteMeasure_iff hp).1 (hint.congr hllr)
    refine Summable.of_abs (hint'.congr fun s => ?_)
    rw [abs_mul, abs_of_nonneg (hp s)]
  · rintro ⟨habs, hsum⟩
    rw [kl_eq_ofReal hp hq hp1 hq1 habs hsum]
    exact ENNReal.ofReal_ne_top

/-- A check of the definition: for `p = (½, ½)` and `q = (1, 0)` on `Bool`, `p` puts mass where
`q` has none, and `kl p q = ∞`. -/
example : kl (S := Bool) (fun _ => 1 / 2) (fun b => if b then 1 else 0) = ∞ := by
  by_contra h
  have hp1 : HasSum (fun _ : Bool => (1 / 2 : ℝ)) 1 := by
    convert hasSum_fintype (fun _ : Bool => (1 / 2 : ℝ)); norm_num
  have hq1 : HasSum (fun b : Bool => if b then (1 : ℝ) else 0) 1 := by
    convert hasSum_fintype (fun b : Bool => if b then (1 : ℝ) else 0); simp
  have := ((kl_ne_top_iff (fun _ => by norm_num) (fun b => by cases b <;> norm_num)
    hp1 hq1).1 h).1 false (by simp)
  norm_num at this

end Divergences

section Core

variable {Ω S : Type*} [MeasurableSpace Ω] [Countable S]
variable {μ : Measure Ω} {f : S → ℝ≥0∞} {fhat : Ω → S → ℝ≥0∞}

lemma ennreal_div_sq (a b : ℝ≥0∞) : (a / b) ^ 2 = a ^ 2 / b ^ 2 := by
  rw [div_eq_mul_inv, div_eq_mul_inv, mul_pow, ENNReal.inv_pow]

omit [Countable S] in
/-- `HasSum` of `toReal` with the value written as `toReal` of the `ENNReal` sum. -/
lemma hasSum_toReal_of_ne_top {f : S → ℝ≥0∞} (h : ∑' s, f s ≠ ∞) :
    HasSum (fun s => (f s).toReal) (∑' s, f s).toReal := by
  have hf : ∀ s, f s ≠ ∞ := fun s => ne_top_of_le_ne_top h (le_tsum _ s)
  rw [ENNReal.tsum_toReal_eq hf]
  exact ENNReal.hasSum_toReal h

/-- Cauchy–Schwarz for lower Lebesgue integrals, in the form we need:
`(∫ g·Y)² ≤ (∫ g) · (∫ g·Y²)`. -/
lemma lintegral_mul_sq_le {g Y : Ω → ℝ≥0∞} (hg : AEMeasurable g μ) (hY : AEMeasurable Y μ) :
    (∫⁻ ω, g ω * Y ω ∂μ) ^ 2 ≤ (∫⁻ ω, g ω ∂μ) * ∫⁻ ω, g ω * Y ω ^ 2 ∂μ := by
  set a : Ω → ℝ≥0∞ := fun ω => g ω ^ (1 / 2 : ℝ)
  set b : Ω → ℝ≥0∞ := fun ω => g ω ^ (1 / 2 : ℝ) * Y ω
  have ha : AEMeasurable a μ := hg.pow_const _
  have hb : AEMeasurable b μ := (hg.pow_const _).mul hY
  have hab : ∀ ω, (a * b) ω = g ω * Y ω := by
    intro ω
    simp only [Pi.mul_apply, a, b, ← mul_assoc]
    rw [← ENNReal.rpow_add_of_nonneg _ _ (by norm_num) (by norm_num)]
    norm_num
  have ha2 : ∀ ω, a ω ^ (2 : ℝ) = g ω := by
    intro ω
    simp only [a, ← ENNReal.rpow_mul]
    norm_num
  have hb2 : ∀ ω, b ω ^ (2 : ℝ) = g ω * Y ω ^ 2 := by
    intro ω
    simp only [b]
    rw [ENNReal.mul_rpow_of_nonneg _ _ (by norm_num), ← ENNReal.rpow_mul]
    norm_num
  have H := ENNReal.lintegral_mul_le_Lp_mul_Lq μ Real.HolderConjugate.two_two ha hb
  simp only [hab, ha2, hb2] at H
  calc (∫⁻ ω, g ω * Y ω ∂μ) ^ 2
      ≤ ((∫⁻ ω, g ω ∂μ) ^ (1 / 2 : ℝ) * (∫⁻ ω, g ω * Y ω ^ 2 ∂μ) ^ (1 / 2 : ℝ)) ^ 2 := by
        gcongr
    _ = (∫⁻ ω, g ω ∂μ) * ∫⁻ ω, g ω * Y ω ^ 2 ∂μ := by
        rw [mul_pow, ← ENNReal.rpow_natCast, ← ENNReal.rpow_natCast, ← ENNReal.rpow_mul,
          ← ENNReal.rpow_mul]
        norm_num

end Core

section Real

variable {Ω S : Type*} [MeasurableSpace Ω] [Countable S]
variable {μ : Measure Ω} {f : S → ℝ≥0∞} {fhat : Ω → S → ℝ≥0∞}

/-- Pointwise lower bound `p log(p/q) ≥ p - q`. -/
lemma mul_log_div_ge {p q : ℝ} (hp : 0 ≤ p) (hq : 0 ≤ q) (hpq : q = 0 → p = 0) :
    p - q ≤ p * Real.log (p / q) := by
  rcases hp.eq_or_lt with rfl | hp'
  · simpa using hq
  have hq' : 0 < q := hq.lt_of_ne (fun h => hp'.ne' (hpq h.symm))
  have h := Real.log_le_sub_one_of_pos (div_pos hq' hp')
  rw [Real.log_div hq'.ne' hp'.ne', ← neg_sub, ← Real.log_div hp'.ne' hq'.ne'] at h
  have : p * (1 - q / p) ≤ p * Real.log (p / q) := by
    apply mul_le_mul_of_nonneg_left _ hp; linarith
  rwa [mul_sub, mul_one, mul_div_cancel₀ _ hp'.ne'] at this

/-- Pointwise upper bound `p log(p/q) - p log c ≤ p²/(c q) - p`. -/
lemma mul_log_div_le {p q c : ℝ} (hp : 0 ≤ p) (hq : 0 ≤ q) (hc : 0 < c)
    (hpq : q = 0 → p = 0) :
    p * Real.log (p / q) - p * Real.log c ≤ p ^ 2 / q / c - p := by
  rcases hp.eq_or_lt with rfl | hp'
  · simp
  have hq' : 0 < q := hq.lt_of_ne (fun h => hp'.ne' (hpq h.symm))
  have hx : 0 < p / q / c := div_pos (div_pos hp' hq') hc
  have h := Real.log_le_sub_one_of_pos hx
  rw [Real.log_div (div_pos hp' hq').ne' hc.ne'] at h
  have := mul_le_mul_of_nonneg_left h hp
  calc p * Real.log (p / q) - p * Real.log c = p * (Real.log (p / q) - Real.log c) := by ring
    _ ≤ p * (p / q / c - 1) := this
    _ = p ^ 2 / q / c - p := by field_simp

end Real

section Variance

/-- `Var(X.toReal) = E[X²] - E[X]²` for a nonnegative extended-real variable with
finite second moment. -/
lemma variance_toReal_eq {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {G : Ω → ℝ≥0∞} (hG : AEMeasurable G μ)
    (h2 : ∫⁻ ω, G ω ^ 2 ∂μ ≠ ∞) :
    variance (fun ω => (G ω).toReal) μ =
      (∫⁻ ω, G ω ^ 2 ∂μ).toReal - ((∫⁻ ω, G ω ∂μ).toReal) ^ 2 := by
  have hlt : ∀ᵐ ω ∂μ, G ω < ∞ := by
    have h := ae_lt_top' (hG.pow_const 2) h2
    filter_upwards [h] with ω hω
    exact (ENNReal.pow_lt_top_iff.1 hω).resolve_right two_ne_zero
  have hX_meas : AEMeasurable (fun ω => (G ω).toReal) μ := hG.ennreal_toReal
  have hX2_int : Integrable (fun ω => (G ω).toReal ^ 2) μ := by
    have := integrable_toReal_of_lintegral_ne_top (hG.pow_const 2) h2
    simpa [ENNReal.toReal_pow] using this
  have hX_L2 : MemLp (fun ω => (G ω).toReal) 2 μ :=
    (memLp_two_iff_integrable_sq hX_meas.aestronglyMeasurable).2 hX2_int
  have hEX : μ[fun ω => (G ω).toReal] = (∫⁻ ω, G ω ∂μ).toReal := integral_toReal hG hlt
  have hEX2 : μ[(fun ω => (G ω).toReal) ^ 2] = (∫⁻ ω, G ω ^ 2 ∂μ).toReal := by
    have h2' : ∀ᵐ ω ∂μ, G ω ^ 2 < ∞ := by
      filter_upwards [hlt] with ω h using ENNReal.pow_lt_top h
    rw [← integral_toReal (hG.pow_const 2) h2']
    simp [ENNReal.toReal_pow]
  rw [variance_eq_sub hX_L2, hEX, hEX2]

/-- `eVar(X.toReal) = E[X²] - E[X]²` in `[0, ∞]`, for a nonnegative extended-real variable
with finite mean; both sides are `∞` when `E[X²] = ∞`. Also `E[X]² ≤ E[X²]`. -/
lemma evariance_toReal_eq {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {G : Ω → ℝ≥0∞} (hG : AEMeasurable G μ)
    (h1 : ∫⁻ ω, G ω ∂μ ≠ ∞) :
    evariance (fun ω => (G ω).toReal) μ = ∫⁻ ω, G ω ^ 2 ∂μ - (∫⁻ ω, G ω ∂μ) ^ 2 ∧
      (∫⁻ ω, G ω ∂μ) ^ 2 ≤ ∫⁻ ω, G ω ^ 2 ∂μ := by
  have hlt : ∀ᵐ ω ∂μ, G ω < ∞ := ae_lt_top' hG h1
  have hX_meas : AEMeasurable (fun ω => (G ω).toReal) μ := hG.ennreal_toReal
  by_cases h2 : ∫⁻ ω, G ω ^ 2 ∂μ = ∞
  · have hnot : ¬ MemLp (fun ω => (G ω).toReal) 2 μ := by
      intro hm
      have hi := (memLp_two_iff_integrable_sq hX_meas.aestronglyMeasurable).1 hm
      have hlt2 : ∫⁻ ω, G ω ^ 2 ∂μ < ∞ := by
        refine lt_of_eq_of_lt (lintegral_congr_ae ?_) hi.lintegral_lt_top
        filter_upwards [hlt] with ω hω
        rw [ENNReal.ofReal_pow ENNReal.toReal_nonneg, ENNReal.ofReal_toReal hω.ne]
      exact hlt2.ne h2
    rw [(evariance_eq_top_iff hX_meas.aestronglyMeasurable).2 hnot, h2,
      ENNReal.top_sub (ENNReal.pow_ne_top h1)]
    exact ⟨rfl, le_top⟩
  · have hX2_int : Integrable (fun ω => (G ω).toReal ^ 2) μ := by
      have := integrable_toReal_of_lintegral_ne_top (hG.pow_const 2) h2
      simpa [ENNReal.toReal_pow] using this
    have hX_L2 : MemLp (fun ω => (G ω).toReal) 2 μ :=
      (memLp_two_iff_integrable_sq hX_meas.aestronglyMeasurable).2 hX2_int
    have hV := variance_toReal_eq hG h2
    have hle : ((∫⁻ ω, G ω ∂μ).toReal) ^ 2 ≤ (∫⁻ ω, G ω ^ 2 ∂μ).toReal := by
      have := variance_nonneg (fun ω => (G ω).toReal) μ
      linarith
    refine ⟨?_, ?_⟩
    · rw [← ofReal_variance hX_L2, hV, ENNReal.ofReal_sub _ (sq_nonneg _),
        ENNReal.ofReal_toReal h2, ENNReal.ofReal_pow ENNReal.toReal_nonneg,
        ENNReal.ofReal_toReal h1]
    · rw [← ENNReal.toReal_le_toReal (ENNReal.pow_ne_top h1) h2, ENNReal.toReal_pow]
      exact hle

end Variance

end DppLm
