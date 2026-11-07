import DppLm.Sketch

/-!
# `thm:var-kl-mixture` (Variance controls bias) and its footnote `fn:wlog`

Paper statement (`thm:var-kl-mixture`): a sketch is supposed, without loss of generality, to have
`Ẑ > 0` almost surely, and its output law is `π̄(S) = E_ω[π̂_ω(S)]`. For any well-behaved sketch,
  `χ²(p ‖ π̄) ≤ Var(Ẑ)/Z²`  and  `KL(p ‖ π̄) ≤ log(1 + Var(Ẑ)/Z²) ≤ Var(Ẑ)/Z²`.

The footnote `fn:wlog` reduces the general case to this one. If `P(Ẑ = 0) > 0`, then `f̂_ω ≡ 0`
on those `ω`, and PoolDPP redraws `ω`; any other rule for these `ω` is a fallback, some
measurable distribution over sets. With
`γ = P(Ẑ > 0)`, which is positive, redrawing runs the sketch `(μ₊, f̂)`, where `μ₊ = μ[|Ẑ ≠ 0]`
is `μ` conditioned on `Ẑ > 0`. That sketch has `Ẑ > 0` almost surely, it is well-behaved for
`f/γ`, which defines the same target, and `E_{μ₊}[Ẑ²]/(Z/γ)² = γ E_μ[Ẑ²]/Z²`. A fallback's law
is at least `γ` times the redraw law, so it multiplies `1 + χ²` by at most `1/γ` and adds at most
`log(1/γ)` to `KL`. Either way, the theorem for `(μ₊, f̂)` gives its bounds for `(μ, f̂)`, with
`Var(Ẑ)` under `μ`.

The definitions (`sketchZ`, `marginalLaw`, `normalize`, `chiSq`, `kl`) are in `Sketch.lean`. Both
divergences take values in `[0, ∞]`, so a finite bound on either one also shows that
`π̄(S) = 0 → p(S) = 0` and that the usual series converges (`chiSq_ne_top_iff`, `kl_ne_top_iff`).
`marginalLaw μ fhat s` is `E_ω[f̂_ω(s)/Ẑ(ω)]` with `ℝ≥0∞` division, where `0/0 = 0`: it is the
paper's `π̄` when `Ẑ > 0` almost surely, and `π̄₀(s) = E[π̂_ω(s); Ẑ ≠ 0]` in general.

* `bias_bound`: `thm:var-kl-mixture`, for a well-behaved sketch with `Ẑ > 0` almost surely.
* `WellBehaved.measure_ne_zero`: `γ = P(Ẑ ≠ 0) > 0`.
* `WellBehaved.cond`: the sketch `(μ₊, f̂)`: `μ₊` is a probability measure, `Ẑ > 0` `μ₊`-almost
  surely, `0 < ∑ f/γ < ∞`, the sketch is well-behaved for `f/γ`, `f/γ` has the same target as `f`,
  and `E_{μ₊}[Ẑ²]/(Z/γ)² = γ E_μ[Ẑ²]/Z²`.
* `redraw_law_eq`: a redraw's output has a fresh run's law `Q`, so `Q = π̄₀ + P(Ẑ = 0) Q`; its only
  finite solution is `π̄₀/γ`, the output law of `(μ₊, f̂)` (`WellBehaved.marginalLaw_cond`).
* `one_add_chiSq_le_of_le`, `kl_le_add_of_le`: if `π̄ ≥ γ π̄₊` pointwise, then
  `1 + χ²(p ‖ π̄) ≤ (1 + χ²(p ‖ π̄₊))/γ` and `KL(p ‖ π̄) ≤ KL(p ‖ π̄₊) + log(1/γ)`.
* `redraw_bound`, `fallback_bound`: the bounds of `thm:var-kl-mixture`, with `Var(Ẑ)` under `μ`,
  for the redraw sampler and for any (measurable) fallback. Both are proved from `bias_bound` for
  `(μ₊, f̂)`, through `WellBehaved.cond` and the two comparison lemmas, as in the footnote.
* `WellBehaved.one_add_variance_div`: `1 + Var(Ẑ)/Z² = E[Ẑ²]/Z²`, used to turn these bounds into
  bounds in terms of `E[Ẑ²]` (`thm:zfree`, `cor:kl-sum`).
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace DppLm

section Defs

variable {Ω S : Type*} [MeasurableSpace Ω]

/-- The paper's well-behaved sketch: unbiased with a finite second moment, plus measurability
of `ω ↦ fhat ω s`, which the paper leaves implicit. `Ẑ = 0` may occur with any probability. -/
structure WellBehaved (μ : Measure Ω) (f : S → ℝ≥0∞) (fhat : Ω → S → ℝ≥0∞) : Prop where
  meas : ∀ s, AEMeasurable (fun ω => fhat ω s) μ
  unbiased : ∀ s, ∫⁻ ω, fhat ω s ∂μ = f s
  sq_finite : ∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ ≠ ∞

omit [MeasurableSpace Ω] in
/-- `Ẑ(ω) = 0` forces every weight `f̂_ω(s)` to be `0`. -/
lemma fhat_eq_zero_of_sketchZ_eq_zero {fhat : Ω → S → ℝ≥0∞} {ω : Ω}
    (h : sketchZ fhat ω = 0) (s : S) : fhat ω s = 0 :=
  le_antisymm ((le_tsum (fun t => fhat ω t) s).trans h.le) zero_le

end Defs

section General

variable {Ω S : Type*} [MeasurableSpace Ω] [Countable S]
variable {μ : Measure Ω} {f : S → ℝ≥0∞} {fhat : Ω → S → ℝ≥0∞}

lemma WellBehaved.aemeasurable_sketchZ (hb : WellBehaved μ f fhat) :
    AEMeasurable (sketchZ fhat) μ := by
  unfold sketchZ
  exact AEMeasurable.tsum hb.meas

lemma WellBehaved.ae_lt_top (hb : WellBehaved μ f fhat) : ∀ᵐ ω ∂μ, sketchZ fhat ω < ∞ := by
  have h := ae_lt_top' (hb.aemeasurable_sketchZ.pow_const 2) hb.sq_finite
  filter_upwards [h] with ω hω
  exact (ENNReal.pow_lt_top_iff.1 hω).resolve_right two_ne_zero

lemma WellBehaved.aemeasurable_pihat (hb : WellBehaved μ f fhat) (s : S) :
    AEMeasurable (fun ω => fhat ω s / sketchZ fhat ω) μ :=
  (hb.meas s).div hb.aemeasurable_sketchZ

/-- `f̂_ω(s) = π̂_ω(s) · Ẑ(ω)` almost surely, including when `Ẑ(ω) = 0`. -/
lemma WellBehaved.ae_div_mul (hb : WellBehaved μ f fhat) :
    ∀ᵐ ω ∂μ, ∀ s, fhat ω s / sketchZ fhat ω * sketchZ fhat ω = fhat ω s := by
  filter_upwards [hb.ae_lt_top] with ω htop s
  by_cases h0 : sketchZ fhat ω = 0
  · rw [fhat_eq_zero_of_sketchZ_eq_zero h0 s, h0, mul_zero]
  · exact ENNReal.div_mul_cancel h0 htop.ne

/-- Pointwise step: `f(s)² ≤ π̄₀(s) · E[π̂(s) Ẑ²]`. -/
lemma WellBehaved.sq_le (hb : WellBehaved μ f fhat) (s : S) :
    f s ^ 2 ≤ marginalLaw μ fhat s *
      ∫⁻ ω, fhat ω s / sketchZ fhat ω * sketchZ fhat ω ^ 2 ∂μ := by
  have hf : f s = ∫⁻ ω, fhat ω s / sketchZ fhat ω * sketchZ fhat ω ∂μ := by
    rw [← hb.unbiased s]
    apply lintegral_congr_ae
    filter_upwards [hb.ae_div_mul] with ω hω
    exact (hω s).symm
  rw [hf]
  exact lintegral_mul_sq_le (hb.aemeasurable_pihat s) hb.aemeasurable_sketchZ

/-- `∑ₛ E[π̂(s) Ẑ²] = E[Ẑ²]`, including the `ω` with `Ẑ(ω) = 0`. -/
lemma WellBehaved.tsum_weighted_sq (hb : WellBehaved μ f fhat) :
    ∑' s, ∫⁻ ω, fhat ω s / sketchZ fhat ω * sketchZ fhat ω ^ 2 ∂μ =
      ∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ := by
  rw [← lintegral_tsum (f := fun s ω => fhat ω s / sketchZ fhat ω * sketchZ fhat ω ^ 2)
    (fun s => (hb.aemeasurable_pihat s).mul (hb.aemeasurable_sketchZ.pow_const 2))]
  apply lintegral_congr_ae
  filter_upwards [hb.ae_lt_top] with ω htop
  rw [ENNReal.tsum_mul_right]
  simp_rw [div_eq_mul_inv]
  rw [ENNReal.tsum_mul_right]
  change sketchZ fhat ω * (sketchZ fhat ω)⁻¹ * _ = _
  by_cases h0 : sketchZ fhat ω = 0
  · rw [h0]; simp
  · rw [ENNReal.mul_inv_cancel h0 htop.ne, one_mul]

/-- `π̄₀(s) = 0 → f(s) = 0`. -/
lemma WellBehaved.abs_cont (hb : WellBehaved μ f fhat) {s : S}
    (hs : marginalLaw μ fhat s = 0) : f s = 0 := by
  have h := hb.sq_le s
  rw [hs, zero_mul, nonpos_iff_eq_zero] at h
  exact (pow_eq_zero_iff two_ne_zero).1 h

/-- `∑ₛ p(s)² / π̄₀(s) ≤ E[Ẑ²] / Z²`. -/
theorem WellBehaved.tsum_sq_div_le (hb : WellBehaved μ f fhat) :
    ∑' s, normalize f s ^ 2 / marginalLaw μ fhat s ≤
      (∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ) / (∑' t, f t) ^ 2 := by
  rw [← hb.tsum_weighted_sq]
  simp_rw [div_eq_mul_inv _ ((∑' t, f t) ^ 2)]
  rw [← ENNReal.tsum_mul_right]
  refine tsum_le_tsum (fun s => ?_)
  apply ENNReal.div_le_of_le_mul
  unfold normalize
  rw [ennreal_div_sq, div_eq_mul_inv, mul_right_comm]
  gcongr
  rw [mul_comm]
  exact hb.sq_le s

/-- `χ²` and KL bounds from the `ℝ≥0∞` inequality `∑ₛ P(s)²/Q(s) ≤ B`, for probability
distributions `P` and `Q` with `Q(s) = 0 → P(s) = 0`: `χ² ≤ B - 1` and `KL ≤ log B`. -/
theorem real_bounds [MeasurableSpace S] [MeasurableSingletonClass S] {P Q : S → ℝ≥0∞}
    (hP : ∑' s, P s = 1) (hQ : ∑' s, Q s = 1)
    (habs : ∀ s, Q s = 0 → P s = 0) {B : ℝ≥0∞} (hB : B ≠ ∞)
    (hc : ∑' s, P s ^ 2 / Q s ≤ B) :
    let p : S → ℝ := fun s => (P s).toReal
    let q : S → ℝ := fun s => (Q s).toReal
    1 ≤ B.toReal ∧ HasSum q 1 ∧ chiSq p q ≤ ENNReal.ofReal (B.toReal - 1) ∧
    kl p q ≤ ENNReal.ofReal (Real.log B.toReal) := by
  intro p q
  have hQ_ne_top : ∀ s, Q s ≠ ∞ := fun s =>
    ne_top_of_le_ne_top (by rw [hQ]; exact ENNReal.one_ne_top) (le_tsum Q s)
  have hc_top : ∑' s, P s ^ 2 / Q s ≠ ∞ := ne_top_of_le_ne_top hB hc
  have hq_nonneg : ∀ s, 0 ≤ q s := fun s => ENNReal.toReal_nonneg
  have hp_nonneg : ∀ s, 0 ≤ p s := fun s => ENNReal.toReal_nonneg
  have hq_hasSum : HasSum q 1 := by
    have := hasSum_toReal_of_ne_top (f := Q) (by rw [hQ]; simp)
    rwa [hQ, ENNReal.toReal_one] at this
  have hp_hasSum : HasSum p 1 := by
    have := hasSum_toReal_of_ne_top (f := P) (by rw [hP]; simp)
    rwa [hP, ENNReal.toReal_one] at this
  have habs' : ∀ s, q s = 0 → p s = 0 := by
    intro s hs
    have h0 : Q s = 0 := by
      rcases (ENNReal.toReal_eq_zero_iff _).1 hs with h | h
      · exact h
      · exact absurd h (hQ_ne_top s)
    simp [p, habs s h0]
  have hterm : ∀ s, (P s ^ 2 / Q s).toReal = p s ^ 2 / q s := by
    intro s
    rw [ENNReal.toReal_div, ENNReal.toReal_pow]
  have hc_hasSum : HasSum (fun s => p s ^ 2 / q s) (∑' s, P s ^ 2 / Q s).toReal := by
    have := hasSum_toReal_of_ne_top hc_top
    simpa only [hterm] using this
  set c := (∑' s, P s ^ 2 / Q s).toReal with hc_def
  have hcB : c ≤ B.toReal := ENNReal.toReal_mono hB hc
  -- χ² = c - 1
  have hchi_term : ∀ s, (p s - q s) ^ 2 / q s = p s ^ 2 / q s - 2 * p s + q s := by
    intro s
    by_cases hs : q s = 0
    · simp [hs, habs' s hs]
    · field_simp; ring
  have hchi_hasSum : HasSum (fun s => (p s - q s) ^ 2 / q s) (c - 2 * 1 + 1) := by
    simp_rw [hchi_term]
    exact (hc_hasSum.sub (hp_hasSum.mul_left 2)).add hq_hasSum
  have hchi_nonneg : 0 ≤ c - 2 * 1 + 1 :=
    hchi_hasSum.nonneg fun s => div_nonneg (sq_nonneg _) (hq_nonneg s)
  have hc_pos : 0 < c := by linarith
  -- KL: summability via the two pointwise bounds
  set t : S → ℝ := fun s => p s * Real.log (p s / q s)
  have hlow : ∀ s, p s - q s ≤ t s :=
    fun s => mul_log_div_ge (hp_nonneg s) (hq_nonneg s) (habs' s)
  have hup : ∀ s, t s - p s * Real.log c ≤ p s ^ 2 / q s / c - p s :=
    fun s => mul_log_div_le (hp_nonneg s) (hq_nonneg s) hc_pos (habs' s)
  have hL : Summable (fun s => p s - q s) := hp_hasSum.summable.sub hq_hasSum.summable
  have hU : HasSum (fun s => p s ^ 2 / q s / c - p s + p s * Real.log c)
      (c / c - 1 + 1 * Real.log c) :=
    ((hc_hasSum.div_const c).sub hp_hasSum).add (hp_hasSum.mul_right (Real.log c))
  have ht_summable : Summable t := by
    have hd : Summable (fun s => t s - (p s - q s)) := by
      refine Summable.of_nonneg_of_le (fun s => by linarith [hlow s]) (fun s => ?_)
        (hU.summable.sub hL)
      linarith [hup s]
    simpa using hd.add hL
  have hkl_le : ∑' s, t s ≤ Real.log c := by
    have h1 : ∑' s, t s ≤ c / c - 1 + 1 * Real.log c :=
      hasSum_le (fun s => by linarith [hup s]) ht_summable.hasSum hU
    rw [div_self hc_pos.ne'] at h1
    simpa using h1
  have hchi_le : ∑' s, (p s - q s) ^ 2 / q s ≤ B.toReal - 1 := by
    rw [hchi_hasSum.tsum_eq]
    linarith
  refine ⟨by linarith, hq_hasSum, ?_, ?_⟩
  · rw [chiSq_eq_ofReal hq_nonneg habs' hchi_hasSum.summable]
    exact ENNReal.ofReal_le_ofReal hchi_le
  · rw [kl_eq_ofReal hp_nonneg hq_nonneg hp_hasSum hq_hasSum habs' ht_summable]
    exact ENNReal.ofReal_le_ofReal (hkl_le.trans (Real.log_le_log hc_pos hcB))

/-- `1 + Var(Ẑ)/Z² = E[Ẑ²]/Z²` for a well-behaved sketch, since `E[Ẑ] = Z`. -/
lemma WellBehaved.one_add_variance_div [IsProbabilityMeasure μ] (hb : WellBehaved μ f fhat)
    (hZ0 : ∑' s, f s ≠ 0) (hZtop : ∑' s, f s ≠ ∞) :
    1 + variance (fun ω => (sketchZ fhat ω).toReal) μ / ((∑' s, f s).toReal) ^ 2 =
      (∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ).toReal / ((∑' s, f s).toReal) ^ 2 := by
  have hZpos : 0 < (∑' s, f s).toReal := ENNReal.toReal_pos hZ0 hZtop
  have hEZ : ∫⁻ ω, sketchZ fhat ω ∂μ = ∑' s, f s := by
    unfold sketchZ
    rw [lintegral_tsum hb.meas]
    exact tsum_congr hb.unbiased
  rw [variance_toReal_eq hb.aemeasurable_sketchZ hb.sq_finite, hEZ]
  field_simp
  ring

/-- `∑ₛ π̄₀(s) = P(Ẑ ≠ 0)`: the weights `π̂_ω` sum to `1` when `0 < Ẑ(ω) < ∞`, to `0` when
`Ẑ(ω) = 0`, and `Ẑ < ∞` almost surely. -/
lemma WellBehaved.tsum_marginalLaw (hb : WellBehaved μ f fhat) :
    ∑' s, marginalLaw μ fhat s = μ {ω | sketchZ fhat ω ≠ 0} := by
  unfold marginalLaw
  rw [← lintegral_tsum (fun s => hb.aemeasurable_pihat s)]
  have hnm : NullMeasurableSet {ω | sketchZ fhat ω ≠ 0} μ :=
    hb.aemeasurable_sketchZ.nullMeasurable (measurableSet_singleton 0).compl
  rw [← lintegral_indicator_one₀ hnm]
  apply lintegral_congr_ae
  filter_upwards [hb.ae_lt_top] with ω htop
  simp_rw [div_eq_mul_inv]
  rw [ENNReal.tsum_mul_right]
  change sketchZ fhat ω * (sketchZ fhat ω)⁻¹ = _
  by_cases h0 : sketchZ fhat ω = 0
  · simp [h0]
  · rw [ENNReal.mul_inv_cancel h0 htop.ne]
    simp [h0]

omit [Countable S] in
/-- `P(Ẑ ≠ 0) ≠ 0`: otherwise `f̂ = 0` almost surely, and `Z = ∑ₛ E[f̂(s)] = 0`. -/
lemma WellBehaved.measure_ne_zero (hb : WellBehaved μ f fhat) (hZ0 : ∑' s, f s ≠ 0) :
    μ {ω | sketchZ fhat ω ≠ 0} ≠ 0 := by
  intro h
  apply hZ0
  have hae : ∀ᵐ ω ∂μ, sketchZ fhat ω = 0 := ae_iff.2 h
  refine tsum_eq_zero_iff.2 fun s => ?_
  rw [← hb.unbiased s]
  calc ∫⁻ ω, fhat ω s ∂μ = ∫⁻ _, 0 ∂μ := by
        refine lintegral_congr_ae ?_
        filter_upwards [hae] with ω hω
        exact fhat_eq_zero_of_sketchZ_eq_zero hω s
    _ = 0 := lintegral_zero


/-- `{ω | Ẑ(ω) = 0}` is null-measurable. -/
lemma WellBehaved.nullMeasurableSet_eq_zero (hb : WellBehaved μ f fhat) :
    NullMeasurableSet {ω | sketchZ fhat ω = 0} μ :=
  hb.aemeasurable_sketchZ.nullMeasurable (measurableSet_singleton 0)

/-- `{ω | Ẑ(ω) ≠ 0}` is null-measurable. -/
lemma WellBehaved.nullMeasurableSet_ne_zero (hb : WellBehaved μ f fhat) :
    NullMeasurableSet {ω | sketchZ fhat ω ≠ 0} μ :=
  hb.nullMeasurableSet_eq_zero.compl

/-- `P(Ẑ = 0) + P(Ẑ ≠ 0) = 1`. -/
lemma WellBehaved.measure_eq_zero_add [IsProbabilityMeasure μ] (hb : WellBehaved μ f fhat) :
    μ {ω | sketchZ fhat ω = 0} + μ {ω | sketchZ fhat ω ≠ 0} = 1 := by
  have h := measure_add_measure_compl₀ hb.nullMeasurableSet_eq_zero
  rw [measure_univ] at h
  exact h

omit [Countable S] in
/-- The target `p = f/Z` sums to `1`. -/
lemma hasSum_normalize (hZ0 : ∑' s, f s ≠ 0) (hZtop : ∑' s, f s ≠ ∞) :
    ∑' s, normalize f s = 1 := by
  unfold normalize
  simp_rw [div_eq_mul_inv]
  rw [ENNReal.tsum_mul_right, ENNReal.mul_inv_cancel hZ0 hZtop]

/-- **`thm:var-kl-mixture`.** Let `f̂` be unbiased with `E[Ẑ²] < ∞` and `Ẑ > 0` almost surely, and
let `π̄(s) = E[π̂_ω(s)]` be its output law (`marginalLaw`). With `Var(Ẑ)` taken under `μ`:
`π̄` sums to `1`, `χ²(p ‖ π̄) ≤ Var(Ẑ)/Z²`, and `KL(p ‖ π̄) ≤ log(1 + Var(Ẑ)/Z²) ≤ Var(Ẑ)/Z²`,
with both divergences in `[0, ∞]`. -/
theorem bias_bound [MeasurableSpace S] [MeasurableSingletonClass S] [IsProbabilityMeasure μ]
    (hZ0 : ∑' s, f s ≠ 0) (hZtop : ∑' s, f s ≠ ∞) (hb : WellBehaved μ f fhat)
    (hpos : ∀ᵐ ω ∂μ, 0 < sketchZ fhat ω) :
    let Z : ℝ := (∑' s, f s).toReal
    let p : S → ℝ := fun s => (normalize f s).toReal
    let q : S → ℝ := fun s => (marginalLaw μ fhat s).toReal
    let V : ℝ := variance (fun ω => (sketchZ fhat ω).toReal) μ
    HasSum q 1 ∧ chiSq p q ≤ ENNReal.ofReal (V / Z ^ 2) ∧
    kl p q ≤ ENNReal.ofReal (Real.log (1 + V / Z ^ 2)) ∧
    Real.log (1 + V / Z ^ 2) ≤ V / Z ^ 2 := by
  intro Z p q V
  -- `P(Ẑ = 0) = 0`, so `π̄` sums to `P(Ẑ ≠ 0) = 1`
  have h0 : μ {ω | sketchZ fhat ω = 0} = 0 := by
    rw [ae_iff] at hpos
    simpa only [not_lt, nonpos_iff_eq_zero] using hpos
  have hQ : ∑' s, marginalLaw μ fhat s = 1 := by
    rw [hb.tsum_marginalLaw]
    simpa [h0] using hb.measure_eq_zero_add
  have habs : ∀ s, marginalLaw μ fhat s = 0 → normalize f s = 0 := by
    intro s hs
    simp [normalize, hb.abs_cont hs]
  have hZ2_ne0 : (∑' t, f t) ^ 2 ≠ 0 := pow_ne_zero 2 hZ0
  have hB : (∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ) / (∑' t, f t) ^ 2 ≠ ∞ :=
    ENNReal.div_ne_top hb.sq_finite hZ2_ne0
  obtain ⟨hB1, hq_hasSum, hchi, hkl⟩ :=
    real_bounds (hasSum_normalize hZ0 hZtop) hQ habs hB hb.tsum_sq_div_le
  have hratio : ((∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ) / (∑' t, f t) ^ 2).toReal = 1 + V / Z ^ 2 := by
    rw [ENNReal.toReal_div, ENNReal.toReal_pow]
    exact (hb.one_add_variance_div hZ0 hZtop).symm
  rw [hratio] at hB1 hchi hkl
  rw [add_sub_cancel_left] at hchi
  refine ⟨hq_hasSum, hchi, hkl, ?_⟩
  have hpos' : 0 < 1 + V / Z ^ 2 := by linarith
  have := Real.log_le_sub_one_of_pos hpos'
  linarith

/-! ### The footnote `fn:wlog`: without loss of generality, `Ẑ > 0` almost surely -/

/-- `E_{μ₊}[g] = E_μ[g]/γ` for every `g` that vanishes where `Ẑ = 0`, where `μ₊ = μ[|Ẑ ≠ 0]` and
`γ = P(Ẑ ≠ 0)`. -/
lemma WellBehaved.lintegral_cond_eq (hb : WellBehaved μ f fhat) {g : Ω → ℝ≥0∞}
    (hg : ∀ ω, sketchZ fhat ω = 0 → g ω = 0) :
    ∫⁻ ω, g ω ∂μ[|{ω | sketchZ fhat ω ≠ 0}] =
      (μ {ω | sketchZ fhat ω ≠ 0})⁻¹ * ∫⁻ ω, g ω ∂μ := by
  rw [ProbabilityTheory.cond, lintegral_smul_measure, smul_eq_mul,
    ← lintegral_indicator₀ hb.nullMeasurableSet_ne_zero]
  congr 1
  refine lintegral_congr fun ω => ?_
  by_cases h : sketchZ fhat ω = 0
  · simp [Set.indicator, h, hg ω h]
  · simp [Set.indicator, h]

/-- The output law of `(μ₊, f̂)` is `π̄₀/γ`, with `π̄₀(s) = E[π̂_ω(s); Ẑ ≠ 0]`. -/
lemma WellBehaved.marginalLaw_cond (hb : WellBehaved μ f fhat) (s : S) :
    marginalLaw (μ[|{ω | sketchZ fhat ω ≠ 0}]) fhat s =
      marginalLaw μ fhat s / μ {ω | sketchZ fhat ω ≠ 0} := by
  unfold marginalLaw
  rw [hb.lintegral_cond_eq (g := fun ω => fhat ω s / sketchZ fhat ω)
    (fun ω h => by simp [fhat_eq_zero_of_sketchZ_eq_zero h s]), ENNReal.div_eq_inv_mul]

/-- **The footnote's sketch `(μ₊, f̂)`** (`fn:wlog`). Let `γ = P(Ẑ ≠ 0)` and let
`μ₊ = μ[|Ẑ ≠ 0]` be `μ` conditioned on `Ẑ > 0`. Then `μ₊` is a probability measure, `Ẑ > 0`
`μ₊`-almost surely, `0 < ∑ f/γ < ∞`, the sketch `(μ₊, f̂)` is well-behaved for `f/γ`, `f/γ`
defines the same target as `f`, and `E_{μ₊}[Ẑ²]/(Z/γ)² = γ E_μ[Ẑ²]/Z²`. -/
theorem WellBehaved.cond [IsProbabilityMeasure μ] (hb : WellBehaved μ f fhat)
    (hZ0 : ∑' s, f s ≠ 0) (hZtop : ∑' s, f s ≠ ∞) :
    let γ := μ {ω | sketchZ fhat ω ≠ 0}
    IsProbabilityMeasure (μ[|{ω | sketchZ fhat ω ≠ 0}]) ∧
      (∀ᵐ ω ∂μ[|{ω | sketchZ fhat ω ≠ 0}], 0 < sketchZ fhat ω) ∧
      ∑' s, f s / γ ≠ 0 ∧ ∑' s, f s / γ ≠ ∞ ∧
      WellBehaved (μ[|{ω | sketchZ fhat ω ≠ 0}]) (fun s => f s / γ) fhat ∧
      normalize (fun s => f s / γ) = normalize f ∧
      (∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ[|{ω | sketchZ fhat ω ≠ 0}]).toReal /
          ((∑' s, f s / γ).toReal) ^ 2 =
        γ.toReal * ((∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ).toReal / ((∑' s, f s).toReal) ^ 2) := by
  intro γ
  have hγ0 : γ ≠ 0 := hb.measure_ne_zero hZ0
  have hγtop : γ ≠ ∞ := measure_ne_top μ _
  have hsum : ∑' s, f s / γ = (∑' s, f s) * γ⁻¹ := by
    simp_rw [div_eq_mul_inv]
    exact ENNReal.tsum_mul_right
  have hE : ∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ[|{ω | sketchZ fhat ω ≠ 0}] =
      γ⁻¹ * ∫⁻ ω, sketchZ fhat ω ^ 2 ∂μ :=
    hb.lintegral_cond_eq (fun ω h => by simp [h])
  refine ⟨cond_isProbabilityMeasure hγ0, ?_, ?_, ?_, ⟨?_, ?_, ?_⟩, ?_, ?_⟩
  · have h := Measure.ae_smul_measure (ae_restrict_mem₀ hb.nullMeasurableSet_ne_zero) γ⁻¹
    filter_upwards [h] with ω hω
    exact pos_iff_ne_zero.2 hω
  · rw [hsum]
    exact mul_ne_zero hZ0 (ENNReal.inv_ne_zero.2 hγtop)
  · rw [hsum]
    exact ENNReal.mul_ne_top hZtop (ENNReal.inv_ne_top.2 hγ0)
  · exact fun s => (hb.meas s).mono_ac cond_absolutelyContinuous
  · intro s
    rw [hb.lintegral_cond_eq (fun ω h => fhat_eq_zero_of_sketchZ_eq_zero h s), hb.unbiased s,
      ENNReal.div_eq_inv_mul]
  · rw [hE]
    exact ENNReal.mul_ne_top (ENNReal.inv_ne_top.2 hγ0) hb.sq_finite
  · funext s
    unfold normalize
    dsimp only
    rw [hsum, div_eq_mul_inv (f s) γ,
      ENNReal.mul_div_mul_right _ _ (ENNReal.inv_ne_zero.2 hγtop) (ENNReal.inv_ne_top.2 hγ0)]
  · have hγr : 0 < γ.toReal := ENNReal.toReal_pos hγ0 hγtop
    have hZr : 0 < (∑' s, f s).toReal := ENNReal.toReal_pos hZ0 hZtop
    rw [hE, hsum, ENNReal.toReal_mul, ENNReal.toReal_mul, ENNReal.toReal_inv]
    field_simp

/-- **The redraw law** (`fn:wlog`). Redrawing `ω` until `Ẑ(ω) ≠ 0` ends with probability `1`
because `γ = P(Ẑ ≠ 0) > 0`, which holds because `E[Ẑ] = Z > 0`. A redraw's output has the law
`Q` of a fresh run, so `Q(s) = π̄₀(s) + P(Ẑ = 0) Q(s)` with `π̄₀(s) = E[π̂_ω(s); Ẑ ≠ 0]`
(`marginalLaw`; the event `Ẑ = ∞` contributes `0` to it and has probability `0` for a
well-behaved sketch). The only finite solution is `Q(s) = π̄₀(s)/γ`, which is the output law of
the sketch `(μ₊, f̂)` with `μ₊ = μ[|Ẑ ≠ 0]`: redrawing runs `(μ₊, f̂)`. -/
theorem redraw_law_eq [IsProbabilityMeasure μ] (hb : WellBehaved μ f fhat)
    (hZ0 : ∑' s, f s ≠ 0) (Q : S → ℝ≥0∞) (hQ : ∀ s, Q s ≠ ∞)
    (hfix : ∀ s, Q s = marginalLaw μ fhat s + μ {ω | sketchZ fhat ω = 0} * Q s) :
    0 < μ {ω | sketchZ fhat ω ≠ 0} ∧
      (∀ s, Q s = marginalLaw μ fhat s / μ {ω | sketchZ fhat ω ≠ 0}) ∧
      ∀ s, Q s = marginalLaw (μ[|{ω | sketchZ fhat ω ≠ 0}]) fhat s := by
  have hE0 := hb.measure_ne_zero hZ0
  have hsol : ∀ s, Q s = marginalLaw μ fhat s / μ {ω | sketchZ fhat ω ≠ 0} := by
    intro s
    have hsum := hb.measure_eq_zero_add
    have hfin : μ {ω | sketchZ fhat ω = 0} * Q s ≠ ∞ :=
      ENNReal.mul_ne_top (measure_ne_top μ _) (hQ s)
    -- `P(Ẑ ≠ 0) Q(s) = π̄₀(s)`: subtract `P(Ẑ = 0) Q(s)` from both sides of the fixed-point equation
    have hkey : μ {ω | sketchZ fhat ω ≠ 0} * Q s = marginalLaw μ fhat s := by
      have h1 : Q s = μ {ω | sketchZ fhat ω = 0} * Q s + μ {ω | sketchZ fhat ω ≠ 0} * Q s := by
        rw [← add_mul, hsum, one_mul]
      have h2 := (hfix s).symm.trans h1
      rw [add_comm (marginalLaw μ fhat s)] at h2
      exact ((ENNReal.add_right_inj hfin).1 h2).symm
    rw [← hkey, mul_comm, ENNReal.mul_div_cancel_right hE0 (measure_ne_top μ _)]
  exact ⟨pos_iff_ne_zero.2 hE0, hsol, fun s => (hsol s).trans (hb.marginalLaw_cond s).symm⟩

omit [Countable S] in
/-- `1 + χ²` under a smaller second argument: if `q ≥ c r` pointwise with `c > 0`, for
distributions `p`, `q`, `r`, then `1 + χ²(p ‖ q) ≤ (1 + χ²(p ‖ r))/c`. -/
lemma one_add_chiSq_le_of_le {p q r : S → ℝ} (hq : ∀ s, 0 ≤ q s) (hr : ∀ s, 0 ≤ r s) (hp1 : HasSum p 1) (hq1 : HasSum q 1) (hr1 : HasSum r 1)
    {c : ℝ} (hc : 0 < c) (hle : ∀ s, c * r s ≤ q s) :
    1 + chiSq p q ≤ ENNReal.ofReal c⁻¹ * (1 + chiSq p r) := by
  by_cases htop : chiSq p r = ∞
  · rw [htop, add_top, ENNReal.mul_top (ENNReal.ofReal_pos.2 (inv_pos.2 hc)).ne']
    exact le_top
  obtain ⟨habs_r, hsum_r⟩ := (chiSq_ne_top_iff hr).1 htop
  have hr_pos : ∀ s, p s ≠ 0 → 0 < r s := fun s hs =>
    (hr s).lt_of_ne fun h => hs (habs_r s h.symm)
  have habs_q : ∀ s, q s = 0 → p s = 0 := by
    intro s hs
    by_contra h
    have := mul_pos hc (hr_pos s h)
    linarith [hle s]
  have hterm : ∀ x : S → ℝ, (∀ s, x s = 0 → p s = 0) → ∀ s,
      (p s - x s) ^ 2 / x s = p s ^ 2 / x s - 2 * p s + x s := by
    intro x hx s
    by_cases hs : x s = 0
    · simp [hs, hx s hs]
    · field_simp
      ring
  have hsq_r : HasSum (fun s => p s ^ 2 / r s) (1 + ∑' s, (p s - r s) ^ 2 / r s) := by
    have h := (hsum_r.hasSum.add (hp1.mul_left 2)).sub hr1
    convert h using 1
    · funext s
      rw [hterm r habs_r s]
      ring
    · ring
  have hle_sq : ∀ s, p s ^ 2 / q s ≤ c⁻¹ * (p s ^ 2 / r s) := by
    intro s
    by_cases hs : p s = 0
    · simp [hs]
    have hcr : 0 < c * r s := mul_pos hc (hr_pos s hs)
    calc p s ^ 2 / q s ≤ p s ^ 2 / (c * r s) :=
          div_le_div_of_nonneg_left (sq_nonneg _) hcr (hle s)
      _ = c⁻¹ * (p s ^ 2 / r s) := by
          field_simp
  have hsq_q : Summable fun s => p s ^ 2 / q s :=
    Summable.of_nonneg_of_le (fun s => div_nonneg (sq_nonneg _) (hq s)) hle_sq
      (hsq_r.summable.mul_left _)
  have hchi_q : HasSum (fun s => (p s - q s) ^ 2 / q s) ((∑' s, p s ^ 2 / q s) - 2 * 1 + 1) := by
    have h := (hsq_q.hasSum.sub (hp1.mul_left 2)).add hq1
    convert h using 1
    funext s
    exact hterm q habs_q s
  have hX : ∑' s, p s ^ 2 / q s ≤ c⁻¹ * (1 + ∑' s, (p s - r s) ^ 2 / r s) :=
    hasSum_le hle_sq hsq_q.hasSum (hsq_r.mul_left c⁻¹)
  have hA0 : 0 ≤ (∑' s, p s ^ 2 / q s) - 2 * 1 + 1 :=
    hchi_q.nonneg fun s => div_nonneg (sq_nonneg _) (hq s)
  have hB0 : 0 ≤ ∑' s, (p s - r s) ^ 2 / r s :=
    tsum_nonneg fun s => div_nonneg (sq_nonneg _) (hr s)
  rw [chiSq_eq_ofReal hq habs_q hchi_q.summable, hchi_q.tsum_eq,
    chiSq_eq_ofReal hr habs_r hsum_r]
  calc 1 + ENNReal.ofReal ((∑' s, p s ^ 2 / q s) - 2 * 1 + 1)
      = ENNReal.ofReal (1 + ((∑' s, p s ^ 2 / q s) - 2 * 1 + 1)) := by
        rw [ENNReal.ofReal_add zero_le_one hA0, ENNReal.ofReal_one]
    _ ≤ ENNReal.ofReal (c⁻¹ * (1 + ∑' s, (p s - r s) ^ 2 / r s)) :=
        ENNReal.ofReal_le_ofReal (by linarith)
    _ = ENNReal.ofReal c⁻¹ * (1 + ENNReal.ofReal (∑' s, (p s - r s) ^ 2 / r s)) := by
        rw [ENNReal.ofReal_mul (inv_nonneg.2 hc.le), ENNReal.ofReal_add zero_le_one hB0,
          ENNReal.ofReal_one]

/-- `KL` under a smaller second argument: if `q ≥ c r` pointwise with `c > 0`, for distributions
`p`, `q`, `r`, then `KL(p ‖ q) ≤ KL(p ‖ r) + log(1/c)`. -/
lemma kl_le_add_of_le [MeasurableSpace S] [MeasurableSingletonClass S] {p q r : S → ℝ}
    (hp : ∀ s, 0 ≤ p s) (hq : ∀ s, 0 ≤ q s) (hr : ∀ s, 0 ≤ r s)
    (hp1 : HasSum p 1) (hq1 : HasSum q 1) (hr1 : HasSum r 1)
    {c : ℝ} (hc : 0 < c) (hle : ∀ s, c * r s ≤ q s) :
    kl p q ≤ kl p r + ENNReal.ofReal (Real.log c⁻¹) := by
  by_cases htop : kl p r = ∞
  · rw [htop, top_add]
    exact le_top
  obtain ⟨habs_r, hsum_r⟩ := (kl_ne_top_iff hp hr hp1 hr1).1 htop
  have hr_pos : ∀ s, p s ≠ 0 → 0 < r s := fun s hs =>
    (hr s).lt_of_ne fun h => hs (habs_r s h.symm)
  have habs_q : ∀ s, q s = 0 → p s = 0 := by
    intro s hs
    by_contra h
    have := mul_pos hc (hr_pos s h)
    linarith [hle s]
  have hup : ∀ s, p s * Real.log (p s / q s) ≤
      p s * Real.log (p s / r s) + p s * Real.log c⁻¹ := by
    intro s
    by_cases hs : p s = 0
    · simp [hs]
    have hp0 : 0 < p s := (hp s).lt_of_ne (Ne.symm hs)
    have hcr : 0 < c * r s := mul_pos hc (hr_pos s hs)
    rw [← mul_add, ← Real.log_mul (div_pos hp0 (hr_pos s hs)).ne' (inv_pos.2 hc).ne']
    refine mul_le_mul_of_nonneg_left (Real.log_le_log (div_pos hp0 (hcr.trans_le (hle s))) ?_)
      hp0.le
    calc p s / q s ≤ p s / (c * r s) := div_le_div_of_nonneg_left hp0.le hcr (hle s)
      _ = p s / r s * c⁻¹ := by field_simp
  have hlow : ∀ s, p s - q s ≤ p s * Real.log (p s / q s) :=
    fun s => mul_log_div_ge (hp s) (hq s) (habs_q s)
  have hU : HasSum (fun s => p s * Real.log (p s / r s) + p s * Real.log c⁻¹)
      ((∑' s, p s * Real.log (p s / r s)) + 1 * Real.log c⁻¹) :=
    hsum_r.hasSum.add (hp1.mul_right _)
  have hL : Summable fun s => p s - q s := hp1.summable.sub hq1.summable
  have ht : Summable fun s => p s * Real.log (p s / q s) := by
    have hd : Summable fun s => p s * Real.log (p s / q s) - (p s - q s) := by
      refine Summable.of_nonneg_of_le (fun s => by linarith [hlow s]) (fun s => ?_)
        (hU.summable.sub hL)
      linarith [hup s]
    simpa using hd.add hL
  have hsum_le : ∑' s, p s * Real.log (p s / q s) ≤
      (∑' s, p s * Real.log (p s / r s)) + 1 * Real.log c⁻¹ :=
    hasSum_le hup ht.hasSum hU
  rw [kl_eq_ofReal hp hq hp1 hq1 habs_q ht, kl_eq_ofReal hp hr hp1 hr1 habs_r hsum_r]
  rw [one_mul] at hsum_le
  exact (ENNReal.ofReal_le_ofReal hsum_le).trans ENNReal.ofReal_add_le

/-- `1 + Var_{μ₊}(Ẑ)/(Z/γ)² = γ (1 + Var_μ(Ẑ)/Z²)`, from `WellBehaved.cond` and
`WellBehaved.one_add_variance_div` for both sketches. -/
lemma WellBehaved.one_add_variance_div_cond [IsProbabilityMeasure μ] (hb : WellBehaved μ f fhat)
    (hZ0 : ∑' s, f s ≠ 0) (hZtop : ∑' s, f s ≠ ∞) :
    1 + variance (fun ω => (sketchZ fhat ω).toReal) (μ[|{ω | sketchZ fhat ω ≠ 0}]) /
        ((∑' s, f s / μ {ω | sketchZ fhat ω ≠ 0}).toReal) ^ 2 =
      (μ {ω | sketchZ fhat ω ≠ 0}).toReal *
        (1 + variance (fun ω => (sketchZ fhat ω).toReal) μ / ((∑' s, f s).toReal) ^ 2) := by
  obtain ⟨hprob, -, hZ0', hZtop', hb', -, hratio⟩ := hb.cond hZ0 hZtop
  rw [hb'.one_add_variance_div hZ0' hZtop', hb.one_add_variance_div hZ0 hZtop]
  exact hratio

omit [Countable S] in
/-- Arithmetic for the reduction: if `1 + a = γ (1 + b)` with `0 ≤ γ ≤ 1` and `0 ≤ b`, then
`a ≤ b`. -/
lemma le_of_one_add_eq_mul {a b γ : ℝ} (h : 1 + a = γ * (1 + b)) (hγ1 : γ ≤ 1) (hb : 0 ≤ b) :
    a ≤ b := by
  nlinarith

omit [Countable S] in
/-- `P(Ẑ ≠ 0)` as a real number lies in `[0, 1]`. -/
lemma toReal_measure_le_one [IsProbabilityMeasure μ] (A : Set Ω) : (μ A).toReal ≤ 1 :=
  ENNReal.toReal_le_of_le_ofReal zero_le_one (by simpa using prob_le_one)

/-- **`thm:var-kl-mixture` for the redraw sampler** (`fn:wlog`). The sampler that redraws `ω` until
`Ẑ(ω) ≠ 0` outputs `π̄(s) = π̄₀(s)/γ` (`redraw_law_eq`), the output law of `(μ₊, f̂)`. This law
satisfies the bounds of `thm:var-kl-mixture` with `Var(Ẑ)` under the unconditioned `μ`: by
`bias_bound` for `(μ₊, f̂)`, whose `1 + Var/Z²` is `γ` times that of `(μ, f̂)`. -/
theorem redraw_bound [MeasurableSpace S] [MeasurableSingletonClass S] [IsProbabilityMeasure μ]
    (hZ0 : ∑' s, f s ≠ 0) (hZtop : ∑' s, f s ≠ ∞)
    (hb : WellBehaved μ f fhat) :
    let Z : ℝ := (∑' s, f s).toReal
    let p : S → ℝ := fun s => (normalize f s).toReal
    let q : S → ℝ := fun s => (marginalLaw μ fhat s / μ {ω | sketchZ fhat ω ≠ 0}).toReal
    let V : ℝ := variance (fun ω => (sketchZ fhat ω).toReal) μ
    HasSum q 1 ∧ chiSq p q ≤ ENNReal.ofReal (V / Z ^ 2) ∧
    kl p q ≤ ENNReal.ofReal (Real.log (1 + V / Z ^ 2)) ∧
    Real.log (1 + V / Z ^ 2) ≤ V / Z ^ 2 := by
  intro Z p q V
  obtain ⟨hprob, hpos, hZ0', hZtop', hb', hnorm, -⟩ := hb.cond hZ0 hZtop
  have hrel := hb.one_add_variance_div_cond hZ0 hZtop
  obtain ⟨hq1, hchi, hkl, -⟩ := bias_bound hZ0' hZtop' hb' hpos
  simp only [hnorm, hb.marginalLaw_cond] at hchi hkl hq1
  have hV : 0 ≤ V / Z ^ 2 := div_nonneg (variance_nonneg _ _) (sq_nonneg _)
  have hV' := div_nonneg (variance_nonneg (fun ω => (sketchZ fhat ω).toReal)
    (μ[|{ω | sketchZ fhat ω ≠ 0}])) (sq_nonneg (∑' s, f s / μ {ω | sketchZ fhat ω ≠ 0}).toReal)
  have hle := le_of_one_add_eq_mul hrel (toReal_measure_le_one _) hV
  have hlog := Real.log_le_log (by linarith [hV']) (add_le_add_right hle 1)
  refine ⟨hq1, hchi.trans (ENNReal.ofReal_le_ofReal hle),
    hkl.trans (ENNReal.ofReal_le_ofReal hlog), ?_⟩
  have := Real.log_le_sub_one_of_pos (show 0 < 1 + V / Z ^ 2 by linarith)
  linarith

/-- **`thm:var-kl-mixture` with any fallback** (`fn:wlog`). The sampler draws from `π̂_ω` when
`0 < Ẑ(ω) < ∞`, and from a distribution `out ω` of its choosing (which may depend on `ω`)
otherwise. Its output law `π̄(s) = E[out ω s]` satisfies the bounds of `thm:var-kl-mixture`, with
`Var(Ẑ)` under `μ`. Proof as in the footnote: `π̄ ≥ π̄₀ = γ π̄₊` pointwise, where `π̄₊` is the
output law of `(μ₊, f̂)`; `bias_bound` bounds `(μ₊, f̂)`, and the comparison lemmas cost a factor
`1/γ` in `1 + χ²` and `log(1/γ)` in `KL`, which `γ (1 + Var/Z²) = 1 + Var₊/Z₊²` pays back. -/
theorem fallback_bound [MeasurableSpace S] [MeasurableSingletonClass S] [IsProbabilityMeasure μ]
    (hZ0 : ∑' s, f s ≠ 0) (hZtop : ∑' s, f s ≠ ∞) (hb : WellBehaved μ f fhat)
    (out : Ω → S → ℝ≥0∞) (hout_meas : ∀ s, AEMeasurable (fun ω => out ω s) μ)
    (hout_sum : ∀ ω, ∑' s, out ω s = 1)
    (hout_eq : ∀ ω, sketchZ fhat ω ≠ 0 → sketchZ fhat ω ≠ ∞ →
      ∀ s, out ω s = fhat ω s / sketchZ fhat ω) :
    let Z : ℝ := (∑' s, f s).toReal
    let p : S → ℝ := fun s => (normalize f s).toReal
    let q : S → ℝ := fun s => (∫⁻ ω, out ω s ∂μ).toReal
    let V : ℝ := variance (fun ω => (sketchZ fhat ω).toReal) μ
    HasSum q 1 ∧ chiSq p q ≤ ENNReal.ofReal (V / Z ^ 2) ∧
    kl p q ≤ ENNReal.ofReal (Real.log (1 + V / Z ^ 2)) ∧
    Real.log (1 + V / Z ^ 2) ≤ V / Z ^ 2 := by
  intro Z p q V
  -- the fallback's law sums to `1` and dominates `π̄₀`
  have hQ : ∑' s, ∫⁻ ω, out ω s ∂μ = 1 := by
    rw [← lintegral_tsum (f := fun s ω => out ω s) hout_meas]
    simp [hout_sum]
  have hQtop : ∀ s, ∫⁻ ω, out ω s ∂μ ≠ ∞ := fun s =>
    ne_top_of_le_ne_top (by rw [hQ]; exact ENNReal.one_ne_top)
      (le_tsum (fun s => ∫⁻ ω, out ω s ∂μ) s)
  have hle : ∀ s, marginalLaw μ fhat s ≤ ∫⁻ ω, out ω s ∂μ := by
    intro s
    unfold marginalLaw
    refine lintegral_mono fun ω => ?_
    by_cases h0 : sketchZ fhat ω = 0
    · rw [fhat_eq_zero_of_sketchZ_eq_zero h0 s, ENNReal.zero_div]
      exact zero_le
    by_cases htop : sketchZ fhat ω = ∞
    · rw [htop, ENNReal.div_top]
      exact zero_le
    exact (hout_eq ω h0 htop s).ge
  -- the theorem for `(μ₊, f̂)`
  obtain ⟨hprob, hpos, hZ0', hZtop', hb', hnorm, -⟩ := hb.cond hZ0 hZtop
  have hrel := hb.one_add_variance_div_cond hZ0 hZtop
  obtain ⟨hr1, hchi', hkl', -⟩ := bias_bound hZ0' hZtop' hb' hpos
  simp only [hnorm] at hchi' hkl'
  have hγ0 : μ {ω | sketchZ fhat ω ≠ 0} ≠ 0 := hb.measure_ne_zero hZ0
  have hγpos : 0 < (μ {ω | sketchZ fhat ω ≠ 0}).toReal :=
    ENNReal.toReal_pos hγ0 (measure_ne_top μ _)
  have hγ1 := toReal_measure_le_one (μ := μ) {ω | sketchZ fhat ω ≠ 0}
  -- `π̄ ≥ γ π̄₊`
  have hdom : ∀ s, (μ {ω | sketchZ fhat ω ≠ 0}).toReal *
      (marginalLaw (μ[|{ω | sketchZ fhat ω ≠ 0}]) fhat s).toReal ≤ q s := by
    intro s
    rw [← ENNReal.toReal_mul, hb.marginalLaw_cond s,
      ENNReal.mul_div_cancel hγ0 (measure_ne_top μ _)]
    exact ENNReal.toReal_mono (hQtop s) (hle s)
  have hp : ∀ s, 0 ≤ p s := fun s => ENNReal.toReal_nonneg
  have hq : ∀ s, 0 ≤ q s := fun s => ENNReal.toReal_nonneg
  have hr : ∀ s, 0 ≤ (marginalLaw (μ[|{ω | sketchZ fhat ω ≠ 0}]) fhat s).toReal :=
    fun s => ENNReal.toReal_nonneg
  have hp1 : HasSum p 1 := by
    have := hasSum_toReal_of_ne_top (f := normalize f) (by rw [hasSum_normalize hZ0 hZtop]; simp)
    rwa [hasSum_normalize hZ0 hZtop, ENNReal.toReal_one] at this
  have hq1 : HasSum q 1 := by
    have := hasSum_toReal_of_ne_top (f := fun s => ∫⁻ ω, out ω s ∂μ) (by rw [hQ]; simp)
    rwa [hQ, ENNReal.toReal_one] at this
  have hV : 0 ≤ V / Z ^ 2 := div_nonneg (variance_nonneg _ _) (sq_nonneg _)
  have hV' := div_nonneg (variance_nonneg (fun ω => (sketchZ fhat ω).toReal)
    (μ[|{ω | sketchZ fhat ω ≠ 0}])) (sq_nonneg (∑' s, f s / μ {ω | sketchZ fhat ω ≠ 0}).toReal)
  refine ⟨hq1, ?_, ?_, ?_⟩
  · -- `1 + χ²(p ‖ π̄) ≤ (1 + χ²(p ‖ π̄₊))/γ ≤ (1 + V₊/Z₊²)/γ = 1 + V/Z²`
    have h : 1 + chiSq p q ≤ ENNReal.ofReal (μ {ω | sketchZ fhat ω ≠ 0}).toReal⁻¹ *
        (1 + ENNReal.ofReal _) :=
      (one_add_chiSq_le_of_le hq hr hp1 hq1 hr1 hγpos hdom).trans (by gcongr)
    rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_add zero_le_one hV',
      ← ENNReal.ofReal_mul (inv_nonneg.2 hγpos.le), hrel, ← mul_assoc,
      inv_mul_cancel₀ hγpos.ne', one_mul, ENNReal.ofReal_add zero_le_one hV,
      ENNReal.ofReal_one] at h
    exact (ENNReal.add_le_add_iff_left ENNReal.one_ne_top).1 h
  · -- `KL(p ‖ π̄) ≤ KL(p ‖ π̄₊) + log(1/γ) ≤ log(1 + V₊/Z₊²) + log(1/γ) = log(1 + V/Z²)`
    have hlogγ : 0 ≤ Real.log (μ {ω | sketchZ fhat ω ≠ 0}).toReal⁻¹ := by
      rw [Real.log_inv]
      linarith [Real.log_nonpos hγpos.le hγ1]
    have h : kl p q ≤ ENNReal.ofReal (Real.log (1 + _)) +
        ENNReal.ofReal (Real.log (μ {ω | sketchZ fhat ω ≠ 0}).toReal⁻¹) :=
      (kl_le_add_of_le hp hq hr hp1 hq1 hr1 hγpos hdom).trans (by gcongr)
    rw [← ENNReal.ofReal_add (Real.log_nonneg (by linarith)) hlogγ, hrel,
      Real.log_mul hγpos.ne' (by linarith), Real.log_inv] at h
    simpa using h
  · have := Real.log_le_sub_one_of_pos (show 0 < 1 + V / Z ^ 2 by linarith)
    linarith

end General

end DppLm
