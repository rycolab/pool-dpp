import DppLm.PoolSketch

/-!
# `thm:zfree` (Finite-sample bound for PoolDPP): the identity and the second-moment bound

Paper statement (`thm:zfree`): for `K ≥ 1`, `N ≥ K`, and a proposal with
`q(y) = 0 → κ(y) = 0`,
  `Var(Ẑ)/Z² = ∑_{m=1}^{K} a_m Var(u_m)`,  `a_m = C(K,m) C(N-K,K-m)/C(N,K)`,
and if `ρ = E_q[u₁²] < ∞`, then for any fallback when `Ẑ = 0`,
`KL(p ‖ π̄) ≤ log(e^θ - 1 + a₀) ≤ θ` with `θ = K² ρ/(N-K+1)` and `a₀ = C(N-K,K)/C(N,K)`.
Here `ρ = 1 + χ²(π₁ ‖ q)`, with `π₁(y) = p(y ∈ S)/K` the law of a uniformly chosen member of a
target set (`secondMomentU1`).

## What is proved here

* `zfree_identity`: the identity in second-moment form, `E[Ẑ²] = Z² ∑_m a_m E[u_m²]`.
  `zfree_variance`: the paper's variance form, with the variances in `[0, ∞]` (Mathlib's
  `evariance`), so it needs no finiteness assumption: when some `E[u_m²] = ∞`, both sides
  are `∞`.
* `secondMoment_le_exp`: under negative correlation and `ρ < ∞`, `E[Ẑ²] < ∞` and
  `E[Ẑ²]/Z² ≤ e^θ - 1 + a₀`. Negative correlation is a hypothesis (`hNA`) because `f` is a
  general target weight here; for the paper's `K`-DPP it is proved (`kdpp_negCorr`,
  `NegCorr.lean`). The KL bound is `thm:var-kl-mixture` applied to this second moment, for any
  fallback: `poolDPP_fallback_bound` and `poolDPP_redraw_bound` in `PoolDPP.lean`.
* The bound: negative correlation gives `u_m(y) ≤ (K^m/(K)_m) ∏ᵢ u₁(yᵢ)` pointwise
  (`inclRatio_le_prod`). Squaring and using the independence of the draws gives
  `E[u_m²] ≤ (K^m/(K)_m)² ρ^m` (`tsum_inclRatio_sq_le`). Applied to every term of the
  identity, this gives `E[Ẑ²]/Z² ≤ ∑_{m=0}^{K} a_m (K^m/(K)_m)² ρ^m`, the sum of
  `cor:kl-sum` (`secondMoment_le_sharper`). Since `C(K,m)/(K)_m = 1/m!`,
  `a_m (K^m/(K)_m)² ρ^m ≤ θ^m/m!` with `θ = K² ρ/(N-K+1)` (`coeff_bound`) for `m ≥ 1`.
  Keeping the `m = 0` term `a₀ = C(N-K,K)/C(N,K)` exact gives
  `∑_{m=0}^{K} a_m (K^m/(K)_m)² ρ^m ≤ a₀ + ∑_{m=1}^{K} θ^m/m! ≤ e^θ - 1 + a₀`
  (`sharperBound_le_exp`), so `E[Ẑ²]/Z² ≤ e^θ - 1 + a₀` (`secondMoment_le_exp`). `thm:var-kl-mixture`
  then gives `KL ≤ log(E[Ẑ²]/Z²) ≤ log(e^θ - 1 + a₀) ≤ θ`, the last step by
  `log_exp_sub_one_add_le`, since `a₀ ≤ 1` (`overlapCoeff_zero_toReal_le_one`).
* `f` is a general target weight on `K`-sets (see `PoolSketch.lean`). `paperFhat_eq`
  (`Determinant.lean`) identifies the paper's `α Σ det(L̂_I)` with `poolFhat`.
-/

open Finset MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace DppLm

variable {Y : Type*} [DecidableEq Y]
variable {q : Y → ℝ≥0∞} {K N : ℕ} {f : Finset Y → ℝ≥0∞}

/-! ### Mean of the inclusion ratios and of the coefficients -/

lemma PoolSetting.inclProb_supp (hs : PoolSetting q K f) {T : Finset Y}
    (h : inclProb f T ≠ 0) : ∀ y ∈ T, q y ≠ 0 := by
  unfold inclProb at h
  have hX : (∑' S, if T ⊆ S then f S else 0) ≠ 0 := fun h0 => h (by rw [h0, ENNReal.zero_div])
  obtain ⟨S, hS⟩ : ∃ S, (if T ⊆ S then f S else 0) ≠ 0 := by
    by_contra hc
    simp only [not_exists, ne_eq, not_not] at hc
    exact hX (tsum_eq_zero_iff.2 hc)
  by_cases hTS : T ⊆ S
  · simp only [hTS, ite_true] at hS
    exact fun y hy => hs.f_supp S hS y (hTS hy)
  · simp [hTS] at hS

omit [DecidableEq Y] in
lemma tupleWeight_ne_zero_of {ι : Type*} [Fintype ι] {y : ι → Y} (h : ∀ i, q (y i) ≠ 0) :
    tupleWeight q y ≠ 0 :=
  Finset.prod_ne_zero_iff.2 fun i _ => h i

/-- `∑_{|T| = m} p(T ⊆ S)·Z = C(K,m) Z`: a `K`-set has `C(K,m)` subsets of size `m`. -/
lemma tsum_card_incl (hs : PoolSetting q K f) (m : ℕ) :
    ∑' T : Finset Y, (if T.card = m then ∑' S, (if T ⊆ S then f S else 0) else 0) =
      K.choose m * ∑' S, f S := by
  have h1 : ∀ T : Finset Y, (if T.card = m then ∑' S, (if T ⊆ S then f S else 0) else 0) =
      ∑' S, (if T.card = m then (if T ⊆ S then f S else 0) else 0) := by
    intro T
    by_cases hT : T.card = m <;> simp [hT]
  simp_rw [h1]
  rw [tsum_comm, ← ENNReal.tsum_mul_left]
  refine tsum_congr fun S => ?_
  rw [tsum_eq_sum (s := powersetCard m S)]
  · have hc : ∀ T ∈ powersetCard m S,
        (if T.card = m then (if T ⊆ S then f S else 0) else 0) = f S := by
      intro T hT
      rw [mem_powersetCard] at hT
      simp [hT.1, hT.2]
    rw [Finset.sum_congr rfl hc, sum_const, card_powersetCard, nsmul_eq_mul]
    by_cases hf : f S = 0
    · simp [hf]
    · rw [hs.f_card S hf]
  · intro T hT
    rw [mem_powersetCard] at hT
    by_cases h1 : T.card = m
    · by_cases h2 : T ⊆ S
      · exact absurd ⟨h2, h1⟩ hT
      · simp [h1, h2]
    · simp [h1]

/-- `E[u_m] = 1` for `m ≤ K`. -/
theorem tsum_inclRatio_eq_one (hs : PoolSetting q K f) {m : ℕ} (hm : m ≤ K) :
    ∑' y : Fin m → Y, tupleWeight q y * inclRatio q K f y = 1 := by
  set D := ((K.descFactorial m : ℕ) : ℝ≥0∞) with hD
  have hD0 : D ≠ 0 := by rw [hD]; exact_mod_cast (Nat.descFactorial_pos.2 hm).ne'
  have hDtop : D ≠ ∞ := ENNReal.natCast_ne_top _
  have h1 : ∀ y : Fin m → Y, tupleWeight q y * inclRatio q K f y =
      (if Function.Injective y then inclProb f (univ.image y) * D⁻¹ else 0) := by
    intro y
    unfold inclRatio
    by_cases hy : Function.Injective y
    · simp only [hy, ite_true, Fintype.card_fin]
      by_cases hP : inclProb f (univ.image y) = 0
      · simp only [hP, ENNReal.zero_div, mul_zero, zero_mul]
      · have hW0 : tupleWeight q y ≠ 0 := tupleWeight_ne_zero_of fun i =>
          hs.inclProb_supp hP (y i) (Finset.mem_image_of_mem y (Finset.mem_univ i))
        rw [div_eq_mul_inv, ENNReal.mul_inv (Or.inl hD0) (Or.inl hDtop)]
        calc tupleWeight q y * (inclProb f (univ.image y) * (D⁻¹ * (tupleWeight q y)⁻¹))
            = inclProb f (univ.image y) * D⁻¹ * (tupleWeight q y * (tupleWeight q y)⁻¹) := by
              ring
          _ = inclProb f (univ.image y) * D⁻¹ := by
              rw [ENNReal.mul_inv_cancel hW0 (hs.tupleWeight_ne_top y), mul_one]
    · simp [hy]
  simp_rw [h1]
  rw [tsum_injective_image (A := Fin m) (fun T => inclProb f T * D⁻¹), Fintype.card_fin]
  have h2 : ∀ T : Finset Y, (if T.card = m then inclProb f T * D⁻¹ else 0) =
      (if T.card = m then ∑' S, (if T ⊆ S then f S else 0) else 0) *
        ((∑' S, f S)⁻¹ * D⁻¹) := by
    intro T
    by_cases hT : T.card = m
    · simp only [hT, ite_true, inclProb, div_eq_mul_inv]; ring
    · simp [hT]
  simp_rw [h2]
  rw [ENNReal.tsum_mul_right, tsum_card_incl hs m]
  have hDm : D = (m.factorial : ℝ≥0∞) * (K.choose m : ℝ≥0∞) := by
    rw [hD, Nat.descFactorial_eq_factorial_mul_choose]; push_cast; ring
  calc _ = (m.factorial * K.choose m : ℝ≥0∞) * D⁻¹ * ((∑' S, f S) * (∑' S, f S)⁻¹) := by ring
    _ = 1 := by
      rw [ENNReal.mul_inv_cancel hs.Z_ne_zero hs.Z_ne_top, mul_one, ← hDm,
        ENNReal.mul_inv_cancel hD0 hDtop]

/-- `∑_{m=0}^{K} a_m = 1` (Vandermonde, via the pair count). -/
theorem sum_overlapCoeff (hKN : K ≤ N) :
    ∑ m ∈ range (K + 1), overlapCoeff K N m = 1 := by
  have h := sum_sum_inter (K := K) (N := N) (fun _ => (1 : ℝ≥0∞))
  simp only [sum_const, card_powersetCard, card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one] at h
  have hC0 : ((N.choose K : ℕ) : ℝ≥0∞) ≠ 0 := by exact_mod_cast (Nat.choose_pos hKN).ne'
  have hCtop : ((N.choose K : ℕ) : ℝ≥0∞) ≠ ∞ := ENNReal.natCast_ne_top _
  have hsum : ∑ m ∈ range (K + 1), ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ≥0∞) =
      (N.choose K : ℝ≥0∞) := ((ENNReal.mul_right_inj hC0 hCtop).1 h).symm
  unfold overlapCoeff
  simp_rw [div_eq_mul_inv]
  rw [← Finset.sum_mul, hsum, ENNReal.mul_inv_cancel hC0 hCtop]

/-! ### The second moment `ρ = E_q[u₁²]` and bounds under negative correlation -/

/-- `ρ = E_q[u₁²] = ∑_y q(y) u₁(y)²`, the second moment of the one-point inclusion ratio.
Since `E_q[u₁] = 1` (`tsum_inclRatio_eq_one`), `ρ = 1 + Var_q(u₁) = 1 + χ²(π₁ ‖ q)` with
`π₁(y) = p(y ∈ S)/K`, the law of a uniformly chosen member of a target set. -/
noncomputable def secondMomentU1 (q : Y → ℝ≥0∞) (K : ℕ) (f : Finset Y → ℝ≥0∞) : ℝ≥0∞ :=
  ∑' y : Fin 1 → Y, tupleWeight q y * inclRatio q K f y ^ 2

/-- `ρ = ∑_x q(x) u₁(x)²`, with `u₁(x)` the inclusion ratio of the one-point tuple `x`. -/
lemma secondMomentU1_eq :
    secondMomentU1 q K f = ∑' x, q x * inclRatio q K f (fun _ : Fin 1 => x) ^ 2 := by
  have h := tsum_fin_prod 1 (fun _ x => q x * inclRatio q K f (fun _ : Fin 1 => x) ^ 2)
  rw [Fin.prod_univ_one] at h
  rw [← h, secondMomentU1]
  refine tsum_congr fun y => ?_
  have hy : y = fun _ : Fin 1 => y 0 := funext fun i => by rw [Subsingleton.elim i 0]
  rw [Fin.prod_univ_one, congrArg (inclRatio q K f) hy]
  unfold tupleWeight
  rw [Fin.prod_univ_one]

/-- `p(x ∈ S) = K q(x) u₁(x)`. (If `q(x) = 0`, both sides are `0`: the support condition gives
`p(x ∈ S) = 0`, and `u₁(x) = 0/0 = 0` in `ℝ≥0∞`.) -/
lemma inclProb_singleton_eq (hs : PoolSetting q K f) (hK : 1 ≤ K) (x : Y) :
    inclProb f {x} = (K : ℝ≥0∞) * q x * inclRatio q K f (fun _ : Fin 1 => x) := by
  have hK0 : (K : ℝ≥0∞) ≠ 0 := by exact_mod_cast (by omega : K ≠ 0)
  by_cases hx : q x = 0
  · have : inclProb f {x} = 0 := by
      by_contra h
      exact hs.inclProb_supp h x (Finset.mem_singleton_self x) hx
    rw [this, hx, mul_zero, zero_mul]
  have hinj : Function.Injective (fun _ : Fin 1 => x) := fun a b _ => Subsingleton.elim a b
  have hu : inclRatio q K f (fun _ : Fin 1 => x) = inclProb f {x} / ((K : ℝ≥0∞) * q x) := by
    unfold inclRatio
    simp only [hinj, ite_true, Fintype.card_fin, Nat.descFactorial_one, tupleWeight,
      Finset.univ_unique, Finset.prod_singleton, Finset.image_singleton]
  rw [hu, ENNReal.mul_div_cancel (mul_ne_zero hK0 hx)
    (ENNReal.mul_ne_top (ENNReal.natCast_ne_top _) (hs.q_ne_top x))]

/-- Under negative correlation, `u_m(y) ≤ (K^m/(K)_m) ∏_i u₁(y_i)` pointwise (the first
inequality of the paper's `eq:incl-product`). -/
theorem inclRatio_le_prod (hs : PoolSetting q K f) (hK : 1 ≤ K)
    (hNA : ∀ T : Finset Y, inclProb f T ≤ ∏ y ∈ T, inclProb f {y}) {m : ℕ} (y : Fin m → Y) :
    inclRatio q K f y ≤ (K : ℝ≥0∞) ^ m / ((K.descFactorial m : ℕ) : ℝ≥0∞) *
      ∏ i, inclRatio q K f (fun _ : Fin 1 => y i) := by
  set D := ((K.descFactorial m : ℕ) : ℝ≥0∞)
  by_cases hy : Function.Injective y
  · have hR : inclRatio q K f y = inclProb f (univ.image y) / (D * tupleWeight q y) := by
      simp only [inclRatio, hy, ite_true, Fintype.card_fin, D]
    rw [hR]
    by_cases hP : inclProb f (univ.image y) = 0
    · rw [hP, ENNReal.zero_div]; exact zero_le
    have hW0 : tupleWeight q y ≠ 0 := tupleWeight_ne_zero_of fun i =>
      hs.inclProb_supp hP (y i) (Finset.mem_image_of_mem y (Finset.mem_univ i))
    have hprod : inclProb f (univ.image y) ≤
        (K : ℝ≥0∞) ^ m * (∏ i, inclRatio q K f (fun _ : Fin 1 => y i)) * tupleWeight q y := by
      calc inclProb f (univ.image y) ≤ ∏ x ∈ univ.image y, inclProb f {x} := hNA _
        _ = ∏ i, inclProb f {y i} := Finset.prod_image (fun a _ b _ h => hy h)
        _ = ∏ i : Fin m, (K : ℝ≥0∞) * q (y i) * inclRatio q K f (fun _ : Fin 1 => y i) :=
            Finset.prod_congr rfl fun i _ => inclProb_singleton_eq hs hK (y i)
        _ = (K : ℝ≥0∞) ^ m * (∏ i, inclRatio q K f (fun _ : Fin 1 => y i)) *
              tupleWeight q y := by
            simp only [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ,
              Fintype.card_fin, tupleWeight]
            ring
    calc inclProb f (univ.image y) / (D * tupleWeight q y)
        ≤ (K : ℝ≥0∞) ^ m * (∏ i, inclRatio q K f (fun _ : Fin 1 => y i)) * tupleWeight q y /
            (D * tupleWeight q y) := by gcongr
      _ = (K : ℝ≥0∞) ^ m * (∏ i, inclRatio q K f (fun _ : Fin 1 => y i)) / D :=
          ENNReal.mul_div_mul_right _ _ hW0 (hs.tupleWeight_ne_top y)
      _ = (K : ℝ≥0∞) ^ m / D * ∏ i, inclRatio q K f (fun _ : Fin 1 => y i) := by
          rw [div_eq_mul_inv, div_eq_mul_inv]; ring
  · have hR : inclRatio q K f y = 0 := by simp only [inclRatio, hy, ite_false]
    rw [hR]; exact zero_le

/-- `E[u_m²] ≤ (K^m/(K)_m)² ρ^m`: square the pointwise bound and use the independence of the
`m` draws (`tsum_fin_prod`). -/
theorem tsum_inclRatio_sq_le (hs : PoolSetting q K f) (hK : 1 ≤ K)
    (hNA : ∀ T : Finset Y, inclProb f T ≤ ∏ y ∈ T, inclProb f {y}) (m : ℕ) :
    ∑' y : Fin m → Y, tupleWeight q y * inclRatio q K f y ^ 2 ≤
      ((K : ℝ≥0∞) ^ m / ((K.descFactorial m : ℕ) : ℝ≥0∞)) ^ 2 * secondMomentU1 q K f ^ m := by
  set c := (K : ℝ≥0∞) ^ m / ((K.descFactorial m : ℕ) : ℝ≥0∞)
  set g : Y → ℝ≥0∞ := fun x => q x * inclRatio q K f (fun _ : Fin 1 => x) ^ 2 with hg
  calc ∑' y : Fin m → Y, tupleWeight q y * inclRatio q K f y ^ 2
      ≤ ∑' y : Fin m → Y, tupleWeight q y *
          (c * ∏ i, inclRatio q K f (fun _ : Fin 1 => y i)) ^ 2 := by
        refine tsum_le_tsum fun y => ?_
        gcongr
        exact inclRatio_le_prod hs hK hNA y
    _ = c ^ 2 * ∑' y : Fin m → Y, ∏ i, g (y i) := by
        rw [← ENNReal.tsum_mul_left]
        refine tsum_congr fun y => ?_
        simp only [hg, tupleWeight, mul_pow, Finset.prod_mul_distrib, Finset.prod_pow]
        ring
    _ = c ^ 2 * secondMomentU1 q K f ^ m := by
        rw [tsum_fin_prod m (fun _ => g), Finset.prod_const, Finset.card_univ, Fintype.card_fin,
          secondMomentU1_eq]

/-- `a_m (K^m/(K)_m)² r^m ≤ θ^m / m!` with `θ = K² r / (N - K + 1)`. The binomials cancel
exactly: `C(K,m)/(K)_m = 1/m!`. -/
lemma coeff_bound {m : ℕ} (hm : m ≤ K) (hKN : K ≤ N) {r : ℝ} (hr : 0 ≤ r) :
    ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ) / (N.choose K : ℝ) *
        (((K : ℝ) ^ m / (K.descFactorial m : ℝ)) ^ 2 * r ^ m) ≤
      ((K : ℝ) ^ 2 * r / ((N : ℝ) - K + 1)) ^ m / (m.factorial : ℝ) := by
  -- the integer inequality `C(N-K,K-m) (N-K+1)^m ≤ (K)_m C(N,K)`
  have hnat : (N - K).choose (K - m) * (N - K + 1) ^ m ≤ K.descFactorial m * N.choose K := by
    have e1 : N.choose K * K.choose m = N.choose m * (N - m).choose (K - m) := Nat.choose_mul hm
    have e2 : (N - K).choose (K - m) ≤ (N - m).choose (K - m) :=
      Nat.choose_le_choose _ (by omega)
    have e4 : (N - K + 1) ^ m ≤ N.descFactorial m :=
      (Nat.pow_le_pow_left (by omega : N - K + 1 ≤ N + 1 - m) m).trans
        (Nat.pow_sub_le_descFactorial N m)
    have e5 : K.descFactorial m = m.factorial * K.choose m :=
      Nat.descFactorial_eq_factorial_mul_choose K m
    have e6 : N.descFactorial m = m.factorial * N.choose m :=
      Nat.descFactorial_eq_factorial_mul_choose N m
    calc (N - K).choose (K - m) * (N - K + 1) ^ m
        ≤ (N - m).choose (K - m) * N.descFactorial m := Nat.mul_le_mul e2 e4
      _ = m.factorial * (N.choose m * (N - m).choose (K - m)) := by rw [e6]; ring
      _ = m.factorial * (N.choose K * K.choose m) := by rw [e1]
      _ = K.descFactorial m * N.choose K := by rw [e5]; ring
  have hcast : ((N - K).choose (K - m) : ℝ) * ((N : ℝ) - K + 1) ^ m ≤
      (K.descFactorial m : ℝ) * (N.choose K : ℝ) := by
    have := (Nat.cast_le (α := ℝ)).2 hnat
    push_cast [Nat.cast_sub hKN] at this
    linarith
  have ha : 0 < (K.choose m : ℝ) := by exact_mod_cast Nat.choose_pos hm
  have hC : 0 < (N.choose K : ℝ) := by exact_mod_cast Nat.choose_pos hKN
  have hfac : 0 < (m.factorial : ℝ) := by exact_mod_cast Nat.factorial_pos m
  have hdpos : 0 < (K.descFactorial m : ℝ) := by exact_mod_cast Nat.descFactorial_pos.2 hm
  have hNK : 0 < (N : ℝ) - K + 1 := by
    have : (K : ℝ) ≤ N := by exact_mod_cast hKN
    linarith
  have hd : (K.descFactorial m : ℝ) = (m.factorial : ℝ) * (K.choose m : ℝ) := by
    rw [Nat.descFactorial_eq_factorial_mul_choose]; push_cast; ring
  set X : ℝ := ((K : ℝ) ^ m) ^ 2 * r ^ m with hX
  have hX0 : 0 ≤ X := by positivity
  have hlhs : ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ) / (N.choose K : ℝ) *
      (((K : ℝ) ^ m / (K.descFactorial m : ℝ)) ^ 2 * r ^ m) =
      ((N - K).choose (K - m) : ℝ) * X /
        ((N.choose K : ℝ) * (m.factorial : ℝ) * (K.descFactorial m : ℝ)) := by
    rw [hX, hd]
    push_cast
    field_simp
  have hrhs : ((K : ℝ) ^ 2 * r / ((N : ℝ) - K + 1)) ^ m / (m.factorial : ℝ) =
      X / (((N : ℝ) - K + 1) ^ m * (m.factorial : ℝ)) := by
    rw [hX, div_pow, mul_pow, ← pow_mul, ← pow_mul, mul_comm 2 m]
    field_simp
  rw [hlhs, hrhs, div_le_div_iff₀ (by positivity) (by positivity)]
  have := mul_le_mul_of_nonneg_left hcast (mul_nonneg hX0 hfac.le)
  nlinarith [this]

/-- `a₀ = C(N-K,K)/C(N,K) ≤ 1`: it is one term of `∑_{m=0}^{K} a_m = 1`. -/
lemma overlapCoeff_zero_toReal_le_one (hKN : K ≤ N) : (overlapCoeff K N 0).toReal ≤ 1 := by
  have h : overlapCoeff K N 0 ≤ 1 := by
    rw [← sum_overlapCoeff hKN]
    exact Finset.single_le_sum (fun m _ => zero_le) (Finset.mem_range.2 (Nat.succ_pos K))
  exact ENNReal.toReal_le_of_le_ofReal zero_le_one (by simpa using h)

/-- `log(e^θ - 1 + a) ≤ θ` for `θ ≥ 0` and `0 ≤ a ≤ 1`. -/
lemma log_exp_sub_one_add_le {θ a : ℝ} (hθ : 0 ≤ θ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    Real.log (Real.exp θ - 1 + a) ≤ θ := by
  have hexp := Real.add_one_le_exp θ
  by_cases h : 0 < Real.exp θ - 1 + a
  · exact (Real.log_le_log h (by linarith)).trans (Real.log_exp θ).le
  · have h0 : Real.exp θ - 1 + a = 0 := by linarith
    rw [h0, Real.log_zero]
    exact hθ

/-- For any `m₁ ≥ 0`, `∑_{c=0}^{K} a_c (K^c/(K)_c)² m₁^c ≤ e^θ - 1 + a₀` with
`θ = K² m₁/(N - K + 1)`. At `m₁ = ρ`, the bound of `cor:kl-sum` is at most that of `thm:zfree`.
The `c = 0` terms agree, and each `c ≥ 1` term is at most `θ^c/c!` (`coeff_bound`). -/
lemma sharperBound_le_exp {K N : ℕ} (hKN : K ≤ N) {m₁ : ℝ} (hm : 0 ≤ m₁) :
    ∑ c ∈ range (K + 1), (overlapCoeff K N c).toReal *
        (((K : ℝ) ^ c / (K.descFactorial c : ℝ)) ^ 2 * m₁ ^ c) ≤
      Real.exp ((K : ℝ) ^ 2 * m₁ / ((N : ℝ) - K + 1)) - 1 + (overlapCoeff K N 0).toReal := by
  set θ : ℝ := (K : ℝ) ^ 2 * m₁ / ((N : ℝ) - K + 1)
  have hθ0 : 0 ≤ θ := by
    have : (K : ℝ) ≤ N := by exact_mod_cast hKN
    exact div_nonneg (mul_nonneg (sq_nonneg _) hm) (by linarith)
  have han : ∀ n, (overlapCoeff K N n).toReal =
      ((K.choose n * (N - K).choose (K - n) : ℕ) : ℝ) / (N.choose K : ℝ) := fun n => by
    simp [overlapCoeff, ENNReal.toReal_div]
  -- `∑_{c=1}^{K} θ^c/c! ≤ e^θ - 1`
  have hexp : ∑ c ∈ range K, θ ^ (c + 1) / ((c + 1).factorial : ℝ) ≤ Real.exp θ - 1 := by
    have h := Real.sum_le_exp_of_nonneg hθ0 (K + 1)
    rw [Finset.sum_range_succ'] at h
    simp only [pow_zero, Nat.factorial_zero, Nat.cast_one, div_one] at h
    linarith
  have hterms : ∑ c ∈ range K, (overlapCoeff K N (c + 1)).toReal *
      (((K : ℝ) ^ (c + 1) / (K.descFactorial (c + 1) : ℝ)) ^ 2 * m₁ ^ (c + 1)) ≤
      ∑ c ∈ range K, θ ^ (c + 1) / ((c + 1).factorial : ℝ) :=
    Finset.sum_le_sum fun c hc => by
      rw [han (c + 1)]
      exact coeff_bound (mem_range.1 hc) hKN hm
  have h0 : (overlapCoeff K N 0).toReal *
      (((K : ℝ) ^ 0 / (K.descFactorial 0 : ℝ)) ^ 2 * m₁ ^ 0) = (overlapCoeff K N 0).toReal := by
    simp
  rw [Finset.sum_range_succ', h0]
  linarith

/-! ### The i.i.d. pool law -/

section Measure

variable [Countable Y] [MeasurableSpace Y] [MeasurableSingletonClass Y]

/-- The i.i.d. pool law `q^{⊗N}` on `Fin N → Y`. -/
noncomputable def poolLaw (p : PMF Y) (N : ℕ) : Measure (Fin N → Y) :=
  Measure.pi fun _ => p.toMeasure

instance (p : PMF Y) (N : ℕ) : IsProbabilityMeasure (poolLaw p N) := by
  unfold poolLaw; infer_instance

omit [DecidableEq Y] in
lemma lintegral_poolLaw (p : PMF Y) (F : (Fin N → Y) → ℝ≥0∞) :
    ∫⁻ ω, F ω ∂poolLaw p N = ∑' ω, tupleWeight p ω * F ω := by
  rw [lintegral_countable']
  refine tsum_congr fun ω => ?_
  rw [mul_comm]
  congr 1
  have : ({ω} : Set (Fin N → Y)) = Set.univ.pi (fun i => {ω i}) := by
    ext x; simp [Set.mem_pi, funext_iff]
  rw [poolLaw, this, Measure.pi_pi]
  simp [tupleWeight, PMF.toMeasure_apply_singleton]

variable (p : PMF Y)

/-- **`thm:zfree`, identity part** (second-moment form; no finiteness assumption).
For `N ≥ K`, under `PoolSetting`:
* `f̂` is unbiased (`prop:PoolDPP-well-behaved`), and `Ẑ = ∑ₛ f̂(S)` is the `U`-statistic `poolZ`;
* `E[Ẑ] = Z`, `E[u_m] = 1` for `m ≤ K`, and `∑_{m=0}^{K} a_m = 1`;
* `E[Ẑ²] = Z² ∑_{m=0}^{K} a_m E[u_m²]`. -/
theorem zfree_identity (hs : PoolSetting p K f) (hKN : K ≤ N) :
    (∀ S, ∫⁻ ω, poolFhat p K f ω S ∂poolLaw p N = f S) ∧
    (∀ ω : Fin N → Y, sketchZ (poolFhat p K f) ω = poolZ p K f ω) ∧
    ∫⁻ ω, poolZ p K f ω ∂poolLaw p N = ∑' S, f S ∧
    (∀ m ≤ K, ∫⁻ y, inclRatio p K f y ∂poolLaw p m = 1) ∧
    ∑ m ∈ range (K + 1), overlapCoeff K N m = 1 ∧
    ∫⁻ ω, poolZ p K f ω ^ 2 ∂poolLaw p N =
      (∑' S, f S) ^ 2 * ∑ m ∈ range (K + 1),
        overlapCoeff K N m * ∫⁻ y, inclRatio p K f y ^ 2 ∂poolLaw p m := by
  refine ⟨fun S => ?_, sketchZ_poolFhat, ?_, fun m hm => ?_, sum_overlapCoeff hKN, ?_⟩
  · rw [lintegral_poolLaw]; exact poolFhat_unbiased hs hKN S
  · rw [lintegral_poolLaw]
    simp_rw [← sketchZ_poolFhat, sketchZ, ← ENNReal.tsum_mul_left]
    rw [tsum_comm]
    exact tsum_congr fun S => poolFhat_unbiased hs hKN S
  · rw [lintegral_poolLaw]; exact tsum_inclRatio_eq_one hs hm
  · simp_rw [lintegral_poolLaw]; exact secondMoment_eq hs hKN

/-- **`thm:zfree`, identity part, variance form** (the paper's eq. `zfree`):
`Var(Ẑ)/Z² = ∑_{m=1}^{K} a_m Var(u_m)`, with the variances in `[0, ∞]` (Mathlib's `evariance`)
and `Ẑ` = `poolZ` under the i.i.d. pool law. No moment is assumed finite: if some `E[u_m²]` is
infinite, both sides are `∞`. -/
theorem zfree_variance (hs : PoolSetting p K f) (hKN : K ≤ N) :
    evariance (fun ω => (poolZ p K f ω).toReal) (poolLaw p N) / (∑' S, f S) ^ 2 =
      ∑ m ∈ Icc 1 K, overlapCoeff K N m *
        evariance (fun y => (inclRatio p K f y).toReal) (poolLaw p m) := by
  obtain ⟨-, -, hEZ, hEu, ha, h2⟩ := zfree_identity p hs hKN
  set Z := ∑' S, f S
  set M : ℕ → ℝ≥0∞ := fun m => ∫⁻ y, inclRatio p K f y ^ 2 ∂poolLaw p m
  have hmeas : ∀ {n : ℕ} (G : (Fin n → Y) → ℝ≥0∞), AEMeasurable G (poolLaw p n) :=
    fun G => Measurable.of_discrete.aemeasurable
  have hZvar := (evariance_toReal_eq (hmeas (poolZ p K f)) (by rw [hEZ]; exact hs.Z_ne_top)).1
  -- `eVar(u_m) = E[u_m²] - 1` and `1 ≤ E[u_m²]`, since `E[u_m] = 1`
  have hu : ∀ m ≤ K, evariance (fun y => (inclRatio p K f y).toReal) (poolLaw p m) = M m - 1 ∧
      1 ≤ M m := by
    intro m hm
    have h := evariance_toReal_eq (hmeas (inclRatio p K f)) (by rw [hEu m hm]; exact ENNReal.one_ne_top)
    rw [hEu m hm, one_pow] at h
    exact h
  -- `u_0 ≡ 1`, so `E[u_0²] = 1`
  have hM0 : M 0 = 1 := by
    have hu0 : ∀ y : Fin 0 → Y, inclRatio p K f y = 1 := by
      intro y
      have h := hEu 0 (Nat.zero_le K)
      rw [lintegral_poolLaw] at h
      have hy : ∀ y' : Fin 0 → Y, y' = y := fun y' => Subsingleton.elim _ _
      rw [tsum_eq_single y (fun b hb => absurd (hy b) hb)] at h
      simpa [tupleWeight] using h
    simp only [M, hu0, one_pow]
    simp
  -- split off the `m = 0` term
  have hsplit : ∀ g : ℕ → ℝ≥0∞, ∑ m ∈ range (K + 1), g m = g 0 + ∑ m ∈ Icc 1 K, g m := by
    intro g
    rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot (by omega)]
    rfl
  -- `∑_{m=0}^{K} a_m E[u_m²] = ∑_{m=1}^{K} a_m (E[u_m²] - 1) + 1`, using `∑_m a_m = 1`
  have hS : ∑ m ∈ range (K + 1), overlapCoeff K N m * M m =
      ∑ m ∈ Icc 1 K, overlapCoeff K N m * (M m - 1) + 1 := by
    have hIcc : ∑ m ∈ Icc 1 K, overlapCoeff K N m * M m =
        ∑ m ∈ Icc 1 K, overlapCoeff K N m * (M m - 1) + ∑ m ∈ Icc 1 K, overlapCoeff K N m := by
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun m hm => ?_
      rw [← mul_add_one, tsub_add_cancel_of_le (hu m (Finset.mem_Icc.1 hm).2).2]
    rw [hsplit, hM0, mul_one, hIcc, ← ha, hsplit (fun m => overlapCoeff K N m)]
    ring
  have hZ2_0 : Z ^ 2 ≠ 0 := pow_ne_zero 2 hs.Z_ne_zero
  have hZ2_top : Z ^ 2 ≠ ∞ := ENNReal.pow_ne_top hs.Z_ne_top
  rw [hZvar, hEZ, h2, hS, mul_add_one, ENNReal.add_sub_cancel_right hZ2_top, mul_comm,
    ENNReal.mul_div_cancel_right hZ2_0 hZ2_top]
  refine Finset.sum_congr rfl fun m hm => ?_
  rw [(hu m (Finset.mem_Icc.1 hm).2).1]

/-- **`cor:kl-sum`, second-moment form**, for any target weight under negative correlation:
if `ρ = E_q[u₁²] ≤ m₁ < ∞`, then `E[Ẑ²] < ∞` and `E[Ẑ²]/Z² ≤ ∑_{c=0}^{K} a_c (K^c/(K)_c)² m₁^c`,
under the i.i.d. pool law. -/
theorem secondMoment_le_sharper (hs : PoolSetting p K f) (hK : 1 ≤ K) (hKN : K ≤ N)
    (hNA : ∀ T : Finset Y, inclProb f T ≤ ∏ y ∈ T, inclProb f {y})
    (m₁ : ℝ≥0) (hm1 : secondMomentU1 p K f ≤ m₁) :
    ∫⁻ ω, poolZ p K f ω ^ 2 ∂poolLaw p N ≠ ∞ ∧
    (∫⁻ ω, poolZ p K f ω ^ 2 ∂poolLaw p N).toReal / ((∑' S, f S).toReal) ^ 2 ≤
      ∑ c ∈ range (K + 1), (overlapCoeff K N c).toReal *
        (((K : ℝ) ^ c / (K.descFactorial c : ℝ)) ^ 2 * (m₁ : ℝ) ^ c) := by
  obtain ⟨-, -, -, -, -, h2⟩ := zfree_identity p hs hKN
  set Z := ∑' S, f S
  set B : ℕ → ℝ≥0∞ := fun m =>
    ((K : ℝ≥0∞) ^ m / ((K.descFactorial m : ℕ) : ℝ≥0∞)) ^ 2 * (m₁ : ℝ≥0∞) ^ m
  have hC0 : ((N.choose K : ℕ) : ℝ≥0∞) ≠ 0 := by exact_mod_cast (Nat.choose_pos hKN).ne'
  have ha_top : ∀ m, overlapCoeff K N m ≠ ∞ := fun m =>
    ENNReal.div_ne_top (ENNReal.natCast_ne_top _) hC0
  have hB_top : ∀ m ≤ K, B m ≠ ∞ := fun m hm =>
    ENNReal.mul_ne_top (ENNReal.pow_ne_top (ENNReal.div_ne_top
      (ENNReal.pow_ne_top (ENNReal.natCast_ne_top _))
      (by exact_mod_cast (Nat.descFactorial_pos.2 hm).ne')))
      (ENNReal.pow_ne_top ENNReal.coe_ne_top)
  -- `E[u_c²] ≤ B_c` for every `c ≤ K`
  have hM_le : ∀ m ≤ K, ∫⁻ y, inclRatio p K f y ^ 2 ∂poolLaw p m ≤ B m := by
    intro m _
    rw [lintegral_poolLaw]
    refine (tsum_inclRatio_sq_le hs hK hNA m).trans ?_
    simp only [B]
    gcongr
  have hsum_le : ∑ m ∈ range (K + 1), overlapCoeff K N m *
      ∫⁻ y, inclRatio p K f y ^ 2 ∂poolLaw p m ≤
      ∑ m ∈ range (K + 1), overlapCoeff K N m * B m :=
    Finset.sum_le_sum fun m hm => by
      gcongr; exact hM_le m (Nat.lt_succ_iff.1 (mem_range.1 hm))
  have hsumB_top : ∑ m ∈ range (K + 1), overlapCoeff K N m * B m ≠ ∞ :=
    ENNReal.sum_ne_top.2 fun m hm =>
      ENNReal.mul_ne_top (ha_top m) (hB_top m (Nat.lt_succ_iff.1 (mem_range.1 hm)))
  have hA_top : ∫⁻ ω, poolZ p K f ω ^ 2 ∂poolLaw p N ≠ ∞ := by
    rw [h2]
    exact ENNReal.mul_ne_top (ENNReal.pow_ne_top hs.Z_ne_top)
      (ne_top_of_le_ne_top hsumB_top hsum_le)
  have hZpos : 0 < Z.toReal := ENNReal.toReal_pos hs.Z_ne_zero hs.Z_ne_top
  have hBn : ∀ n, (B n).toReal =
      ((K : ℝ) ^ n / (K.descFactorial n : ℝ)) ^ 2 * (m₁ : ℝ) ^ n := fun n => by
    simp [B, ENNReal.toReal_div, ENNReal.toReal_pow, ENNReal.toReal_mul]
  refine ⟨hA_top, ?_⟩
  rw [h2, ENNReal.toReal_mul, ENNReal.toReal_pow, mul_div_cancel_left₀ _ (by positivity)]
  calc (∑ m ∈ range (K + 1), overlapCoeff K N m *
        ∫⁻ y, inclRatio p K f y ^ 2 ∂poolLaw p m).toReal
      ≤ (∑ m ∈ range (K + 1), overlapCoeff K N m * B m).toReal :=
        ENNReal.toReal_mono hsumB_top hsum_le
    _ = ∑ m ∈ range (K + 1), (overlapCoeff K N m).toReal * (B m).toReal := by
        rw [ENNReal.toReal_sum fun m hm =>
          ENNReal.mul_ne_top (ha_top m) (hB_top m (Nat.lt_succ_iff.1 (mem_range.1 hm)))]
        simp [ENNReal.toReal_mul]
    _ = _ := Finset.sum_congr rfl fun m _ => by rw [hBn m]

/-- The second-moment bound behind `thm:zfree`, for the plain i.i.d. pool law: under `hNA`
and `ρ = E_q[u₁²] ≤ m₁ < ∞`, `E[Ẑ²] < ∞` and `E[Ẑ²]/Z² ≤ e^θ - 1 + a₀` with
`θ = K² m₁/(N - K + 1)` and `a₀ = C(N-K,K)/C(N,K)` (`overlapCoeff K N 0`). It is
`secondMoment_le_sharper` (`cor:kl-sum`'s sum) followed by `sharperBound_le_exp`: the `c = 0`
term `a₀` is kept exact, and each `c ≥ 1` term is at most `θ^c/c!`. -/
theorem secondMoment_le_exp (hs : PoolSetting p K f) (hK : 1 ≤ K) (hKN : K ≤ N)
    (hNA : ∀ T : Finset Y, inclProb f T ≤ ∏ y ∈ T, inclProb f {y})
    (m₁ : ℝ≥0) (hm1 : secondMomentU1 p K f ≤ m₁) :
    ∫⁻ ω, poolZ p K f ω ^ 2 ∂poolLaw p N ≠ ∞ ∧
    (∫⁻ ω, poolZ p K f ω ^ 2 ∂poolLaw p N).toReal / ((∑' S, f S).toReal) ^ 2 ≤
      Real.exp ((K : ℝ) ^ 2 * m₁ / ((N : ℝ) - K + 1)) - 1 + (overlapCoeff K N 0).toReal := by
  obtain ⟨hA_top, hratio⟩ := secondMoment_le_sharper p hs hK hKN hNA m₁ hm1
  exact ⟨hA_top, hratio.trans (sharperBound_le_exp hKN m₁.2)⟩

end Measure

end DppLm
