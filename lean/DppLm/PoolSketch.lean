import DppLm.Tuples
import DppLm.Sketch

/-!
# The pool sketch and the exact second-moment identity (`thm:zfree`, first part)

## Setting (`PoolSetting`)

* `Y` is a countable set of strings and `q : Y → ℝ≥0∞` is the proposal (`∑ q = 1`).
* `f : Finset Y → ℝ≥0∞` is the unnormalized target on sets. For PoolDPP,
  `f S = det(L_S)` when `|S| = K` and `0` otherwise. Everything in this file uses only:
  - `f S ≠ 0 → |S| = K` (the target lives on `K`-sets);
  - `f S ≠ 0 → ∀ y ∈ S, q y ≠ 0`. This follows from the paper's
    `q(y) = 0 → κ(y) = 0`, because then `L_S` has a zero row;
  - `0 < Z = ∑ f < ∞`.
* A pool is `ω : Fin N → Y`, drawn i.i.d. from `q`. Expectations are written as
  `∑' ω, tupleWeight q ω * F ω`, which equals `∫⁻ F` under the i.i.d. law (`lintegral_pool`
  in `FiniteSample.lean`).

## The sketch

The paper defines `f̂_ω(S) = α_{N,K} ∑_{I : ω(I) = S} det(L̂_I)` with `α = N^K/(N)_K`.
With `L̂ = B̂ᵀB̂` and the columns of `B̂` equal to `√(w/N) φ`, each term with `ω` injective
on `I` equals `N^{-K} det(L_{ω(I)}) / ∏_{i∈I} q(ω_i)`, and the others vanish (repeated
column). So `f̂_ω(S) = (N)_K⁻¹ ∑_{|I|=K, ω(I)=S} f(ω(I)) / ∏_{i∈I} q(ω_i)`. That last
expression, for a general `f`, is `poolFhat` below. The determinant step is ordinary linear
algebra and is **not** formalized here.

## Lean conventions

In `ℝ≥0∞`, `x / 0 = ∞` for `x ≠ 0` and `0 / 0 = 0`. Terms with `q(ω_i) = 0` have zero
weight in every expectation, and the support hypothesis makes their numerators vanish too,
so these conventions do not affect any expectation below.
-/

open Finset
open scoped ENNReal

namespace DppLm

variable {Y : Type*} [DecidableEq Y]

/-- Standing hypotheses of `thm:zfree` (see the module docstring). -/
structure PoolSetting (q : Y → ℝ≥0∞) (K : ℕ) (f : Finset Y → ℝ≥0∞) : Prop where
  q_sum : ∑' y, q y = 1
  f_card : ∀ S, f S ≠ 0 → S.card = K
  f_supp : ∀ S, f S ≠ 0 → ∀ y ∈ S, q y ≠ 0
  Z_ne_zero : ∑' S, f S ≠ 0
  Z_ne_top : ∑' S, f S ≠ ∞

section Defs

variable (q : Y → ℝ≥0∞) (K : ℕ) (f : Finset Y → ℝ≥0∞)

/-- `H_I(ω) = f(ω(I)) / ∏_{i∈I} q(ω_i)` (equal to `K! · h(Y_I)` in the paper's notation). -/
noncomputable def idxWeight {N : ℕ} (ω : Fin N → Y) (I : Finset (Fin N)) : ℝ≥0∞ :=
  f (I.image ω) / ∏ i ∈ I, q (ω i)

/-- The PoolDPP sketch `f̂_ω(S) = (N)_K⁻¹ ∑_{|I| = K, ω(I) = S} H_I(ω)`. -/
noncomputable def poolFhat {N : ℕ} (ω : Fin N → Y) (S : Finset Y) : ℝ≥0∞ :=
  ((N.descFactorial K : ℕ) : ℝ≥0∞)⁻¹ *
    ∑ I ∈ powersetCard K (univ : Finset (Fin N)), if I.image ω = S then idxWeight q f ω I else 0

/-- The `U`-statistic `Ẑ(ω) = (N)_K⁻¹ ∑_{|I| = K} H_I(ω)`. -/
noncomputable def poolZ {N : ℕ} (ω : Fin N → Y) : ℝ≥0∞ :=
  ((N.descFactorial K : ℕ) : ℝ≥0∞)⁻¹ * ∑ I ∈ powersetCard K (univ : Finset (Fin N)),
    idxWeight q f ω I

/-- Target inclusion probability `p(T ⊆ S)`. -/
noncomputable def inclProb (T : Finset Y) : ℝ≥0∞ :=
  (∑' S, if T ⊆ S then f S else 0) / ∑' S, f S

/-- Inclusion ratio `u_m(y₁,…,y_m) = p({y₁,…,y_m} ⊆ S) / ((K)_m q(y₁)⋯q(y_m))`,
and `0` on tuples with a repeat. Defined for any finite index type; `thm:zfree` uses
`ι = Fin m`. -/
noncomputable def inclRatio {ι : Type*} [Fintype ι] (y : ι → Y) : ℝ≥0∞ :=
  if Function.Injective y then
    inclProb f (univ.image y) / (((K.descFactorial (Fintype.card ι) : ℕ) : ℝ≥0∞) * tupleWeight q y)
  else 0

/-- Overlap coefficient `a_m = C(K,m) C(N-K,K-m) / C(N,K)`. -/
noncomputable def overlapCoeff (N m : ℕ) : ℝ≥0∞ :=
  ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ≥0∞) / ((N.choose K : ℕ) : ℝ≥0∞)

/-- Conditional mean of `H_I` given the coordinates in `C`. -/
noncomputable def condMean {N : ℕ} (I C : Finset (Fin N)) (ω : Fin N → Y) : ℝ≥0∞ :=
  ∑' b, tupleWeight q b * idxWeight q f (C.piecewise ω b) I

end Defs

variable {q : Y → ℝ≥0∞} {K : ℕ} {f : Finset Y → ℝ≥0∞}

/-! ### Small facts -/

omit [DecidableEq Y] in
lemma PoolSetting.q_ne_top (hs : PoolSetting q K f) (y : Y) : q y ≠ ∞ :=
  ne_top_of_le_ne_top (by rw [hs.q_sum]; exact ENNReal.one_ne_top) (le_tsum _ y)

omit [DecidableEq Y] in
lemma PoolSetting.f_eq_zero_of_card_ne (hs : PoolSetting q K f) {S : Finset Y}
    (h : S.card ≠ K) : f S = 0 := by
  by_contra h0
  exact h (hs.f_card S h0)

omit [DecidableEq Y] in
lemma PoolSetting.tupleWeight_ne_top (hs : PoolSetting q K f) {ι : Type*} [Fintype ι]
    (c : ι → Y) : tupleWeight q c ≠ ∞ :=
  ENNReal.prod_ne_top fun i _ => hs.q_ne_top (c i)

omit [DecidableEq Y] in
lemma PoolSetting.prod_ne_top (hs : PoolSetting q K f) {ι : Type*} (s : Finset ι) (c : ι → Y) :
    ∏ i ∈ s, q (c i) ≠ ∞ :=
  ENNReal.prod_ne_top fun i _ => hs.q_ne_top (c i)

/-- If `f X ≠ 0` then every `q (c i)` with `c i ∈ X` is nonzero. -/
lemma PoolSetting.prod_ne_zero (hs : PoolSetting q K f) {ι : Type*} (s : Finset ι) (c : ι → Y)
    {X : Finset Y} (hX : f X ≠ 0) (hsub : s.image c ⊆ X) : ∏ i ∈ s, q (c i) ≠ 0 :=
  Finset.prod_ne_zero_iff.2 fun i hi =>
    hs.f_supp X hX (c i) (hsub (Finset.mem_image_of_mem c hi))

omit [DecidableEq Y] in
lemma mul_div_cancel_of_imp {x b : ℝ≥0∞} (hb : b ≠ ∞) (h : x ≠ 0 → b ≠ 0) : b * (x / b) = x := by
  by_cases hx : x = 0
  · simp [hx]
  · exact ENNReal.mul_div_cancel (h hx) hb

/-- Injectivity of a tuple is read off from the size of its image. -/
lemma card_image_lt_of_not_injective {A : Type*} [Fintype A] {c : A → Y}
    (h : ¬ Function.Injective c) : (univ.image c).card < Fintype.card A := by
  have hle : (univ.image c).card ≤ (univ : Finset A).card := Finset.card_image_le
  have hne : (univ.image c).card ≠ (univ : Finset A).card := fun he =>
    h fun x y hxy => Finset.card_image_iff.1 he (Finset.mem_coe.2 (Finset.mem_univ x))
      (Finset.mem_coe.2 (Finset.mem_univ y)) hxy
  rw [← Finset.card_univ]
  exact lt_of_le_of_ne hle hne

lemma image_eq_univ_image {ι : Type*} [DecidableEq ι] (s : Finset ι) (b : ι → Y) :
    s.image b = (univ : Finset s).image (fun i : s => b i) := by
  ext y
  simp

omit [DecidableEq Y] in
lemma prod_eq_tupleWeight {ι : Type*} (s : Finset ι) (b : ι → Y) :
    ∏ i ∈ s, q (b i) = tupleWeight q (fun i : s => b i) := by
  rw [tupleWeight, Finset.prod_coe_sort s (fun i => q (b i))]

/-! ### Locality -/

variable {N : ℕ}

lemma idxWeight_congr {ω ω' : Fin N → Y} {I : Finset (Fin N)} (h : ∀ i ∈ I, ω i = ω' i) :
    idxWeight q f ω I = idxWeight q f ω' I := by
  unfold idxWeight
  rw [Finset.image_congr (fun i hi => h i hi), Finset.prod_congr rfl (fun i hi => by rw [h i hi])]

lemma condMean_congr {ω ω' : Fin N → Y} {I C : Finset (Fin N)} (h : ∀ i ∈ C, ω i = ω' i) :
    condMean q f I C ω = condMean q f I C ω' := by
  unfold condMean
  congr 1
  funext b
  congr 2
  funext i
  by_cases hi : i ∈ C <;> simp [Finset.piecewise, hi, h]

/-! ### `Ẑ = ∑ₛ f̂(S)` -/

lemma sketchZ_poolFhat (ω : Fin N → Y) : sketchZ (poolFhat q K f) ω = poolZ q K f ω := by
  unfold sketchZ poolFhat poolZ
  rw [ENNReal.tsum_mul_left]
  congr 1
  rw [Summable.tsum_finsetSum (fun _ _ => CompleteLattice.summable)]
  refine Finset.sum_congr rfl fun I _ => ?_
  simp_rw [eq_comm (a := I.image ω)]
  exact tsum_ite_eq _ _

/-! ### Unbiasedness (`prop:PoolDPP-well-behaved`, for a general `f`) -/

/-- `E[1{ω(I) = S} H_I(ω)] = K! f(S)` for every index set of size `K`. -/
lemma tsum_weight_indicator (hs : PoolSetting q K f) {I : Finset (Fin N)} (hI : I.card = K)
    (S : Finset Y) :
    ∑' ω, tupleWeight q ω * (if I.image ω = S then idxWeight q f ω I else 0) =
      K.factorial * f S := by
  obtain ⟨G, hG⟩ : ∃ G : (I → Y) → ℝ≥0∞,
      G = fun c => if univ.image c = S then f (univ.image c) / tupleWeight q c else 0 := ⟨_, rfl⟩
  have hrw : ∀ ω : Fin N → Y, (if I.image ω = S then idxWeight q f ω I else 0) =
      G (fun i => ω i) := by
    intro ω
    simp only [hG, idxWeight, image_eq_univ_image I ω, prod_eq_tupleWeight I ω]
  simp_rw [hrw]
  rw [tsum_weight_restrict hs.q_sum I G, hG]
  have hcancel : ∀ c : I → Y,
      tupleWeight q c * (if univ.image c = S then f (univ.image c) / tupleWeight q c else 0) =
        if Function.Injective c then (if univ.image c = S then f (univ.image c) else 0) else 0 := by
    intro c
    by_cases h2 : Function.Injective c
    · simp only [h2, ite_true]
      by_cases h1 : univ.image c = S
      · simp only [h1, ite_true]
        refine mul_div_cancel_of_imp (hs.tupleWeight_ne_top c) fun hf => ?_
        exact hs.prod_ne_zero univ c hf (by rw [h1])
      · simp only [h1, ite_false, mul_zero]
    · simp only [h2, ite_false]
      by_cases h1 : univ.image c = S
      · -- not injective: the image is too small, so `f` vanishes
        have hlt : (univ.image c).card < K := by
          have := card_image_lt_of_not_injective h2
          rwa [Fintype.card_coe, hI] at this
        simp only [h1, ite_true]
        rw [h1] at hlt
        rw [hs.f_eq_zero_of_card_ne hlt.ne, ENNReal.zero_div, mul_zero]
      · simp only [h1, ite_false, mul_zero]
  simp_rw [hcancel]
  rw [tsum_injective_image (fun R => if R = S then f R else 0)]
  simp only [Fintype.card_coe, hI]
  congr 1
  rw [tsum_eq_single S]
  · by_cases hS : S.card = K
    · simp [hS]
    · simp [hS, hs.f_eq_zero_of_card_ne hS]
  · intro R hR
    simp [hR]

theorem poolFhat_unbiased (hs : PoolSetting q K f) (hKN : K ≤ N) (S : Finset Y) :
    ∑' ω : Fin N → Y, tupleWeight q ω * poolFhat q K f ω S = f S := by
  unfold poolFhat
  have hrw : ∀ ω : Fin N → Y, tupleWeight q ω * (((N.descFactorial K : ℕ) : ℝ≥0∞)⁻¹ *
      ∑ I ∈ powersetCard K (univ : Finset (Fin N)),
        if I.image ω = S then idxWeight q f ω I else 0) =
      ((N.descFactorial K : ℕ) : ℝ≥0∞)⁻¹ * ∑ I ∈ powersetCard K (univ : Finset (Fin N)),
        tupleWeight q ω * (if I.image ω = S then idxWeight q f ω I else 0) := by
    intro ω
    rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun I _ => by ring
  simp_rw [hrw]
  rw [ENNReal.tsum_mul_left, Summable.tsum_finsetSum (fun _ _ => CompleteLattice.summable)]
  rw [Finset.sum_congr rfl fun I hI =>
    tsum_weight_indicator hs (Finset.mem_powersetCard.1 hI).2 S]
  rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  have hd : ((N.descFactorial K : ℕ) : ℝ≥0∞) = (N.choose K : ℝ≥0∞) * (K.factorial : ℝ≥0∞) := by
    rw [Nat.descFactorial_eq_factorial_mul_choose]
    push_cast; ring
  have hd0 : ((N.descFactorial K : ℕ) : ℝ≥0∞) ≠ 0 := by
    exact_mod_cast (Nat.descFactorial_pos.2 hKN).ne'
  rw [← mul_assoc (N.choose K : ℝ≥0∞), ← hd, ← mul_assoc,
    ENNReal.inv_mul_cancel hd0 (ENNReal.natCast_ne_top _), one_mul]

/-! ### Conditional means -/

/-- `∑_{|R| = K-m} f(T ∪ R) = ∑_{S ⊇ T} f(S)` when `|T| = m ≤ K`. -/
lemma tsum_union_eq (hs : PoolSetting q K f) {T : Finset Y} {m : ℕ} (hT : T.card = m)
    (hm : m ≤ K) :
    ∑' R : Finset Y, (if R.card = K - m then f (T ∪ R) else 0) =
      ∑' S, if T ⊆ S then f S else 0 := by
  symm
  refine tsum_eq_tsum_of_ne_zero_bij (fun R => T ∪ R.1) ?_ ?_ ?_
  · rintro ⟨R, hR⟩ ⟨R', hR'⟩ h
    simp only [Function.mem_support, ne_eq, ite_eq_right_iff, not_forall] at hR hR'
    obtain ⟨hRc, hRf⟩ := hR
    obtain ⟨hRc', hRf'⟩ := hR'
    have hd : ∀ {R : Finset Y}, R.card = K - m → f (T ∪ R) ≠ 0 → Disjoint T R := by
      intro R hc hf
      have := hs.f_card _ hf
      exact Finset.card_union_eq_card_add_card.1 (by omega)
    simp only [Subtype.mk.injEq]
    have h1 := Finset.union_sdiff_cancel_left (hd hRc hRf)
    have h2 := Finset.union_sdiff_cancel_left (hd hRc' hRf')
    simp only at h
    rw [← h1, ← h2, h]
  · intro S hS
    simp only [Function.mem_support, ne_eq, ite_eq_right_iff, not_forall] at hS
    obtain ⟨hTS, hfS⟩ := hS
    have hU : T ∪ (S \ T) = S := Finset.union_sdiff_of_subset hTS
    have hSc := hs.f_card S hfS
    have hcard : (S \ T).card = K - m := by
      rw [Finset.card_sdiff, Finset.inter_eq_left.2 hTS]
      omega
    refine ⟨⟨S \ T, ?_⟩, ?_⟩
    · simp only [Function.mem_support, ne_eq, hcard, ite_true, hU]
      exact hfS
    · simp only [hU]
  · rintro ⟨R, hR⟩
    simp only [Function.mem_support, ne_eq, ite_eq_right_iff, not_forall] at hR
    simp [hR.1]

/-- **Conditional mean** (paper, eq. `proj-is-incl`): integrating out the coordinates of
`I` outside `C` gives `E[H_I | ω_C] = K! · Z · u_{|C|}(ω_C)`. -/
theorem condMean_eq (hs : PoolSetting q K f) {I C : Finset (Fin N)} (hI : I.card = K)
    (hC : C ⊆ I) (ω : Fin N → Y) :
    condMean q f I C ω = K.factorial * (∑' S, f S) * inclRatio q K f (fun i : C => ω i) := by
  set A := I \ C with hA
  set T := C.image ω with hT
  set P := ∏ i ∈ C, q (ω i) with hP
  have hIu : I = C ∪ A := (Finset.union_sdiff_of_subset hC).symm
  obtain ⟨G, hG⟩ : ∃ G : (A → Y) → ℝ≥0∞,
      G = fun c => f (T ∪ univ.image c) / (P * tupleWeight q c) := ⟨_, rfl⟩
  have h1 : ∀ b : Fin N → Y, idxWeight q f (C.piecewise ω b) I = G (fun i => b i) := by
    intro b
    simp only [idxWeight, hG]
    have himg : I.image (C.piecewise ω b) = T ∪ univ.image (fun i : A => b i) := by
      rw [← image_eq_univ_image, hIu, Finset.image_union]
      congr 1
      · exact Finset.image_congr (fun i hi => by simp [Finset.piecewise, Finset.mem_coe.1 hi])
      · exact Finset.image_congr (fun i hi => by
          simp [Finset.piecewise, (Finset.mem_sdiff.1 (Finset.mem_coe.1 hi)).2])
    have hprod : ∏ i ∈ I, q (C.piecewise ω b i) = P * tupleWeight q (fun i : A => b i) := by
      rw [← prod_eq_tupleWeight, hIu, Finset.prod_union Finset.disjoint_sdiff]
      congr 1
      · exact Finset.prod_congr rfl (fun i hi => by simp [Finset.piecewise, hi])
      · exact Finset.prod_congr rfl (fun i hi => by
          simp [Finset.piecewise, (Finset.mem_sdiff.1 hi).2])
    rw [himg, hprod]
  unfold condMean
  simp_rw [h1]
  rw [tsum_weight_restrict hs.q_sum A G, hG]
  have h3 : ∀ c : A → Y, tupleWeight q c * (f (T ∪ univ.image c) / (P * tupleWeight q c)) =
      f (T ∪ univ.image c) * P⁻¹ := by
    intro c
    by_cases hf : f (T ∪ univ.image c) = 0
    · simp only [hf, ENNReal.zero_div, mul_zero, zero_mul]
    · have hW0 : tupleWeight q c ≠ 0 :=
        hs.prod_ne_zero univ c hf Finset.subset_union_right
      have hP0 : P ≠ 0 := hs.prod_ne_zero C ω hf Finset.subset_union_left
      rw [div_eq_mul_inv, ENNReal.mul_inv (Or.inl hP0) (Or.inl (hs.prod_ne_top C ω))]
      calc tupleWeight q c * (f (T ∪ univ.image c) * (P⁻¹ * (tupleWeight q c)⁻¹))
          = f (T ∪ univ.image c) * P⁻¹ * (tupleWeight q c * (tupleWeight q c)⁻¹) := by ring
        _ = f (T ∪ univ.image c) * P⁻¹ := by
          rw [ENNReal.mul_inv_cancel hW0 (hs.tupleWeight_ne_top c), mul_one]
  simp_rw [h3]
  rw [ENNReal.tsum_mul_right]
  have hCK : C.card ≤ K := hI ▸ Finset.card_le_card hC
  have hAc : Fintype.card A = K - C.card := by
    rw [Fintype.card_coe, hA, Finset.card_sdiff, Finset.inter_eq_left.2 hC, hI]
  by_cases hinj : Function.Injective (fun i : C => ω i)
  · have hTc : T.card = C.card := by
      rw [hT, image_eq_univ_image, Finset.card_image_of_injective _ hinj, Finset.card_univ,
        Fintype.card_coe]
    have hins : ∀ c : A → Y,
        f (T ∪ univ.image c) = if Function.Injective c then f (T ∪ univ.image c) else 0 := by
      intro c
      by_cases hc : Function.Injective c
      · simp [hc]
      · simp only [hc, ite_false]
        apply hs.f_eq_zero_of_card_ne
        apply ne_of_lt
        have h1 := card_image_lt_of_not_injective hc
        have h2 := Finset.card_union_le T (univ.image c)
        omega
    rw [tsum_congr hins, tsum_injective_image (fun R => f (T ∪ R)), hAc,
      tsum_union_eq hs hTc hCK]
    unfold inclRatio
    simp only [hinj, ite_true]
    rw [← image_eq_univ_image C ω, ← prod_eq_tupleWeight C ω, Fintype.card_coe]
    unfold inclProb
    rw [← hT, ← hP]
    set X := ∑' S, if T ⊆ S then f S else 0 with hX
    set Z := ∑' S, f S
    have hKf : (K.factorial : ℝ≥0∞) =
        ((K - C.card).factorial : ℝ≥0∞) * ((K.descFactorial C.card : ℕ) : ℝ≥0∞) := by
      rw [← Nat.factorial_mul_descFactorial hCK]
      push_cast; ring
    have hD0 : ((K.descFactorial C.card : ℕ) : ℝ≥0∞) ≠ 0 := by
      exact_mod_cast (Nat.descFactorial_pos.2 hCK).ne'
    by_cases hP0 : P = 0
    · -- some coordinate of `ω` on `C` has zero proposal mass; then `X = 0`
      have hX0 : X = 0 := by
        refine tsum_eq_zero_iff.2 fun S => ?_
        by_cases hTS : T ⊆ S
        · simp only [hTS, ite_true]
          by_contra hfS
          exact hs.prod_ne_zero C ω hfS hTS hP0
        · simp [hTS]
      simp [hX0, hP0]
    · rw [div_eq_mul_inv, div_eq_mul_inv,
        ENNReal.mul_inv (Or.inl hD0) (Or.inl (ENNReal.natCast_ne_top _)), hKf]
      calc ((K - C.card).factorial : ℝ≥0∞) * X * P⁻¹
          = ((K - C.card).factorial : ℝ≥0∞) * X * P⁻¹ *
              (((K.descFactorial C.card : ℕ) : ℝ≥0∞) * ((K.descFactorial C.card : ℕ) : ℝ≥0∞)⁻¹) *
              (Z * Z⁻¹) := by
            rw [ENNReal.mul_inv_cancel hD0 (ENNReal.natCast_ne_top _),
              ENNReal.mul_inv_cancel hs.Z_ne_zero hs.Z_ne_top, mul_one, mul_one]
        _ = _ := by ring
  · have h0 : ∀ c : A → Y, f (T ∪ univ.image c) = 0 := by
      intro c
      apply hs.f_eq_zero_of_card_ne
      apply ne_of_lt
      have hTlt : T.card < C.card := by
        rw [hT, image_eq_univ_image]
        have := card_image_lt_of_not_injective hinj
        rwa [Fintype.card_coe] at this
      have h1 : (univ.image c).card ≤ Fintype.card A := by
        rw [← Finset.card_univ]; exact Finset.card_image_le
      have h2 := Finset.card_union_le T (univ.image c)
      omega
    simp only [h0, tsum_zero, zero_mul, inclRatio, hinj, ite_false, mul_zero]

/-- Two applications of the independent-copies identity:
`E[H_I H_J] = E[E[H_I | ω_C] E[H_J | ω_C]]` with `C = I ∩ J`. -/
theorem tsum_pair (hq : ∑' y, q y = 1) (I J : Finset (Fin N)) :
    ∑' ω, tupleWeight q ω * (idxWeight q f ω I * idxWeight q f ω J) =
      ∑' ω, tupleWeight q ω * (condMean q f I (I ∩ J) ω * condMean q f J (I ∩ J) ω) := by
  rw [tsum_weight_piecewise hq I (fun ω => idxWeight q f ω I * idxWeight q f ω J)]
  have h1 : ∀ a b : Fin N → Y, idxWeight q f (I.piecewise a b) I = idxWeight q f a I :=
    fun a b => idxWeight_congr (fun i hi => by simp [hi])
  have h2 : ∀ a b : Fin N → Y,
      idxWeight q f (I.piecewise a b) J = idxWeight q f ((I ∩ J).piecewise a b) J :=
    fun a b => idxWeight_congr (fun j hj => by by_cases hi : j ∈ I <;> simp [hi, hj])
  simp_rw [h1, h2]
  have h3 : ∀ a : Fin N → Y, ∑' b, tupleWeight q a * tupleWeight q b *
      (idxWeight q f a I * idxWeight q f ((I ∩ J).piecewise a b) J) =
      tupleWeight q a * (idxWeight q f a I * condMean q f J (I ∩ J) a) := by
    intro a
    rw [condMean, ← ENNReal.tsum_mul_left, ← ENNReal.tsum_mul_left]
    exact tsum_congr fun b => by ring
  simp_rw [h3]
  rw [tsum_weight_piecewise hq (I ∩ J)
    (fun ω => idxWeight q f ω I * condMean q f J (I ∩ J) ω)]
  have h4 : ∀ a b : Fin N → Y,
      condMean q f J (I ∩ J) ((I ∩ J).piecewise a b) = condMean q f J (I ∩ J) a :=
    fun a b => condMean_congr (fun i hi => by simp [hi])
  simp_rw [h4]
  refine tsum_congr fun a => ?_
  have hI : condMean q f I (I ∩ J) a =
      ∑' b, tupleWeight q b * idxWeight q f ((I ∩ J).piecewise a b) I := rfl
  rw [hI, ← ENNReal.tsum_mul_right, ← ENNReal.tsum_mul_left]
  exact tsum_congr fun b => by ring

/-! ### Relabeling tuples -/

omit [DecidableEq Y] in
lemma tupleWeight_comp_equiv {ι ι' : Type*} [Fintype ι] [Fintype ι'] (e : ι ≃ ι')
    (c : ι' → Y) : tupleWeight q (c ∘ e) = tupleWeight q c :=
  Fintype.prod_equiv e _ _ (fun _ => rfl)

lemma inclRatio_comp_equiv {ι ι' : Type*} [Fintype ι] [Fintype ι'] [DecidableEq ι']
    (e : ι ≃ ι') (c : ι' → Y) : inclRatio q K f (c ∘ e) = inclRatio q K f c := by
  unfold inclRatio
  have himg : univ.image (c ∘ e) = univ.image c := by
    rw [← Finset.image_image, Finset.image_univ_equiv]
  have hinj : Function.Injective (c ∘ e) ↔ Function.Injective c := Equiv.injective_comp e c
  by_cases h : Function.Injective c
  · simp only [hinj.2 h, h, ite_true, himg, Fintype.card_congr e, tupleWeight_comp_equiv]
  · simp only [mt hinj.1 h, h, ite_false]

lemma tsum_inclRatio_sq_equiv {ι ι' : Type*} [Fintype ι] [Fintype ι'] [DecidableEq ι']
    (e : ι ≃ ι') :
    ∑' c : ι → Y, tupleWeight q c * inclRatio q K f c ^ 2 =
      ∑' c : ι' → Y, tupleWeight q c * inclRatio q K f c ^ 2 := by
  rw [← (e.arrowCongr (Equiv.refl Y)).symm.tsum_eq]
  refine tsum_congr fun c => ?_
  have : (e.arrowCongr (Equiv.refl Y)).symm c = c ∘ e := by
    funext i; simp [Equiv.arrowCongr]
  rw [this, tupleWeight_comp_equiv, inclRatio_comp_equiv]

/-- `E[H_I H_J] = (K! Z)² E[u_m²]` with `m = |I ∩ J|`. -/
theorem tsum_pair_eq (hs : PoolSetting q K f) {I J : Finset (Fin N)} (hI : I.card = K)
    (hJ : J.card = K) :
    ∑' ω, tupleWeight q ω * (idxWeight q f ω I * idxWeight q f ω J) =
      (K.factorial * ∑' S, f S) ^ 2 *
        ∑' y : Fin (I ∩ J).card → Y, tupleWeight q y * inclRatio q K f y ^ 2 := by
  rw [tsum_pair hs.q_sum I J]
  obtain ⟨G, hG⟩ : ∃ G : ((I ∩ J : Finset (Fin N)) → Y) → ℝ≥0∞,
      G = fun c => inclRatio q K f c ^ 2 := ⟨_, rfl⟩
  have hc : ∀ ω : Fin N → Y,
      tupleWeight q ω * (condMean q f I (I ∩ J) ω * condMean q f J (I ∩ J) ω) =
        (K.factorial * ∑' S, f S) ^ 2 * (tupleWeight q ω * G (fun i => ω i)) := by
    intro ω
    rw [condMean_eq hs hI Finset.inter_subset_left, condMean_eq hs hJ Finset.inter_subset_right,
      hG]
    ring
  simp_rw [hc]
  rw [ENNReal.tsum_mul_left, tsum_weight_restrict hs.q_sum (I ∩ J) G, hG]
  congr 1
  exact tsum_inclRatio_sq_equiv (I ∩ J).equivFin

/-! ### Counting pairs of index sets by overlap -/

omit [DecidableEq Y] in
lemma card_filter_inter {I : Finset (Fin N)} (hI : I.card = K) {m : ℕ} (hm : m ≤ K) :
    ((powersetCard K (univ : Finset (Fin N))).filter (fun J => (I ∩ J).card = m)).card =
      K.choose m * (N - K).choose (K - m) := by
  calc _ = (powersetCard m I ×ˢ powersetCard (K - m) Iᶜ).card := by
        refine Finset.card_nbij' (fun J => (I ∩ J, J \ I)) (fun p => p.1 ∪ p.2) ?_ ?_ ?_ ?_
        · intro J hJ
          simp only [coe_filter, mem_powersetCard, Set.mem_ofPred_eq, subset_univ,
            true_and] at hJ
          simp only [coe_product, Set.mem_prod, mem_coe, mem_powersetCard]
          refine ⟨⟨Finset.inter_subset_left, hJ.2⟩, ⟨?_, ?_⟩⟩
          · intro x hx
            simp only [mem_sdiff] at hx
            simpa using hx.2
          · have := Finset.card_sdiff_add_card_inter J I
            rw [Finset.inter_comm] at this
            omega
        · rintro ⟨B, C⟩ h
          simp only [coe_product, Set.mem_prod, mem_coe, mem_powersetCard] at h
          obtain ⟨⟨hB, hBc⟩, ⟨hC, hCc⟩⟩ := h
          have hdisj : Disjoint B C :=
            Finset.disjoint_of_subset_left hB (Finset.disjoint_of_subset_right hC
              disjoint_compl_right)
          have hIBC : I ∩ (B ∪ C) = B := by
            ext x
            simp only [mem_inter, mem_union]
            constructor
            · rintro ⟨hxI, hxB | hxC⟩
              · exact hxB
              · exact absurd hxI (by simpa using hC hxC)
            · intro hxB
              exact ⟨hB hxB, Or.inl hxB⟩
          simp only [coe_filter, mem_powersetCard, Set.mem_ofPred_eq, subset_univ, true_and]
          refine ⟨by rw [Finset.card_union_of_disjoint hdisj]; omega, by rw [hIBC, hBc]⟩
        · intro J _
          ext x
          simp only [mem_union, mem_inter, mem_sdiff]
          tauto
        · rintro ⟨B, C⟩ h
          simp only [coe_product, Set.mem_prod, mem_coe, mem_powersetCard] at h
          obtain ⟨⟨hB, _⟩, ⟨hC, _⟩⟩ := h
          have hCI : ∀ x ∈ C, x ∉ I := fun x hx => by simpa using hC hx
          simp only [Prod.mk.injEq]
          constructor
          · ext x
            simp only [mem_inter, mem_union]
            constructor
            · rintro ⟨hxI, hxB | hxC⟩
              · exact hxB
              · exact absurd hxI (hCI x hxC)
            · intro hxB
              exact ⟨hB hxB, Or.inl hxB⟩
          · ext x
            simp only [mem_sdiff, mem_union]
            constructor
            · rintro ⟨hxB | hxC, hxI⟩
              · exact absurd (hB hxB) hxI
              · exact hxC
            · intro hxC
              exact ⟨Or.inr hxC, hCI x hxC⟩
    _ = K.choose m * (N - K).choose (K - m) := by
        rw [card_product, card_powersetCard, card_powersetCard, hI, card_compl,
          Fintype.card_fin, hI]

omit [DecidableEq Y] in
lemma sum_sum_inter (M : ℕ → ℝ≥0∞) :
    ∑ I ∈ powersetCard K (univ : Finset (Fin N)), ∑ J ∈ powersetCard K univ, M (I ∩ J).card =
      (N.choose K : ℝ≥0∞) * ∑ m ∈ range (K + 1),
        ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ≥0∞) * M m := by
  have hinner : ∀ I ∈ powersetCard K (univ : Finset (Fin N)),
      ∑ J ∈ powersetCard K univ, M (I ∩ J).card =
        ∑ m ∈ range (K + 1), ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ≥0∞) * M m := by
    intro I hI
    have hIc : I.card = K := (mem_powersetCard.1 hI).2
    rw [← Finset.sum_fiberwise_of_maps_to (g := fun J => (I ∩ J).card) (t := range (K + 1))]
    · refine Finset.sum_congr rfl fun m hm => ?_
      have hmK : m ≤ K := Nat.lt_succ_iff.1 (mem_range.1 hm)
      rw [Finset.sum_congr rfl (fun J hJ => by rw [(mem_filter.1 hJ).2]), sum_const,
        card_filter_inter hIc hmK, nsmul_eq_mul]
    · intro J _
      rw [mem_range, Nat.lt_succ_iff, ← hIc]
      exact card_le_card Finset.inter_subset_left
  rw [Finset.sum_congr rfl hinner, sum_const, card_powersetCard, card_univ, Fintype.card_fin,
    nsmul_eq_mul]

/-! ### The second-moment identity -/

/-- **`thm:zfree`, identity (second-moment form).**
`E[Ẑ²] = Z² ∑_{m=0}^{K} a_m E[u_m²]`. Since `E[Ẑ] = Z`, `E[u_m] = 1`, `u_0 ≡ 1` and
`∑ a_m = 1`, subtracting `Z² = Z² ∑ a_m · 1` gives the paper's variance form. -/
theorem secondMoment_eq (hs : PoolSetting q K f) (hKN : K ≤ N) :
    ∑' ω : Fin N → Y, tupleWeight q ω * poolZ q K f ω ^ 2 =
      (∑' S, f S) ^ 2 * ∑ m ∈ range (K + 1),
        overlapCoeff K N m * ∑' y : Fin m → Y, tupleWeight q y * inclRatio q K f y ^ 2 := by
  set c := ((N.descFactorial K : ℕ) : ℝ≥0∞)⁻¹ with hc
  set P := powersetCard K (univ : Finset (Fin N)) with hP
  set M : ℕ → ℝ≥0∞ := fun m => ∑' y : Fin m → Y, tupleWeight q y * inclRatio q K f y ^ 2
    with hM
  have hexp : ∀ ω : Fin N → Y, tupleWeight q ω * poolZ q K f ω ^ 2 =
      c ^ 2 * ∑ I ∈ P, ∑ J ∈ P,
        tupleWeight q ω * (idxWeight q f ω I * idxWeight q f ω J) := by
    intro ω
    unfold poolZ
    rw [mul_pow, sq (∑ I ∈ P, _), Finset.sum_mul_sum]
    simp_rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun I _ => Finset.sum_congr rfl fun J _ => by rw [hc]; ring
  simp_rw [hexp]
  rw [ENNReal.tsum_mul_left, Summable.tsum_finsetSum (fun _ _ => CompleteLattice.summable)]
  simp_rw [Summable.tsum_finsetSum (fun _ _ => CompleteLattice.summable)]
  rw [Finset.sum_congr rfl fun I hI => Finset.sum_congr rfl fun J hJ =>
    tsum_pair_eq hs (mem_powersetCard.1 hI).2 (mem_powersetCard.1 hJ).2]
  simp_rw [← Finset.mul_sum]
  rw [sum_sum_inter M]
  have hC0 : ((N.choose K : ℕ) : ℝ≥0∞) ≠ 0 := by exact_mod_cast (Nat.choose_pos hKN).ne'
  have hCtop : ((N.choose K : ℕ) : ℝ≥0∞) ≠ ∞ := ENNReal.natCast_ne_top _
  have hcK : c * ((K.factorial : ℝ≥0∞) * (N.choose K : ℝ≥0∞)) = 1 := by
    rw [hc, show ((K.factorial : ℝ≥0∞) * (N.choose K : ℝ≥0∞)) =
      ((N.descFactorial K : ℕ) : ℝ≥0∞) by
        rw [Nat.descFactorial_eq_factorial_mul_choose]; push_cast; ring]
    exact ENNReal.inv_mul_cancel (by exact_mod_cast (Nat.descFactorial_pos.2 hKN).ne')
      (ENNReal.natCast_ne_top _)
  have hsum : ∑ m ∈ range (K + 1), overlapCoeff K N m * M m =
      ((N.choose K : ℕ) : ℝ≥0∞)⁻¹ * ∑ m ∈ range (K + 1),
        ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ≥0∞) * M m := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun m _ => ?_
    rw [overlapCoeff, div_eq_mul_inv]
    ring
  rw [hsum]
  calc c ^ 2 * ((K.factorial * ∑' S, f S) ^ 2 * ((N.choose K : ℝ≥0∞) *
        ∑ m ∈ range (K + 1), ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ≥0∞) * M m))
      = c ^ 2 * ((K.factorial * ∑' S, f S) ^ 2 * ((N.choose K : ℝ≥0∞) *
        ∑ m ∈ range (K + 1), ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ≥0∞) * M m)) *
          ((N.choose K : ℝ≥0∞) * ((N.choose K : ℕ) : ℝ≥0∞)⁻¹) := by
        rw [ENNReal.mul_inv_cancel hC0 hCtop, mul_one]
    _ = (c * ((K.factorial : ℝ≥0∞) * (N.choose K : ℝ≥0∞))) ^ 2 * ((∑' S, f S) ^ 2 *
        (((N.choose K : ℕ) : ℝ≥0∞)⁻¹ * ∑ m ∈ range (K + 1),
          ((K.choose m * (N - K).choose (K - m) : ℕ) : ℝ≥0∞) * M m)) := by ring
    _ = _ := by rw [hcK, one_pow, one_mul]

end DppLm
