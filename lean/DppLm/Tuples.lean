import Mathlib

/-!
# Sums over tuples with product weights

Generic lemmas used in the proof of `thm:zfree`. Throughout, `q : Y → ℝ≥0∞` is a
probability mass function (`∑' y, q y = 1`) and a tuple `ω : ι → Y` (with `ι` finite)
has weight `W(ω) = ∏ i, q (ω i)`; sums `∑' ω, W ω * F ω` are expectations under i.i.d.
draws from `q`.
-/

open Finset
open scoped ENNReal

namespace DppLm

variable {Y : Type*}

/-- `∑_{ω : Fin n → Y} ∏ᵢ gᵢ(ωᵢ) = ∏ᵢ ∑_y gᵢ(y)`. -/
theorem tsum_fin_prod : ∀ (n : ℕ) (g : Fin n → Y → ℝ≥0∞),
    ∑' ω : Fin n → Y, ∏ i, g i (ω i) = ∏ i, ∑' y, g i y
  | 0, g => by simp
  | n + 1, g => by
    rw [← (Fin.consEquiv (fun _ : Fin (n + 1) => Y)).tsum_eq]
    simp only [Fin.consEquiv, Equiv.coe_fn_mk, Fin.prod_univ_succ, Fin.cons_zero, Fin.cons_succ]
    rw [tsum_prod (f := fun (a : Y) (b : Fin n → Y) => g 0 a * ∏ i : Fin n, g i.succ (b i))]
    simp_rw [ENNReal.tsum_mul_left, ENNReal.tsum_mul_right]
    rw [tsum_fin_prod n (fun i => g i.succ)]

/-- `∑_{ω : ι → Y} ∏ᵢ gᵢ(ωᵢ) = ∏ᵢ ∑_y gᵢ(y)` for any finite index type. -/
theorem tsum_pi_prod {ι : Type*} [Fintype ι] (g : ι → Y → ℝ≥0∞) :
    ∑' ω : ι → Y, ∏ i, g i (ω i) = ∏ i, ∑' y, g i y := by
  let e := Fintype.equivFin ι
  rw [← (e.arrowCongr (Equiv.refl Y)).symm.tsum_eq]
  have h1 : ∀ ω : Fin (Fintype.card ι) → Y,
      ∏ i, g i (((e.arrowCongr (Equiv.refl Y)).symm ω) i) = ∏ j, g (e.symm j) (ω j) := by
    intro ω
    simp only [Equiv.arrowCongr_symm, Equiv.arrowCongr_apply, Equiv.symm_symm, Equiv.refl_symm,
      Equiv.coe_refl, Function.comp_apply, id]
    exact Fintype.prod_equiv e _ _ (fun i => by simp)
  simp_rw [h1]
  rw [tsum_fin_prod]
  exact (Fintype.prod_equiv e _ _ (fun i => by simp)).symm

variable (q : Y → ℝ≥0∞)

/-- Product weight of a tuple. -/
noncomputable def tupleWeight {ι : Type*} [Fintype ι] (ω : ι → Y) : ℝ≥0∞ := ∏ i, q (ω i)

variable {q}

theorem tsum_tupleWeight {ι : Type*} [Fintype ι] (hq : ∑' y, q y = 1) :
    ∑' ω : ι → Y, tupleWeight q ω = 1 := by
  unfold tupleWeight
  rw [tsum_pi_prod (fun _ => q)]
  simp [hq]

/-- **Independent-copies identity.** Integrating `F` against one i.i.d. tuple equals
integrating `F (s.piecewise a b)` against two independent i.i.d. tuples `a, b`. -/
theorem tsum_weight_piecewise {ι : Type*} [Fintype ι] [DecidableEq ι] (hq : ∑' y, q y = 1)
    (s : Finset ι) (F : (ι → Y) → ℝ≥0∞) :
    ∑' ω, tupleWeight q ω * F ω =
      ∑' a, ∑' b, tupleWeight q a * tupleWeight q b * F (s.piecewise a b) := by
  let σ : (ι → Y) × (ι → Y) → (ι → Y) × (ι → Y) :=
    fun p => (s.piecewise p.1 p.2, s.piecewise p.2 p.1)
  have hσ : Function.Involutive σ := by
    intro p
    ext i <;> by_cases hi : i ∈ s <;> simp [σ, Finset.piecewise, hi]
  have hW : ∀ p : (ι → Y) × (ι → Y),
      tupleWeight q (σ p).1 * tupleWeight q (σ p).2 = tupleWeight q p.1 * tupleWeight q p.2 := by
    intro p
    simp only [tupleWeight, σ, ← Finset.prod_mul_distrib]
    refine Finset.prod_congr rfl fun i _ => ?_
    by_cases hi : i ∈ s <;> simp [Finset.piecewise, hi, mul_comm]
  rw [← tsum_prod (f := fun a b => tupleWeight q a * tupleWeight q b * F (s.piecewise a b))]
  rw [← (hσ.toPerm σ).tsum_eq]
  simp only [Function.Involutive.coe_toPerm]
  have : ∀ p : (ι → Y) × (ι → Y),
      tupleWeight q (σ p).1 * tupleWeight q (σ p).2 * F (s.piecewise (σ p).1 (σ p).2) =
        tupleWeight q p.1 * tupleWeight q p.2 * F p.1 := by
    intro p
    have h2 : σ (σ p) = p := hσ p
    have h3 : (s.piecewise (σ p).1 (σ p).2) = p.1 := by
      have := congrArg Prod.fst h2
      simpa [σ] using this
    rw [hW, h3]
  simp_rw [this]
  rw [tsum_prod (f := fun a b => tupleWeight q a * tupleWeight q b * F a)]
  simp_rw [mul_right_comm _ _ (F _), ENNReal.tsum_mul_left, tsum_tupleWeight hq, mul_one]

/-- **Marginalization.** If `G` depends only on the coordinates in `s`, the expectation over
the full tuple equals the expectation over the `s`-coordinates alone. -/
theorem tsum_weight_restrict {ι : Type*} [Fintype ι] [DecidableEq ι] (hq : ∑' y, q y = 1)
    (s : Finset ι) (G : (s → Y) → ℝ≥0∞) :
    ∑' ω : ι → Y, tupleWeight q ω * G (fun i => ω i) =
      ∑' c : s → Y, tupleWeight q c * G c := by
  let e := Equiv.piEquivPiSubtypeProd (· ∈ s) (fun _ : ι => Y)
  rw [← e.symm.tsum_eq]
  have hW : ∀ p : ({i // i ∈ s} → Y) × ({i // i ∉ s} → Y),
      tupleWeight q (e.symm p) = tupleWeight q p.1 * tupleWeight q p.2 := by
    intro p
    simp only [tupleWeight]
    rw [← Fintype.prod_subtype_mul_prod_subtype (· ∈ s) (fun i => q (e.symm p i))]
    congr 1
    · refine Finset.prod_congr (by ext; simp) fun i _ => ?_
      simp [e, Equiv.piEquivPiSubtypeProd, i.2]
    · refine Finset.prod_congr rfl fun i _ => ?_
      simp [e, Equiv.piEquivPiSubtypeProd, i.2]
  have hG : ∀ p : ({i // i ∈ s} → Y) × ({i // i ∉ s} → Y),
      G (fun i => e.symm p i) = G p.1 := by
    intro p
    congr 1
    funext i
    simp [e, Equiv.piEquivPiSubtypeProd, i.2]
  simp_rw [hW, hG]
  rw [tsum_prod (f := fun a b => tupleWeight q a * tupleWeight q b * G a)]
  simp_rw [mul_right_comm _ _ (G _), ENNReal.tsum_mul_left, tsum_tupleWeight hq, mul_one]

/-! ### Counting injective tuples by their image -/

section Count

variable [DecidableEq Y]

/-- Tuples with image `R` (when `|R| = |A|`) are the bijections `A ≃ R`. -/
noncomputable def fiberEquiv {A : Type*} [Fintype A] (R : Finset Y)
    (hR : R.card = Fintype.card A) :
    {c : A → Y // Finset.univ.image c = R} ≃ (A ≃ R) where
  toFun c :=
    have hinj : Function.Injective c.1 := by
      have h : (Finset.univ.image c.1).card = (Finset.univ : Finset A).card := by
        rw [c.2, hR, Finset.card_univ]
      intro x y hxy
      exact Finset.card_image_iff.1 h (Finset.mem_coe.2 (Finset.mem_univ x))
        (Finset.mem_coe.2 (Finset.mem_univ y)) hxy
    Equiv.ofBijective (fun a => ⟨c.1 a, by
        have h := Finset.mem_image_of_mem c.1 (Finset.mem_univ a)
        rwa [c.2] at h⟩)
      ⟨fun x y h => hinj (congrArg Subtype.val h), fun ⟨y, hy⟩ => by
        rw [← c.2] at hy
        obtain ⟨a, _, ha⟩ := Finset.mem_image.1 hy
        exact ⟨a, Subtype.ext ha⟩⟩
  invFun σ := ⟨fun a => (σ a : Y), by
    ext y
    simp only [Finset.mem_image, Finset.mem_univ, true_and]
    constructor
    · rintro ⟨a, rfl⟩; exact (σ a).2
    · intro hy; exact ⟨σ.symm ⟨y, hy⟩, by simp⟩⟩
  left_inv c := by ext a; rfl
  right_inv σ := by ext a; rfl

/-- **Counting lemma.** Summing a function of the image over injective tuples `c : A → Y`
counts each `|A|`-set `R` exactly `|A|!` times (once per bijection `A ≃ R`). -/
theorem tsum_injective_image {A : Type*} [Fintype A] [DecidableEq A] (G : Finset Y → ℝ≥0∞) :
    ∑' c : A → Y, (if Function.Injective c then G (Finset.univ.image c) else 0) =
      (Fintype.card A).factorial *
        ∑' R : Finset Y, (if R.card = Fintype.card A then G R else 0) := by
  set img : (A → Y) → Finset Y := fun c => Finset.univ.image c
  rw [← (Equiv.sigmaFiberEquiv img).tsum_eq, tsum_sigma', ← ENNReal.tsum_mul_left]
  refine tsum_congr fun R => ?_
  by_cases hR : R.card = Fintype.card A
  · simp only [hR, ite_true]
    have hinj : ∀ c : {c // img c = R}, Function.Injective c.1 := by
      intro c x y hxy
      have h : (Finset.univ.image c.1).card = (Finset.univ : Finset A).card := by
        rw [show Finset.univ.image c.1 = R from c.2, hR, Finset.card_univ]
      exact Finset.card_image_iff.1 h (Finset.mem_coe.2 (Finset.mem_univ x))
        (Finset.mem_coe.2 (Finset.mem_univ y)) hxy
    have : ∀ c : {c // img c = R},
        (if Function.Injective ((Equiv.sigmaFiberEquiv img) ⟨R, c⟩) then
          G (Finset.univ.image ((Equiv.sigmaFiberEquiv img) ⟨R, c⟩)) else 0) = G R := by
      intro c
      simp only [Equiv.sigmaFiberEquiv_apply]
      simp only [hinj c, ite_true]
      exact congrArg G c.2
    simp_rw [this]
    rw [← (fiberEquiv R hR).symm.tsum_eq, tsum_fintype, Finset.sum_const, Finset.card_univ,
      nsmul_eq_mul]
    congr 1
    rw [Fintype.card_equiv (Fintype.equivOfCardEq (by simpa using hR.symm))]
  · simp only [hR, ite_false, mul_zero]
    refine tsum_eq_zero_iff.2 fun c => ?_
    simp only [Equiv.sigmaFiberEquiv_apply]
    refine ite_eq_right_iff.2 fun hc => ?_
    exfalso
    apply hR
    rw [← show Finset.univ.image c.1 = R from c.2, Finset.card_image_of_injective _ hc,
      Finset.card_univ]

end Count

end DppLm
