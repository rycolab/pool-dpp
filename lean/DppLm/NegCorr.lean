import DppLm.GramDPP
import DppLm.Determinant
import DppLm.Truncation

/-!
# Negative correlation for the paper's `K`-DPP

`kdpp_negCorr` proves `p(T ⊆ S) ≤ ∏_{y∈T} p(y ∈ S)` for the `K`-DPP target
`f(S) = det(L_S) 1{|S| = K}` with the quality–diversity kernel
`L(x, y) = √κ(x) φ(x)ᵀφ(y) √κ(y)`, on the countable base set `Y`. It needs only `0 < Z < ∞`.

This is the inequality that `thm:zfree` uses. The paper cites Borcea–Brändén–Liggett for it;
here it is proved:
* on finite base sets by `gram_negCorr` (`GramDPP.lean`), with `ψ(y) = √κ(y) φ(y)`;
* on `Y` by the limit step `negCorr_of_restrict` (`Truncation.lean`).
-/

open Finset Matrix
open scoped ENNReal

namespace DppLm

variable {Y : Type*} [DecidableEq Y] {D : ℕ}

omit [DecidableEq Y] in
/-- `L = [ψ(x)ᵀψ(y)]` with `ψ(y) = √κ(y) φ(y)`. -/
lemma kernelOn_eq_gramOn (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (S : Finset Y) :
    kernelOn κ φ S = gramOn (fun y => Real.sqrt (κ y) • φ y) S := by
  ext x y
  simp only [kernelOn, gramOn, of_apply, smul_dotProduct, dotProduct_smul, smul_eq_mul]
  ring

/-- The inclusion sums of the restricted `K`-DPP weight are the real sums `inclN`. -/
lemma tsum_restrict_kdpp (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (K : ℕ) (U T : Finset Y) :
    ∑' S, (if T ⊆ S then restrictW (kdppWeight κ φ K) U S else 0) =
      ENNReal.ofReal (inclN (fun y => Real.sqrt (κ y) • φ y) K U T) := by
  set ψ : Y → Fin D → ℝ := fun y => Real.sqrt (κ y) • φ y
  rw [tsum_eq_sum (s := U.powerset)]
  · unfold inclN
    rw [ENNReal.ofReal_sum_of_nonneg (fun S _ => gw_nonneg ψ S), powersetCard_eq_filter,
      filter_filter, sum_filter]
    refine sum_congr rfl fun S hS => ?_
    rw [mem_powerset] at hS
    by_cases hT : T ⊆ S <;> by_cases hK : S.card = K <;>
      simp [restrictW, kdppWeight, hS, hT, hK, gw, kernelOn_eq_gramOn, ψ]
  · intro S hS
    rw [mem_powerset] at hS
    simp [restrictW, hS]

lemma tsum_restrict_kdpp_univ (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (K : ℕ) (U : Finset Y) :
    ∑' S, restrictW (kdppWeight κ φ K) U S =
      ENNReal.ofReal (partZ (fun y => Real.sqrt (κ y) • φ y) K U) := by
  rw [← inclN_empty, ← tsum_restrict_kdpp]
  simp

/-- **Negative correlation on finite base sets**, in the form `negCorr_of_restrict` takes. -/
theorem kdpp_negCorr_finite (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (K : ℕ) (U : Finset Y)
    (hU : ∑' S, restrictW (kdppWeight κ φ K) U S ≠ 0) (T : Finset Y) :
    inclProb (restrictW (kdppWeight κ φ K) U) T ≤
      ∏ y ∈ T, inclProb (restrictW (kdppWeight κ φ K) U) {y} := by
  set ψ : Y → Fin D → ℝ := fun y => Real.sqrt (κ y) • φ y
  rw [tsum_restrict_kdpp_univ] at hU
  have hZ : 0 < partZ ψ K U := by
    by_contra h
    exact hU (ENNReal.ofReal_eq_zero.2 (not_lt.1 h))
  have hp : ∀ A : Finset Y, inclProb (restrictW (kdppWeight κ φ K) U) A =
      ENNReal.ofReal (inclN ψ K U A / partZ ψ K U) := by
    intro A
    unfold inclProb
    rw [tsum_restrict_kdpp, tsum_restrict_kdpp_univ, ENNReal.ofReal_div_of_pos hZ]
  simp_rw [hp]
  rw [← ENNReal.ofReal_prod_of_nonneg fun y _ => div_nonneg (inclN_nonneg ψ K U {y}) hZ.le]
  exact ENNReal.ofReal_le_ofReal (gram_negCorr ψ K U T)

/-- **Negative correlation for the paper's `K`-DPP** on the countable base set:
`p(T ⊆ S) ≤ ∏_{y∈T} p(y ∈ S)` whenever `0 < Z < ∞`. -/
theorem kdpp_negCorr (κ : Y → ℝ) (φ : Y → Fin D → ℝ) (K : ℕ)
    (hZ0 : ∑' S, kdppWeight κ φ K S ≠ 0) (hZtop : ∑' S, kdppWeight κ φ K S ≠ ∞)
    (T : Finset Y) :
    inclProb (kdppWeight κ φ K) T ≤ ∏ y ∈ T, inclProb (kdppWeight κ φ K) {y} :=
  negCorr_of_restrict _ hZ0 hZtop (fun U hU T => kdpp_negCorr_finite κ φ K U hU T) T

end DppLm
