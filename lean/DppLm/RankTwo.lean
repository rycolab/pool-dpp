import DppLm.Newton

/-!
# Two-string negative correlation in the diagonal case

Let `M = diag(μ)` with `μ ≥ 0`, and let `u, w` be vectors. Write `[P]_K` for the coefficient of
`X^K` in a polynomial `P`, and
  `P(u, w) = det(1 + X (diag μ + u uᵀ + w wᵀ))`.
`rank_two_rayleigh` proves
  `[P(0,0)]_K [P(u,w)]_K ≤ [P(u,0)]_K [P(0,w)]_K`.
In `GramDPP.lean` the four coefficients are the `K`-DPP normalizers with neither, one, or both
of two strings `a, b` in the base set. The inequality is then
`p(a, b ∈ S) ≤ p(a ∈ S) p(b ∈ S)`.

The proof:
* `det_diagonal_add_rank_two`: for invertible `diag g`, the matrix determinant lemma gives
  `det(diag g + c (u uᵀ + w wᵀ)) = ∏ g + c ∑_i (u_i² + w_i²) ∏_{l≠i} g_l
     + c² ∑_{i,j} T_ij ∏_{l≠i,j} g_l`, with `T_ij = u_i² w_j² - u_i w_i u_j w_j`.
* `det_one_add_X_smul_rank_two`: with `g = 1 + x μ` (invertible for `x ≥ 0`), the identity
  holds at every `x ≥ 0`, so it holds as an identity of polynomials in `X`.
* `rank_two_rayleigh`: reading off the coefficient of `X^K` gives `A`, `A + B`, `A + C`,
  `A + B + C + E`, with `A = e_K(μ)`, `B = ∑ u_i² c_i`, `C = ∑ w_i² c_i`,
  `E = ∑ T_ij d_ij`, `c_i = e_{K-1}(μ_{-i})`, `d_ij = e_{K-2}(μ_{-ij})`. Then
  `BC - AE = (∑ u_i w_i c_i)² + ½ ∑_{i,j} (u_i w_j - u_j w_i)² (c_i c_j - A d_ij)`,
  and for `i ≠ j`, `c_i c_j - A d_ij = e_{K-1}(μ_{-ij})² - e_K(μ_{-ij}) e_{K-2}(μ_{-ij}) ≥ 0`
  by Newton's inequality.
-/

open Finset Polynomial Matrix

namespace DppLm

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- `T_ij = u_i² w_j² - u_i w_i u_j w_j`. Its symmetrization `T_ij + T_ji` is
`(u_i w_j - u_j w_i)²`, and `T_ii = 0`. -/
def cross (u w : ι → ℝ) (i j : ι) : ℝ := u i ^ 2 * w j ^ 2 - u i * w i * (u j * w j)

omit [Fintype ι] [DecidableEq ι] in
lemma cross_self (u w : ι → ℝ) (i : ι) : cross u w i i = 0 := by unfold cross; ring

omit [Fintype ι] [DecidableEq ι] in
lemma cross_add_cross (u w : ι → ℝ) (i j : ι) :
    cross u w i j + cross u w j i = (u i * w j - u j * w i) ^ 2 := by unfold cross; ring

omit [Fintype ι] [DecidableEq ι] in
@[simp] lemma cross_zero_left (w : ι → ℝ) (i j : ι) : cross 0 w i j = 0 := by simp [cross]

omit [Fintype ι] [DecidableEq ι] in
@[simp] lemma cross_zero_right (u : ι → ℝ) (i j : ι) : cross u 0 i j = 0 := by simp [cross]

/-- `∏ g · gᵢ⁻¹ gⱼ⁻¹ = ∏_{l ≠ i, j} g_l` for `i ≠ j`. -/
lemma prod_mul_inv_mul_inv {g : ι → ℝ} (hg : ∀ i, g i ≠ 0) {i j : ι} (hij : i ≠ j) :
    (∏ l, g l) * (g i)⁻¹ * (g j)⁻¹ = ∏ l ∈ (univ.erase i).erase j, g l := by
  rw [← mul_prod_erase univ g (mem_univ i),
    ← mul_prod_erase (univ.erase i) g (mem_erase.2 ⟨hij.symm, mem_univ j⟩)]
  field_simp [hg i, hg j]

lemma prod_mul_inv {g : ι → ℝ} (hg : ∀ i, g i ≠ 0) (i : ι) :
    (∏ l, g l) * (g i)⁻¹ = ∏ l ∈ univ.erase i, g l := by
  rw [← mul_prod_erase univ g (mem_univ i)]
  field_simp [hg i]

/-- **Determinant of a diagonal plus rank-two matrix.** -/
theorem det_diagonal_add_rank_two (g u w : ι → ℝ) (hg : ∀ i, g i ≠ 0) (c : ℝ) :
    (diagonal g + c • (vecMulVec u u + vecMulVec w w)).det =
      ∏ i, g i + c * ∑ i, (u i ^ 2 + w i ^ 2) * ∏ l ∈ univ.erase i, g l
        + c ^ 2 * ∑ i, ∑ j, cross u w i j * ∏ l ∈ (univ.erase i).erase j, g l := by
  let U : Matrix ι (Fin 2) ℝ := Matrix.of fun i k => ![u i, w i] k
  let V : Matrix (Fin 2) ι ℝ := c • Uᵀ
  have hUV : U * V = c • (vecMulVec u u + vecMulVec w w) := by
    ext i j
    simp [U, V, Matrix.mul_apply, Fin.sum_univ_two, vecMulVec_apply, Function.swap]
    ring
  have hdet : IsUnit (diagonal g).det := by
    rw [det_diagonal, isUnit_iff_ne_zero]
    exact prod_ne_zero_iff.2 fun i _ => hg i
  have hinv : (diagonal g)⁻¹ = diagonal fun i => (g i)⁻¹ := by
    apply inv_eq_right_inv
    rw [diagonal_mul_diagonal]
    ext i j
    by_cases h : i = j
    · subst h; simp [hg i]
    · simp [diagonal_apply_ne _ h, one_apply_ne h]
  rw [← hUV, det_add_mul U V hdet, hinv, det_diagonal, det_fin_two]
  simp only [U, V, Matrix.add_apply, Matrix.mul_apply, Matrix.smul_apply, transpose_apply,
    of_apply, diagonal_apply, one_apply_eq, smul_eq_mul, mul_ite, mul_zero, sum_ite_eq',
    mem_univ, ite_true, Fin.isValue]
  simp only [Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one,
    Matrix.one_apply_ne (show (0 : Fin 2) ≠ 1 by decide),
    Matrix.one_apply_ne (show (1 : Fin 2) ≠ 0 by decide), zero_add]
  -- the three terms, after multiplying by `∏ g`
  have h1 : (∏ i, g i) * (∑ x, c * u x * (g x)⁻¹ * u x + ∑ x, c * w x * (g x)⁻¹ * w x) =
      c * ∑ i, (u i ^ 2 + w i ^ 2) * ∏ l ∈ univ.erase i, g l := by
    rw [← sum_add_distrib, mul_sum, mul_sum]
    refine sum_congr rfl fun i _ => ?_
    rw [← prod_mul_inv hg i]
    ring
  have h2 : (∏ i, g i) * ((∑ x, c * u x * (g x)⁻¹ * u x) * (∑ x, c * w x * (g x)⁻¹ * w x) -
      (∑ x, c * u x * (g x)⁻¹ * w x) * (∑ x, c * w x * (g x)⁻¹ * u x)) =
      c ^ 2 * ∑ i, ∑ j, cross u w i j * ∏ l ∈ (univ.erase i).erase j, g l := by
    rw [sum_mul_sum, sum_mul_sum, ← sum_sub_distrib, mul_sum, mul_sum]
    refine sum_congr rfl fun i _ => ?_
    rw [← sum_sub_distrib, mul_sum, mul_sum]
    refine sum_congr rfl fun j _ => ?_
    by_cases hij : i = j
    · subst hij; rw [cross_self]; ring
    · rw [← prod_mul_inv_mul_inv hg hij]
      unfold cross
      ring
  linear_combination h1 + h2

/-- The polynomial identity: `det(1 + X (diag μ + u uᵀ + w wᵀ))
= ∏(1 + μ_i X) + X ∑_i (u_i² + w_i²) ∏_{l≠i} (1 + μ_l X) + X² ∑_{i,j} T_ij ∏_{l≠i,j} (1 + μ_l X)`
for `μ ≥ 0`. -/
theorem det_one_add_X_smul_rank_two (μ u w : ι → ℝ) (hμ : ∀ i, 0 ≤ μ i) :
    (1 + (X : ℝ[X]) • (diagonal μ + vecMulVec u u + vecMulVec w w).map C).det =
      esymPoly univ μ + X * ∑ i, C (u i ^ 2 + w i ^ 2) * esymPoly (univ.erase i) μ
        + X ^ 2 * ∑ i, ∑ j, C (cross u w i j) * esymPoly ((univ.erase i).erase j) μ := by
  apply eq_of_infinite_eval_eq
  refine Set.Infinite.mono (s := Set.Ici (0 : ℝ)) ?_ (Set.Ici_infinite 0)
  intro x hx
  rw [Set.mem_Ici] at hx
  simp only [Set.mem_ofPred_eq]
  have hg : ∀ i, (1 + x * μ i) ≠ 0 := fun i => by nlinarith [hμ i]
  have hmat : (evalRingHom x).mapMatrix
      (1 + (X : ℝ[X]) • (diagonal μ + vecMulVec u u + vecMulVec w w).map C) =
      diagonal (fun i => 1 + x * μ i) + x • (vecMulVec u u + vecMulVec w w) := by
    ext i j
    by_cases h : i = j
    · subst h; simp [vecMulVec_apply]; ring
    · simp [one_apply_ne h, diagonal_apply_ne _ h, vecMulVec_apply]; ring
  rw [← coe_evalRingHom, RingHom.map_det, hmat, det_diagonal_add_rank_two _ u w hg x]
  simp only [esymPoly, eval_add, eval_mul, eval_X, eval_pow, eval_finsetSum, eval_prod, eval_C,
    eval_one, coe_evalRingHom]
  ring_nf

/-- `P(u, w) = det(1 + X (diag μ + u uᵀ + w wᵀ))`. -/
noncomputable def rkPoly (μ u w : ι → ℝ) : ℝ[X] :=
  (1 + (X : ℝ[X]) • (diagonal μ + vecMulVec u u + vecMulVec w w).map C).det

/-- The coefficient of `X^{k+2}` in `P(u, w)`. -/
lemma rkPoly_coeff (μ u w : ι → ℝ) (hμ : ∀ i, 0 ≤ μ i) (k : ℕ) :
    (rkPoly μ u w).coeff (k + 2) = esym univ μ (k + 2)
      + ∑ i, (u i ^ 2 + w i ^ 2) * esym (univ.erase i) μ (k + 1)
      + ∑ i, ∑ j, cross u w i j * esym ((univ.erase i).erase j) μ k := by
  rw [rkPoly, det_one_add_X_smul_rank_two μ u w hμ, coeff_add, coeff_add,
    show k + 2 = (k + 1) + 1 from rfl, coeff_X_mul, show k + 1 + 1 = k + 2 from rfl,
    coeff_X_pow_mul, finsetSum_coeff, finsetSum_coeff]
  simp only [coeff_C_mul, finsetSum_coeff, esym]

/-- The Newton gap `c_i c_j - A d_ij` is nonnegative for `i ≠ j`. -/
lemma newton_gap_nonneg {μ : ι → ℝ} (hμ : ∀ i, 0 ≤ μ i) (k : ℕ) {i j : ι} (hij : i ≠ j) :
    0 ≤ esym (univ.erase i) μ (k + 1) * esym (univ.erase j) μ (k + 1) -
      esym univ μ (k + 2) * esym ((univ.erase i).erase j) μ k := by
  set s := (univ.erase i).erase j with hs
  have hj : j ∈ univ.erase i := mem_erase.2 ⟨hij.symm, mem_univ j⟩
  have hi : i ∈ univ.erase j := mem_erase.2 ⟨hij, mem_univ i⟩
  have ci := esym_erase_succ hj μ k
  have cj := esym_erase_succ hi μ k
  rw [erase_right_comm, ← hs] at cj
  have A1 := esym_erase_succ (mem_univ i) μ (k + 1)
  have A2 := esym_erase_succ hj μ (k + 1)
  have A3 := esym_erase_succ hj μ k
  rw [← hs] at ci A2 A3
  have hN := esym_newton (s := s) (μ := μ) (fun l _ => hμ l) k
  rw [ci, cj, A1, A2, A3]
  nlinarith [hN]

/-- **Rayleigh inequality for diagonal plus two rank-one terms.**
`[P(0,0)]_K [P(u,w)]_K ≤ [P(u,0)]_K [P(0,w)]_K` for `K ≥ 2` and `μ ≥ 0`. -/
theorem rank_two_rayleigh (μ u w : ι → ℝ) (hμ : ∀ i, 0 ≤ μ i) (k : ℕ) :
    (rkPoly μ 0 0).coeff (k + 2) * (rkPoly μ u w).coeff (k + 2) ≤
      (rkPoly μ u 0).coeff (k + 2) * (rkPoly μ 0 w).coeff (k + 2) := by
  simp only [rkPoly_coeff μ _ _ hμ, Pi.zero_apply, cross_zero_left, cross_zero_right,
    zero_mul, sum_const_zero, add_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
    zero_pow, zero_add]
  set A := esym univ μ (k + 2)
  set c : ι → ℝ := fun i => esym (univ.erase i) μ (k + 1) with hc
  set d : ι → ι → ℝ := fun i j => esym ((univ.erase i).erase j) μ k with hd
  change A * (A + ∑ i, (u i ^ 2 + w i ^ 2) * c i + ∑ i, ∑ j, cross u w i j * d i j) ≤
    (A + ∑ i, u i ^ 2 * c i) * (A + ∑ i, w i ^ 2 * c i)
  -- `G_ij = c_i c_j - A d_ij` is symmetric, and nonnegative off the diagonal
  set G : ι → ι → ℝ := fun i j => c i * c j - A * d i j with hG
  have hGsymm : ∀ i j, G i j = G j i := by
    intro i j
    simp only [hG, hd, erase_right_comm (a := i) (b := j)]
    ring
  have hGnn : ∀ i j, i ≠ j → 0 ≤ G i j := fun i j hij => newton_gap_nonneg hμ k hij
  -- `∑ T_ij G_ij ≥ 0`, by symmetrizing
  have hTG : 0 ≤ ∑ i, ∑ j, cross u w i j * G i j := by
    have h2 : 2 * ∑ i, ∑ j, cross u w i j * G i j =
        ∑ i, ∑ j, (u i * w j - u j * w i) ^ 2 * G i j := by
      have hswap : ∑ i, ∑ j, cross u w i j * G i j = ∑ i, ∑ j, cross u w j i * G i j := by
        rw [sum_comm]
        exact sum_congr rfl fun i _ => sum_congr rfl fun j _ => by rw [hGsymm]
      rw [two_mul]
      nth_rewrite 2 [hswap]
      rw [← sum_add_distrib]
      refine sum_congr rfl fun i _ => ?_
      rw [← sum_add_distrib]
      refine sum_congr rfl fun j _ => ?_
      rw [← cross_add_cross]; ring
    have : 0 ≤ ∑ i, ∑ j, (u i * w j - u j * w i) ^ 2 * G i j := by
      refine sum_nonneg fun i _ => sum_nonneg fun j _ => ?_
      by_cases hij : i = j
      · subst hij; simp
      · exact mul_nonneg (sq_nonneg _) (hGnn i j hij)
    linarith
  -- `BC - AE = (∑ u_i w_i c_i)² + ∑ T_ij G_ij`
  have key : (∑ i, u i ^ 2 * c i) * (∑ i, w i ^ 2 * c i) - A * ∑ i, ∑ j, cross u w i j * d i j =
      (∑ i, u i * w i * c i) ^ 2 + ∑ i, ∑ j, cross u w i j * G i j := by
    rw [sq, sum_mul_sum, sum_mul_sum, mul_sum, ← sum_sub_distrib, ← sum_add_distrib]
    refine sum_congr rfl fun i _ => ?_
    rw [mul_sum, ← sum_sub_distrib, ← sum_add_distrib]
    refine sum_congr rfl fun j _ => ?_
    simp only [hG, cross]
    ring
  have hsplit : ∑ i, (u i ^ 2 + w i ^ 2) * c i = ∑ i, u i ^ 2 * c i + ∑ i, w i ^ 2 * c i := by
    rw [← sum_add_distrib]
    exact sum_congr rfl fun i _ => by ring
  rw [hsplit]
  nlinarith [key, hTG, sq_nonneg (∑ i, u i * w i * c i)]

end DppLm
