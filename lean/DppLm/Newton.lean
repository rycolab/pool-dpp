import Mathlib

/-!
# Elementary symmetric polynomials and Newton's inequality

`esym s μ k` is the elementary symmetric polynomial `e_k` of the numbers `μ i`, `i ∈ s`. We
define it as the coefficient of `X^k` in `∏_{i ∈ s} (1 + μ i X)`, which is how it appears in
`det(1 + X M)` for a diagonal matrix `M`.

`esym_newton` is Newton's inequality `e_k e_{k+2} ≤ e_{k+1}²` for nonnegative numbers, in the
unnormalized form. Mathlib does not have it. The proof is by induction on `s`, adding one
number at a time; the induction needs the more general statement
`e_i e_{j+2} ≤ e_{i+1} e_{j+1}` for all `i ≤ j` (`esym_mul_le`).
-/

open Finset Polynomial

namespace DppLm

variable {ι : Type*} [DecidableEq ι]

/-- `∏_{i ∈ s} (1 + μ i X)`. -/
noncomputable def esymPoly (s : Finset ι) (μ : ι → ℝ) : ℝ[X] := ∏ i ∈ s, (1 + C (μ i) * X)

/-- The elementary symmetric polynomial `e_k(μ_i : i ∈ s)`. -/
noncomputable def esym (s : Finset ι) (μ : ι → ℝ) (k : ℕ) : ℝ := (esymPoly s μ).coeff k

lemma coeff_one_add_C_mul_X_mul_succ (a : ℝ) (p : ℝ[X]) (k : ℕ) :
    ((1 + C a * X) * p).coeff (k + 1) = p.coeff (k + 1) + a * p.coeff k := by
  rw [add_mul, one_mul, coeff_add, mul_assoc, coeff_C_mul, coeff_X_mul]

lemma coeff_one_add_C_mul_X_mul_zero (a : ℝ) (p : ℝ[X]) :
    ((1 + C a * X) * p).coeff 0 = p.coeff 0 := by
  rw [add_mul, one_mul, coeff_add, mul_assoc, coeff_C_mul, coeff_X_mul_zero, mul_zero, add_zero]

lemma esymPoly_insert {s : Finset ι} {x : ι} (hx : x ∉ s) (μ : ι → ℝ) :
    esymPoly (insert x s) μ = (1 + C (μ x) * X) * esymPoly s μ := by
  unfold esymPoly
  rw [prod_insert hx]

lemma esym_insert_succ {s : Finset ι} {x : ι} (hx : x ∉ s) (μ : ι → ℝ) (k : ℕ) :
    esym (insert x s) μ (k + 1) = esym s μ (k + 1) + μ x * esym s μ k := by
  unfold esym
  rw [esymPoly_insert hx, coeff_one_add_C_mul_X_mul_succ]

lemma esym_insert_zero {s : Finset ι} {x : ι} (hx : x ∉ s) (μ : ι → ℝ) :
    esym (insert x s) μ 0 = esym s μ 0 := by
  unfold esym
  rw [esymPoly_insert hx, coeff_one_add_C_mul_X_mul_zero]

@[simp] lemma esym_zero (s : Finset ι) (μ : ι → ℝ) : esym s μ 0 = 1 := by
  induction s using Finset.induction_on with
  | empty => simp [esym, esymPoly]
  | insert x s hx ih => rw [esym_insert_zero hx, ih]

omit [DecidableEq ι] in
lemma esym_empty_succ (μ : ι → ℝ) (k : ℕ) : esym (∅ : Finset ι) μ (k + 1) = 0 := by
  simp [esym, esymPoly, coeff_one]

/-- For `i ∈ s`: `e_{k+1}(s) = e_{k+1}(s ∖ i) + μ i e_k(s ∖ i)`. -/
lemma esym_erase_succ {s : Finset ι} {x : ι} (hx : x ∈ s) (μ : ι → ℝ) (k : ℕ) :
    esym s μ (k + 1) = esym (s.erase x) μ (k + 1) + μ x * esym (s.erase x) μ k := by
  conv_lhs => rw [← insert_erase hx]
  exact esym_insert_succ (notMem_erase x s) μ k

lemma esym_nonneg {s : Finset ι} {μ : ι → ℝ} (hμ : ∀ i ∈ s, 0 ≤ μ i) (k : ℕ) :
    0 ≤ esym s μ k := by
  induction s using Finset.induction_on generalizing k with
  | empty =>
    cases k with
    | zero => simp
    | succ k => rw [esym_empty_succ]
  | insert x s hx ih =>
    have hs : ∀ i ∈ s, 0 ≤ μ i := fun i hi => hμ i (mem_insert_of_mem hi)
    cases k with
    | zero => simp
    | succ k =>
      rw [esym_insert_succ hx]
      exact add_nonneg (ih hs _) (mul_nonneg (hμ x (mem_insert_self x s)) (ih hs k))

/-- The generalized Newton inequality `e_i e_{j+2} ≤ e_{i+1} e_{j+1}` for `i ≤ j` and
nonnegative numbers. -/
theorem esym_mul_le {s : Finset ι} {μ : ι → ℝ} (hμ : ∀ i ∈ s, 0 ≤ μ i) :
    ∀ i j : ℕ, i ≤ j → esym s μ i * esym s μ (j + 2) ≤ esym s μ (i + 1) * esym s μ (j + 1) := by
  induction s using Finset.induction_on with
  | empty =>
    intro i j _
    rw [esym_empty_succ, esym_empty_succ, mul_zero, zero_mul]
  | insert x s hx ih =>
    have hs : ∀ i ∈ s, 0 ≤ μ i := fun i hi => hμ i (mem_insert_of_mem hi)
    have hm : 0 ≤ μ x := hμ x (mem_insert_self x s)
    have H := ih hs
    have e0 := esym_nonneg hs
    set m := μ x
    intro i j hij
    rcases i with _ | i₀
    · -- `e'_{j+2} ≤ e'_1 e'_{j+1}`
      rw [esym_zero, one_mul, esym_insert_succ hx, esym_insert_succ hx, esym_insert_succ hx,
        esym_zero]
      have h0 := H 0 j (Nat.zero_le j)
      rw [esym_zero, one_mul] at h0
      have := e0 1; have := e0 j; have := e0 (j + 1)
      nlinarith [mul_nonneg hm (e0 j), mul_nonneg (mul_nonneg hm (e0 1)) (e0 j),
        mul_nonneg (mul_nonneg hm hm) (e0 j)]
    · obtain ⟨j₀, rfl⟩ : ∃ j₀, j = j₀ + 1 := ⟨j - 1, by omega⟩
      have hij' : i₀ ≤ j₀ := by omega
      rw [esym_insert_succ hx, esym_insert_succ hx, esym_insert_succ hx, esym_insert_succ hx]
      -- the coefficients of `1`, `m` and `m²`
      have c0 := H (i₀ + 1) (j₀ + 1) (by omega)
      have c2 := H i₀ j₀ hij'
      have c1 : esym s μ i₀ * esym s μ (j₀ + 1 + 2) ≤
          esym s μ (i₀ + 2) * esym s μ (j₀ + 1) := by
        have h1 := H i₀ (j₀ + 1) (by omega)
        rcases Nat.eq_or_lt_of_le hij' with h | h
        · subst h
          calc _ ≤ esym s μ (i₀ + 1) * esym s μ (i₀ + 1 + 1) := h1
            _ = _ := by ring
        · have h2 := H (i₀ + 1) j₀ (by omega)
          calc _ ≤ esym s μ (i₀ + 1) * esym s μ (j₀ + 1 + 1) := h1
            _ ≤ esym s μ (i₀ + 1 + 1) * esym s μ (j₀ + 1) := h2
            _ = _ := by ring
      have hsq : 0 ≤ m * m := mul_nonneg hm hm
      have k0 : (j₀ + 1 + 2) = j₀ + 2 + 1 := by ring
      have k1 : (j₀ + 1 + 1) = j₀ + 2 := by ring
      have k2 : (i₀ + 1 + 1) = i₀ + 2 := by ring
      simp only [k0, k1, k2] at c0 c1 c2 ⊢
      nlinarith [mul_le_mul_of_nonneg_left c1 hm, mul_le_mul_of_nonneg_left c2 hsq]

/-- **Newton's inequality** (unnormalized): `e_k e_{k+2} ≤ e_{k+1}²` for nonnegative numbers. -/
theorem esym_newton {s : Finset ι} {μ : ι → ℝ} (hμ : ∀ i ∈ s, 0 ≤ μ i) (k : ℕ) :
    esym s μ k * esym s μ (k + 2) ≤ esym s μ (k + 1) ^ 2 := by
  rw [sq]
  exact esym_mul_le hμ k k le_rfl

end DppLm
