import DppLm.PoolSketch

/-!
# The paper's determinant sketch is `poolFhat` with `f(S) = det(L_S) 1{|S| = K}`

The paper defines (eq. `PoolDPP-fhat`)
  `f̂_ω(S) = α_{N,K} ∑_{|I| = K, ω(I) = S} det(L̂_I)`,  `α_{N,K} = N^K / (N)_K`,
with `L̂ = B̂ᵀ B̂` and `B̂ = [√(w⁽ⁿ⁾/N) φ(y⁽ⁿ⁾)]`, `w = κ/q` (eq. `B-matrix`), and
`L(x,y) = √κ(x) φ(x)ᵀφ(y) √κ(y)` (eq. `quality-diversity`).

`paperFhat_eq` shows that this equals `poolFhat q K f` with `f = kdppWeight` (the `K`-DPP
weight), for every pool whose strings all have positive proposal mass. Pools containing a
string with `q = 0` have probability zero, and there `w = κ/q` is undefined.
So Theorems 5.1 and 5.5, proved for `poolFhat`, apply to the paper's `f̂`.

`kdppWeight_supp` derives `PoolSetting.f_supp` from the paper's condition `q(y) = 0 → κ(y) = 0`.
-/

open Finset Matrix
open scoped ENNReal

namespace DppLm

variable {Y : Type*} [DecidableEq Y] {D : ℕ}

section Defs

variable (κ : Y → ℝ) (φ : Y → Fin D → ℝ)

/-- Gram matrix `[φ(yᵢ)ᵀ φ(yⱼ)]` of a tuple. -/
def gram {ι : Type*} (y : ι → Y) : Matrix ι ι ℝ := Matrix.of fun i j => φ (y i) ⬝ᵥ φ (y j)

/-- `L_S = [√κ(x) φ(x)ᵀφ(y) √κ(y)]_{x,y ∈ S}`. -/
noncomputable def kernelOn (S : Finset Y) : Matrix S S ℝ :=
  Matrix.of fun x y => Real.sqrt (κ x) * (φ x ⬝ᵥ φ y) * Real.sqrt (κ y)

/-- The `K`-DPP weight `f(S) = det(L_S) 1{|S| = K}`. -/
noncomputable def kdppWeight (K : ℕ) (S : Finset Y) : ℝ≥0∞ :=
  if S.card = K then ENNReal.ofReal (kernelOn κ φ S).det else 0

variable (q : Y → ℝ)

/-- `B̂ = [√(w⁽ⁿ⁾/N) φ(y⁽ⁿ⁾)]ₙ ∈ ℝ^{D×N}`, `w = κ/q`. -/
noncomputable def Bhat {N : ℕ} (ω : Fin N → Y) : Matrix (Fin D) (Fin N) ℝ :=
  Matrix.of fun d n => Real.sqrt (κ (ω n) / q (ω n) / N) * φ (ω n) d

/-- `L̂ = B̂ᵀ B̂`. -/
noncomputable def Lhat {N : ℕ} (ω : Fin N → Y) : Matrix (Fin N) (Fin N) ℝ :=
  (Bhat κ φ q ω)ᵀ * Bhat κ φ q ω

/-- `det(L̂_I)`. -/
noncomputable def detLhat {N : ℕ} (ω : Fin N → Y) (I : Finset (Fin N)) : ℝ :=
  ((Lhat κ φ q ω).submatrix (fun i : I => (i : Fin N)) (fun i : I => (i : Fin N))).det

/-- The paper's sketch `f̂_ω(S) = (N^K/(N)_K) ∑_{|I| = K, ω(I) = S} det(L̂_I)`. -/
noncomputable def paperFhat (K N : ℕ) (ω : Fin N → Y) (S : Finset Y) : ℝ :=
  ((N : ℝ) ^ K / (N.descFactorial K : ℝ)) *
    ∑ I ∈ powersetCard K (univ : Finset (Fin N)),
      if I.image ω = S then detLhat κ φ q ω I else 0

end Defs

variable {κ : Y → ℝ} {φ : Y → Fin D → ℝ}

omit [DecidableEq Y] in
lemma det_of_diag_mul {n : Type*} [Fintype n] [DecidableEq n] (s : n → ℝ) (G : Matrix n n ℝ) :
    (Matrix.of fun i j => s i * G i j * s j).det = (∏ i, s i) ^ 2 * G.det := by
  have : (Matrix.of fun i j => s i * G i j * s j) = diagonal s * G * diagonal s := by
    ext i j; simp [diagonal_mul, mul_diagonal]
  rw [this, det_mul, det_mul, det_diagonal]; ring

omit [DecidableEq Y] in
lemma gram_det_nonneg {ι : Type*} [Fintype ι] [DecidableEq ι] (y : ι → Y) :
    0 ≤ (gram φ y).det := by
  let Φ : Matrix (Fin D) ι ℝ := Matrix.of fun d i => φ (y i) d
  have : gram φ y = Φᴴ * Φ := by
    ext i j
    rw [Matrix.mul_apply]
    simp [gram, Φ, dotProduct]
  rw [this]
  exact (posSemidef_conjTranspose_mul_self Φ).det_nonneg

omit [DecidableEq Y] in
lemma prod_sqrt_sq {ι : Type*} [Fintype ι] {c : ι → ℝ} (hc : ∀ i, 0 ≤ c i) :
    (∏ i, Real.sqrt (c i)) ^ 2 = ∏ i, c i := by
  rw [← Finset.prod_pow]
  exact Finset.prod_congr rfl fun i _ => Real.sq_sqrt (hc i)

lemma det_kernelOn (hκ : ∀ y, 0 ≤ κ y) (S : Finset Y) :
    (kernelOn κ φ S).det = (∏ x : S, κ x) * (gram φ (fun x : S => (x : Y))).det := by
  have := det_of_diag_mul (fun x : S => Real.sqrt (κ x)) (gram φ (fun x : S => (x : Y)))
  rw [prod_sqrt_sq (c := fun x : S => κ x) (fun x => hκ x)] at this
  rw [← this]
  rfl

omit [DecidableEq Y] in
lemma detLhat_eq {q : Y → ℝ} {N : ℕ} (ω : Fin N → Y) (I : Finset (Fin N))
    (hw : ∀ i, 0 ≤ κ (ω i) / q (ω i) / N) :
    detLhat κ φ q ω I =
      (∏ i : I, κ (ω i) / q (ω i) / N) * (gram φ (fun i : I => ω i)).det := by
  have hentry : ((Lhat κ φ q ω).submatrix (fun i : I => (i : Fin N)) (fun i : I => (i : Fin N))) =
      Matrix.of fun i j : I => Real.sqrt (κ (ω i) / q (ω i) / N) *
        gram φ (fun i : I => ω i) i j * Real.sqrt (κ (ω j) / q (ω j) / N) := by
    ext i j
    rw [submatrix_apply, Lhat, Matrix.mul_apply]
    simp only [Bhat, transpose_apply, of_apply, gram, dotProduct, Finset.mul_sum, Finset.sum_mul]
    exact Finset.sum_congr rfl fun d _ => by ring
  rw [detLhat, hentry, det_of_diag_mul,
    prod_sqrt_sq (c := fun i : I => κ (ω i) / q (ω i) / N) (fun i => hw i)]

lemma gram_det_reindex {N : ℕ} (ω : Fin N → Y) (I : Finset (Fin N)) (hinj : Set.InjOn ω I) :
    (gram φ (fun i : I => ω i)).det = (gram φ (fun x : I.image ω => (x : Y))).det := by
  let e : I ≃ I.image ω := Equiv.ofBijective (fun i => ⟨ω i, mem_image_of_mem ω i.2⟩)
    ⟨fun a b h => Subtype.ext (hinj a.2 b.2 (congrArg Subtype.val h)), fun ⟨y, hy⟩ => by
      obtain ⟨i, hi, rfl⟩ := mem_image.1 hy
      exact ⟨⟨i, hi⟩, rfl⟩⟩
  rw [← det_submatrix_equiv_self e]
  rfl

/-- The paper's support condition `q(y) = 0 → κ(y) = 0` gives `PoolSetting.f_supp`. -/
lemma kdppWeight_supp (hκ : ∀ y, 0 ≤ κ y) {q : Y → ℝ≥0∞} (hq : ∀ y, q y = 0 → κ y = 0)
    {K : ℕ} (S : Finset Y) (hS : kdppWeight κ φ K S ≠ 0) : ∀ y ∈ S, q y ≠ 0 := by
  intro y hy hqy
  apply hS
  unfold kdppWeight
  split_ifs
  · rw [det_kernelOn hκ, Finset.prod_eq_zero (i := ⟨y, hy⟩) (Finset.mem_univ _) (hq y hqy),
      zero_mul, ENNReal.ofReal_zero]
  · rfl

/-- **The paper's `f̂` is `poolFhat`** with the `K`-DPP weight, on every pool whose strings
have positive proposal mass. -/
theorem paperFhat_eq (hκ : ∀ y, 0 ≤ κ y) (p : PMF Y) {K N : ℕ} (hK : 1 ≤ K) (hKN : K ≤ N)
    (ω : Fin N → Y) (hω : ∀ i, p (ω i) ≠ 0) (S : Finset Y) :
    ENNReal.ofReal (paperFhat κ φ (fun y => (p y).toReal) K N ω S) =
      poolFhat p K (kdppWeight κ φ K) ω S := by
  set qR : Y → ℝ := fun y => (p y).toReal
  have hqpos : ∀ i, 0 < qR (ω i) := fun i => ENNReal.toReal_pos (hω i) (PMF.apply_ne_top p _)
  have hN : (0 : ℝ) < N := by exact_mod_cast (by omega : 0 < N)
  have hw : ∀ i, 0 ≤ κ (ω i) / qR (ω i) / N := fun i =>
    div_nonneg (div_nonneg (hκ _) (hqpos i).le) hN.le
  have hdet_nonneg : ∀ I : Finset (Fin N), 0 ≤ detLhat κ φ qR ω I := by
    intro I
    rw [detLhat_eq ω I hw]
    exact mul_nonneg (Finset.prod_nonneg fun i _ => hw i) (gram_det_nonneg _)
  -- per-index-set identity: `N^K ofReal(det L̂_I) = f(ω(I)) / ∏ q(ω_i)`
  have hterm : ∀ I ∈ powersetCard K (univ : Finset (Fin N)),
      (N : ℝ≥0∞) ^ K * ENNReal.ofReal (detLhat κ φ qR ω I) =
        idxWeight p (kdppWeight κ φ K) ω I := by
    intro I hI
    have hIc : I.card = K := (mem_powersetCard.1 hI).2
    unfold idxWeight
    by_cases hinj : Set.InjOn ω I
    · have hcard : (I.image ω).card = K := by rw [card_image_of_injOn hinj, hIc]
      have hprodp : ∏ i ∈ I, p (ω i) = ENNReal.ofReal (∏ i ∈ I, qR (ω i)) := by
        rw [ENNReal.ofReal_prod_of_nonneg fun i _ => (hqpos i).le]
        exact Finset.prod_congr rfl fun i _ => (ENNReal.ofReal_toReal (PMF.apply_ne_top p _)).symm
      have hprodκ : ∏ i : I, κ (ω i) = ∏ x : I.image ω, κ x := by
        refine Fintype.prod_equiv (Equiv.ofBijective (fun i : I => (⟨ω i, mem_image_of_mem ω i.2⟩ :
          I.image ω)) ⟨fun a b h => Subtype.ext (hinj a.2 b.2 (congrArg Subtype.val h)),
            fun ⟨y, hy⟩ => by
              obtain ⟨i, hi, rfl⟩ := mem_image.1 hy
              exact ⟨⟨i, hi⟩, rfl⟩⟩) _ _ (fun i => rfl)
      have hreal : (N : ℝ) ^ K * detLhat κ φ qR ω I =
          (kernelOn κ φ (I.image ω)).det / ∏ i ∈ I, qR (ω i) := by
        rw [detLhat_eq ω I hw, det_kernelOn hκ, ← gram_det_reindex ω I hinj, ← hprodκ,
          Finset.prod_div_distrib, Finset.prod_div_distrib, Finset.prod_const, Finset.card_univ,
          Fintype.card_coe, hIc, Finset.prod_coe_sort I (fun i => qR (ω i))]
        have hprodpos : 0 < ∏ i ∈ I, qR (ω i) := Finset.prod_pos fun i _ => hqpos i
        field_simp
      simp only [kdppWeight, hcard, ↓reduceIte]
      rw [hprodp, ← ENNReal.ofReal_div_of_pos
        (Finset.prod_pos fun i _ => hqpos i), ← hreal, ENNReal.ofReal_mul (by positivity),
        ENNReal.ofReal_pow hN.le, ENNReal.ofReal_natCast]
    · -- a repeated string gives two equal rows, so `det(L̂_I) = 0`; and `|ω(I)| < K`
      have h0 : detLhat κ φ qR ω I = 0 := by
        simp only [Set.InjOn, not_forall] at hinj
        obtain ⟨i, hi, j, hj, hij, hne⟩ := hinj
        refine det_zero_of_row_eq (i := ⟨i, hi⟩) (j := ⟨j, hj⟩)
          (fun h => hne (congrArg Subtype.val h)) ?_
        ext k
        simp only [submatrix_apply]
        rw [Lhat, Matrix.mul_apply, Matrix.mul_apply]
        simp [Bhat, hij]
      have hlt : (I.image ω).card < K := by
        rw [← hIc]
        exact lt_of_le_of_ne card_image_le (fun h => hinj (card_image_iff.1 h))
      simp only [kdppWeight, hlt.ne, ↓reduceIte]
      rw [h0, ENNReal.ofReal_zero, mul_zero, ENNReal.zero_div]
  unfold paperFhat poolFhat
  have hα : (0 : ℝ) ≤ (N : ℝ) ^ K / (N.descFactorial K : ℝ) := by positivity
  have hD0 : (0 : ℝ) < (N.descFactorial K : ℝ) := by
    exact_mod_cast Nat.descFactorial_pos.2 hKN
  rw [ENNReal.ofReal_mul hα, ENNReal.ofReal_sum_of_nonneg (fun I _ => by
      split_ifs
      · exact hdet_nonneg I
      · exact le_rfl),
    ENNReal.ofReal_div_of_pos hD0, ENNReal.ofReal_pow hN.le, ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast, div_eq_mul_inv, mul_comm ((N : ℝ≥0∞) ^ K), mul_assoc,
    Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl fun I hI => ?_
  split_ifs
  · exact hterm I hI
  · simp

end DppLm
