import DppLm.RankTwo

/-!
# Negative correlation for `K`-DPPs with Gram kernels on finite base sets

Let `ψ : Y → ℝ^ι` assign a vector to each string, and let `L(x, y) = ψ(x)ᵀψ(y)`. The
`K`-DPP on a finite base set `V` has weight `det(L_S)` on each `K`-subset `S ⊆ V`.
`gram_negCorr` proves `p(T ⊆ S) ≤ ∏_{y∈T} p(y ∈ S)` for every `T`.

This replaces the citation of Borcea–Brändén–Liggett (`K`-DPPs are strongly Rayleigh, hence
negatively associated) with an elementary proof. The steps:

* **Bridge** (`partZ_eq_coeff`): the normalizer `Z_K(V) = ∑_{|S|=K, S⊆V} det(L_S)` is the
  coefficient of `X^K` in `det(1 + X M(V))`, where `M(V) = ∑_{y∈V} ψ(y)ψ(y)ᵀ` is `ι × ι`.
  Proof: Weinstein–Aronszajn (`det(1 + AB) = det(1 + BA)`) and the expansion of `det(1 + X L)`
  in principal minors, both in Mathlib.
* **Rotation** (`exists_rotation`): replacing `ψ` by `Qᵀψ` for an orthogonal `Q` leaves every
  `L(x, y)` unchanged. Choosing `Q` from the spectral theorem makes `M(V')` diagonal.
* **Two strings** (`inclN_pair_le`): with `V' = V ∖ {a, b}` and `M(V') = diag μ`, the four
  normalizers `Z(V')`, `Z(V' + a)`, `Z(V' + b)`, `Z(V)` are the coefficients in
  `rank_two_rayleigh`, which gives `p(a, b ∈ S) ≤ p(a ∈ S) p(b ∈ S)`.
* **Conditioning** (`gw_insert`): `det(L_{R+a}) = ‖ψ(a)‖² det(L'_R)`, where `L'` is the Gram
  kernel of the vectors projected onto `ψ(a)^⊥` (Schur complement). So given `a ∈ S`, the rest
  of `S` is a `(K-1)`-DPP of the same kind.
* **Induction on `|T|`** (`gram_negCorr`):
  `p(T + a) = p(a) p(T | a) ≤ p(a) ∏_{y∈T} p(y | a) ≤ p(a) ∏_{y∈T} p(y)`.
-/

open Finset Polynomial Matrix

namespace DppLm

variable {Y : Type*} [DecidableEq Y] {ι : Type*} [Fintype ι] [DecidableEq ι]

section Defs

variable (ψ : Y → ι → ℝ)

/-- The Gram matrix `[ψ(x)ᵀψ(y)]_{x,y ∈ S}`. -/
def gramOn (S : Finset Y) : Matrix S S ℝ := Matrix.of fun x y => ψ x ⬝ᵥ ψ y

/-- The weight `det(L_S)`. -/
noncomputable def gw (S : Finset Y) : ℝ := (gramOn ψ S).det

/-- The normalizer `Z_K(V) = ∑_{S ⊆ V, |S| = K} det(L_S)`. -/
noncomputable def partZ (K : ℕ) (V : Finset Y) : ℝ := ∑ S ∈ V.powersetCard K, gw ψ S

/-- The inclusion sum `∑_{S ⊆ V, |S| = K, T ⊆ S} det(L_S)`. -/
noncomputable def inclN (K : ℕ) (V T : Finset Y) : ℝ :=
  ∑ S ∈ (V.powersetCard K).filter (T ⊆ ·), gw ψ S

/-- `M(V) = ∑_{y∈V} ψ(y)ψ(y)ᵀ`. -/
def outer (V : Finset Y) : Matrix ι ι ℝ := ∑ y ∈ V, vecMulVec (ψ y) (ψ y)

/-- `ψ` projected onto `ψ(a)^⊥`. -/
noncomputable def proj (a : Y) : Y → ι → ℝ :=
  fun y => ψ y - ((ψ y ⬝ᵥ ψ a) / (ψ a ⬝ᵥ ψ a)) • ψ a

end Defs

variable {ψ : Y → ι → ℝ}

omit [DecidableEq Y] [DecidableEq ι] in
/-- The matrix with columns `ψ(y)`, `y ∈ V`. -/
lemma gramOn_eq_mul (ψ : Y → ι → ℝ) (V : Finset Y) :
    gramOn ψ V = (Matrix.of fun (d : ι) (y : V) => ψ y d)ᵀ * Matrix.of fun (d : ι) (y : V) => ψ y d := by
  ext x y
  simp [gramOn, Matrix.mul_apply, dotProduct]

omit [DecidableEq ι] in
lemma gw_nonneg (ψ : Y → ι → ℝ) (S : Finset Y) : 0 ≤ gw ψ S := by
  unfold gw
  rw [gramOn_eq_mul, ← conjTranspose_eq_transpose_of_trivial]
  exact (posSemidef_conjTranspose_mul_self _).det_nonneg

omit [DecidableEq ι] in
lemma partZ_nonneg (ψ : Y → ι → ℝ) (K : ℕ) (V : Finset Y) : 0 ≤ partZ ψ K V :=
  sum_nonneg fun S _ => gw_nonneg ψ S

omit [DecidableEq ι] in
lemma inclN_nonneg (ψ : Y → ι → ℝ) (K : ℕ) (V T : Finset Y) : 0 ≤ inclN ψ K V T :=
  sum_nonneg fun S _ => gw_nonneg ψ S

omit [DecidableEq ι] in
/-- The weights depend on `ψ` only through the inner products `ψ(x)ᵀψ(y)`. -/
lemma gw_congr {ψ ψ' : Y → ι → ℝ} (h : ∀ x y, ψ' x ⬝ᵥ ψ' y = ψ x ⬝ᵥ ψ y) (S : Finset Y) :
    gw ψ' S = gw ψ S := by
  unfold gw gramOn
  simp_rw [h]

omit [DecidableEq Y] [DecidableEq ι] in
/-- Gram determinants agree on equivalent index sets. -/
lemma det_gram_equiv {α β : Type*} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]
    (ψ : Y → ι → ℝ) (f : α → Y) (g : β → Y) (e : α ≃ β) (h : ∀ x, g (e x) = f x) :
    (Matrix.of fun x y : α => ψ (f x) ⬝ᵥ ψ (f y)).det =
      (Matrix.of fun x y : β => ψ (g x) ⬝ᵥ ψ (g y)).det := by
  rw [← det_submatrix_equiv_self e]
  congr 1
  ext x y
  simp [h]

omit [DecidableEq ι] in
/-- A principal minor of the Gram matrix on `V` is the weight of the corresponding subset. -/
lemma det_gramOn_submatrix (ψ : Y → ι → ℝ) (V : Finset Y) (s : Finset V) :
    ((gramOn ψ V).submatrix (Subtype.val : s → V) Subtype.val).det =
      gw ψ (s.map (Function.Embedding.subtype _)) := by
  unfold gw gramOn
  let e : s ≃ (s.map (Function.Embedding.subtype (· ∈ V))) :=
    { toFun := fun x => ⟨x.1.1, mem_map_of_mem _ x.2⟩
      invFun := fun y => ⟨⟨y.1, by
          obtain ⟨z, -, hz⟩ := mem_map.1 y.2
          rw [← hz]; exact z.2⟩, by
          obtain ⟨z, hzs, hz⟩ := mem_map.1 y.2
          have : z = ⟨y.1, by rw [← hz]; exact z.2⟩ := Subtype.ext hz
          rw [← this]; exact hzs⟩
      left_inv := fun x => rfl
      right_inv := fun y => rfl }
  exact det_gram_equiv ψ (fun x : s => (x : V).1) (fun y => y.1) e (fun x => rfl)

omit [DecidableEq Y] [Fintype ι] [DecidableEq ι] in
lemma sum_powersetCard_subtype (V : Finset Y) (K : ℕ) (F : Finset Y → ℝ) :
    ∑ s ∈ (univ : Finset V).powersetCard K, F (s.map (Function.Embedding.subtype _)) =
      ∑ S ∈ V.powersetCard K, F S := by
  have hV : (univ : Finset V).map (Function.Embedding.subtype (· ∈ V)) = V := by
    rw [univ_eq_attach, attach_map_val]
  conv_rhs => rw [← hV, powersetCard_map, sum_map]
  rfl

/-- **Bridge.** `Z_K(V)` is the coefficient of `X^K` in `det(1 + X M(V))`. -/
theorem partZ_eq_coeff (ψ : Y → ι → ℝ) (K : ℕ) (V : Finset Y) :
    partZ ψ K V = (1 + (X : ℝ[X]) • (outer ψ V).map C).det.coeff K := by
  set Ψ : Matrix ι V ℝ := Matrix.of fun d y => ψ y d
  have h1 : (1 + (X : ℝ[X]) • (gramOn ψ V).map C).det =
      (1 + (X : ℝ[X]) • (outer ψ V).map C).det := by
    have h2 : outer ψ V = Ψ * Ψᵀ := by
      ext d e
      simp only [outer, Matrix.sum_apply, vecMulVec_apply, Ψ, Matrix.mul_apply, transpose_apply,
        of_apply]
      exact (sum_coe_sort V (fun y => ψ y d * ψ y e)).symm
    rw [gramOn_eq_mul, h2, Matrix.map_mul, Matrix.map_mul, ← Matrix.smul_mul,
      det_one_add_mul_comm, Matrix.mul_smul]
  rw [← h1, coeff_det_one_add_X_smul_eq_sum_minors, partZ]
  simp_rw [det_gramOn_submatrix]
  exact (sum_powersetCard_subtype V K (gw ψ)).symm

/-! ### Rotation -/

omit [DecidableEq Y] [Fintype ι] [DecidableEq ι] in
lemma outer_isHermitian (ψ : Y → ι → ℝ) (V : Finset Y) : (outer ψ V).IsHermitian := by
  ext i j
  simp [outer, conjTranspose_apply, Matrix.sum_apply, vecMulVec_apply, mul_comm]

omit [DecidableEq Y] in
lemma dotProduct_mulVec_mulVec {A : Matrix ι ι ℝ} (hA : Aᵀ * A = 1) (x y : ι → ℝ) :
    (A *ᵥ x) ⬝ᵥ (A *ᵥ y) = x ⬝ᵥ y := by
  rw [dotProduct_comm, dotProduct_mulVec, ← mulVec_transpose, mulVec_mulVec, hA, one_mulVec,
    dotProduct_comm]

omit [DecidableEq Y] [DecidableEq ι] in
lemma outer_mulVec (A : Matrix ι ι ℝ) (ψ : Y → ι → ℝ) (V : Finset Y) :
    outer (fun y => A *ᵥ ψ y) V = A * outer ψ V * Aᵀ := by
  simp only [outer, mul_sum, sum_mul, mul_vecMulVec, vecMulVec_mul, vecMul_transpose]

omit [DecidableEq Y] in
/-- **Rotation.** Some rotation of the vectors leaves every inner product unchanged and makes
`M(V)` diagonal, with nonnegative diagonal. -/
lemma exists_rotation (ψ : Y → ι → ℝ) (V : Finset Y) :
    ∃ ψ' : Y → ι → ℝ, (∀ x y, ψ' x ⬝ᵥ ψ' y = ψ x ⬝ᵥ ψ y) ∧
      ∃ μ : ι → ℝ, (∀ i, 0 ≤ μ i) ∧ outer ψ' V = diagonal μ := by
  have hA := outer_isHermitian ψ V
  set Q : Matrix ι ι ℝ := (hA.eigenvectorUnitary : Matrix ι ι ℝ)
  have hstar : star Q = Qᵀ := conjTranspose_eq_transpose_of_trivial Q
  have hQ1 : Qᵀ * Q = 1 := by rw [← hstar]; exact Unitary.coe_star_mul_self _
  have hQ2 : Q * Qᵀ = 1 := by rw [← hstar]; exact Unitary.coe_mul_star_self _
  set μ : ι → ℝ := RCLike.ofReal ∘ hA.eigenvalues
  have hspec : outer ψ V = Q * diagonal μ * Qᵀ := by
    conv_lhs => rw [hA.spectral_theorem, Unitary.conjStarAlgAut_apply]
    rw [hstar]
  refine ⟨fun y => Qᵀ *ᵥ ψ y, fun x y => ?_, μ, ?_, ?_⟩
  · exact dotProduct_mulVec_mulVec (by rw [transpose_transpose, hQ2]) _ _
  · have hdiag : outer (fun y => Qᵀ *ᵥ ψ y) V = diagonal μ := by
      rw [outer_mulVec, hspec, transpose_transpose, ← Matrix.mul_assoc, ← Matrix.mul_assoc, hQ1,
        Matrix.one_mul, Matrix.mul_assoc, hQ1, Matrix.mul_one]
    intro i
    have := congrFun (congrFun hdiag i) i
    rw [diagonal_apply_eq] at this
    rw [← this]
    simp only [outer, Matrix.sum_apply, vecMulVec_apply]
    exact sum_nonneg fun y _ => mul_self_nonneg _
  · rw [outer_mulVec, hspec, transpose_transpose, ← Matrix.mul_assoc, ← Matrix.mul_assoc, hQ1,
      Matrix.one_mul, Matrix.mul_assoc, hQ1, Matrix.mul_one]

omit [DecidableEq ι] in
lemma partZ_congr {ψ ψ' : Y → ι → ℝ} (h : ∀ x y, ψ' x ⬝ᵥ ψ' y = ψ x ⬝ᵥ ψ y) (K : ℕ)
    (V : Finset Y) : partZ ψ' K V = partZ ψ K V := by
  unfold partZ; simp_rw [gw_congr h]

omit [DecidableEq ι] in
lemma inclN_congr {ψ ψ' : Y → ι → ℝ} (h : ∀ x y, ψ' x ⬝ᵥ ψ' y = ψ x ⬝ᵥ ψ y) (K : ℕ)
    (V T : Finset Y) : inclN ψ' K V T = inclN ψ K V T := by
  unfold inclN; simp_rw [gw_congr h]

/-! ### Inclusion–exclusion -/

omit [Fintype ι] [DecidableEq ι] in
lemma filter_notMem_powersetCard (V : Finset Y) (K : ℕ) (a : Y) :
    (V.powersetCard K).filter (fun S => a ∉ S) = (V.erase a).powersetCard K := by
  ext S
  simp only [mem_filter, mem_powersetCard]
  constructor
  · rintro ⟨⟨hS, hc⟩, ha⟩
    exact ⟨fun x hx => mem_erase.2 ⟨fun h => ha (h ▸ hx), hS hx⟩, hc⟩
  · rintro ⟨hS, hc⟩
    exact ⟨⟨hS.trans (erase_subset a V), hc⟩, fun ha => notMem_erase a V (hS ha)⟩

omit [DecidableEq ι] in
/-- `Z(V) = N_a(V) + Z(V ∖ a)`. -/
lemma partZ_eq_inclN_add (ψ : Y → ι → ℝ) (K : ℕ) (V : Finset Y) (a : Y) :
    partZ ψ K V = inclN ψ K V {a} + partZ ψ K (V.erase a) := by
  unfold partZ inclN
  rw [← filter_notMem_powersetCard, ← sum_filter_add_sum_filter_not _ (fun S => a ∈ S)]
  simp only [singleton_subset_iff]

omit [DecidableEq ι] in
/-- `N_a(V) = N_{ab}(V) + N_a(V ∖ b)`. -/
lemma inclN_eq_inclN_pair_add (ψ : Y → ι → ℝ) (K : ℕ) (V : Finset Y) (a b : Y) :
    inclN ψ K V {a} = inclN ψ K V {a, b} + inclN ψ K (V.erase b) {a} := by
  unfold inclN
  rw [← filter_notMem_powersetCard, filter_filter,
    ← sum_filter_add_sum_filter_not _ (fun S => b ∈ S), filter_filter]
  congr 2
  · ext S; simp only [mem_filter, insert_subset_iff, singleton_subset_iff]
  · ext S; simp only [mem_filter]; tauto

/-! ### Two strings -/

omit [DecidableEq ι] in
lemma inclN_pair_eq_zero_of_notMem (ψ : Y → ι → ℝ) (K : ℕ) {V : Finset Y} {a b : Y}
    (h : a ∉ V ∨ b ∉ V) : inclN ψ K V {a, b} = 0 := by
  unfold inclN
  rw [sum_eq_zero]
  intro S hS
  simp only [mem_filter, mem_powersetCard, insert_subset_iff, singleton_subset_iff] at hS
  rcases h with h | h
  · exact absurd (hS.1.1 hS.2.1) h
  · exact absurd (hS.1.1 hS.2.2) h

omit [DecidableEq ι] in
lemma inclN_pair_eq_zero_of_lt (ψ : Y → ι → ℝ) {K : ℕ} (hK : K < 2) (V : Finset Y) {a b : Y}
    (hab : a ≠ b) : inclN ψ K V {a, b} = 0 := by
  unfold inclN
  rw [sum_eq_zero]
  intro S hS
  simp only [mem_filter, mem_powersetCard] at hS
  have := card_le_card hS.2
  rw [card_pair hab, hS.1.2] at this
  omega

/-- **Two-string negative correlation.** `N_{ab} Z ≤ N_a N_b`, i.e.
`p(a, b ∈ S) ≤ p(a ∈ S) p(b ∈ S)`. -/
theorem inclN_pair_le (ψ : Y → ι → ℝ) (K : ℕ) (V : Finset Y) {a b : Y} (hab : a ≠ b) :
    inclN ψ K V {a, b} * partZ ψ K V ≤ inclN ψ K V {a} * inclN ψ K V {b} := by
  have hnn := mul_nonneg (inclN_nonneg ψ K V {a}) (inclN_nonneg ψ K V {b})
  by_cases hV : a ∈ V ∧ b ∈ V
  swap
  · rw [inclN_pair_eq_zero_of_notMem ψ K (by tauto), zero_mul]; exact hnn
  obtain ⟨ha, hb⟩ := hV
  rcases lt_or_ge K 2 with hK | hK
  · rw [inclN_pair_eq_zero_of_lt ψ hK V hab, zero_mul]; exact hnn
  obtain ⟨k, rfl⟩ : ∃ k, K = k + 2 := ⟨K - 2, by omega⟩
  -- rotate so that `M(V ∖ {a, b})` is diagonal
  set V' := (V.erase a).erase b with hV'
  obtain ⟨ψ', hψ', μ, hμ, hdiag⟩ := exists_rotation ψ V'
  simp only [← inclN_congr hψ', ← partZ_congr hψ']
  -- inclusion–exclusion
  have e1 := partZ_eq_inclN_add ψ' (k + 2) V a
  have e2 := partZ_eq_inclN_add ψ' (k + 2) V b
  have e3 := inclN_eq_inclN_pair_add ψ' (k + 2) V a b
  have e4 := partZ_eq_inclN_add ψ' (k + 2) (V.erase b) a
  rw [erase_right_comm, ← hV'] at e4
  -- the four normalizers, by the bridge
  have hb' : b ∈ V.erase a := mem_erase.2 ⟨hab.symm, hb⟩
  have ha' : a ∈ V.erase b := mem_erase.2 ⟨hab, ha⟩
  have hVa : V.erase a = insert b V' := (insert_erase hb').symm
  have hVb : V.erase b = insert a V' := by
    rw [hV', erase_right_comm]; exact (insert_erase ha').symm
  have hVab : V = insert a (insert b V') := by rw [← hVa, insert_erase ha]
  have nb : b ∉ V' := notMem_erase b _
  have na : a ∉ V' := by
    rw [hV']; exact fun h => notMem_erase a V (mem_of_mem_erase h)
  have nab : a ∉ insert b V' := by
    rw [mem_insert, not_or]; exact ⟨hab, na⟩
  have Z0 : partZ ψ' (k + 2) V' = (rkPoly μ 0 0).coeff (k + 2) := by
    rw [partZ_eq_coeff, hdiag, rkPoly, vecMulVec_zero, add_zero, add_zero]
  have Za : partZ ψ' (k + 2) (V.erase a) = (rkPoly μ 0 (ψ' b)).coeff (k + 2) := by
    have hM : outer ψ' (V.erase a) = diagonal μ + vecMulVec 0 0 + vecMulVec (ψ' b) (ψ' b) := by
      rw [hVa, outer, sum_insert nb, ← outer, hdiag, vecMulVec_zero, add_zero, add_comm]
    rw [partZ_eq_coeff, hM, rkPoly]
  have Zb : partZ ψ' (k + 2) (V.erase b) = (rkPoly μ (ψ' a) 0).coeff (k + 2) := by
    have hM : outer ψ' (V.erase b) = diagonal μ + vecMulVec (ψ' a) (ψ' a) + vecMulVec 0 0 := by
      rw [hVb, outer, sum_insert na, ← outer, hdiag, vecMulVec_zero, add_zero, add_comm]
    rw [partZ_eq_coeff, hM, rkPoly]
  have Zab : partZ ψ' (k + 2) V = (rkPoly μ (ψ' a) (ψ' b)).coeff (k + 2) := by
    have hM : outer ψ' V =
        diagonal μ + vecMulVec (ψ' a) (ψ' a) + vecMulVec (ψ' b) (ψ' b) := by
      rw [hVab, outer, sum_insert nab, sum_insert nb, ← outer, hdiag]
      abel
    rw [partZ_eq_coeff, hM, rkPoly]
  have key := rank_two_rayleigh μ (ψ' a) (ψ' b) hμ k
  rw [← Z0, ← Za, ← Zb, ← Zab] at key
  have hNa : inclN ψ' (k + 2) V {a} = partZ ψ' (k + 2) V - partZ ψ' (k + 2) (V.erase a) := by
    linarith
  have hNb : inclN ψ' (k + 2) V {b} = partZ ψ' (k + 2) V - partZ ψ' (k + 2) (V.erase b) := by
    linarith
  have hNab : inclN ψ' (k + 2) V {a, b} = partZ ψ' (k + 2) V - partZ ψ' (k + 2) (V.erase a) -
      partZ ψ' (k + 2) (V.erase b) + partZ ψ' (k + 2) V' := by
    linarith
  rw [hNa, hNb, hNab]
  nlinarith [key]

/-! ### Conditioning on one string -/

omit [Fintype ι] [DecidableEq ι] in
/-- `R ⊕ {a} ≃ R + a` for `a ∉ R`. -/
def insertSumEquiv {a : Y} {R : Finset Y} (ha : a ∉ R) : R ⊕ Unit ≃ (insert a R : Finset Y) where
  toFun := Sum.elim (fun r => ⟨r, mem_insert_of_mem r.2⟩) (fun _ => ⟨a, mem_insert_self a R⟩)
  invFun y := if h : (y : Y) = a then Sum.inr () else
    Sum.inl ⟨y, (mem_insert.1 y.2).resolve_left h⟩
  left_inv := by
    rintro (r | _)
    · have : (r : Y) ≠ a := fun h => ha (h ▸ r.2)
      simp [this]
    · simp
  right_inv := by
    rintro ⟨y, hy⟩
    by_cases h : y = a
    · subst h; simp
    · simp [h]

omit [DecidableEq ι] in
/-- **Conditioning.** `det(L_{R+a}) = ‖ψ(a)‖² det(L'_R)` for `a ∉ R`, where `L'` is the Gram
kernel of the vectors projected onto `ψ(a)^⊥` (Schur complement). -/
theorem gw_insert (ψ : Y → ι → ℝ) {a : Y} {R : Finset Y} (ha : a ∉ R) :
    gw ψ (insert a R) = (ψ a ⬝ᵥ ψ a) * gw (proj ψ a) R := by
  set c := ψ a ⬝ᵥ ψ a with hc
  by_cases h0 : c = 0
  · -- `ψ(a) = 0`, so the row of `a` is zero
    have hψa : ψ a = 0 := dotProduct_self_eq_zero.1 h0
    rw [h0, zero_mul, gw]
    refine det_eq_zero_of_row_eq_zero ⟨a, mem_insert_self a R⟩ fun j => ?_
    simp [gramOn, hψa]
  · let e := insertSumEquiv ha
    unfold gw
    rw [← det_submatrix_equiv_self e]
    let Dm : Matrix Unit Unit ℝ := Matrix.of fun _ _ => c
    let _ : Invertible Dm :=
      ⟨Matrix.of fun _ _ => c⁻¹, by ext; simp [Dm, Matrix.mul_apply, h0],
        by ext; simp [Dm, Matrix.mul_apply, h0]⟩
    have hblocks : (gramOn ψ (insert a R)).submatrix e e =
        fromBlocks (gramOn ψ R) (Matrix.of fun (r : R) (_ : Unit) => ψ r ⬝ᵥ ψ a)
          (Matrix.of fun (_ : Unit) (r : R) => ψ a ⬝ᵥ ψ r) Dm := by
      ext (i | i) (j | j) <;> rfl
    rw [hblocks, det_fromBlocks₂₂]
    have hD : Dm.det = c := by simp [Dm]
    rw [hD]
    congr 2
    ext r s
    have hinv : (⅟Dm : Matrix Unit Unit ℝ) = Matrix.of fun _ _ => c⁻¹ := rfl
    simp only [hinv, gramOn, proj, Matrix.sub_apply, Matrix.mul_apply, of_apply,
      Finset.univ_unique, sum_singleton]
    simp only [sub_dotProduct, dotProduct_sub, smul_dotProduct, dotProduct_smul, smul_eq_mul,
      dotProduct_comm (ψ a) (ψ s), ← hc]
    field_simp
    ring

omit [DecidableEq ι] in
/-- Conditioning on `a ∈ S`: `N_{T + a}(V, K+1, ψ) = ‖ψ(a)‖² N_T(V ∖ a, K, proj ψ a)`. -/
theorem inclN_insert (ψ : Y → ι → ℝ) {a : Y} {V : Finset Y} (ha : a ∈ V) (k : ℕ)
    {T : Finset Y} (haT : a ∉ T) :
    inclN ψ (k + 1) V (insert a T) = (ψ a ⬝ᵥ ψ a) * inclN (proj ψ a) k (V.erase a) T := by
  unfold inclN
  rw [mul_sum]
  refine sum_nbij' (fun S => S.erase a) (fun R => insert a R) ?_ ?_ ?_ ?_ ?_
  · intro S hS
    simp only [mem_filter, mem_powersetCard] at hS ⊢
    have haS : a ∈ S := hS.2 (mem_insert_self a T)
    refine ⟨⟨erase_subset_erase a hS.1.1, by rw [card_erase_of_mem haS, hS.1.2]; rfl⟩, ?_⟩
    intro x hx
    exact mem_erase.2 ⟨fun h => haT (h ▸ hx), hS.2 (mem_insert_of_mem hx)⟩
  · intro R hR
    simp only [mem_filter, mem_powersetCard] at hR ⊢
    have haR : a ∉ R := fun h => notMem_erase a V (hR.1.1 h)
    exact ⟨⟨insert_subset ha (hR.1.1.trans (erase_subset a V)),
      by rw [card_insert_of_notMem haR, hR.1.2]⟩, insert_subset_insert a hR.2⟩
  · intro S hS
    simp only [mem_filter] at hS
    exact insert_erase (hS.2 (mem_insert_self a T))
  · intro R hR
    simp only [mem_filter, mem_powersetCard] at hR
    exact erase_insert fun h => notMem_erase a V (hR.1.1 h)
  · intro S hS
    simp only [mem_filter] at hS
    have haS : a ∈ S := hS.2 (mem_insert_self a T)
    conv_lhs => rw [← insert_erase haS]
    exact gw_insert ψ (notMem_erase a S)

omit [DecidableEq ι] in
lemma inclN_empty (ψ : Y → ι → ℝ) (K : ℕ) (V : Finset Y) : inclN ψ K V ∅ = partZ ψ K V := by
  unfold inclN partZ
  rw [filter_true_of_mem fun S _ => empty_subset S]

omit [DecidableEq ι] in
lemma inclN_anti (ψ : Y → ι → ℝ) (K : ℕ) (V : Finset Y) {T T' : Finset Y} (h : T ⊆ T') :
    inclN ψ K V T' ≤ inclN ψ K V T := by
  unfold inclN
  refine sum_le_sum_of_subset_of_nonneg ?_ fun S _ _ => gw_nonneg ψ S
  intro S hS
  simp only [mem_filter] at hS ⊢
  exact ⟨hS.1, h.trans hS.2⟩

omit [DecidableEq ι] in
lemma inclN_le_partZ (ψ : Y → ι → ℝ) (K : ℕ) (V T : Finset Y) :
    inclN ψ K V T ≤ partZ ψ K V := by
  rw [← inclN_empty]; exact inclN_anti ψ K V (empty_subset T)

/-! ### Negative correlation -/

/-- **Negative correlation for Gram `K`-DPPs on finite base sets.**
`p(T ⊆ S) ≤ ∏_{y∈T} p(y ∈ S)`, with `p(T ⊆ S) = N_T / Z`. -/
theorem gram_negCorr (ψ : Y → ι → ℝ) (K : ℕ) (V T : Finset Y) :
    inclN ψ K V T / partZ ψ K V ≤ ∏ y ∈ T, inclN ψ K V {y} / partZ ψ K V := by
  induction T using Finset.induction_on generalizing ψ K V with
  | empty =>
    rw [inclN_empty, prod_empty]
    exact div_self_le_one _
  | insert a T haT ih =>
    rw [prod_insert haT]
    set Z := partZ ψ K V
    set Na := inclN ψ K V {a}
    have hZ0 := partZ_nonneg ψ K V
    have hprod0 : 0 ≤ ∏ y ∈ T, inclN ψ K V {y} / Z :=
      prod_nonneg fun y _ => div_nonneg (inclN_nonneg ψ K V {y}) hZ0
    have hNa0 : 0 ≤ Na := inclN_nonneg ψ K V {a}
    rcases hNa0.eq_or_lt with hNa | hNa
    · -- `p(a ∈ S) = 0`
      have : inclN ψ K V (insert a T) = 0 := le_antisymm
        ((inclN_anti ψ K V (singleton_subset_iff.2 (mem_insert_self a T))).trans hNa.symm.le)
        (inclN_nonneg ψ K V _)
      rw [this, zero_div, ← hNa, zero_div, zero_mul]
    -- `p(a ∈ S) > 0`, so `a ∈ V` and `K ≥ 1`
    have haV : a ∈ V := by
      by_contra h
      have : Na = 0 := by
        unfold Na inclN
        refine sum_eq_zero fun S hS => ?_
        simp only [mem_filter, mem_powersetCard, singleton_subset_iff] at hS
        exact absurd (hS.1.1 hS.2) h
      linarith
    obtain ⟨k, rfl⟩ : ∃ k, K = k + 1 := by
      rcases K with _ | k
      · exfalso
        have : Na = 0 := by
          unfold Na inclN
          refine sum_eq_zero fun S hS => ?_
          simp only [mem_filter, mem_powersetCard, card_eq_zero, singleton_subset_iff] at hS
          rw [hS.1.2] at hS; exact absurd hS.2 (notMem_empty a)
        linarith
      · exact ⟨k, rfl⟩
    -- the conditional `k`-DPP
    set ψ' := proj ψ a
    set V' := V.erase a
    set c := ψ a ⬝ᵥ ψ a
    set Z' := partZ ψ' k V'
    have hN : inclN ψ (k + 1) V (insert a T) = c * inclN ψ' k V' T :=
      inclN_insert ψ haV k haT
    have hNa' : Na = c * Z' := by
      have := inclN_insert ψ haV k (notMem_empty a)
      rw [insert_empty, inclN_empty] at this
      exact this
    have hNay : ∀ y ∈ T, inclN ψ (k + 1) V {a, y} = c * inclN ψ' k V' {y} := by
      intro y hy
      have hya : a ∉ ({y} : Finset Y) := by
        rw [mem_singleton]; exact fun h => haT (h ▸ hy)
      exact inclN_insert ψ haV k hya
    have hZ : 0 < Z := lt_of_lt_of_le hNa (inclN_le_partZ ψ (k + 1) V {a})
    have hc0 : 0 ≤ c := sum_nonneg fun i _ => mul_self_nonneg _
    have hZ'0 : 0 ≤ Z' := partZ_nonneg ψ' k V'
    have hcZ : c ≠ 0 ∧ Z' ≠ 0 := mul_ne_zero_iff.1 (hNa' ▸ hNa.ne')
    have hZ' : 0 < Z' := lt_of_le_of_ne hZ'0 (Ne.symm hcZ.2)
    -- `p(y | a) ≤ p(y)`, from the two-string inequality
    have hpair : ∀ y ∈ T, inclN ψ' k V' {y} / Z' ≤ inclN ψ (k + 1) V {y} / Z := by
      intro y hy
      have hay : a ≠ y := fun h => haT (h ▸ hy)
      have h2 := inclN_pair_le ψ (k + 1) V hay
      rw [hNay y hy] at h2
      rw [div_le_div_iff₀ hZ' hZ]
      have hc : 0 < c := lt_of_le_of_ne hc0 (Ne.symm hcZ.1)
      have : c * (inclN ψ' k V' {y} * Z) ≤ c * (inclN ψ (k + 1) V {y} * Z') := by
        change c * inclN ψ' k V' {y} * Z ≤ Na * inclN ψ (k + 1) V {y} at h2
        rw [hNa'] at h2
        nlinarith [h2]
      exact le_of_mul_le_mul_left this hc
    have hih := ih ψ' k V'
    calc inclN ψ (k + 1) V (insert a T) / Z
        = Na / Z * (inclN ψ' k V' T / Z') := by
          rw [hN, hNa']; field_simp
      _ ≤ Na / Z * ∏ y ∈ T, inclN ψ' k V' {y} / Z' :=
          mul_le_mul_of_nonneg_left hih (div_nonneg hNa.le hZ.le)
      _ ≤ Na / Z * ∏ y ∈ T, inclN ψ (k + 1) V {y} / Z :=
          mul_le_mul_of_nonneg_left
            (Finset.prod_le_prod₀ (fun y _ => div_nonneg (inclN_nonneg ψ' k V' {y}) hZ'0) hpair)
            (div_nonneg hNa.le hZ.le)

end DppLm
