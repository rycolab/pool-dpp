import DppLm.PoolSketch

/-!
# Negative correlation on an infinite base set, from finite base sets

The proof of `thm:zfree` uses `p(T ⊆ S) ≤ ∏_{y∈T} p(y ∈ S)` for the `K`-DPP on `Σ*`. The
cited result (Borcea–Brändén–Liggett: `K`-DPPs are strongly Rayleigh, hence negatively
associated) is stated for finite base sets, and `Σ*` is countably infinite.
`negCorr_of_restrict` is the limit step: if the inequality holds for the restriction of `f`
to every finite base set `U`, it holds for `f`.

The restriction is `restrictW f U S = f S` for `S ⊆ U` and `0` otherwise. For the `K`-DPP
weight `kdppWeight κ φ K`, normalizing `restrictW` gives the `K`-DPP on the finite base set
`U` whose kernel is `L_U`, the principal submatrix of `L` on `U` (again a Gram matrix). That
is the setting of the cited result.

Instead of the first `n` strings of an enumeration, the proof takes the limit along all finite
`U`, ordered by inclusion. This is the same argument and needs no countability: each sum
`∑_{S ⊆ U} f(S)` increases to `∑_S f(S)`, and `0 < Z < ∞`.
-/

open Finset Filter Topology
open scoped ENNReal

namespace DppLm

variable {Y : Type*} [DecidableEq Y]

/-- The weight `f` restricted to subsets of the finite base set `U`. -/
def restrictW (f : Finset Y → ℝ≥0∞) (U : Finset Y) (S : Finset Y) : ℝ≥0∞ :=
  if S ⊆ U then f S else 0

/-- `∑_{S ⊆ U} g(S) → ∑_S g(S)` as the finite base set `U` grows. -/
lemma tendsto_tsum_restrictW (g : Finset Y → ℝ≥0∞) :
    Tendsto (fun U => ∑' S, restrictW g U S) atTop (𝓝 (∑' S, g S)) := by
  have h1 : ∀ U : Finset Y, ∑' S, restrictW g U S = ∑ S ∈ U.powerset, g S := by
    intro U
    rw [tsum_eq_sum (s := U.powerset)]
    · refine Finset.sum_congr rfl fun S hS => ?_
      simp [restrictW, Finset.mem_powerset.1 hS]
    · intro S hS
      rw [Finset.mem_powerset] at hS
      simp [restrictW, hS]
  simp_rw [h1]
  have hs : Tendsto (fun s : Finset (Finset Y) => ∑ S ∈ s, g S) atTop (𝓝 (∑' S, g S)) :=
    CompleteLattice.summable.hasSum
  exact hs.comp tendsto_finset_powerset_atTop_atTop

omit [DecidableEq Y] in
lemma tsum_ite_le (f : Finset Y → ℝ≥0∞) (P : Finset Y → Prop) [DecidablePred P] :
    ∑' S, (if P S then f S else 0) ≤ ∑' S, f S :=
  tsum_le_tsum fun S => by split_ifs <;> simp

lemma inclProb_le_one (f : Finset Y → ℝ≥0∞) (T : Finset Y) : inclProb f T ≤ 1 := by
  unfold inclProb
  exact ENNReal.div_le_of_le_mul (by rw [one_mul]; exact tsum_ite_le f (T ⊆ ·))

/-- **Limit step.** Let `0 < ∑ f < ∞`. If, for every finite base set `U` on which the
restriction of `f` is not identically zero, the restriction satisfies
`p_U(T ⊆ S) ≤ ∏_{y∈T} p_U(y ∈ S)`, then so does `f`. -/
theorem negCorr_of_restrict (f : Finset Y → ℝ≥0∞) (hZ0 : ∑' S, f S ≠ 0)
    (hZtop : ∑' S, f S ≠ ∞)
    (h : ∀ U : Finset Y, ∑' S, restrictW f U S ≠ 0 →
      ∀ T : Finset Y, inclProb (restrictW f U) T ≤ ∏ y ∈ T, inclProb (restrictW f U) {y})
    (T : Finset Y) : inclProb f T ≤ ∏ y ∈ T, inclProb f {y} := by
  -- the inclusion sums of the restrictions increase to those of `f`
  have hnum : ∀ A : Finset Y, Tendsto (fun U => ∑' S, (if A ⊆ S then restrictW f U S else 0))
      atTop (𝓝 (∑' S, if A ⊆ S then f S else 0)) := by
    intro A
    have e : ∀ U, (fun S => if A ⊆ S then restrictW f U S else 0) =
        restrictW (fun S => if A ⊆ S then f S else 0) U := by
      intro U; funext S
      by_cases hA : A ⊆ S <;> by_cases hU : S ⊆ U <;> simp [restrictW, hA, hU]
    simp_rw [e]
    exact tendsto_tsum_restrictW _
  have hden := tendsto_tsum_restrictW f
  -- so the inclusion probabilities converge, since `0 < Z < ∞`
  have hlim : ∀ A : Finset Y,
      Tendsto (fun U => inclProb (restrictW f U) A) atTop (𝓝 (inclProb f A)) := by
    intro A
    unfold inclProb
    exact ENNReal.Tendsto.div (hnum A) (Or.inr hZ0) hden (Or.inl hZtop)
  have hprod : Tendsto (fun U => ∏ y ∈ T, inclProb (restrictW f U) {y}) atTop
      (𝓝 (∏ y ∈ T, inclProb f {y})) :=
    ENNReal.tendsto_finsetProd_of_ne_top T (fun y _ => hlim {y})
      (fun y _ => ((inclProb_le_one f {y}).trans_lt ENNReal.one_lt_top).ne)
  -- eventually the restriction is not identically zero, so the hypothesis applies
  have hev : ∀ᶠ U in atTop, ∑' S, restrictW f U S ≠ 0 := hden.eventually_ne hZ0
  exact le_of_tendsto_of_tendsto (hlim T) hprod (hev.mono fun U hU => h U hU T)

end DppLm
