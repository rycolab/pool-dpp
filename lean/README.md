# Lean formalization of `thm:var-kl-mixture`, `thm:zfree` and `cor:kl-sum`

Machine-checked proofs (Lean 4 + Mathlib) of three results in `sec:analysis` of
the paper: `thm:var-kl-mixture` (variance controls bias), `thm:zfree` (the
finite-sample bound for PoolDPP) and `cor:kl-sum` (the sharper bound), together with
`prop:PoolDPP-well-behaved` (PoolDPP is well-behaved). This folder names paper results by their
LaTeX labels, not their numbers, because the numbers change when the paper is edited.

## Check it

```
sh setup.sh                       # one time: installs elan + pinned Mathlib cache (~8 GB in .lake/)
~/.elan/bin/lake build            # compiles everything
~/.elan/bin/lake env lean Check.lean   # prints the axioms each main theorem uses
```

Every main theorem depends only on Lean's standard axioms `propext`, `Classical.choice`, and `Quot.sound`.
There is no `sorry`.

## Where each statement lives

| Paper | Lean | File |
|---|---|---|
| χ² and KL, valued in `[0, ∞]`: KL is Mathlib's `klDiv`; each is finite exactly when `π̄(S) = 0 → p(S) = 0` and the usual series converges, and then equals it | `chiSq`, `kl` (`discreteMeasure`); `chiSq_ne_top_iff`, `chiSq_eq_ofReal`, `kl_ne_top_iff`, `kl_eq_ofReal` | `Sketch.lean` |
| `thm:var-kl-mixture`, any well-behaved sketch with `Ẑ > 0` almost surely: χ² ≤ Var/Z², KL ≤ log(1+Var/Z²) ≤ Var/Z² | `bias_bound` | `VarianceBias.lean` |
| Footnote `fn:wlog`, the reduction: `γ = P(Ẑ > 0) > 0`; `(μ₊, f̂)` with `μ₊ = μ[|Ẑ > 0]` has `Ẑ > 0` a.s., is well-behaved for `f/γ`, has the same target, and `E_{μ₊}[Ẑ²]/(Z/γ)² = γ E_μ[Ẑ²]/Z²` | `WellBehaved.measure_ne_zero`, `WellBehaved.cond` | `VarianceBias.lean` |
| … redrawing ends, and its law `Q = π̄₀ + P(Ẑ = 0) Q` is `π̄₀/γ`, the output law of `(μ₊, f̂)` | `redraw_law_eq`, `WellBehaved.marginalLaw_cond` | `VarianceBias.lean` |
| … if `π̄ ≥ γ π̄₊`, then `1 + χ²(p ‖ π̄) ≤ (1 + χ²(p ‖ π̄₊))/γ` and `KL(p ‖ π̄) ≤ KL(p ‖ π̄₊) + log(1/γ)` | `one_add_chiSq_le_of_le`, `kl_le_add_of_le` | `VarianceBias.lean` |
| … so the bounds hold for `(μ, f̂)`, `Var(Ẑ)` under `μ`, for redrawing and for any fallback; both proved from `bias_bound` for `(μ₊, f̂)` | `redraw_bound`, `fallback_bound` | `VarianceBias.lean` |
| Well-behaved sketch: `E[f̂_ω(S)] = f(S)`, `E[Ẑ²] < ∞` | `WellBehaved` (plus measurability) | `VarianceBias.lean` |
| `prop:PoolDPP-well-behaved` (unbiasedness), for any target weight `f` | `poolFhat_unbiased` | `PoolSketch.lean` |
| `prop:PoolDPP-well-behaved` for the paper's sketch: unbiased; if `ρ < ∞`, then `E[Ẑ²] < ∞` and the sketch is well-behaved | `poolDPP_unbiased`, `poolDPP_wellBehaved` | `PoolDPP.lean` |
| `α Σ det(L̂_I)` equals the general sketch | `paperFhat_eq` | `Determinant.lean` |
| `thm:zfree` identity, second-moment form `E[Ẑ²] = Z² Σ a_m E[u_m²]` | `zfree_identity` | `FiniteSample.lean` |
| `thm:zfree` identity, variance form `Var(Ẑ)/Z² = Σ a_m Var(u_m)`, variances in `[0, ∞]` | `zfree_variance` | `FiniteSample.lean` |
| Var of the paper's sketch = Var of the U-statistic in `zfree_variance` | `poolDPP_variance_eq` | `PoolDPP.lean` |
| `thm:zfree` KL bound for PoolDPP, plain i.i.d. pool law, any fallback: `KL ≤ log(e^θ − 1 + a₀) ≤ θ`, `θ = K² ρ/(N−K+1)`, `ρ = E_q[u₁²] = 1 + χ²(π₁ ‖ q)`, `a₀ = C(N−K,K)/C(N,K)` | `poolDPP_fallback_bound` (any fallback), `poolDPP_redraw_bound` (redraw) | `PoolDPP.lean` |
| `thm:zfree` proof: `u_m ≤ (K^m/(K)_m) ∏ᵢ u₁(yᵢ)`, `E[u_m²] ≤ (K^m/(K)_m)² ρ^m`, `E[Ẑ²]/Z² ≤ Σ_m a_m (K^m/(K)_m)² ρ^m ≤ e^θ − 1 + a₀`, `log(e^θ − 1 + a₀) ≤ θ` | `inclRatio_le_prod`, `tsum_inclRatio_sq_le`, `secondMoment_le_sharper`, `sharperBound_le_exp`, `secondMoment_le_exp`, `log_exp_sub_one_add_le` | `FiniteSample.lean` |
| `cor:kl-sum` (sharper bound) for PoolDPP, plain i.i.d. pool law, any fallback: `KL ≤ log Σ_{c=0}^{K} a_c (K^c/(K)_c)² ρ^c`; the sum equals `1 + a₁ Var_q(u₁) + Σ_{c=2}^{K} a_c ((K^c/(K)_c)² ρ^c − 1)`; its log is at most `thm:zfree`'s `log(e^θ − 1 + a₀)` | `poolDPP_sharper_fallback_bound` (any fallback), `poolDPP_sharper_redraw_bound` (redraw) | `SharperBound.lean` |
| `cor:kl-sum` proof: `E[Ẑ²]/Z² ≤ Σ_{c=0}^{K} a_c (K^c/(K)_c)² ρ^c`; the second line; the comparison with `thm:zfree` | `secondMoment_le_sharper` (any target weight), `poolDPP_secondMoment_le_sharper` (the paper's sketch), `sharperBound_eq`, `sharperBound_le_exp`, `poolDPP_sharper_sum` | `FiniteSample.lean`, `PoolDPP.lean`, `SharperBound.lean` |
| Negative correlation `p(T ⊆ S) ≤ ∏_{y∈T} p(y ∈ S)` for the `K`-DPP on Σ* (cited from Borcea–Brändén–Liggett, with an elementary proof in the appendix) | `kdpp_negCorr` | `NegCorr.lean` |
| … on finite base sets, any Gram kernel | `gram_negCorr` | `GramDPP.lean` |
| … for two strings | `inclN_pair_le` | `GramDPP.lean` |
| … limit step from finite base sets to Σ* | `negCorr_of_restrict` | `Truncation.lean` |
| Newton's inequality `e_k e_{k+2} ≤ e_{k+1}²` | `esym_newton` | `Newton.lean` |

Every declaration in `DppLm/` is used by one of the theorems in this table. The definitions
`thm:var-kl-mixture` is stated with (`sketchZ`, `marginalLaw`, `normalize`, `discreteMeasure`,
`chiSq`, `kl`) are in `Sketch.lean`; `Tuples.lean` has the sums over tuples of pool draws.

## Where the Lean statements differ from the paper

1. **Negative correlation is proved, not cited.** The paper cites Borcea–Brändén–Liggett for
   `p(T ⊆ S) ≤ ∏_{y∈T} p(y ∈ S)`. That result is not in Mathlib, and it is stated for finite
   base sets. `kdpp_negCorr` proves the inequality for the paper's `K`-DPP on Σ* by an
   elementary argument (section "Negative correlation without Borcea–Brändén–Liggett"
   below), so the PoolDPP theorems have no negative-correlation hypothesis. The
   general-weight lemmas `secondMoment_le_exp` and `secondMoment_le_sharper` hold for any
   target weight `f`, so they take the inequality as a hypothesis (`hNA`).
2. **The fallback must be measurable in Lean.** `fallback_bound` asks that the fallback's probability of
   each `S` be a.e.-measurable in `ω` (`hout_meas`), the per-`S` form of the paper's "some
   measurable distribution over sets".
3. **Measurability is weaker in Lean.** `thm:var-kl-mixture` assumes `ω ↦ f̂_ω(S)` is a.e.-measurable
   for each `S` (`WellBehaved.meas`). The paper's definition of a sketch asks for a measurable
   map `f̂`, which for the product σ-algebra on weight functions means each `ω ↦ f̂_ω(S)` is
   measurable, so the paper's condition implies the Lean one.

## `Ẑ = 0` and the fallback (`VarianceBias.lean`)

`thm:var-kl-mixture` supposes, without loss of generality, that `Ẑ > 0` almost surely, so the
output law is `π̄ = E_ω[π̂_ω]` (`bias_bound`). The paper's footnote `fn:wlog` reduces the general
case to this one, and the Lean follows it step by step. Weights are nonnegative, so `Ẑ(ω) = 0`
forces `f̂_ω = 0`, and `γ = P(Ẑ > 0)` is positive because `E[Ẑ] = Z > 0`
(`WellBehaved.measure_ne_zero`). Conditioning `μ` on `Ẑ > 0` gives the sketch `(μ₊, f̂)`, which
has `Ẑ > 0` almost surely, is well-behaved for `f/γ` (a rescaling of `f` with the same target),
and has `E_{μ₊}[Ẑ²]/(Z/γ)² = γ E_μ[Ẑ²]/Z²` (`WellBehaved.cond`). Redrawing `ω` until `Ẑ > 0`
runs this sketch: a redraw's output has a fresh run's law `Q`, so `Q = π̄₀ + P(Ẑ = 0) Q` with
`π̄₀(S) = E[π̂_ω(S); Ẑ ≠ 0]`, and the only finite solution is `π̄₀/γ`, the output law of
`(μ₊, f̂)` (`redraw_law_eq`, `WellBehaved.marginalLaw_cond`). The loop itself, a sequence of
independent draws of `ω` stopped at the first with `Ẑ > 0`, is not modeled in Lean. Any other
fallback outputs a law `π̄ ≥ π̄₀ = γ π̄₊`, which costs at most a factor `1/γ` in `1 + χ²` and
`log(1/γ)` in KL (`one_add_chiSq_le_of_le`, `kl_le_add_of_le`). Since `1 + Var/Z²` of `(μ₊, f̂)`
is `γ` times that of `(μ, f̂)`, `bias_bound` for `(μ₊, f̂)` gives the bounds for `(μ, f̂)` with
`Var(Ẑ)` under `μ`, for redrawing and for any fallback (`redraw_bound`, `fallback_bound`).
For PoolDPP, `μ` is the plain i.i.d. pool law, the law of `thm:zfree`'s identity, and no
assumption on the embeddings is needed.

Two scripts check these statements with exact rational arithmetic, enumerating every pool:

* `check_fallback.py`: `thm:var-kl-mixture` for the output laws of several samplers (redraw, uniform
  fallback, and every fixed fallback set) on 277 random instances with collinear embeddings and
  `P(Ẑ = 0)` up to 0.895 (about 2 minutes: `python3 check_fallback.py`).
* `check_m1.py`: `thm:zfree`'s identity, the second-moment bound
  `Var(u_c) ≤ (K^c/(K)_c)² ρ^c − 1` (`eq:second-moment`), `cor:kl-sum`, the closed form
  `Σ_c θ^c/c!`, and `ρ = 1 + Var_q(u₁)`, on 300 random instances (about 5 seconds:
  `python3 check_m1.py`).

## Negative correlation without Borcea–Brändén–Liggett (`Newton.lean`, `RankTwo.lean`, `GramDPP.lean`, `NegCorr.lean`)

The paper needs `p(T ⊆ S) ≤ ∏_{y∈T} p(y ∈ S)` for the `K`-DPP with kernel
`L(x, y) = ψ(x)ᵀψ(y)`, `ψ(y) = √κ(y) φ(y)`. The cited route (strongly Rayleigh measures are
negatively associated) needs the theory of real-stable polynomials. The proof here uses only
linear algebra and Newton's inequality:

1. **Bridge** (`partZ_eq_coeff`). On a finite base set `V`, the normalizer
   `Z_K(V) = Σ_{S⊆V, |S|=K} det(L_S)` is the coefficient of `X^K` in `det(1 + X M(V))`, where
   `M(V) = Σ_{y∈V} ψ(y)ψ(y)ᵀ` is `D × D`. (Weinstein–Aronszajn plus the principal-minor
   expansion of `det(1 + X L)`; both are in Mathlib.)
2. **Rotation** (`exists_rotation`). Rotating every `ψ(y)` by the same orthogonal matrix leaves
   `L` unchanged, so we may assume `M(V ∖ {a, b}) = diag(μ)`.
3. **Two strings** (`rank_two_rayleigh`, `inclN_pair_le`). Write `u = ψ(a)`, `w = ψ(b)`,
   `c_i = e_{K-1}(μ_{-i})`, `d_ij = e_{K-2}(μ_{-ij})`. The four normalizers with neither, one, or
   both of `a, b` are `A`, `A + B`, `A + C`, `A + B + C + E` with `A = e_K(μ)`,
   `B = Σ u_i² c_i`, `C = Σ w_i² c_i`, `E = ½ Σ_{i,j} (u_i w_j − u_j w_i)² d_ij`
   (`det_diagonal_add_rank_two`). `p(a,b ∈ S) ≤ p(a ∈ S) p(b ∈ S)` is `AE ≤ BC`, and
   `BC − AE = (Σ u_i w_i c_i)² + ½ Σ_{i,j} (u_i w_j − u_j w_i)² (c_i c_j − A d_ij)`,
   where for `i ≠ j`, `c_i c_j − A d_ij = e_{K-1}(ν)² − e_K(ν) e_{K-2}(ν) ≥ 0` with `ν = μ_{-ij}`
   (Newton's inequality, `esym_newton`, proved by induction on the number of variables).
4. **Conditioning** (`gw_insert`, `inclN_insert`). `det(L_{R+a}) = ‖ψ(a)‖² det(L'_R)` where `L'`
   is the Gram kernel of the vectors projected onto `ψ(a)^⊥` (Schur complement). So given
   `a ∈ S`, the rest of `S` is a `(K−1)`-DPP of the same kind.
5. **Induction on `|T|`** (`gram_negCorr`): `p(T+a) = p(a) p(T | a) ≤ p(a) ∏_{y∈T} p(y | a)
   ≤ p(a) ∏_{y∈T} p(y)`, the last step by step 3.
6. **Limit to Σ*** (`negCorr_of_restrict`), since `0 < Z < ∞`.

A quick web search did not find this proof; the identity in step 3 may be known.

## Lean conventions that could hide a failure, and how they are ruled out

* `x / 0 = 0` and `log 0 = 0` in ℝ, and `tsum` of a divergent series is `0`: a real-valued KL or
  χ² sum would count a term with `π̄(S) = 0 < p(S)` as `0`, and a divergent series as `0`. The
  divergences are therefore valued in `[0, ∞]`. `kl` is Mathlib's `klDiv`, which is `∞` unless
  `p ≪ π̄` and `log(p/π̄)` is integrable, and `chiSq` is summed in `ℝ≥0∞`, where `x / 0 = ∞` for
  `x ≠ 0`. A finite bound on either one implies `π̄(S) = 0 → p(S) = 0` and that the real series
  converges (`kl_ne_top_iff`, `chiSq_ne_top_iff`).
* Mathlib's `variance` is `0` when the variance is infinite: `thm:var-kl-mixture`'s `Var(Ẑ)` is of
  a well-behaved sketch, whose `E[Ẑ²] < ∞`. `zfree_variance` uses `evariance`, the variance in
  `[0, ∞]`.
* `x / 0 = ∞` in `ℝ≥0∞` (for `x ≠ 0`): only pools containing a string with `q = 0` hit this. They
  have probability zero, and the support hypothesis makes those numerators zero.

## Proof route (where it departs from the paper's text)

* `thm:var-kl-mixture`: the paper uses conditional Jensen; we use the equivalent Cauchy–Schwarz step
  `f(S)² ≤ π̄(S) · E[π̂(S) Ẑ²]`. For `KL ≤ log(1+χ²)` we apply `log x ≤ x − 1` at
  `x = p/(c π̄)`, with `c = 1 + χ²`, instead of Jensen over an infinite sum.
* `thm:zfree`: Hoeffding's decomposition, proved by an independent-copies identity
  (`tsum_weight_piecewise`) plus counting pairs of index sets by overlap
  (`card_filter_inter`). For the KL bound, negative correlation gives
  `u_m ≤ (K^m/(K)_m) ∏ᵢ u₁(yᵢ)`; squaring it and using the independence of the draws gives
  `E[u_m²] ≤ (K^m/(K)_m)² ρ^m`, and since `C(K,m)/(K)_m = 1/m!`, `a_m E[u_m²] ≤ θ^m/m!` with
  `θ = K² ρ/(N−K+1)` for `m ≥ 1`. The `m = 0` term is kept exact: `a₀ E[u₀²] = a₀ =
  C(N−K,K)/C(N,K)`. So `E[Ẑ²]/Z² ≤ Σ_m a_m (K^m/(K)_m)² ρ^m` (the sum of `cor:kl-sum`)
  `≤ a₀ + Σ_{m=1}^{K} θ^m/m! ≤ e^θ − 1 + a₀`, and
  `KL ≤ log(E[Ẑ²]/Z²) ≤ log(e^θ − 1 + a₀) ≤ θ`, using `E[Ẑ²] ≥ Z²` and `a₀ ≤ 1`.
* `cor:kl-sum`: the `thm:zfree` steps up to the sum, without the closed form (the Lean proof
  of `thm:zfree` goes through this sum: `secondMoment_le_exp` is `secondMoment_le_sharper`
  followed by `sharperBound_le_exp`). Every term of
  `E[Ẑ²]/Z² = Σ_{c=0}^{K} a_c E[u_c²]` gets `E[u_c²] ≤ (K^c/(K)_c)² ρ^c`, and `thm:var-kl-mixture`
  gives `KL ≤ log(E[Ẑ²]/Z²)`. The paper's appendix writes the same sum with the `−1` of each
  variance kept; `sharperBound_eq` shows the two forms agree.
