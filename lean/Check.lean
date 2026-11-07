import DppLm
-- the divergences: Mathlib's `klDiv` and a `[0, ∞]`-valued χ², equal to the usual series
#print axioms DppLm.kl_eq_ofReal
#print axioms DppLm.kl_ne_top_iff
#print axioms DppLm.chiSq_eq_ofReal
#print axioms DppLm.chiSq_ne_top_iff
-- `thm:var-kl-mixture` (a well-behaved sketch with Ẑ > 0 almost surely)
#print axioms DppLm.bias_bound
-- footnote `fn:wlog`: the conditioned sketch, the redraw law, the comparison lemmas, and the
-- bounds for the redraw sampler and any fallback, proved from `bias_bound`
#print axioms DppLm.WellBehaved.measure_ne_zero
#print axioms DppLm.WellBehaved.cond
#print axioms DppLm.WellBehaved.marginalLaw_cond
#print axioms DppLm.redraw_law_eq
#print axioms DppLm.one_add_chiSq_le_of_le
#print axioms DppLm.kl_le_add_of_le
#print axioms DppLm.redraw_bound
#print axioms DppLm.fallback_bound
-- `prop:PoolDPP-well-behaved`
#print axioms DppLm.poolFhat_unbiased
#print axioms DppLm.poolDPP_unbiased
#print axioms DppLm.poolDPP_wellBehaved
#print axioms DppLm.paperFhat_eq
-- `thm:zfree`: the identity, and the KL bound for PoolDPP (any fallback)
#print axioms DppLm.zfree_identity
#print axioms DppLm.zfree_variance
#print axioms DppLm.poolDPP_variance_eq
#print axioms DppLm.poolDPP_fallback_bound
#print axioms DppLm.poolDPP_redraw_bound
-- `thm:zfree`: steps of the proof
#print axioms DppLm.inclRatio_le_prod
#print axioms DppLm.tsum_inclRatio_sq_le
#print axioms DppLm.secondMoment_le_exp
#print axioms DppLm.overlapCoeff_zero_toReal_le_one
#print axioms DppLm.log_exp_sub_one_add_le
-- `cor:kl-sum` (sharper bound): the corollary for PoolDPP, and steps of its proof
#print axioms DppLm.poolDPP_sharper_fallback_bound
#print axioms DppLm.poolDPP_sharper_redraw_bound
#print axioms DppLm.poolDPP_sharper_sum
#print axioms DppLm.sharperBound_eq
#print axioms DppLm.sharperBound_le_exp
#print axioms DppLm.secondMoment_le_sharper
#print axioms DppLm.poolDPP_secondMoment_le_sharper
-- negative correlation of the K-DPP (the paper cites Borcea–Brändén–Liggett)
#print axioms DppLm.kdpp_negCorr
#print axioms DppLm.negCorr_of_restrict
#print axioms DppLm.gram_negCorr
#print axioms DppLm.inclN_pair_le
#print axioms DppLm.gw_insert
#print axioms DppLm.partZ_eq_coeff
#print axioms DppLm.rank_two_rayleigh
#print axioms DppLm.det_diagonal_add_rank_two
#print axioms DppLm.esym_newton
