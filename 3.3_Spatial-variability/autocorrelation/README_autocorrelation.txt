The summary columns
-------------------
The min/max columns give ranges, since each quantity has one value per station (r₁) or per pair (the rest).

- r1_min/max: the range of lag-1 autocorrelation across stations.
- r_min/max: the range of the Fig. 9 correlations.
- n_min/max: the range of paired days.
- neff_min/max: the range of effective sample sizes across pairs.
- p_eff_max: the largest p-value after the correction, i.e. the weakest pair. (If this is below 0.05, every pair passes.) 
- pairs_total: the number of pairs tested. (36 for nine stations, and 28 for precipitation)
- pairs_p_ge_0.05: how many pairs are no longer significant
- failing_pairs: which pairs are no longer significant