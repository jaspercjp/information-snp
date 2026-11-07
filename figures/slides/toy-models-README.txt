ToyModels: rho and lambda on synthetic data
===========================================

monotonic-lambda-rho   A monotonic non-linearity lowers Pearson rho but not lambda.

Model, per draw of N = 51 pairs (the T of SLP lead 2-4), 500 seeded draws:
  x, z ~ N(0, 1) independent          x plays the predictable signal (ensemble mean)
  u = r x + sqrt(1 - r^2) z, r = 0.8  (x, u) bivariate Gaussian, correlation r
  y = g_a(u) = (exp(a u) - 1) / a     y plays the observations; g_0(u) = u; increasing, skewed
  a = 0, 0.25, ..., 2                 strength of the non-linearity
The noise enters before the transform, so the mutual information I(x; y) = I(x; u) is the same
for every a: the exact lambda = sqrt(1 - exp(-2 I)) = r = 0.8, while the exact Pearson
rho(x, y) = a r / sqrt(exp(a^2) - 1) falls (0.80, 0.61, 0.22 at a = 0, 1, 2).
Every a reuses the same x and z of a draw (common random numbers).

Panels
  (a) one draw at a = 0, 1, 2; y standardised.
  (b) rho: exact curve and the sample value (median and interquartile range over the draws);
      lambda: exact (= r) and the rank-bin estimate.
  (c) three lambda estimators against the exact lambda:
      rank bins    plug-in MI on 4 x 4 equal-count bins; sees only ranks, which g_a keeps, so it
                   is exactly flat (its level, 0.77, is below r: coarse bins on 51 points)
      Knuth bins   the production estimator (equal-width bins, Knuth's count per variable on its
                   51 values); it picks a single bin for a variable in 14-28% of draws, which
                   makes lambda exactly 0
      KSG, k = 4   nearest-neighbour estimator (infomeasure); drifts down as y skews

Reading: at N = 51 the sample rho falls much less than the exact rho (0.52 vs 0.22 at a = 2):
with a skewed y the population rho is set by rare extremes that 51 samples seldom contain.

Regenerate (compute node, repository root):
  python scripts/monotonic_demo.py [--recompute] [--draws 500] [--n 51] [--r 0.8]
  python scripts/figure_set.py Presentation refresh && ... verify
Curves: figures/data/monotonic-lambda-rho.npz; tests: scripts/tests/test_monotonic_demo.py.
