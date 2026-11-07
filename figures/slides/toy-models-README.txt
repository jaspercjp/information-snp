ToyModels: rho and lambda on synthetic data
===========================================

monotonic-lambda-rho   A monotonic non-linearity lowers Pearson rho but not lambda.
estimator-toys         Which lambda estimator for N ~ 50, T ~ 30 (also in Manuscript/SuppInformation).

Model, one seeded draw of N = 1,000,000 pairs (enough that one sample suffices):
  x, z ~ N(0, 1) independent          x plays the predictable signal (ensemble mean)
  u = r x + sqrt(1 - r^2) z, r = 0.8  (x, u) bivariate Gaussian, correlation r
  y = g_a(u) = (exp(a u) - 1) / a     y plays the observations; g_0(u) = u; increasing, skewed
  a = 0, 0.25, ..., 2                 strength of the non-linearity
The noise enters before the transform, so the mutual information I(x; y) = I(x; u) is the same
for every a: the exact lambda = sqrt(1 - exp(-2 I)) = r = 0.8, while the exact Pearson
rho(x, y) = a r / sqrt(exp(a^2) - 1) falls (0.80, 0.61, 0.22 at a = 0, 1, 2).
Every a reuses the same x and z of a draw (common random numbers).

Panels
  (a)-(c) y = g_a(u) against x for a = 0, 1, 2, one panel each: 100,000 samples of one draw,
      one grey; each panel has its own y axis
      (0.1-99.9% range shown): the cloud goes from symmetric to strongly skewed as a grows.
  (d) rho (Pearson) and lambda (KSG on normal scores, the estimator the maps now use) from the
      one sample (the draw of panel a), as curves: rho 0.800 -> 0.613 -> 0.240 at a = 0, 1, 2
      (the exact curve, not drawn: 0.80, 0.61, 0.22); lambda 0.800 at every a (exact: r = 0.8).
  (e) lambda estimators against the exact lambda:
      KSG on normal scores   KSG, k = 4, after the rank transform of x and y (the maps): 0.800
                   at every a
      rank bins    plug-in MI on 100 x 100 equal-count bins; sees only ranks, which g_a keeps,
                   so it is exactly flat: 0.801 at every a
      Knuth bins   the old production estimator (equal-width bins, Knuth's count per variable, cap
                   200); not invariant: as y skews, most of it falls in the lowest bins and lambda
                   collapses, 0.80 -> 0.77 -> 0.28 at a = 0, 1, 2
      Knuth bins on normal scores   the proposed fix: the production estimator after a rank
                   transform of each variable to normal scores, Phi^-1((rank - 1/2)/N). It depends
                   only on ranks, so it is exactly the same at every a: 0.801 (it overlaps the
                   rank-bin and KSG points at this N). Knuth on the plain ranks fails: a uniform
                   histogram is best fitted by one bin, so lambda = 0 in every draw.
      KSG on raw values   KSG, k = 4, without the rank transform (infomeasure): 0.80 at every a
                   at this N, but it drifts with the skew at small N

Reading: the sample rho matches the exact curve (0.800, 0.613, 0.240 vs 0.80, 0.61, 0.22). It
needs this N: with a skewed y the correlation is set by rare extremes (at a = 2 the median sample
rho is 0.30 at N = 1e4, 0.25 at 1e5). At N = 51 (the T of SLP lead 2-4; --n 51 --draws 500) the
sample rho falls only to 0.52 at a = 2, KSG drifts to 0.68, and Knuth's rule picks a single bin
for a variable in 14-28% of draws, which makes the production lambda exactly 0. Knuth on normal
scores at N = 51 (300 draws): 0.726 at every a and never 0 (Knuth picks 4 bins on the Gaussian
marginals), against the production 0.73 / 0.66 / 0.67 at a = 0 / 1 / 2 with 16-26% zeros.

Regenerate (compute node, repository root):
  python scripts/monotonic_demo.py [--recompute] [--draws 1] [--n 1000000] [--r 0.8]
      (~1 min on 16 cores for one draw; KSG takes ~25 s per call at this N)
  python scripts/figure_set.py Presentation refresh && ... verify
Curves: figures/data/monotonic-lambda-rho.npz; tests: scripts/tests/test_monotonic_demo.py.


estimator-toys
--------------
Seven lambda estimators on four toy problems with a known exact lambda, n = 33, 51, 120 pairs
(the map sample sizes), 500 seeded draws; top row median and 5-95% of each estimate, bottom row
the shape of each case (20,000 samples in grey, one n = 33 draw in black).
  independent    y = z                                      exact lambda 0
  gaussian       y = 0.8 x + 0.6 z            (g_0(u) above) exact lambda 0.8
  skewed         y = (exp(2u) - 1)/2          (g_2(u) above) exact lambda 0.8
  non-monotonic  y = x^2 + 0.75 z, rho = 0                   exact lambda 0.825 (numerical)
Estimators: production Knuth bins on raw values; Knuth on normal scores (>= 3 bins); equal-count
rank bins; rank bins minus a 200-permutation null; Gaussian copula (|rank correlation|: monotonic
only); KSG, k = 4, on normal scores; KSG on raw values.
Reading: the production lambda is 0 in 55% of Gaussian draws at n = 33; the Gaussian copula misses
the non-monotonic case (0.15); KSG on normal scores is closest in every dependent case (0.76,
0.74, 0.65 at n = 33) with median 0 when independent.
Regenerate: python scripts/estimator_toys.py [--recompute] (~1 min on 16 cores);
numbers in figures/data/estimator-toys.json.
