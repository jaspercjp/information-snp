RhoLambdaMaps: signal-to-noise maps, rho RPC and lambda RPC
===========================================================

Each slide: columns SLP, PRECT, TREFHT; top row rho RPC, bottom row lambda RPC.
Colour: RPC from 0 to >=2, white at 1.

  rho RPC     Pearson rho_o / rho_m (ensemble mean vs observations; vs the members).
  lambda RPC  Granger-Lin lambda_o / lambda_m, lambda = sqrt(1 - exp(-2 I)), with I from the KSG
              nearest-neighbour estimator (k = 4) on normal scores, Phi^-1((rank - 1/2)/n), of each
              variable (scripts/ksg_lambda.py): rank-based, so no assumption on the distributions
              and no change under any monotonic distortion (skew):
              lambda_o from I(o; s) on the T pairs of observations and ensemble mean (never
              repeated per member); lambda_m from I(f_n; s_-n), each member against the mean of the
              others on its T pairs, averaged over a seeded 16 members, then one lambda.
              Until 2026-11-07 these slides used Knuth-binned MI on raw values, which is biased at
              T ~ 30-60 (zero in many cells, inflated lambda_o) and collapses under skew; the
              comparison is in figures/slides/lambda-ksg-vs-knuth-* and the choice in
              ToyModels/estimator-toys.

Marks
  black dot   not significant: the 5-95% interval of the Eade et al. (2014) bootstrap draws
              contains 1 (time resampled with replacement -- start dates, or whole years for the
              monthly decade -- and N-3 members without); rho 1000 draws, lambda 200. At these T the
              lambda intervals are wide: a dot means "cannot tell", not "RPC = 1".
  grey        rho_o <= 0, or lambda_m ~ 0 (<= 0.02, where lambda_o / lambda_m is not defined).
  under each map: share of the area not significant, significantly > 1, significantly < 1.

Slides
  rho-lambda-lead2-4   DJF lead months 2-4, one value per start date; five November models pooled
                       (HadGEM3-GC31-MM, EC-Earth3, MIROC6, MPI-ESM1-2-LR, MRI-ESM2-0), N = 62;
                       T = 51 (SLP), 33 (PRECT, GPCP from 1979), 52 (TREFHT).
  rho-lambda-moy       The monthly decade (s1961; PRECT s1978), each model's month-of-year
                       climatology removed, then a line removed per member; six models pooled,
                       N = 92 (SLP), 112 (PRECT, TREFHT); T = 120.
  rho-lambda-canesm5   CanESM5 alone, DJF lead months 12-14: CanESM5 starts in January, so this is
                       its first full DJF, the single-model counterpart of lead 2-4 (one year longer
                       lead). N = 20 (SLP), 40 (PRECT, TREFHT); T = 58, 42, 60.

No-stipple versions: rho-lambda-<slide>-nostipple are the same maps without the dots, to see the
colours; the shares under each map are unchanged (they still say how much is significant).

Regenerate (on a compute node, from the repository root)
  bootstraps  rho: scripts/boot.py VAR CASE fd (.claude/scripts/rho_lambda_boot*.sbatch);
              lambda: scripts/ksg_lambda.py run/merge VAR CASE (.claude/scripts/ksg_lambda.sbatch)
  slides      python scripts/plots.py slides [lead2-4] [moy] [canesm5]
  set         python scripts/figure_set.py Presentation refresh && ... verify
Area shares: figures/data/rho-lambda-<slide>.json.
