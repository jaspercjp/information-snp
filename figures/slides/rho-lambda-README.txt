RhoLambdaMaps: signal-to-noise maps, rho RPC and lambda RPC
===========================================================

Each slide: columns SLP, PRECT, TREFHT; top row rho RPC, bottom row lambda RPC.
Colour: RPC from 0 to >=2, white at 1.

  rho RPC     Pearson rho_o / rho_m (ensemble mean vs observations; vs the members).
  lambda RPC  Granger-Lin lambda_o / lambda_m from Knuth-binned mutual information, with lambda_m
              and lambda_o both from the leave-one-out ensemble mean s_-n:
              lambda_o from I(s_-n; o), using the observations' T values only (never repeated per
              member); lambda_m from I(f; s_-n).

Marks
  black dot   not significant: the 5-95% interval of 1000 Eade et al. (2014) bootstrap draws
              contains 1 (time resampled with replacement -- start dates, or whole years for the
              monthly decade -- and N-3 members without). rho and lambda have separate bootstraps;
              lambda's keeps the full-data bins.
  green ring  on a dot: lambda changes by more than itself between 0.5x and 1.5x the bin count.
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

Regenerate (on a compute node, from the repository root)
  bootstraps  scripts/boot.py VAR CASE {fd,olmloo}  (.claude/scripts/rho_lambda_boot*.sbatch)
  slides      python scripts/plots.py slides [lead2-4] [moy] [canesm5]
  set         python scripts/figure_set.py Presentation refresh && ... verify
Area shares: figures/data/rho-lambda-<slide>.json.
