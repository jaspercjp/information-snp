"""Pairwise mutual information between ensemble members -- the MI twin of the notebook's
`pearson_coeff_F_pairwise`.

    import sys; sys.path.insert(0, "../scripts")
    import dcpp_handles as G, mi_pairwise as MP, smyle_metrics as M

    c = G.get(lead="13-60", var="SLP", remove_gm=True)
    I_m = MP.MI_F_pairwise(c.F)          # (N, N, lat, lon), NATS
    lam_m = M.lam_of(I_m)                # the analogue of rho_m_pairwise

Same contract as the notebook's Pearson version: in (N, J, lat, lon), out
(N, N, lat, lon), entry [i, j] the statistic between members i and j at that cell.

NATS, not bits, KSG at k=4, copula-transformed -- i.e. the convention of
`smyle_metrics.calc_MI_sG(..., use_copula=True)`, so `smyle_metrics.lam_of` applies
directly and the numbers are comparable to the notebook's `MI_FG_pairwise`.

Where the speed comes from
--------------------------
The estimator call is `infomeasure`'s KSG, once per (pair, cell). That count is
irreducible, so everything else is about making it smaller and the calls cheaper:

* **Symmetry.** KSG's max-norm neighbourhoods are symmetric in its two arguments, so
  I(f_i; f_j) == I(f_j; f_i) exactly (checked in the test script). Only the N(N-1)/2
  upper triangle is estimated and then mirrored -- a clean factor of 2.
* **Transform once.** The copula transform and the tie-breaking dither are hoisted out
  of the inner loop and applied to the whole cube in two vectorised calls. Left inside,
  they would run 2 * N(N-1)/2 times per cell instead of N -- and `infomeasure`'s own
  dither is a *per-sample* `rng.normal` call, a third of its runtime all by itself.
  With the cube pre-dithered, the estimator is called with `noise_level=0`.
* **One task per grid cell.** 2664 tasks of ~1035 calls is ~0.5 s of work each, so
  joblib dispatch is noise and the load balances itself. Each task ships a (N, J) slab
  of ~17 kB, not the cube.

For (N, J) = (46, 48) on a 37x72 grid that is 1035 * 2664 = 2.76M KSG calls, ~21 min
of core time: ~80 s on 16 cores, ~20 s on 64.

The dither is not cosmetic -- read this before changing `noise_level`
--------------------------------------------------------------------
Ranks land on a lattice of spacing 1/(J+1), so the copula transform makes max-norm
distances massively degenerate: at J=48 only **46 of the 1128 pairwise distances are
distinct**. KSG counts neighbours by comparing distances, so its answer is decided by
how those exact ties break, and that is the dither's job. Consequences:

* `noise_level=0` with `use_copula=True` is a different estimator, not a faster one.
  It sits ~0.05 nats from the dithered answer -- comparable to the signal in a
  well-correlated pair. Do not set it to zero to save time.
* Even at 1e-10 the dither realisation is worth ~0.03 nats peak-to-peak at J=48. That
  is a floor on this estimator, inherited from the notebook's convention, not
  introduced here. `seed` makes it reproducible; vary it to see the spread.
* With `use_copula=False` the data is continuous, every distance is distinct, and the
  dither really is an exact no-op (~1e-18).

What this can and cannot resolve at J=48
----------------------------------------
On the DCPP grand ensemble (N=101, J=48, SLP lead 13-60) the off-diagonal MI runs
-0.29 .. 0.54 nats with a mean of 0.0022 -- i.e. the typical inter-member signal is an
order of magnitude below the dither floor above, and 52% of the estimates come out
negative. A single (pair, cell) number is therefore mostly noise: its per-entry
correlation with |Pearson| is only 0.22. What survives is the conditional mean, which
tracks |rho| cleanly and monotonically:

    |rho| bin   0-.1   .1-.2   .2-.3   .3-.4   .4-.6   .6-1
    mean |rho|  0.048  0.145   0.242   0.339   0.444   0.630
    mean lam    0.124  0.144   0.189   0.266   0.387   0.621

So aggregate before interpreting -- over cells, over pairs, or over a region. The
floor of ~0.12 in the lowest bin is `lam_of` rectifying negative MI to its 1e-3 floor,
not shared information. On synthetic data with real spatial signal the per-entry
correlation with |rho| is 0.85 and the field-level correlation 0.97, so the estimator
is fine; J=48 is the binding constraint.

The diagonal
------------
I(f_i; f_i) is infinite, not 1. `pearson_coeff_F_pairwise` puts 1.0 there; the honest
analogue is `np.inf`, which is what is returned, and which `lam_of` maps to 1.0 -- so
the notebook's `rho_o / rho_m_pairwise[i, i]` pattern behaves the same after `lam_of`.
Raw MI diagonals will overflow anything that averages them.
"""
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata

import infomeasure


def MI_F_pairwise(F, k=4, use_copula=True, noise_level=1e-10, seed=0, n_jobs=-1):
    """MI between every pair of members, per grid cell -> (N, N, lat, lon) in NATS.

    F : (N, J, lat, lon). Diagonal is +inf (self-information); off-diagonal is
    symmetric by construction. See the module docstring, especially on `noise_level`.
    """
    N, J = F.shape[:2]
    X = np.asarray(F, dtype=float).reshape(N, J, -1)
    if use_copula:
        X = rankdata(X, method="average", axis=1) / (J + 1.0)
    if noise_level:
        X = X + np.random.default_rng(seed).normal(0.0, noise_level, X.shape)
    X = np.ascontiguousarray(X.transpose(2, 0, 1))          # (cell, N, J)
    i, j = np.triu_indices(N, k=1)

    def calc_point(xy):                                     # xy: (N, J) at one cell
        return [infomeasure.mutual_information(xy[a], xy[b], approach="metric",
                                               k=k, noise_level=0.0)
                for a, b in zip(i, j)]

    upper = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(calc_point)(cell) for cell in X)
    upper = np.asarray(upper, dtype=float).T                 # (n_pairs, cell)

    out = np.empty((N, N, X.shape[0]))
    out[i, j] = out[j, i] = upper
    out[np.diag_indices(N)] = np.inf
    return out.reshape((N, N) + F.shape[2:])
