"""Eade et al. (2014) spatial smoothing: an area-weighted box mean over lat-lon.

    from _smooth import smooth_box, box_for

    dlat, dlon = box_for("SLP")               # (11.25, 12.5); (15, 15) for TREFHT
    Xs = smooth_box(X, lats, lons, dlat, dlon)

Eade et al. state: "Model and observed data are smoothed over regions of 11.25 deg
latitude by 12.5 deg longitude (15 deg by 15 deg for SAT)". That is a running box
mean -- the grid is preserved, each point becomes the average over a box centred on
it -- and it is applied to MODEL AND OBSERVATIONS ALIKE. Applying it to one side only
would be worse than not applying it at all, since the whole point is to compare like
with like.

Why this matters for RPC
-----------------------
Smoothing removes spatially incoherent variance while leaving the large-scale
predictable signal. It therefore moves both rho_o and rho_m, and it is the single
largest preprocessing difference between this project and Eade's. Without it the
model retains sub-grid variance the observational analyses have already smoothed
away -- measured here as sd_tot(model)/sd_tot(obs) = 1.31 for DJF SLP -- which
inflates sigma_tot, depresses rho_m and so inflates RPC.

What "area-weighted" means here
-------------------------------
The weight of source cell i in target cell j's box is the AREA of the overlap
between cell i and the box, on the sphere. Because the box is a lat-lon rectangle
the operator separates into a longitude pass and a latitude pass:

* longitude: overlap in degrees between cell i's span and the box, wrapped
  cyclically. Uniform spacing is not assumed but is the normal case.
* latitude: overlap in `sin(lat)`, not degrees. sin-differences ARE the spherical
  area element, so this gets the cos(lat) weighting exactly right rather than
  approximating it, and it degrades gracefully at the poles.

Consequences worth knowing:

* **Fractional windows are handled exactly.** On the 5 deg HadSLP2r grid the box is
  11.25/5 = 2.25 cells by 12.5/5 = 2.5 cells. Rounding that to an odd cell count
  would smooth by 15x15 deg or 10x10 deg instead of what Eade specifies; edge cells
  here get partial weight instead.
* **Grids may be non-uniform in latitude.** NCEP's T62 Gaussian grid is, so cell
  edges are taken as midpoints between neighbouring latitudes rather than assumed.
* **NaNs are skipped, not spread.** Weights are renormalised over the finite cells
  in each box, so one missing cell does not blank its whole neighbourhood. A box
  with no finite cell returns NaN.
* **Neighbouring points become strongly correlated.** The box spans several cells,
  so a pointwise significance test over the smoothed field has far fewer effective
  degrees of freedom than it has grid points. Eade's own maps stipple at a pointwise
  90% level with no multiple-testing correction, which is worth remembering when
  comparing against an FDR-controlled result.
"""
import numpy as np

# Eade's boxes. TREFHT is this project's name for near-surface air temperature, the
# "SAT" that gets the larger box.
BOX_DEFAULT = (11.25, 12.5)
BOX_BY_VAR = {"TREFHT": (15.0, 15.0)}


def box_for(var):
    """(dlat, dlon) in degrees for a variable name."""
    return BOX_BY_VAR.get(var, BOX_DEFAULT)


def _edges(x, lo=None, hi=None):
    """Cell edges from centres: midpoints inside, half-steps at the ends."""
    x = np.asarray(x, float)
    mid = 0.5 * (x[1:] + x[:-1])
    first = x[0] - (mid[0] - x[0]) if x.size > 1 else x[0] - 0.5
    last = x[-1] + (x[-1] - mid[-1]) if x.size > 1 else x[-1] + 0.5
    e = np.concatenate([[first], mid, [last]])
    if lo is not None:
        e = np.clip(e, lo, hi)
    return e


def lat_weights(lats, dlat):
    """(nlat, nlat) area weights: row j is the box centred on lats[j].

    Overlap is measured in sin(lat) because that is the spherical area element, so
    the cos(lat) weighting is exact rather than approximated.
    """
    lats = np.asarray(lats, float)
    e = _edges(lats, -90.0, 90.0)
    bot, top = e[:-1], e[1:]
    half = dlat / 2.0
    lo = np.clip(lats - half, -90.0, 90.0)[:, None]
    hi = np.clip(lats + half, -90.0, 90.0)[:, None]
    s_lo = np.sin(np.deg2rad(np.maximum(bot[None, :], lo)))
    s_hi = np.sin(np.deg2rad(np.minimum(top[None, :], hi)))
    return np.maximum(s_hi - s_lo, 0.0)


def lon_weights(lons, dlon):
    """(nlon, nlon) weights, cyclic in longitude. Overlap in degrees."""
    lons = np.asarray(lons, float)
    n = lons.size
    span = 360.0
    e = _edges(lons)
    width = np.diff(e)                                   # per-cell width
    half = dlon / 2.0
    # signed angular distance from each target centre to each source centre, wrapped
    d = lons[None, :] - lons[:, None]
    d = (d + span / 2.0) % span - span / 2.0
    bot = d - width[None, :] / 2.0
    top = d + width[None, :] / 2.0
    return np.maximum(np.minimum(top, half) - np.maximum(bot, -half), 0.0)


def smooth_box(X, lats, lons, dlat, dlon):
    """Area-weighted running box mean over the last two axes of `X`.

    X    : (..., nlat, nlon), any number of leading axes. NaNs are skipped.
    Returns an array of the same shape and dtype float.
    """
    X = np.asarray(X, float)
    nlat, nlon = X.shape[-2], X.shape[-1]
    if len(lats) != nlat or len(lons) != nlon:
        raise ValueError(f"grid {len(lats)}x{len(lons)} does not match field "
                         f"{nlat}x{nlon}")
    WL = lat_weights(lats, dlat)                          # (nlat, nlat)
    WO = lon_weights(lons, dlon)                          # (nlon, nlon)

    good = np.isfinite(X)
    Z = np.where(good, X, 0.0)
    M = good.astype(float)

    # longitude pass, then latitude pass -- the box separates, so this is exact
    Z = np.einsum("...ab,cb->...ac", Z, WO)
    M = np.einsum("...ab,cb->...ac", M, WO)
    Z = np.einsum("ca,...ab->...cb", WL, Z)
    M = np.einsum("ca,...ab->...cb", WL, M)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = Z / M
    out[M <= 0] = np.nan
    return out


if __name__ == "__main__":
    # a constant field must survive untouched, everywhere, including at the poles
    lats = np.arange(-90.0, 90.1, 5.0)
    lons = np.arange(0.0, 360.0, 5.0)
    C = np.full((len(lats), len(lons)), 3.7)
    S = smooth_box(C, lats, lons, 11.25, 12.5)
    print("constant field preserved : max|S - 3.7| = %.3e" % np.abs(S - 3.7).max())

    # a longitude-only wave must stay a wave of the same phase, damped not shifted
    k = 4
    W = np.tile(np.cos(k * np.deg2rad(lons)), (len(lats), 1))
    SW = smooth_box(W, lats, lons, 11.25, 12.5)
    a, b = W[len(lats) // 2], SW[len(lats) // 2]
    print("cyclic lon wave k=4     : amplitude %.4f, shape corr %.8f (no phase shift)"
          % (b.max(), np.corrcoef(a, b)[0, 1]))

    # smoothing must reduce variance, and more so for small scales than large
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((200, len(lats), len(lons)))
    big = np.tile(np.cos(np.deg2rad(lons)), (200, len(lats), 1))
    for name, F in (("white noise", noise), ("k=1 wave", big)):
        r = smooth_box(F, lats, lons, 11.25, 12.5).std() / F.std()
        print("sd ratio after smoothing: %-12s %.4f" % (name, r))

    # NaNs must not spread
    G = np.full((len(lats), len(lons)), 1.0)
    G[10, 10] = np.nan
    SG = smooth_box(G, lats, lons, 11.25, 12.5)
    print("NaN containment         : %d non-finite cell(s) out of %d"
          % ((~np.isfinite(SG)).sum(), SG.size))

    # weights must normalise to 1 per target row
    print("row sums                : lat %.6f..%.6f  lon %.6f..%.6f (pre-normalise)"
          % (lat_weights(lats, 11.25).sum(1).min(), lat_weights(lats, 11.25).sum(1).max(),
             lon_weights(lons, 12.5).sum(1).min(), lon_weights(lons, 12.5).sum(1).max()))
    print("box_for('SLP') =", box_for("SLP"), "  box_for('TREFHT') =", box_for("TREFHT"))
