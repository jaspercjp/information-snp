"""Eade et al. (2014) spatial smoothing: an area-weighted box mean over lat-lon.

    from _smooth import smooth_box, box_for

    dlat, dlon = box_for("SLP")               # (11.25, 12.5); (15, 15) for TREFHT
    Xs = smooth_box(X, lats, lons, dlat, dlon)

Eade et al.: "Model and observed data are smoothed over regions of 11.25 deg latitude by
12.5 deg longitude (15 deg by 15 deg for SAT)". A running box mean -- the grid is
preserved, each point becomes the average over a box centred on it -- applied to MODEL
AND OBSERVATIONS ALIKE.

The weight of source cell i in target cell j's box is the AREA of their overlap on the
sphere. The box separates into a longitude pass (overlap in degrees, cyclic) and a
latitude pass (overlap in sin(lat), the spherical area element, so cos(lat) weighting is
exact). Fractional windows get partial edge weights (11.25/5 = 2.25 cells on HadSLP2r);
non-uniform latitudes (NCEP T62) take edges at midpoints; NaNs are skipped and weights
renormalised over the finite cells, and a box with no finite cell returns NaN.
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
    """(nlat, nlat) area weights: row j is the box centred on lats[j]. Overlap in sin(lat)."""
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
