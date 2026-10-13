"""Loading, subsetting and detrending one per-model lead-window cube. Behind `dcpp_handles`.

Cubes are built by `build_dcpp_cube.py` / `build_depresys4_cube.py`, one npz per (model,
variable, lead window): cube (J, N, lat, lon) raw members, obs (J, lat, lon), year,
month, obs_ok (J,). The set of initialization months is derived from the cube itself.

On load, in this order and identically for model AND observations:

1. `remove_gm` (off by default): subtract the area-weighted global mean of each map. See
   `remove_global_mean` for why SLP needs this.
2. Eade et al. (2014)'s box mean -- 11.25 deg latitude by 12.5 deg longitude, 15 x 15 for
   TREFHT. See `_smooth.py`. Pass `smooth=False` to skip.
3. Detrend within each init month: remove a least-squares line over the start dates,
   per cell (the mean goes with it).

Samples with no observations (`obs_ok` False, the window runs past the record) are
dropped from `J` whenever the cube has any observations at all.
"""

import os

import numpy as np

import _smooth

MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# CESM calls sea-level pressure PSL; this project calls it SLP. Accept either spelling.
_ALIAS = {"PSL": "SLP"}


def canon(var):
    return _ALIAS.get(var, var)


def _cube_path(var, lead, cubes):
    return os.path.join(cubes, f"{canon(var)}_lead{lead}.npz")


def _line(X):
    """Remove the least-squares line along axis 0, per grid cell. Mean goes with it."""
    t = np.arange(X.shape[0], dtype=float)
    t -= t.mean()
    flat = X.reshape(X.shape[0], -1)
    slope = (t[:, None] * flat).sum(axis=0) / (t ** 2).sum()
    return (flat - t[:, None] * slope - flat.mean(axis=0)).reshape(X.shape)


def _line_nan(X):
    """`_line` fitted per cell over the finite samples only; non-finite stay NaN."""
    t = np.arange(X.shape[0], dtype=float)
    flat = X.reshape(X.shape[0], -1).astype(float)
    m = np.isfinite(flat)
    n = m.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tbar = np.where(m, t[:, None], 0.0).sum(axis=0) / n
        ybar = np.where(m, flat, 0.0).sum(axis=0) / n
        dt = np.where(m, t[:, None] - tbar, 0.0)
        dy = np.where(m, flat - ybar, 0.0)
        slope = (dt * dy).sum(axis=0) / (dt ** 2).sum(axis=0)
        out = flat - t[:, None] * slope - (ybar - tbar * slope)
    out[~m] = np.nan
    out[:, n < 3] = np.nan               # too few samples to detrend meaningfully
    return out.reshape(X.shape)


def area_weights(lats, lons):
    """cos(lat) weights on a (lat, lon) grid, normalised to sum to 1."""
    w = np.cos(np.deg2rad(np.asarray(lats)))[:, None] * np.ones(len(lons))
    return w / w.sum()


def remove_global_mean(X, lats, lons):
    """Subtract the area-weighted global mean of each map. NaN-aware.

    For SLP the area mean is fixed by the mass of the atmosphere, so the global mean of an
    SLP anomaly is ~0 (HadSLP2r: sd 6.3 Pa on 48-month means). EC-Earth3's dcppA-hindcast
    `psl` instead carries a spatially uniform offset that varies by start date (sd 28.8
    Pa), identical in every member, so it survives ensemble averaging and lands in the
    ensemble mean of every cell. Applied to model and observations alike.
    """
    w = area_weights(lats, lons)
    finite = np.isfinite(X)
    ww = np.where(finite, w, 0.0)
    tot = ww.sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        gm = np.nansum(np.where(finite, X, 0.0) * ww, axis=(-2, -1)) / tot
    return X - gm[..., None, None]


def detrend(X, month, nan_aware=False):
    """`_line` within each init month, so the seasonal cycle goes with the trend."""
    fit = _line_nan if nan_aware else _line
    out = np.empty_like(X, dtype=float)
    for m in np.unique(month):
        k = month == m
        out[k] = fit(X[k])
    return out


class Cube:
    """One (variable, lead window, init month) slice of one model, detrended.

        F       (N, J, lat, lon)   members
        G       (J, lat, lon)      observations
        YEAR    (J,)               initialization year
        MONTH   (J,)               initialization month
    """

    def __init__(self, var, lead, season, cubes, require_obs=None, detrend_data=True,
                 smooth=True, years=None, remove_gm=False):
        var = canon(var)
        z = np.load(_cube_path(var, lead, cubes))
        year, month, ok = z["year"], z["month"], z["obs_ok"]

        # None means "require observations if this variable has any".
        if require_obs is None:
            require_obs = bool(ok.any())

        seasons = {MONTH_ABBR[m]: int(m) for m in sorted(np.unique(month))}
        keep = np.ones(len(year), bool)
        if season != "all":
            if season not in seasons:
                raise KeyError(f"season {season!r} not in this cube; use one of "
                               f"{list(seasons)} or 'all'")
            keep &= month == seasons[season]
        if require_obs:
            keep &= ok
        # Pooling models: each member set is detrended on its own (Eade corrects each
        # model separately) but over the SAME start dates. Filtering here, before
        # detrend(), is what makes that work.
        if years is not None:
            keep &= np.isin(year, np.asarray(list(years)))
        if not keep.any():
            raise ValueError(f"no samples for {var} lead {lead} season {season} "
                             f"(require_obs={require_obs})")

        self.var, self.lead, self.season = var, lead, season
        self.units = str(z["units"]) if "units" in z else ""
        self.lats, self.lons = z["lat"], z["lon"]
        self.members = z["members"]
        self.YEAR, self.MONTH = year[keep], month[keep]

        cube = z["cube"][keep].astype(float)                 # (J, N, lat, lon)
        obs = z["obs"][keep].astype(float)                   # (J, lat, lon)

        # Mass offset, removed from MODEL AND OBSERVATIONS IDENTICALLY and before
        # smoothing. Off by default: for TREFHT the global mean is real skill.
        self.gm_removed = bool(remove_gm)
        if remove_gm:
            cube = remove_global_mean(cube, self.lats, self.lons)
            obs = remove_global_mean(obs, self.lats, self.lons)

        # Eade's spatial smoothing, MODEL AND OBSERVATIONS IDENTICALLY.
        self.smooth_box_deg = _smooth.box_for(var) if smooth is True else (
            tuple(smooth) if smooth else None)
        if self.smooth_box_deg:
            dlat, dlon = self.smooth_box_deg
            cube = _smooth.smooth_box(cube, self.lats, self.lons, dlat, dlon)
            obs = _smooth.smooth_box(obs, self.lats, self.lons, dlat, dlon)

        if detrend_data:
            cube = detrend(cube, self.MONTH)          # asserted finite at build time
            obs = detrend(obs, self.MONTH, nan_aware=True)
        self.F = cube.transpose(1, 0, 2, 3)                  # (N, J, lat, lon)
        self.G = obs
        self.J, self.N = self.F.shape[1], self.F.shape[0]
