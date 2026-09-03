"""Loading, subsetting and detrending behind the handle modules. Not a public interface.

Import `smyle_handles` or `depresys4_handles` instead; everything here is an
implementation detail of them. This file exists so those interfaces stay small enough to
read in one sitting, and so both prediction systems share one engine.

Cubes are built by `build_smyle_cube.py` / `build_depresys4_cube.py`. Which system a
handle reads is set by the `cubes` directory passed in; the set of initialization months
is then derived from the cube itself, so SMYLE (quarterly Feb/May/Aug/Nov) and DePreSys4
(annual November) need no special-casing. No estimators live here.

Seasons
-------
SMYLE initializes quarterly, so a "season" here is an INITIALIZATION month, and the
target season it verifies into is the init month shifted by the lead window:

    S.get("Nov", lead="2-4")   Nov init, lead months 2-4  ->  Dec/Jan/Feb   (DJF)
    S.get("Feb", lead="1-3")   Feb init, lead months 1-3  ->  Feb/Mar/Apr   (FMA)

`c.target` reports that mapping so a plot label never has to be worked out by hand.
`S.get("all", ...)` pools all four, which is what the 247-sample runs in the git
history did.

Spatial smoothing is ON by default
----------------------------------
Every handle applies Eade et al. (2014)'s box mean -- 11.25 deg latitude by 12.5 deg
longitude, 15 x 15 for TREFHT -- to the model AND the observations, identically, on
load. See `_smooth.py`. This is deliberately a load-time step, not a build-time one,
so it reaches every cube already on disk without a rebuild and cannot be applied to
one side only.

It changes every number this module produces, so `repr(cube)` states the box, and
`cube.smoothed` / `cube.smooth_box_deg` record it. Pass `smooth=False` for the old
unsmoothed behaviour, or `smooth=(dlat, dlon)` for a different box. Results computed
before this became the default are NOT comparable to results after it.

Two things the data forces
--------------------------
**Detrend within each init month.** The sample axis cycles Feb/May/Aug/Nov, and model
and obs share the seasonal cycle (their climatologies correlate 0.97), so one line
fitted over all 247 samples leaves that cycle in and fakes skill: rho_o comes out 0.759
that way against 0.450 done properly. `detrend()` therefore fits per init month always,
including in the pooled "all" case.

**Pooled seasons stay heteroscedastic.** Removing the seasonal cycle does not equalise
the four seasons' anomaly variances, so a pooled correlation is still a variance-weighted
blend. Prefer one season at a time (~62 samples each); pool only to compare against the
older 247-sample numbers.

Samples with no observations are dropped from `J` (`obs_ok` in the npz). At both lead 1-3
and lead 2-4 that is the single 2019-11 init, whose window reaches Jan 2020 while
HadSLP2r stops at Dec 2019 -- the J=247-not-248 note in the git history. Longer or later
windows drop more; `inventory()` reports how many per cube. Pass `require_obs=False` to
keep them, e.g. for model-only work on TREFHT or PRECT.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import os
import glob
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _smooth

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
CUBES = os.path.join(_DATA, "smyle_cubes")

# SMYLE's four initialization months. DePreSys4 has only November; `seasons_of()`
# reads whichever a given cube actually holds, so nothing here is system-specific.
SEASONS = {"Feb": 2, "May": 5, "Aug": 8, "Nov": 11}
MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# CESM calls sea-level pressure PSL; this project calls it SLP. Accept either spelling
# so old notebooks keep working, but resolve to SLP everywhere.
_ALIAS = {"PSL": "SLP"}


def canon(var):
    return _ALIAS.get(var, var)


def _cube_path(var, lead, cubes=CUBES):
    return os.path.join(cubes, f"{canon(var)}_lead{lead}.npz")


def seasons_of(var, lead, cubes=CUBES):
    """{name: month} for the initialization months this cube actually contains."""
    z = np.load(_cube_path(var, lead, cubes), mmap_mode="r")
    return {MONTH_ABBR[m]: int(m) for m in sorted(np.unique(np.asarray(z["month"])))}


def inventory(cubes=CUBES):
    """Every (var, lead) cube on disk. Prints, and returns the list."""
    found = []
    for p in sorted(glob.glob(os.path.join(cubes, "*_lead*.npz"))):
        m = re.match(r"(.+)_lead(\d+-\d+)\.npz$", os.path.basename(p))
        if m:
            found.append((m.group(1), m.group(2)))
    if not found:
        print(f"no cubes in {cubes}\n  build one, e.g.: "
              f"python scripts/build_smyle_cube.py --var SLP --leads 1-3")
        return found
    print(f"{len(found)} cube(s) in {cubes}:")
    for var, lead in found:
        z = np.load(_cube_path(var, lead, cubes), mmap_mode="r")
        J, N = z["cube"].shape[0], z["cube"].shape[1]
        u = str(z["units"]) if "units" in z else "?"
        print(f"  {var:7s} lead {lead:5s}  J={J} N={N}  {u:7s} "
              f"obs {int(np.asarray(z['obs_ok']).sum())}/{J} usable")
    return found


def target_season(init_month, lead):
    """('Nov', '2-4') -> 'DJF'; long decadal windows -> 'yr2-5'."""
    a, b = (int(x) for x in lead.split("-")) if "-" in lead else (int(lead), int(lead))
    # Anything longer than a season is a multi-year mean; naming its 48 constituent
    # months is useless, so report the lead YEARS it spans, as Eade et al. do.
    if b - a + 1 > 4:
        y0, y1 = (a + 11) // 12, (b + 11) // 12
        return f"yr{y0}-{y1}" if y0 != y1 else f"yr{y0}"
    months = [((init_month - 1 + k - 1) % 12) + 1 for k in range(a, b + 1)]
    if len(months) == 3:
        return "".join(MONTH_ABBR[m][0] for m in months)
    return "-".join(MONTH_ABBR[m] for m in months)


def _line(X):
    """Remove the least-squares line along axis 0, per grid cell. Mean goes with it."""
    t = np.arange(X.shape[0], dtype=float)
    t -= t.mean()
    flat = X.reshape(X.shape[0], -1)
    slope = (t[:, None] * flat).sum(axis=0) / (t ** 2).sum()
    return (flat - t[:, None] * slope - flat.mean(axis=0)).reshape(X.shape)


def _line_nan(X):
    """`_line` fitted per cell over the finite samples only; non-finite stay NaN.

    Needed for the observations when `require_obs=False` keeps a sample whose
    verification window runs past the obs record. Without this, one NaN row makes
    `flat.mean(axis=0)` NaN and the whole season's obs come out NaN.
    """
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

    For SLP this is a physical constraint, not a preference: the area mean of a
    sea-level-pressure field is fixed by the mass of the atmosphere, so the global
    mean of an SLP *anomaly* is ~0 (HadSLP2r: sd 6.3 Pa on 48-month means). A model
    whose global mean wanders instead carries a spatially uniform offset that is
    identical in every member, so it survives ensemble averaging intact and lands
    in the ensemble mean of every grid cell while correlating with nothing.

    EC-Earth3's dcppA-hindcast `psl` does exactly that: sd 28.8 Pa across start
    dates, range 101054-101225 Pa, against 1.1 Pa for DePreSys4. The offset is the
    same size at every lead, but the field it contaminates is not -- a 48-month mean
    has far less real variance than a DJF mean, so the offset is 5% of the lead 2-4
    ensemble-mean variance and 50% of the lead 13-60 ensemble-mean variance. That is
    why removing it barely moves lead 2-4 and rescues lead 13-60.

    Applied to model and observations alike, like the smoothing.
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
    """One (variable, lead window, season) slice, detrended and ready to estimate on.

        F       (N, J, lat, lon)   members
        G       (J, lat, lon)      observations, NaN if none are wired up
        s       (J, lat, lon)      ensemble mean
        YEAR    (J,)               initialization year
        MONTH   (J,)               initialization month
        target  str                the calendar months verified, e.g. "DJF"
    """

    def __init__(self, var, lead, season, require_obs=None, detrend_data=True,
                 cubes=CUBES, smooth=True, years=None, remove_gm=False):
        var = canon(var)
        z = np.load(_cube_path(var, lead, cubes))
        year, month, ok = z["year"], z["month"], z["obs_ok"]

        # None means "require observations if this variable has any". TREFHT and PRECT
        # have no observational counterpart wired up, so demanding obs there would drop
        # every sample; SLP behaves exactly as before.
        if require_obs is None:
            require_obs = bool(ok.any())

        # the initialization months this cube holds, not a hard-coded list
        seasons = {MONTH_ABBR[m]: int(m) for m in sorted(np.unique(month))}
        keep = np.ones(len(year), bool)
        if season != "all":
            if season not in seasons:
                raise KeyError(f"season {season!r} not in this cube; use one of "
                               f"{list(seasons)} or 'all'")
            keep &= month == seasons[season]
        if require_obs:
            keep &= ok
        # Restricting to a given set of init years matters for pooling models: each
        # member set must be detrended on its own (Eade corrects each model
        # separately) but over the SAME start dates, or the anomalies are not
        # commensurable. Filtering here, before detrend(), is what makes that work.
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
        self.seasons = seasons
        self.n_dropped = int((~keep).sum() if season == "all"
                             else ((month == seasons[season]) & ~keep).sum())

        cube = z["cube"][keep].astype(float)                 # (J, N, lat, lon)
        obs = z["obs"][keep].astype(float)                   # (J, lat, lon)

        # Mass offset, removed from MODEL AND OBSERVATIONS IDENTICALLY and before
        # smoothing, so what is taken out is the offset in the real field rather than
        # in a smoothed proxy of it. See `remove_global_mean` for why SLP needs this
        # and the other variables do not. Off by default: it changes every number,
        # and for TREFHT the global mean is real skill, not an artefact.
        self.gm_removed = bool(remove_gm)
        if remove_gm:
            cube = remove_global_mean(cube, self.lats, self.lons)
            obs = remove_global_mean(obs, self.lats, self.lons)

        # Eade's spatial smoothing, applied to MODEL AND OBSERVATIONS IDENTICALLY.
        # Done here rather than in the cube builders so it reaches every cube already
        # on disk without a rebuild, and so the box is never applied to one side
        # only. Smoothing and detrending are both linear and act on different axes,
        # so they commute -- the order below is Eade's wording, not a constraint.
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
        self.detrended = detrend_data
        self.smoothed = self.smooth_box_deg is not None
        self.has_obs = bool(np.isfinite(self.G).any())

        # cos(lat) weights; N48 spans both poles so plain means over-weight them
        self.W = (np.cos(np.deg2rad(self.lats))[:, None]
                  * np.ones(len(self.lons)))
        # in init-month order, not alphabetical, so "all" reads Feb->May->Aug->Nov
        self.target = ("/".join(target_season(m, lead)
                                for m in sorted(np.unique(self.MONTH)))
                       if season == "all" else target_season(seasons[season], lead))

    @property
    def s(self):
        """Ensemble mean, (J, lat, lon)."""
        return self.F.mean(axis=0)

    def loo_mean(self, n):
        """s_-n, the ensemble mean over the other N-1 members. (J, lat, lon)."""
        return (self.F.sum(axis=0) - self.F[n]) / (self.N - 1)

    def gm(self, a):
        """Area-weighted global mean, NaNs skipped."""
        m = np.isfinite(a)
        return float((a[m] * self.W[m]).sum() / self.W[m].sum())

    @property
    def label(self):
        return f"{self.var} {self.season} init, lead {self.lead} -> {self.target}"

    def __repr__(self):
        box = (f", smooth {self.smooth_box_deg[0]:g}x{self.smooth_box_deg[1]:g}deg"
               if self.smoothed else ", UNSMOOTHED")
        return (f"<Cube {self.label} [{self.units}]: N={self.N} J={self.J}"
                f"{'' if self.has_obs else ', no obs'}"
                f"{'' if self.detrended else ', RAW'}{box}"
                f"{', gm removed' if self.gm_removed else ''}>")


def summary(var, lead, cubes=CUBES):
    """One printed block per lead window, for the handle modules' __main__."""
    print(f"\n{var} lead {lead}")
    names = list(seasons_of(var, lead, cubes))
    rows = [Cube(var, lead, n, require_obs=False, cubes=cubes) for n in names]
    if len(names) > 1:
        rows.append(Cube(var, lead, "all", require_obs=False, cubes=cubes))
    for c in rows:
        print(f"  {c.season:3s} -> {c.target:16s} N={c.N} J={c.J} "
              f"dropped={c.n_dropped}  anomaly sd {c.gm(c.s.std(axis=0)):9.4g} "
              f"{c.units}")
