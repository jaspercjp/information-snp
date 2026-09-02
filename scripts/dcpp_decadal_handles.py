"""One DCPP-A decade at monthly resolution: the sample axis is TIME WITHIN a hindcast.

    import sys; sys.path.insert(0, "../scripts")
    import dcpp_decadal_handles as D

    D.inventory()                          # start dates, models, run lengths, obs cover
    c = D.get(1961, var="SLP")             # Jan 1962 .. Dec 1971, T=120, all long models
    c = D.get(1990, var="TREFHT", models=["HadGEM3-GC31-MM", "MIROC6"])

    rho_o = M.pearson_coeff(c.s, c.G)      # correlation ALONG TIME, not across starts

This is the analogue of the original thesis dataset. That study used CMIP5
`decadal1961` -- five initialised hindcasts, monthly means over the whole ~120-month
run, one start date -- and correlated along the month axis. `dcpp_handles` cannot
express that: it collapses each run to one lead-window mean and its sample axis is the
48-60 start dates. This module keeps the months and fixes the start date instead.

    dcpp_handles         F (N, J, lat, lon)   J = start dates,  one lead-window mean
    dcpp_decadal_handles F (N, T, lat, lon)   T = months within ONE hindcast

Both are initialised hindcasts of the same kind; DCPP-A `sYYYY` is the CMIP6 successor
of CMIP5 `decadalYYYY`. Nothing here is uninitialised.

Which models, and why one is missing
------------------------------------
DCPP-A only requires 5-year hindcasts, and MRI-ESM2-0 took that option: its runs are 62
months, so it cannot support a decade and is excluded. Run lengths, read off the
manifests rather than assumed:

    HadGEM3-GC31-MM  125     MIROC6         122     CanESM5   120
    NorCPM1          123     MPI-ESM1-2-LR  122     EC-Earth3 122 (132 at two starts)
    MRI-ESM2-0        62  <- excluded by MIN_MONTHS

The shared calendar window
--------------------------
Initialisation months differ -- November for DePreSys4, EC-Earth3, MIROC6 and
MPI-ESM1-2-LR, October for NorCPM1, January of YYYY+1 for CanESM5. Because the sample
axis is time and there is only one observed series, the models must be aligned on
CALENDAR months, not on lead. The intersection over all six, for every `sYYYY`, is
exactly

    Jan(YYYY+1) .. Dec(YYYY+10),  T = 120

set at the front by CanESM5's January start and at the back by its 120-month length.
`window="canonical"` (the default) always uses that window, so results from different
model subsets sit on the same time axis and are comparable. `window="max"` instead
takes the largest window the SELECTED models share -- DePreSys4 alone gives
Nov(YYYY) .. Mar(YYYY+10), T=125 -- which is better use of the data but not comparable
across subsets.

`s1961` therefore covers Jan 1962 - Dec 1971, which is precisely the window CMIP5
`decadal1961` covered for CanCM4, MIROC5 and MPI-ESM-LR (`196201-197112`). The thesis
comparison is exact.

Anomalies: month-of-year climatology, not a trend fit
----------------------------------------------------
`dcpp_handles` detrends across start dates, which removes the seasonal cycle because
each sample is the same season. Here successive samples are successive MONTHS, so the
annual cycle is the dominant signal in both model and observations and a correlation on
raw fields is mostly a correlation of two annual cycles. Measured on s1961 SLP, pooled
over all six models:

    anom="none"   area-weighted mean rho_o along time = +0.7101
    anom="moy"                                          +0.0569

Almost all of that 0.71 is the seasonal cycle, not skill. Before comparing anything to
a number from the original thesis, establish which of those two it corresponds to.

`anom="moy"` (the default) removes the month-of-year climatology, per cell, from the
model and the observations, computed on the same 120 months on both sides so the two are
treated identically. The model climatology is computed per model over its members and
the 10 occurrences of each calendar month (10 x N samples); the observed one has only
the 10.

**The limitation this design cannot escape.** An initialised hindcast drifts, and the
correct correction is a lead-dependent climatology averaged over start dates. One start
date gives nothing to average over, so drift stays in the series as a slow trend. That
was true of the thesis too. `detrend_time=True` removes a per-cell linear trend in
time afterwards, which absorbs most of it; it is off by default because it also removes
real low-frequency signal. If you want the proper correction, use `dcpp_handles`.

Two further notes
-----------------
`remove_gm` exists for parity with `dcpp_handles`, but EC-Earth3's spurious mass offset
is CONSTANT WITHIN a start date, so `anom="moy"` already absorbs it entirely. Verify
with `D.mass_offset_check()` rather than taking that on trust.

Observations come from the same products and the same native grids as everywhere else
in this project (`OBS_SPEC`): HadSLP2r 5 deg for SLP, NCEP R1 T62 for TREFHT, GPCP
2.5 deg for PRECT. The model is interpolated onto them; they are never resampled. That
caps the usable start dates: SLP s1960-s2009, TREFHT s1960-s2015, PRECT s1978-s2015.
`inventory()` prints it.
"""
import json
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _smooth                                                    # noqa: E402
from _cube_io import canon, remove_global_mean                    # noqa: E402
from build_smyle_cube import OBS_SPEC, UNITS, obs_grid, to_grid    # noqa: E402

warnings.simplefilter("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
_DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(_DATA, "dcpp_monthly_cache")

CMIP_NAME = {"SLP": "psl", "TREFHT": "tas", "PRECT": "pr"}
SCALE = {"PRECT": 86400.0}              # kg m-2 s-1 -> mm/day, as in build_dcpp_cube
MIN_MONTHS = 120                        # a "decade"; excludes MRI-ESM2-0's 62-month runs
ALIAS = {"DePreSys4": "HadGEM3-GC31-MM"}

_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)
_SPAN = re.compile(r"_(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$")

# Where the raw monthly files live. DePreSys4 predates the multi-model tree.
_DCPP_RAW = os.path.join(os.environ.get("SCRATCH", _DATA), "dcpp")
_DEPRESYS_RAW = os.path.join(_DATA, "depresys4")
_MANIFEST = os.path.join(_DATA, "dcpp_files.json")
_DEPRESYS_MANIFEST = os.path.join(_DATA, "depresys4_files.json")


# ------------------------------------------------------------------ month helpers
def _mi(year, month):
    return year * 12 + month - 1


def _lab(m):
    return f"{m // 12:04d}-{m % 12 + 1:02d}"


def canonical_window(start, months=MIN_MONTHS):
    """Jan(start+1) .. Dec(start+10) as absolute month indices. See the docstring."""
    base = _mi(start + 1, 1)
    return [base + k for k in range(months)]


# ------------------------------------------------------------------- file indexes
_INDEX = None


def _build_index():
    """{model: {var: {(start_year, member): [(lo, hi, path), ...]}}} from the manifests.

    Spans come from the filename, which both manifests carry, so DePreSys4's schema
    (no `first_month` field) and the DCPP one are handled by the same code.
    """
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(_MANIFEST) as f:
        man = json.load(f)
    for model, byvar in man.items():
        for cm, rows in byvar.items():
            for r in rows:
                m = _SPAN.search(r["title"])
                if not m or not r["start"].startswith("s"):
                    continue
                lo = _mi(int(m.group(1)), int(m.group(2)))
                hi = _mi(int(m.group(3)), int(m.group(4)))
                p = os.path.join(_DCPP_RAW, model, r["start"], cm, r["title"])
                idx[model][cm][(int(r["start"][1:]), r["member"])].append((lo, hi, p))

    if os.path.exists(_DEPRESYS_MANIFEST):
        with open(_DEPRESYS_MANIFEST) as f:
            dp = json.load(f)
        for r in dp:
            m = _SPAN.search(r["filename"])
            if not m:
                continue
            lo = _mi(int(m.group(1)), int(m.group(2)))
            hi = _mi(int(m.group(3)), int(m.group(4)))
            p = os.path.join(_DEPRESYS_RAW, r["start"], r["var"], r["filename"])
            idx["HadGEM3-GC31-MM"][r["var"]][(int(r["start"][1:]), r["member"])].append(
                (lo, hi, p))
    return idx


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    return _INDEX


def _member_span(chunks):
    """(first, last) absolute month a member's chunks cover, contiguously."""
    return min(c[0] for c in chunks), max(c[1] for c in chunks)


def run_lengths(var="SLP"):
    """{model: {start_year: n_months}} -- the span common to all of a start's members."""
    cm = CMIP_NAME[canon(var)]
    out = {}
    for model, byvar in index().items():
        if cm not in byvar:
            continue
        byst = defaultdict(list)
        for (st, mem), chunks in byvar[cm].items():
            byst[st].append(_member_span(chunks))
        out[model] = {st: min(b for _, b in v) - max(a for a, _ in v) + 1
                      for st, v in byst.items()}
    return out


def long_models(var="SLP", min_months=MIN_MONTHS):
    """Models whose EVERY start date runs at least `min_months`. Read, not hard-coded."""
    rl = run_lengths(var)
    return sorted(m for m, d in rl.items() if d and min(d.values()) >= min_months)


def _covers(model, cm, start, want):
    """Members of (model, start) whose chunks span every month in `want`, and are on disk."""
    lo, hi = min(want), max(want)
    good = []
    for (st, mem), chunks in index()[model][cm].items():
        if st != start:
            continue
        a, b = _member_span(chunks)
        if a > lo or b < hi:
            continue
        if not all(os.path.exists(p) for _, _, p in chunks
                   if not (p is None) and not (_ > hi)):
            pass                                     # existence checked at read time
        good.append(mem)
    return sorted(good)


# ------------------------------------------------------------------------ reading
def _read_member(chunks, cm, want, lat, lon, scale):
    """(T, lat, lon) on the target grid, in the order of `want`. Opens only what it needs.

    Regridding is done with the time axis intact -- one `interp` call per member rather
    than per month, which is ~100x fewer calls and gives bit-identical results (checked).
    """
    lo, hi = min(want), max(want)
    got = {}
    native = None
    for a, b, p in sorted(chunks):
        if b < lo or a > hi:
            continue
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with xr.open_dataset(p, decode_times=_CODER) as d:
            stamps = d.time_bnds.values[:, 0] if "time_bnds" in d else d.time.values
            months = np.array([t.year * 12 + t.month - 1 for t in stamps])
            sel = np.where(np.isin(months, want))[0]
            if sel.size == 0:
                continue
            arr = to_grid(d[cm].isel(time=sel), lat, lon).values
            if native is None:
                native = (d["lat"].values, d["lon"].values)
            for k, m in enumerate(months[sel]):
                got[int(m)] = arr[k]
    missing = [m for m in want if m not in got]
    if missing:
        raise ValueError(f"{len(missing)} of {len(want)} months missing "
                         f"({[_lab(m) for m in missing[:3]]}...)")
    return np.stack([got[m] for m in want]).astype(np.float32) * scale


def _cache_path(model, var, start, want):
    return os.path.join(CACHE, model,
                        f"{var}_s{start}_{_lab(min(want))}_{len(want)}mo.npz")


def _model_cube(model, var, start, want, lat, lon, use_cache=True, verbose=True):
    """(N, T, lat, lon) raw -- no anomaly, no smoothing. Cached, since it is the slow part."""
    cm = CMIP_NAME[var]
    cp = _cache_path(model, var, start, want)
    if use_cache and os.path.exists(cp):
        z = np.load(cp, allow_pickle=False)
        if np.array_equal(z["months"], np.asarray(want)):
            return z["cube"], [str(s) for s in z["members"]]
    mems = _covers(model, cm, start, want)
    if not mems:
        raise ValueError(f"{model} has no member covering {_lab(min(want))}.."
                         f"{_lab(max(want))} from s{start}")
    if verbose:
        print(f"    reading {model} s{start} {var}: {len(mems)} members", flush=True)
    rows = []
    for mem in mems:
        rows.append(_read_member(index()[model][cm][(start, mem)], cm, want,
                                 lat, lon, SCALE.get(var, 1.0)))
    cube = np.stack(rows)
    if use_cache:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        np.savez_compressed(cp, cube=cube, members=np.array(mems),
                            months=np.asarray(want), lat=lat, lon=lon)
    return cube, mems


def _obs(var, want, lat, lon):
    """(T, lat, lon) observations on exactly `want`, or None if the product cannot cover it."""
    spec = OBS_SPEC.get(var)
    if spec is None:
        return None
    path = os.path.join(_DATA, "obs", spec[0])
    if not os.path.exists(path):
        return None
    d = xr.open_dataset(path, decode_times=_CODER,
                        drop_variables=["time_bnds", "climatology_bounds"])[spec[1]]
    idx = np.array([t.year * 12 + t.month - 1 for t in d["time"].values])
    lut = {int(m): i for i, m in enumerate(idx)}
    if not all(m in lut for m in want):
        return None
    sub = d.isel(time=[lut[m] for m in want])
    native = (len(lat) == d.sizes["lat"] and np.allclose(lat, d["lat"].values)
              and len(lon) == d.sizes["lon"] and np.allclose(lon, d["lon"].values))
    return (sub.values if native else to_grid(sub, lat, lon).values).astype(float)


def model_starts(var="SLP"):
    """Every start year any long model publishes, so spans are reported over real data."""
    rl = run_lengths(var)
    out = set()
    for m in long_models(var):
        out |= set(rl.get(m, {}))
    return out


def obs_span(var, restrict_to_models=True):
    """(first, last) start year whose canonical window the observations fully cover.

    Intersected with the start dates a model actually exists for by default -- HadSLP2r
    begins in 1850 and would otherwise report a floor of s1851, which is true of the
    observations and useless as guidance.
    """
    spec = OBS_SPEC.get(canon(var))
    if spec is None:
        return None
    path = os.path.join(_DATA, "obs", spec[0])
    if not os.path.exists(path):
        return None
    with xr.open_dataset(path, decode_times=_CODER,
                         drop_variables=["time_bnds", "climatology_bounds"]) as d:
        t = d["time"].values
    lo, hi = _mi(t[0].year, t[0].month), _mi(t[-1].year, t[-1].month)
    ok = [y for y in range(1840, 2060)
          if min(canonical_window(y)) >= lo and max(canonical_window(y)) <= hi]
    if restrict_to_models:
        have = model_starts(canon(var))
        if have:
            ok = [y for y in ok if min(have) <= y <= max(have)]
    return (min(ok), max(ok)) if ok else None


# ---------------------------------------------------------------------- anomalies
def _moy_of(want):
    return np.array([m % 12 + 1 for m in want])


def _remove_moy(X, moy):
    """Remove the month-of-year mean along the time axis (axis -3 for F, -3 for G).

    X is (..., T, lat, lon); the climatology is pooled over every leading axis, so a
    model's climatology uses its members and the 10 occurrences of each calendar month.
    """
    out = np.array(X, dtype=float, copy=True)
    tax = out.ndim - 3
    for m in range(1, 13):
        k = moy == m
        if not k.any():
            continue
        sl = [slice(None)] * out.ndim
        sl[tax] = k
        sl = tuple(sl)
        blk = out[sl]
        clim = np.nanmean(blk, axis=tuple(range(out.ndim - 2)), keepdims=True)
        out[sl] = blk - clim
    return out


def _detrend_time(X):
    """Remove a per-cell least-squares line along the time axis (axis -3)."""
    out = np.array(X, dtype=float, copy=True)
    tax = out.ndim - 3
    T = out.shape[tax]
    t = np.arange(T, dtype=float)
    t = t - t.mean()
    shape = [1] * out.ndim
    shape[tax] = T
    tt = t.reshape(shape)
    denom = (t ** 2).sum()
    slope = np.nansum(tt * out, axis=tax, keepdims=True) / denom
    return out - tt * slope


# ------------------------------------------------------------------- the interface
class Decade:
    """One start date, monthly, pooled over models. Built by `get`.

        F  (N, T, lat, lon)   members          G  (T, lat, lon)  observations
        s  (T, lat, lon)      ensemble mean    W  (lat, lon)     cos(lat) weights
        months (T,) absolute   dates (T,) 'YYYY-MM'   moy (T,) 1..12
    """

    def __init__(self, start, var="SLP", models=None, months=MIN_MONTHS,
                 window="canonical", anom="moy", detrend_time=False, smooth=True,
                 remove_gm=False, max_members=None, require_obs=True, cache=True,
                 verbose=True):
        var = canon(var)
        if var not in CMIP_NAME:
            raise ValueError(f"var must be one of {list(CMIP_NAME)}, got {var!r}")
        start = int(start)
        avail = long_models(var, months)
        models = [ALIAS.get(m, m) for m in (models or avail)]
        bad = [m for m in models if m not in avail]
        if bad:
            rl = run_lengths(var)
            why = {m: (min(rl[m].values()) if m in rl and rl[m] else None) for m in bad}
            raise ValueError(f"{bad} cannot supply {months} months "
                             f"(shortest run: {why}); see D.run_lengths()")
        models = [m for m in models if start in run_lengths(var).get(m, {})]
        if not models:
            raise ValueError(f"no requested model has start s{start}")

        LAT, LON = obs_grid(var)
        if window == "canonical":
            want = canonical_window(start, months)
        elif window == "max":
            spans = []
            for m in models:
                sp = [_member_span(c) for (st, _), c in index()[m][CMIP_NAME[var]].items()
                      if st == start]
                spans.append((max(a for a, _ in sp), min(b for _, b in sp)))
            a, b = max(x for x, _ in spans), min(y for _, y in spans)
            want = list(range(a, b + 1))
        else:
            raise ValueError("window must be 'canonical' or 'max'")

        g = _obs(var, want, LAT, LON)
        if g is None:
            sp = obs_span(var)
            msg = (f"{var} observations do not cover {_lab(min(want))}.."
                   f"{_lab(max(want))}" + (f"; usable starts s{sp[0]}..s{sp[1]}"
                                           if sp else ""))
            if require_obs:
                raise ValueError(msg)
            if verbose:
                print(f"  {msg}; obs left as NaN")
            g = np.full((len(want), len(LAT), len(LON)), np.nan)

        parts, names, owner = [], [], []
        for m in models:
            cube, mems = _model_cube(m, var, start, want, LAT, LON,
                                     use_cache=cache, verbose=verbose)
            if max_members is not None and cube.shape[0] > max_members:
                cube, mems = cube[:max_members], mems[:max_members]
            parts.append(cube.astype(float))
            names += [f"{m}:{s}" for s in mems]
            owner += [m] * cube.shape[0]

        self.months = np.asarray(want)
        self.moy = _moy_of(want)
        self.dates = np.array([_lab(m) for m in want])
        self.var, self.start, self.window = var, start, window
        self.units = UNITS.get(var, "")
        self.lats, self.lons = LAT, LON
        self.models = models

        F = np.concatenate(parts, axis=0)

        self.gm_removed = bool(remove_gm)
        if remove_gm:
            F = remove_global_mean(F, LAT, LON)
            g = remove_global_mean(g, LAT, LON)

        self.smooth_box_deg = (_smooth.box_for(var) if smooth is True
                               else (tuple(smooth) if smooth else None))
        if self.smooth_box_deg:
            dlat, dlon = self.smooth_box_deg
            F = _smooth.smooth_box(F, LAT, LON, dlat, dlon)
            g = _smooth.smooth_box(g, LAT, LON, dlat, dlon)

        # Anomalies per MODEL, so each model's own climatology and drift go with it --
        # the same reason dcpp_handles detrends per model before pooling.
        self.anom = anom
        if anom == "moy":
            i = 0
            for m, p in zip(models, parts):
                n = p.shape[0]
                F[i:i + n] = _remove_moy(F[i:i + n], self.moy)
                i += n
            g = _remove_moy(g, self.moy)
        elif anom not in (None, "none"):
            raise ValueError("anom must be 'moy' or 'none'")

        self.detrended_time = bool(detrend_time)
        if detrend_time:
            i = 0
            for m, p in zip(models, parts):
                n = p.shape[0]
                F[i:i + n] = _detrend_time(F[i:i + n])
                i += n
            g = _detrend_time(g)

        self.F, self.G = F, g
        self.members = np.array(names)
        self.model_of = np.array(owner)
        self.by_model = {}
        i = 0
        for m, p in zip(models, parts):
            self.by_model[m] = slice(i, i + p.shape[0])
            i += p.shape[0]
        self.N, self.T = self.F.shape[0], self.F.shape[1]
        self.has_obs = bool(np.isfinite(self.G).any())
        self.W = np.cos(np.deg2rad(LAT))[:, None] * np.ones(len(LON))
        self.init_months = {m: self.lead1(m)[1] for m in models}
        if verbose:
            print(repr(self))
            for m in models:
                sl = self.by_model[m]
                l1 = self.lead1(m)
                print(f"    {m:18s} N={sl.stop - sl.start:3d}  lead 1 = {_lab(l1[0])}"
                      f"  target months are leads "
                      f"{self.lead_of(m)[0]}..{self.lead_of(m)[-1]}")

    # ------------------------------------------------------------------- lead info
    def lead1(self, model):
        """(absolute month of lead 1, calendar month) for one model at this start."""
        cm = CMIP_NAME[self.var]
        sp = [_member_span(c) for (st, _), c in index()[model][cm].items()
              if st == self.start]
        a = max(x for x, _ in sp)
        return a, a % 12 + 1

    def lead_of(self, model):
        """(T,) lead month of every target month for one model. Lead 1 = first output."""
        return self.months - self.lead1(model)[0] + 1

    # ------------------------------------------------------------- Cube-like extras
    @property
    def s(self):
        return self.F.mean(axis=0)

    def loo_mean(self, n):
        return (self.F.sum(axis=0) - self.F[n]) / (self.N - 1)

    def gm(self, a):
        m = np.isfinite(a)
        return float((a[m] * self.W[m]).sum() / self.W[m].sum())

    def model_mean(self, model):
        return self.F[self.by_model[model]].mean(axis=0)

    def season(self, name="DJF"):
        """Boolean (T,) mask for a named season, for subsetting the time axis."""
        table = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8),
                 "SON": (9, 10, 11)}
        if name not in table:
            raise KeyError(f"season must be one of {list(table)}")
        return np.isin(self.moy, table[name])

    @property
    def label(self):
        return (f"DCPP-A s{self.start} ({len(self.models)} models) {self.var}, "
                f"{self.dates[0]}..{self.dates[-1]} monthly")

    def __repr__(self):
        box = (f", smooth {self.smooth_box_deg[0]:g}x{self.smooth_box_deg[1]:g}deg"
               if self.smooth_box_deg else ", UNSMOOTHED")
        return (f"<Decade {self.label} [{self.units}]: N={self.N} T={self.T}"
                f"{'' if self.has_obs else ', no obs'}"
                f", anom={self.anom}{box}"
                f"{', detrended in time' if self.detrended_time else ''}"
                f"{', gm removed' if self.gm_removed else ''}>")


def get(start=1961, var="SLP", models=None, **kw):
    """One decade of monthly DCPP-A output, pooled over models. See `Decade`."""
    return Decade(start, var=var, models=models, **kw)


# --------------------------------------------------------------------- inventory
def inventory(var=None):
    """What decades can be built, for which models, and where the observations stop."""
    vars_ = [canon(var)] if var else list(CMIP_NAME)
    out = {}
    for v in vars_:
        rl = run_lengths(v)
        keep = long_models(v)
        sp = obs_span(v)
        print(f"\n{v}: obs = {OBS_SPEC[v][0]}"
              + (f", canonical window covered for s{sp[0]}..s{sp[1]}" if sp else
                 ", NO usable window"))
        print(f"  {'model':18s} {'run (months)':>13s} {'starts':>7s}  usable for a decade")
        for m in sorted(rl):
            if not rl[m]:
                continue
            lo, hi = min(rl[m].values()), max(rl[m].values())
            yrs = sorted(rl[m])
            ok = m in keep
            print(f"  {m:18s} {(str(lo) if lo == hi else f'{lo}-{hi}'):>13s} "
                  f"{len(yrs):7d}  {'yes' if ok else 'NO -- shorter than ' + str(MIN_MONTHS)}"
                  f"   [s{yrs[0]}..s{yrs[-1]}]")
        common = None
        for m in keep:
            s = set(rl[m])
            common = s if common is None else (common & s)
        if common and sp:
            usable = sorted(y for y in common if sp[0] <= y <= sp[1])
            print(f"  -> {len(usable)} decade(s) with all {len(keep)} long models "
                  f"AND observations: s{usable[0]}..s{usable[-1]}")
            out[v] = usable
    return out


def mass_offset_check(start=1990, var="SLP", **kw):
    """Is EC-Earth3's uniform mass offset really constant within one start date?

    If it is, `anom="moy"` removes it and `remove_gm` is unnecessary here. Prints the
    sd over TIME of each model's area-weighted global mean, before and after anomalies.
    """
    raw = get(start, var=var, anom="none", smooth=False, verbose=False, **kw)
    ano = get(start, var=var, anom="moy", smooth=False, verbose=False, **kw)
    w = raw.W / raw.W.sum()
    print(f"{var} s{start}: sd over the {raw.T} months of the area-weighted global mean.")
    print("  Per SINGLE MEMBER (mean over members of each member's own sd) as well as")
    print("  per ensemble mean -- the observations are one realisation, so only the")
    print("  single-member column is comparable to them.")
    print(f"  {'model':18s} {'1 member raw':>13s} {'1 member moy':>13s}"
          f" {'ens mean raw':>13s} {'ens mean moy':>13s}")
    for m in raw.models:
        sl = raw.by_model[m]
        a1 = ((raw.F[sl] * w).sum(axis=(2, 3))).std(axis=1).mean()
        b1 = ((ano.F[sl] * w).sum(axis=(2, 3))).std(axis=1).mean()
        a = (raw.model_mean(m) * w).sum(axis=(1, 2)).std()
        b = (ano.model_mean(m) * w).sum(axis=(1, 2)).std()
        print(f"  {m:18s} {a1:13.4f} {b1:13.4f} {a:13.4f} {b:13.4f}")
    go = (np.nan_to_num(raw.G) * w).sum(axis=(1, 2)).std()
    gn = (np.nan_to_num(ano.G) * w).sum(axis=(1, 2)).std()
    print(f"  {'observations':18s} {go:13.4f} {gn:13.4f} {'--':>13s} {'--':>13s}"
          f"   {raw.units}")
    print("\n  EC-Earth3 is an outlier ACROSS start dates (sd 28.8 Pa, see")
    print("  FINDING-SLP-rho_o-EC-Earth3-mass-offset.md) but should NOT be an outlier")
    print("  here: within one start date the offset is a constant, which anom='moy'")
    print("  removes. If it is an outlier above, that assumption is wrong.")


if __name__ == "__main__":
    inventory()
    print()
    get(1961, var="SLP")
