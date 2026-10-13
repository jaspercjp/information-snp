"""One DCPP-A decade at monthly resolution: the sample axis is TIME WITHIN a hindcast.

    import dcpp_decadal_handles as D
    c = D.get(1961, var="SLP", anom="none")   # Jan 1962 .. Dec 1971, T=120, six models
    c.F (N, T, lat, lon)   c.G (T, lat, lon)   c.lats, c.lons, c.models, c.dates

The analogue of the original thesis dataset (CMIP5 `decadal1961`: monthly means over the
~120-month run of one start date, correlated along the month axis). DCPP-A `sYYYY` is
its CMIP6 successor.

Models (`MODELS`, pinned): every model whose every start runs at least 120 months --
HadGEM3-GC31-MM 125, MIROC6 122, CanESM5 120, NorCPM1 123, MPI-ESM1-2-LR 122,
EC-Earth3 122. MRI-ESM2-0's 62-month runs cannot support a decade.

The shared calendar window. Initialisation months differ (Nov for DePreSys4, EC-Earth3,
MIROC6, MPI-ESM1-2-LR; Oct for NorCPM1; Jan of YYYY+1 for CanESM5), and there is one
observed series, so the models are aligned on CALENDAR months: Jan(YYYY+1) ..
Dec(YYYY+10), T = 120. s1961 is therefore Jan 1962 - Dec 1971, exactly CMIP5
decadal1961's `196201-197112`.

Processing, in order: interpolate each member onto the observations' grid (`obs_grid`;
the slow part, cached in data/dcpp_monthly_cache/ -- see build_dcpp_monthly_cache.py),
Eade's box smoothing on model and obs alike (`smooth=True`), then optionally
`anom="moy"`: remove the month-of-year climatology per cell, per MODEL over its members
and the 10 occurrences of each calendar month, and from the observations over the same
120 months. Observations are HadSLP2r (SLP), NCEP R1 (TREFHT), GPCP (PRECT, from 1979, so
PRECT starts at s1978).
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
from _cube_io import canon                                        # noqa: E402
from obs_grid import OBS_SPEC, UNITS, obs_grid, to_grid           # noqa: E402
from paths import (DCPP_MANIFEST, DCPP_RAW, DEPRESYS_MANIFEST,     # noqa: E402
                   DEPRESYS_RAW, MONTHLY_CACHE, OBS)

warnings.simplefilter("ignore")

CMIP_NAME = {"SLP": "psl", "TREFHT": "tas", "PRECT": "pr"}
SCALE = {"PRECT": 86400.0}              # kg m-2 s-1 -> mm/day, as in build_dcpp_cube
MIN_MONTHS = 120                        # a "decade"
# Pooling order (sorted). Pinned: the old code derived the same list from the manifests'
# run lengths (every start >= MIN_MONTHS long).
MODELS = ["CanESM5", "EC-Earth3", "HadGEM3-GC31-MM", "MIROC6", "MPI-ESM1-2-LR", "NorCPM1"]

_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)
_SPAN = re.compile(r"_(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$")


# ------------------------------------------------------------------ month helpers
def _mi(year, month):
    return year * 12 + month - 1


def _lab(m):
    return f"{m // 12:04d}-{m % 12 + 1:02d}"


def canonical_window(start, months=MIN_MONTHS):
    """Jan(start+1) .. Dec(start+10) as absolute month indices."""
    base = _mi(start + 1, 1)
    return [base + k for k in range(months)]


# ------------------------------------------------------------------- file indexes
_INDEX = None


def _build_index():
    """{model: {var: {(start_year, member): [(lo, hi, path), ...]}}} from the manifests.

    Spans come from the filename, which both manifests carry, so DePreSys4's schema and
    the DCPP one are handled by the same code.
    """
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(DCPP_MANIFEST) as f:
        man = json.load(f)
    for model, byvar in man.items():
        for cm, rows in byvar.items():
            for r in rows:
                m = _SPAN.search(r["title"])
                if not m or not r["start"].startswith("s"):
                    continue
                lo = _mi(int(m.group(1)), int(m.group(2)))
                hi = _mi(int(m.group(3)), int(m.group(4)))
                p = os.path.join(DCPP_RAW, model, r["start"], cm, r["title"])
                idx[model][cm][(int(r["start"][1:]), r["member"])].append((lo, hi, p))

    with open(DEPRESYS_MANIFEST) as f:
        dp = json.load(f)
    for r in dp:
        m = _SPAN.search(r["filename"])
        if not m:
            continue
        lo = _mi(int(m.group(1)), int(m.group(2)))
        hi = _mi(int(m.group(3)), int(m.group(4)))
        p = os.path.join(DEPRESYS_RAW, r["start"], r["var"], r["filename"])
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


def _covers(model, cm, start, want):
    """Members of (model, start) whose chunks span every month in `want`."""
    lo, hi = min(want), max(want)
    good = []
    for (st, mem), chunks in index()[model][cm].items():
        if st != start:
            continue
        a, b = _member_span(chunks)
        if a > lo or b < hi:
            continue
        good.append(mem)
    return sorted(good)


# ------------------------------------------------------------------------ reading
def _read_member(chunks, cm, want, lat, lon, scale):
    """(T, lat, lon) on the target grid, in the order of `want`. Opens only what it needs.

    One `interp` call per member with the time axis intact (bit-identical to per month).
    """
    lo, hi = min(want), max(want)
    got = {}
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
            for k, m in enumerate(months[sel]):
                got[int(m)] = arr[k]
    missing = [m for m in want if m not in got]
    if missing:
        raise ValueError(f"{len(missing)} of {len(want)} months missing "
                         f"({[_lab(m) for m in missing[:3]]}...)")
    return np.stack([got[m] for m in want]).astype(np.float32) * scale


def _cache_path(model, var, start, want, cache_dir=MONTHLY_CACHE):
    return os.path.join(cache_dir, model,
                        f"{var}_s{start}_{_lab(min(want))}_{len(want)}mo.npz")


def model_cube(model, var, start, want, lat, lon, use_cache=True, verbose=True,
               cache_dir=MONTHLY_CACHE, rebuild=False):
    """(N, T, lat, lon) raw on the obs grid -- no anomaly, no smoothing. Cached, since it
    is the slow part: read from data/dcpp_monthly_cache/ if there (unless `rebuild`), else
    built from the raw files and written there."""
    cm = CMIP_NAME[var]
    cp = _cache_path(model, var, start, want, cache_dir)
    if use_cache and not rebuild and os.path.exists(cp):
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
    spec = OBS_SPEC[var]
    path = os.path.join(OBS, spec[0])
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


# ---------------------------------------------------------------------- anomalies
def _moy_of(want):
    return np.array([m % 12 + 1 for m in want])


def _remove_moy(X, moy):
    """Remove the month-of-year mean along the time axis (axis -3).

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


# ------------------------------------------------------------------- the interface
class Decade:
    """One start date, monthly, pooled over models. Built by `get`.

        F  (N, T, lat, lon)   members          G  (T, lat, lon)  observations
        months (T,) absolute   dates (T,) 'YYYY-MM'   moy (T,) 1..12
    """

    def __init__(self, start, var="SLP", models=None, anom="moy", smooth=True,
                 cache=True, verbose=True):
        var = canon(var)
        if var not in CMIP_NAME:
            raise ValueError(f"var must be one of {list(CMIP_NAME)}, got {var!r}")
        start = int(start)
        models = list(models or MODELS)

        LAT, LON = obs_grid(var)
        want = canonical_window(start)
        g = _obs(var, want, LAT, LON)
        if g is None:
            raise ValueError(f"{var} observations do not cover {_lab(min(want))}.."
                             f"{_lab(max(want))}")

        parts, sizes, names, owner = [], [], [], []
        for m in models:
            cube, mems = model_cube(m, var, start, want, LAT, LON,
                                    use_cache=cache, verbose=verbose)
            parts.append(cube)
            sizes.append(cube.shape[0])
            names += [f"{m}:{s}" for s in mems]
            owner += [m] * cube.shape[0]

        self.months = np.asarray(want)
        self.moy = _moy_of(want)
        self.dates = np.array([_lab(m) for m in want])
        self.var, self.start = var, start
        self.units = UNITS.get(var, "")
        self.lats, self.lons = LAT, LON
        self.models = models

        # float32 -> float64 is exact, so converting after the concatenation gives the
        # same array as converting each part first, with one float64 copy fewer
        F = np.concatenate(parts, axis=0).astype(float)
        del parts

        self.smooth_box_deg = (_smooth.box_for(var) if smooth is True
                               else (tuple(smooth) if smooth else None))
        if self.smooth_box_deg:
            dlat, dlon = self.smooth_box_deg
            F = _smooth.smooth_box(F, LAT, LON, dlat, dlon)
            g = _smooth.smooth_box(g, LAT, LON, dlat, dlon)

        # Anomalies per MODEL, so each model's own climatology and drift go with it.
        self.anom = anom
        if anom == "moy":
            i = 0
            for n in sizes:
                F[i:i + n] = _remove_moy(F[i:i + n], self.moy)
                i += n
            g = _remove_moy(g, self.moy)
        elif anom not in (None, "none"):
            raise ValueError("anom must be 'moy' or 'none'")

        self.F, self.G = F, g
        self.members = np.array(names)
        self.model_of = np.array(owner)
        self.N, self.T = self.F.shape[0], self.F.shape[1]

    def __repr__(self):
        return (f"<Decade DCPP-A s{self.start} ({len(self.models)} models) {self.var} "
                f"{self.dates[0]}..{self.dates[-1]} [{self.units}]: N={self.N} T={self.T}"
                f", anom={self.anom}>")


def get(start=1961, var="SLP", models=None, **kw):
    """One decade of monthly DCPP-A output, pooled over models. See `Decade`."""
    return Decade(start, var=var, models=models, **kw)
