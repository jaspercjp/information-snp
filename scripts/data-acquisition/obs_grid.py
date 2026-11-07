"""The verifying observations, their grids, and the regridding onto them.

Shared by the DCPP builders and the monthly decadal loader (formerly the obs-grid part of
build_smyle_cube.py).

Grid: the model is coarsened onto the VERIFYING OBSERVATIONS' own grid, and the
observations are never resampled -- they are the ground truth, so their resolution is the
resolution the comparison can support. That is 5 deg for SLP (HadSLP2r), T62 Gaussian
for TREFHT (NCEP R1) and 2.5 deg for PRECT (GPCP), so the three variables are NOT on a
common grid.

Interpolation is bilinear, not conservative. For a fine model field onto a coarse grid
that point-samples rather than area-averages, so it retains sub-grid variance an area
average would remove.
"""

import os
import warnings

import numpy as np
import xarray as xr

from paths import OBS as OBSDIR

warnings.simplefilter("ignore")

# Which observational field verifies each model variable: (filename in data/obs, varname).
# SLP comes from fetch_hadslp2r.py, the other two from fetch_obs.py.
OBS_SPEC = {
    "SLP":    ("hadslp2r_monthly_1850_2019.nc", "psl"),      # 1850-2019, 5 deg
    "TREFHT": ("ncep_air2m_monthly.nc", "obs"),              # NCEP R1, 1948-, K
    "PRECT":  ("gpcp_precip_monthly.nc", "obs"),             # GPCP v2.3, 1979-, mm/day
}
UNITS = {"SLP": "Pa", "TREFHT": "K", "PRECT": "mm/day"}


def pad_cyclic(da):
    """Pad longitude periodically and latitude to the poles, so interp never extrapolates."""
    lon = da["lon"].values
    lo = da.isel(lon=[-1]).assign_coords(lon=[lon[-1] - 360.0])
    hi = da.isel(lon=[0]).assign_coords(lon=[lon[0] + 360.0])
    da = xr.concat([lo, da, hi], dim="lon")
    lat = da["lat"].values
    if lat[0] > -90.0:
        s = da.isel(lat=[0]).mean("lon").expand_dims(lon=da["lon"]).transpose(*da.dims)
        da = xr.concat([s.assign_coords(lat=[-90.0]), da], dim="lat")
    if lat[-1] < 90.0:
        n = da.isel(lat=[-1]).mean("lon").expand_dims(lon=da["lon"]).transpose(*da.dims)
        da = xr.concat([da, n.assign_coords(lat=[90.0])], dim="lat")
    return da


def to_grid(da, lat, lon):
    """Bilinear onto a target grid, after cyclic/pole padding so interp never extrapolates."""
    return pad_cyclic(da).interp(lat=lat, lon=lon)


def obs_grid(var):
    """(lat, lon) of the verifying observations' native grid. Models are coarsened onto THIS."""
    path = os.path.join(OBSDIR, OBS_SPEC[var][0])
    if not os.path.exists(path):
        raise SystemExit(f"need {path} to define the {var} grid; run fetch_obs.py "
                         f"(or fetch_hadslp2r.py for SLP)")
    with xr.open_dataset(path, decode_times=False,
                         drop_variables=["time_bnds", "climatology_bounds"]) as d:
        return d["lat"].values, d["lon"].values


def target_months(year, mm, lead):
    """The calendar months a lead window verifies into, as absolute month indices.
    Lead 1 is month `mm` of `year`."""
    a, b = lead
    base = year * 12 + int(mm) - 1
    return [base + k - 1 for k in range(a, b + 1)]


def obs_windows(inits, leads, var, lat=None, lon=None):
    """The verifying observations on the same windows -> {lead: ((J,lat,lon), (J,) bool)}.

    An unweighted mean of the monthly means in each window; obs_ok is False where the
    window runs past the record.
    """
    spec = OBS_SPEC.get(var)
    path = os.path.join(OBSDIR, spec[0])
    if not os.path.exists(path):
        print(f"  no observations at {path}; run fetch_obs.py "
              f"(or fetch_hadslp2r.py for SLP). obs left as NaN.")
        return None
    # time_bnds in the NOAA PSL files has a fill value that overflows the CF decoder
    d = xr.open_dataset(path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                        drop_variables=["time_bnds", "climatology_bounds"])[spec[1]]
    native = (lat is None
              or (len(lat) == d.sizes["lat"] and np.allclose(lat, d["lat"].values)
                  and len(lon) == d.sizes["lon"] and np.allclose(lon, d["lon"].values)))
    lat = d["lat"].values if lat is None else lat
    lon = d["lon"].values if lon is None else lon
    if not native:
        print(f"  NOTE: regridding observations onto a non-native grid")
    idx = np.array([t.year * 12 + t.month - 1 for t in d["time"].values])
    lut = {m: i for i, m in enumerate(idx)}
    out = {}
    for lead in leads:
        arr = np.full((len(inits), len(lat), len(lon)), np.nan, np.float32)
        ok = np.zeros(len(inits), bool)
        for j, (y, mm) in enumerate(inits):
            want = target_months(y, mm, lead)
            if not all(m in lut for m in want):          # window runs past the record
                continue
            sel = [lut[m] for m in want]
            m = d.isel(time=sel).mean("time")
            arr[j] = m.values if native else to_grid(m, lat, lon).values
            ok[j] = True
        out[lead] = (arr, ok)
        n_bad = int((~ok).sum())
        if n_bad:
            print(f"  lead {lead[0]}-{lead[1]}: {n_bad} init(s) verify past the obs "
                  f"record, obs_ok=False there")
    return out
