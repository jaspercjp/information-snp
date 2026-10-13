"""The thesis CMIP5 decadal1961 SLP case, loaded exactly as calc_RPC.ipynb cell 3.

    import cmip5_thesis
    F, o, s, lats, lons = cmip5_thesis.arrays()

Reads the two files build_cmip5_thesis_ensemble.py writes. `load()` is calc_RPC cell 3
verbatim, except `warnings.catch_warnings(action=...)` (Python >= 3.11) becomes
`simplefilter`: HadSLP2 restricted to 1962-1971 and interpolated onto the model grid and
mid-month stamps, a least-squares line in time (intercept and slope) removed from obs and
from every member and cell (NaN-skipping), and s the xarray member mean (NaN-skipping).
"""
import os
import sys
import warnings

import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import ENSEMBLES, HADSLP2                               # noqa: E402


def load():
    """calc_RPC.ipynb cell 3. Returns (f, obs, s) as xarray Datasets."""
    f = xr.open_dataset(os.path.join(ENSEMBLES, "large_psl_decadal_ensemble_no_extrapolate.nc")
                        ).drop_dims(["bnds"])
    obs = xr.open_dataset(os.path.join(HADSLP2, "hadslp2_monthly_1850_2004.nc"))
    obs = obs.where(obs["time.year"] >= 1962, drop=True)
    obs = obs.where(obs["time.year"] < 1972, drop=True)
    obs = obs.sortby("lon")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs["time"] = obs.indexes["time"].to_datetimeindex()
    obs = obs.interp_like(f, kwargs={"fill_value": "extrapolate"})
    obs_lin_trend = xr.polyval(obs["time"], obs.psl.polyfit(dim='time', deg=1).polyfit_coefficients)
    obs["psl"] = obs["psl"] - obs_lin_trend
    f_lin_trend = xr.polyval(f["time"], f.psl.polyfit(dim='time', skipna=True, deg=1).polyfit_coefficients)
    f["psl"] = f["psl"] - f_lin_trend
    s = f.mean("n")
    return f, obs, s


def arrays():
    """(F (N,T,ny,nx), o (T,ny,nx), s (T,ny,nx), lats, lons)."""
    f, obs, s = load()
    return (f.psl.transpose("n", "time", "lat", "lon").to_numpy(),
            obs.psl.transpose("time", "lat", "lon").to_numpy(),
            s.psl.transpose("time", "lat", "lon").to_numpy(),
            f.lat.to_numpy(), f.lon.to_numpy())
