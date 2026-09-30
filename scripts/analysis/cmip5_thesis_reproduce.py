"""Reproduce the thesis CMIP5 jugaad numbers from the rebuilt data, against calc_RPC's saved outputs.

    python scripts/analysis/cmip5_thesis_reproduce.py

`load()` is calc_RPC.ipynb cell 3 verbatim, except `warnings.catch_warnings(action=...)`
(Python >= 3.11) becomes `simplefilter` for SNP_env's 3.10. The run then prints the
numbers the notebook's saved outputs carry -- cell 8 (flattened moments), cell 16 (gamma
maxima and means), cell 20 (mean alt=0.5 vs 1.5 binning ranges) -- next to the recomputed
ones, runs the thesis `info_RPC` itself and checks `jugaad_thesis` against it bitwise.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import os
import time
import types
import warnings

import numpy as np
import xarray as xr
from scipy.stats import kurtosis, skew

for _name, _attr, _obj in (("tqdm", "tqdm", lambda it, **kw: it),
                           ("sklearn.feature_selection", "mutual_info_regression", None)):
    try:
        __import__(_name)
    except ImportError:
        _sys.modules.setdefault(_name.split(".")[0], types.ModuleType(_name.split(".")[0]))
        setattr(_sys.modules.setdefault(_name, types.ModuleType(_name)), _attr, _obj)

import InfoTheoryMetrics as ITM                                      # noqa: E402
import jugaad_thesis as JT                                           # noqa: E402

DATA = os.path.join(_snp_path.REPO if hasattr(_snp_path, "REPO") else
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data")

# calc_RPC.ipynb saved outputs (the thesis run)
SAVED = {"cell8": {"sig_var": 104535.758, "sig_skew": -0.286, "sig_kurt": 2.572,
                   "obs_var": 158673.163, "obs_skew": 0.129, "obs_kurt": 2.216,
                   "ens_var": 261392.168, "ens_skew": 0.016, "ens_kurt": 1.66},
         "cell16": {"gam_o_max": 0.9922160431674707, "gam_m_max": 0.9935209217108472,
                    "gam_o_mean": 0.7090980982558687, "gam_m_mean": 0.7675887635696372},
         "cell20": {"gam_o_range": 0.04458541204522211, "gam_m_range": 0.032688386660848806,
                    "gam_range": 0.30342196792556947, "lam_o_range": 0.19558260227226498,
                    "lam_m_range": 0.11124330111259985, "lam_range": 0.14038587876406738}}


def load():
    """calc_RPC.ipynb cell 3. Returns (f, obs, s) as xarray Datasets."""
    f = xr.open_dataset(os.path.join(DATA, "ensembles",
                                     "large_psl_decadal_ensemble_no_extrapolate.nc")
                        ).drop_dims(["bnds"])
    obs = xr.open_dataset(os.path.join(DATA, "obs", "hadslp2", "hadslp2_monthly_1850_2004.nc"))
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
    """(F (N,T,ny,nx), o (T,ny,nx), s (T,ny,nx), lats, lons) for jugaad_fd_optbins_run.py."""
    f, obs, s = load()
    return (f.psl.transpose("n", "time", "lat", "lon").to_numpy(),
            obs.psl.transpose("time", "lat", "lon").to_numpy(),
            s.psl.transpose("time", "lat", "lon").to_numpy(),
            f.lat.to_numpy(), f.lon.to_numpy())


def cell8(f, obs, s):
    out = {}
    for k, a in (("sig", s.psl), ("obs", obs.psl), ("ens", f.psl)):
        x = a.to_numpy().flatten()
        x = x[~np.isnan(x)]
        out[f"{k}_var"] = round(float(np.nanvar(x)), 3)
        out[f"{k}_skew"] = round(float(skew(x)), 3)
        out[f"{k}_kurt"] = round(float(kurtosis(x)), 3)
    return out


def edge_fix(a):
    a = a.copy()
    a[:, 0], a[:, -1] = a[:, 1], a[:, -2]                            # as cell 19
    return a


def compare(tag, got, ref):
    print(f"\n{tag}")
    for k, v in ref.items():
        g = got[k]
        print(f"  {k:<12} thesis {v:>14.6f}   now {g:>14.6f}   diff {g - v:>+10.2e}")


def main():
    t0 = time.time()
    f, obs, s = load()
    print(f"loaded N={f.sizes['n']} T={f.sizes['time']} grid={f.sizes['lat']}x{f.sizes['lon']} "
          f"({time.time() - t0:.0f}s)")
    compare("cell 8 (flattened moments)", cell8(f, obs, s), SAVED["cell8"])

    F, o, S, lats, lons = arrays()
    # rho BEFORE info_RPC, as in the notebook (cell 10 < cell 12): info_RPC writes NaN -> 0
    # into `ensemble.psl` in place (`.to_numpy()` is a view), so anything computed from f
    # afterwards sees zero-filled members.
    W = np.cos(np.deg2rad(lats))[:, None] * np.ones(len(lons))
    rho_o = xr.corr(s.psl, obs.psl, dim="time")                      # calc_RPC cell 10
    rho_o = xr.where(rho_o > 0, rho_o, 0)
    rho_m = np.sqrt(s.psl.var(dim="time", skipna=True)
                    / f.psl.var(dim="time", skipna=True).mean("n"))
    rho = (rho_o / rho_m).transpose("lat", "lon").to_numpy()
    print(f"\nrho > 1 over {100 * (W * (rho > 1)).sum() / W.sum():.2f}% area-weighted "
          f"(published 72.1%), {100 * (rho > 1).mean():.2f}% unweighted (66.4%)")
    nan_before = int(np.isnan(f.psl.values).sum())
    t1 = time.time()
    I_fg, I_fs, H_f, H_g = ITM.info_RPC(f, s, obs, method="histogram", quant_uncert=True)
    print(f"\ninfo_RPC {time.time() - t1:.0f}s; NaNs in f before/after: "
          f"{nan_before}/{int(np.isnan(f.psl.values).sum())} (in-place NaN -> 0)")
    r = JT.jugaad_maps(F, o, s=S, rule="fd")
    print(f"jugaad_thesis vs info_RPC bitwise: I_o {np.array_equal(r['I_o'], I_fg)}, "
          f"I_m {np.array_equal(r['I_m'], I_fs)}")

    gam_o, gam_m = I_fg / H_g, I_fs / H_f
    compare("cell 16 (gamma)", {"gam_o_max": np.max(gam_o), "gam_m_max": np.max(gam_m),
                                "gam_o_mean": np.mean(gam_o), "gam_m_mean": np.mean(gam_m)},
            SAVED["cell16"])
    lam_o = np.sqrt(1 - np.power(2, -2 * I_fg))
    lam_m = np.sqrt(1 - np.power(2, -2 * I_fs))
    got = {"gam_o_range": np.mean(edge_fix(np.abs(gam_o[2] - gam_o[0]))),
           "gam_m_range": np.mean(edge_fix(np.abs(gam_m[2] - gam_m[0]))),
           "gam_range": np.mean(edge_fix(np.abs(gam_o[2] / gam_m[2] - gam_o[0] / gam_m[0]))),
           "lam_o_range": np.mean(edge_fix(np.abs(lam_o[2] - lam_o[0]))),
           "lam_m_range": np.mean(edge_fix(np.abs(lam_m[2] - lam_m[0]))),
           "lam_range": np.mean(edge_fix(np.abs(lam_o[2] / lam_m[2] - lam_o[0] / lam_m[0])))}
    compare("cell 20 (alt 0.5 vs 1.5 ranges)", got, SAVED["cell20"])

    lam = lam_o[1] / lam_m[1]
    print(f"\nalt=1: mean lam_o {lam_o[1].mean():.4f}, lam_m {lam_m[1].mean():.4f}, "
          f"lam {lam.mean():.4f}; lam>1 over {100 * (W * (lam > 1)).sum() / W.sum():.1f}% "
          f"(area-weighted), {100 * (lam > 1).mean():.1f}% (unweighted)")
    print(f"FD bins f/o/s median {[int(np.median(b)) for b in r['bins']]}, "
          f"ranges {[(int(b.min()), int(b.max())) for b in r['bins']]}")


if __name__ == "__main__":
    main()
