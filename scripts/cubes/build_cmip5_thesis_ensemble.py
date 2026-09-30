"""Rebuild the thesis CMIP5 decadal1961 psl ensemble and the HadSLP2 obs file.

    python scripts/cubes/build_cmip5_thesis_ensemble.py [--obs-day 16] [--skip-ensemble]

Writes, under data/ of the checkout this script lives in (or $SNP_DATA if set):
    data/ensembles/large_psl_decadal_ensemble_no_extrapolate.nc
    data/obs/hadslp2/hadslp2_monthly_1850_2004.nc

Inputs: the 49 psl files in data/models/<model>/psl/ (fetched from the CEDA ESGF replica,
checksums in data/models/PROVENANCE_psl.json) and data/obs/hadslp2/hadslp2.asc.gz
(https://www.metoffice.gov.uk/hadobs/hadslp2/data/hadslp2.asc.gz, HadSLP2 1850-2004).

The ensemble follows notebooks/create_ensemble_dataset.ipynb step for step: drop the
bounds variables, convert every time axis to a DatetimeIndex, `interp_like` each member
onto the first dataset (linear, no extrapolation -> NaN outside a member's grid or time
span), and concat along a new `n` dimension. The notebook's first dataset came from
`os.listdir` order; the thesis grid was 64 x 128 with T = 120 (see
scripts/metrics/build_calc_RPC_infomeasure.py), which only CanCM4 matches, so CanCM4 r1i1p1
is the template here and the remaining paths are sorted.

The notebook's drop of the bounds variables leaves no `bnds` dimension, yet
notebooks/calc_RPC.ipynb opens the file with `.drop_dims(["bnds"])`; the template's
`time_bnds` is carried along so that call works unchanged. It is never read.

HadSLP2 parse is the one in scripts/fetch/fetch_hadslp2r.py (hPa*100 -> Pa, lat ascending,
lon 0..355). The time axis is written in a noleap calendar so it decodes to a CFTimeIndex
and calc_RPC's `.to_datetimeindex()` works as written; the day of month is a parameter
because calc_RPC interpolates obs onto the model's mid-month stamps. Day 16 is the default
because it is the one that reproduces calc_RPC's saved cell-8 output: flattened obs variance
after cell 3 is 158673.2 Pa^2 (saved: 158673.163), against 155033.9 for day 15, 150370.8
for day 14 and 121629.7 for day 1.
"""

import argparse
import gzip
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SNP_DATA", os.path.join(_HERE, "..", "..", "data"))
MODEL_DIR = os.path.join(DATA, "models")
ENS_OUT = os.path.join(DATA, "ensembles", "large_psl_decadal_ensemble_no_extrapolate.nc")
OBS_RAW = os.path.join(DATA, "obs", "hadslp2", "hadslp2.asc.gz")
OBS_OUT = os.path.join(DATA, "obs", "hadslp2", "hadslp2_monthly_1850_2004.nc")
TEMPLATE = os.path.join("CanCM4", "psl", "psl_Amon_CanCM4_decadal1961_r1i1p1_196201-197112.nc")
NLON, NLAT = 72, 37


def _to_datetime(ds):
    idx = ds.indexes["time"]
    ds["time"] = idx if isinstance(idx, pd.DatetimeIndex) else idx.to_datetimeindex()
    return ds


def build_ensemble():
    paths = sorted(os.path.join(m, "psl", f)
                   for m in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, m))
                   for f in os.listdir(os.path.join(MODEL_DIR, m, "psl")) if f.endswith(".nc"))
    paths.remove(TEMPLATE)
    paths = [TEMPLATE] + paths
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        example = _to_datetime(xr.open_dataset(os.path.join(MODEL_DIR, TEMPLATE)))
        members = []
        for p in paths:
            ds = xr.open_dataset(os.path.join(MODEL_DIR, p))
            ds = ds.drop_vars(["lon_bnds", "lat_bnds", "time_bnds"])
            if "average_DT" in ds.variables:
                ds = ds.drop_vars(["average_DT", "average_T1", "average_T2"])
            members.append(_to_datetime(ds).interp_like(example))
    ens = xr.concat(members, dim="n")
    ens["time_bnds"] = example["time_bnds"]
    ens = ens.assign_coords(member=("n", [os.path.basename(p).split("_")[2] + "_"
                                          + os.path.basename(p).split("_")[4] for p in paths]))
    os.makedirs(os.path.dirname(ENS_OUT), exist_ok=True)
    ens.to_netcdf(ENS_OUT)
    return ens


def build_obs(day):
    with gzip.open(OBS_RAW, "rt") as fh:
        tokens = fh.read().split()
    vals, stamps, i = [], [], 0
    while i < len(tokens):
        year, month = int(tokens[i]), int(tokens[i + 1])
        block = np.array(tokens[i + 2: i + 2 + NLON * NLAT], dtype=np.float64)
        vals.append(block.reshape(NLAT, NLON) * 0.01 * 100.0)    # hPa*100 -> hPa -> Pa
        stamps.append((year, month))
        i += 2 + NLON * NLAT
    times = xr.date_range(f"{stamps[0][0]}-{stamps[0][1]:02d}-01", periods=len(stamps),
                          freq="MS", calendar="noleap", use_cftime=True)
    times = [t.replace(day=day) for t in times]
    assert [(t.year, t.month) for t in times] == stamps
    obs = xr.Dataset({"psl": (("time", "lat", "lon"), np.stack(vals))},
                     coords={"time": times,
                             "lat": np.arange(90.0, -95.0, -5.0),
                             "lon": np.arange(-180.0, 180.0, 5.0)})
    obs = obs.reindex(lat=obs.lat[::-1])
    obs = obs.assign_coords(lon=(obs.lon % 360)).sortby("lon")
    obs.psl.attrs = {"units": "Pa", "long_name": "mean sea level pressure"}
    obs.attrs["source"] = "HadSLP2 (1850-2004) bulk ASCII hadslp2.asc.gz, Met Office Hadley Centre"
    obs.attrs["time_note"] = f"monthly means stamped on day {day} of each month (noleap calendar)"
    obs.to_netcdf(OBS_OUT)
    return obs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-day", type=int, default=16)
    ap.add_argument("--skip-ensemble", action="store_true")
    a = ap.parse_args()
    if not a.skip_ensemble:
        e = build_ensemble()
        print(ENS_OUT, dict(e.sizes))
    o = build_obs(a.obs_day)
    print(OBS_OUT, dict(o.sizes), o.time.values[[0, -1]])
