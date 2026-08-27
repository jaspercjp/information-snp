"""Build the SMYLE hindcast cube: one lead-1-3 mean per (initialization, member), N48 grid.

    python scripts/build_smyle_cube.py          # -> data/smyle_cube.npz

CESM2 SMYLE initializes four times a year (Feb/May/Aug/Nov), so the sample axis here is
248 initializations rather than the 45 annual start dates of `build_djf_cube.py`.

Lead convention, asserted per file: CESM stamps a monthly mean at the END of its
averaging period, so record 0 of a YYYY-MM init carries time=YYYY-(MM+1) but IS the
YYYY-MM mean. `time_bnds[:, 0]` gives the true month. Records 0,1,2 are lead months
1,2,3 -- the init month and the two after it.

The last init, 2019-11, is dropped: its lead-1-3 window ends Jan 2020 and HadSLP2r
stops at Dec 2019. That leaves J=247.

Grid is HadCM3's N48, as in `build_djf_cube.py` -- the observations are 5 degree, so
interpolating SMYLE's f09 down to N48 loses nothing that the obs could verify, and it
keeps these results comparable with the existing cubes. `pad_cyclic`/`to_n48` are that
file's, unchanged.
"""
import os
import sys
import warnings

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")

ROOT = "/Users/jasperchen/Academics/Research/SNP/information-snp"
SMYLE = f"{ROOT}/data/smyle/PSL"
# Prefer the tracked copy built by scripts/fetch_hadslp2r.py; fall back to the
# original under .scratch, which is untracked and absent in a fresh clone.
_OBS_CANDIDATES = [f"{ROOT}/data/obs/hadslp2r_monthly_1850_2019.nc",
                   f"{ROOT}/.scratch/eade_replication/data/hadslp2r_monthly_1850_2019.nc"]
OBS = next((p for p in _OBS_CANDIDATES if os.path.exists(p)), _OBS_CANDIDATES[0])
OUT = f"{ROOT}/data/smyle_cube.npz"

LAT48 = np.arange(-90.0, 90.0 + 1e-9, 2.5)          # 73
LON48 = np.arange(0.0, 360.0 - 1e-9, 3.75)          # 96
INITS = ["02", "05", "08", "11"]
MEMBERS = range(1, 21)
NLEAD = 3                                           # lead months 1-3


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


def to_n48(da):
    return pad_cyclic(da).interp(lat=LAT48, lon=LON48)


def span(year, mm):
    """The file's month range: init .. init+23, as YYYYMM-YYYYMM."""
    tot = year * 12 + int(mm) - 1 + 23
    return f"{year}{int(mm):02d}-{tot // 12}{tot % 12 + 1:02d}"


def init_list():
    """(year, mm) for every usable init: quarterly 1958-2019, less 2019-11."""
    out = [(y, mm) for y in range(1958, 2020) for mm in INITS]
    return [t for t in out if t != (2019, "11")]        # obs stop at Dec 2019


def member_lead13(year, mm, n):
    """The lead-1-3 PSL mean for one (init, member), on N48. Asserts the lead convention."""
    case = f"b.e21.BSMYLE.f09_g17.{year}-{mm}.{n:03d}"
    p = f"{SMYLE}/{case}.cam.h0.PSL.{span(year, mm)}.nc"
    d = xr.open_dataset(p, use_cftime=True)
    b0 = d.time_bnds.values[:NLEAD, 0]                  # start of each averaging period
    assert b0[0].year == year and b0[0].month == int(mm), (p, b0[0])
    return to_n48(d.PSL.isel(time=slice(0, NLEAD)).mean("time")).values


def obs_lead13(inits):
    """HadSLP2r lead-1-3 means over the same windows and grid, shape (J, 73, 96)."""
    d = xr.open_dataset(OBS, use_cftime=True).psl
    yrs = np.array([t.year for t in d["time"].values])
    mos = np.array([t.month for t in d["time"].values])
    out = []
    for y, mm in inits:
        want = [(y * 12 + int(mm) - 1 + k) for k in range(NLEAD)]
        mask = np.isin(yrs * 12 + mos - 1, want)
        assert mask.sum() == NLEAD, (y, mm, mask.sum())
        out.append(to_n48(d.isel(time=np.where(mask)[0]).mean("time")).values)
    return np.stack(out)


def main():
    inits = init_list()
    J, N = len(inits), len(MEMBERS)
    print(f"{J} initializations x {N} members, lead months 1-{NLEAD}, N48 "
          f"{LAT48.size}x{LON48.size}", flush=True)

    cube = np.full((J, N, LAT48.size, LON48.size), np.nan, np.float32)
    for j, (y, mm) in enumerate(inits):
        for i, n in enumerate(MEMBERS):
            cube[j, i] = member_lead13(y, mm, n)
        if (j + 1) % 20 == 0 or j + 1 == J:
            print(f"  {j+1}/{J} inits", flush=True)

    print("observations ...", flush=True)
    obs = obs_lead13(inits).astype(np.float32)

    assert np.isfinite(cube).all() and np.isfinite(obs).all(), "NaNs in the cube"
    np.savez_compressed(
        OUT, cube=cube, obs=obs,
        year=np.array([y for y, _ in inits]),
        month=np.array([int(m) for _, m in inits]),
        members=np.array([f"{n:03d}" for n in MEMBERS]),
        lat=LAT48, lon=LON48, nlead=np.array(NLEAD))
    print(f"\nwrote {OUT}")
    print(f"  cube {cube.shape}  mean {cube.mean():.1f} Pa")
    print(f"  obs  {obs.shape}  mean {obs.mean():.1f} Pa")


if __name__ == "__main__":
    main()
