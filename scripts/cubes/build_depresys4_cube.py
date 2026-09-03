"""Reduce the raw DePreSys4 files to (init, member) cubes on the observations' grid.

    python scripts/build_depresys4_cube.py                        # SLP, lead 2-4 (DJF)
    python scripts/build_depresys4_cube.py --leads 2-4,13-60       # + Eade's years 2-5
    python scripts/build_depresys4_cube.py --var TREFHT --leads 2-4
    python scripts/build_depresys4_cube.py --obs-only --leads 2-4

Same output schema, same target grid and the same variable names as
`build_smyle_cube.py`, so both systems load through the identical handle interface and
their numbers are directly comparable. The model is coarsened onto the verifying
observations' own grid; the observations are never resampled:

    data/depresys4_cubes/<VAR>_lead<a>-<b>.npz

DePreSys4 is initialised on 1 November once a year, s1960..s2018, 10 members, run 10
years forward. So lead month 1 is November of the start year and DJF is lead months 2-4,
the same offset as SMYLE's November starts. Eade et al.'s decadal window, annual mean
over lead YEARS 2-5, is lead months 13-60.

Three things that differ from SMYLE and are handled here:

* **360-day calendar.** HadGEM3-GC31-MM gives every month 30 days. Month labels come
  from `time_bnds[:, 0]` decoded with cftime; the year*12+month arithmetic is unaffected
  because months are still 1-12.
* **Time is stamped mid-month**, not at the end of the averaging period as CESM does. So
  reading `time` directly would NOT shift a month here -- but `time_bnds[:, 0]` is used
  anyway, so the same code is correct for both systems.
* **Each (start, member, variable) is split into ~12 chunks.** Only the chunks whose
  span overlaps the requested lead window are opened, which is why building lead 2-4
  costs 2 file opens per member rather than 12.

CMIP variable names (psl/tas/pr) map onto this project's vocabulary (SLP/TREFHT/PRECT).
TREFHT is CESM's name for near-surface air temperature; it is used here for HadGEM3's
`tas` too so that one name means one physical field across both systems. `pr` is
converted from kg m-2 s-1 to mm/day, matching the SMYLE PRECT cubes.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import argparse
import glob
import os
import re
import warnings

import numpy as np
import xarray as xr

from build_smyle_cube import (LAT48, LON48, OBS_SPEC, UNITS, obs_grid,
                              obs_windows, target_months, to_grid)

warnings.simplefilter("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
RAW = os.path.join(ROOT, "data", "depresys4")
OUTDIR = os.path.join(ROOT, "data", "depresys4_cubes")

YEARS = range(1960, 2019)                 # s1960..s2018
INIT_MONTH = 11                           # 1 November, every year
MEMBERS = [f"r{i}i1p1f2" for i in range(1, 11)]
NMONTH = 120                              # 10-year runs

CMIP_NAME = {"SLP": "psl", "TREFHT": "tas", "PRECT": "pr"}
SCALE = {"PRECT": 86400.0}                # kg m-2 s-1 -> mm/day
VARS = list(CMIP_NAME)

_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


def parse_leads(text):
    """'2-4,13-60' -> [(2, 4), (13, 60)]. Inclusive, 1-based; lead 1 = the init month."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        a, b = int(a), int(b or a)
        if not 1 <= a <= b <= NMONTH:
            raise SystemExit(f"lead window {part!r} outside 1-{NMONTH}")
        out.append((a, b))
    return out


def chunk_span(path):
    """'..._196101-196112.nc' -> (first month index, last month index), absolute."""
    m = re.search(r"_(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$", os.path.basename(path))
    if not m:
        return None
    y0, m0, y1, m1 = (int(g) for g in m.groups())
    return y0 * 12 + m0 - 1, y1 * 12 + m1 - 1


def member_files(year, var, member):
    cm = CMIP_NAME[var]
    pat = os.path.join(RAW, f"s{year}", cm, f"{cm}_*_s{year}-{member}_*.nc")
    return sorted(glob.glob(pat))


def read_member(year, var, member, leads, lat, lon):
    """One (start, member) -> {lead: (lat, lon) mean on the target grid}. Opens only
    the chunks it needs."""
    cm = CMIP_NAME[var]
    base = year * 12 + INIT_MONTH - 1
    wanted = {lead: set(target_months(year, INIT_MONTH, lead)) for lead in leads}
    need = set().union(*wanted.values())

    files = member_files(year, var, member)
    if not files:
        raise FileNotFoundError(f"no files for s{year} {var} {member}")

    acc = {lead: None for lead in leads}
    cnt = {lead: 0 for lead in leads}
    for p in files:
        sp = chunk_span(p)
        if sp is None or sp[1] < min(need) or sp[0] > max(need):
            continue
        with xr.open_dataset(p, decode_times=_CODER) as d:
            b0 = d.time_bnds.values[:, 0]
            months = np.array([t.year * 12 + t.month - 1 for t in b0])
            a = d[cm].values                                  # (t, lat, lon), native
            for lead in leads:
                sel = np.where(np.isin(months, list(wanted[lead])))[0]
                if sel.size == 0:
                    continue
                part = a[sel].sum(axis=0)
                acc[lead] = part if acc[lead] is None else acc[lead] + part
                cnt[lead] += sel.size

    out = {}
    for lead in leads:
        n_expect = lead[1] - lead[0] + 1
        if cnt[lead] != n_expect:
            raise ValueError(f"s{year} {var} {member} lead {lead}: found "
                             f"{cnt[lead]} of {n_expect} months")
        mean = acc[lead] / cnt[lead] * SCALE.get(var, 1.0)
        da = xr.DataArray(mean, dims=("lat", "lon"),
                          coords={"lat": _LAT, "lon": _LON})
        out[lead] = to_grid(da, lat, lon).values
    return out


# native grid, read once from any file
_LAT = _LON = None


def _load_grid(var):
    global _LAT, _LON
    f = member_files(YEARS[0], var, MEMBERS[0])
    with xr.open_dataset(f[0], decode_times=_CODER) as d:
        _LAT, _LON = d["lat"].values, d["lon"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="SLP", choices=VARS,
                    help="SLP (psl), TREFHT (tas) or PRECT (pr)")
    ap.add_argument("--leads", default="2-4",
                    help="inclusive lead-month windows; 2-4 = DJF, 13-60 = years 2-5")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--grid", default="obs", choices=["obs", "n48"],
                    help="'obs' coarsens the model onto the observations' own grid")
    ap.add_argument("--obs-only", action="store_true",
                    help="attach/refresh observations on cubes that already exist")
    a = ap.parse_args()

    leads = parse_leads(a.leads)
    inits = [(y, INIT_MONTH) for y in YEARS]
    var, J, N = a.var, len(inits), len(MEMBERS)
    os.makedirs(a.outdir, exist_ok=True)
    LAT, LON = obs_grid(var) if a.grid == "obs" else (LAT48, LON48)

    print(f"DePreSys4 {var}: {J} start dates x {N} members, "
          f"lead windows {', '.join(f'{x}-{y}' for x, y in leads)}, "
          f"grid {a.grid} {len(LAT)}x{len(LON)}", flush=True)

    if a.obs_only:
        obs = obs_windows(inits, leads, var, LAT, LON)
        if obs is None:
            raise SystemExit(f"nothing to attach for {var}")
        for lead in leads:
            out = os.path.join(a.outdir, f"{var}_lead{lead[0]}-{lead[1]}.npz")
            if not os.path.exists(out):
                raise SystemExit(f"{out} does not exist; build it first")
            z = dict(np.load(out))
            g, ok = obs[lead]
            z["obs"], z["obs_ok"] = g, ok
            np.savez_compressed(out, **z)
            print(f"attached obs to {out}:  {int(ok.sum())}/{J} usable")
        return

    _load_grid(var)
    print(f"  native grid {len(_LAT)}x{len(_LON)}", flush=True)

    cubes = {lead: np.full((J, N, len(LAT), len(LON)), np.nan, np.float32)
             for lead in leads}
    for j, (y, _) in enumerate(inits):
        for i, mem in enumerate(MEMBERS):
            got = read_member(y, var, mem, leads, LAT, LON)
            for lead in leads:
                cubes[lead][j, i] = got[lead]
        if (j + 1) % 5 == 0 or j + 1 == J:
            print(f"  {j+1}/{J} start dates", flush=True)

    print("observations ...", flush=True)
    obs = obs_windows(inits, leads, var, LAT, LON)

    for lead in leads:
        cube = cubes[lead]
        assert np.isfinite(cube).all(), f"NaNs in the {var} lead {lead} cube"
        if obs is not None:
            g, ok = obs[lead]
        else:
            g = np.full((J, len(LAT), len(LON)), np.nan, np.float32)
            ok = np.zeros(J, bool)
        out = os.path.join(a.outdir, f"{var}_lead{lead[0]}-{lead[1]}.npz")
        np.savez_compressed(
            out, cube=cube, obs=g, obs_ok=ok,
            year=np.array([y for y, _ in inits]),
            month=np.array([m for _, m in inits]),
            members=np.array(MEMBERS),
            lat=LAT, lon=LON, lead=np.array(lead), var=np.array(var),
            units=np.array(UNITS.get(var, "")))
        print(f"wrote {out}")
        print(f"  cube {cube.shape}  mean {cube.mean():.4g} {UNITS.get(var, '')}   "
              f"obs {int(ok.sum())}/{J} usable")


if __name__ == "__main__":
    main()
