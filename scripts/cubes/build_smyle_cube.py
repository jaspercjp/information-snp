"""Reduce the raw SMYLE files to (init, member) cubes on the observations' grid.

    python scripts/build_smyle_cube.py                          # SLP, lead 1-3
    python scripts/build_smyle_cube.py --leads 1-3,2-4,1-12      # several windows, one pass
    python scripts/build_smyle_cube.py --var TREFHT --leads 1-3

Variables are SLP, TREFHT and PRECT. SLP is CESM's PSL -- the raw tree under
data/smyle/ keeps NCAR's spelling, everything from the cube outwards says SLP.

Writes one file per (variable, lead window):

    data/smyle_cubes/<VAR>_lead<a>-<b>.npz
        cube   (J, N, lat, lon)   float32, raw -- NOT detrended
        obs    (J, lat, lon)      float32, HadSLP2r on the same windows (SLP only)
        year   (J,)              initialization year
        month  (J,)              initialization month: 2, 5, 8 or 11
        obs_ok (J,)              bool, False where the window runs past the obs record

Detrending is deliberately NOT done here -- it depends on which seasons you keep, and
`smyle_handles` does it per init month at load time. See that module's docstring.

Every requested lead window is extracted in the same pass over the netCDF files, since
opening 4,960 files is the whole cost: ~13 min for one variable, whether you ask for one
window or six.

Lead convention, asserted per file: CESM stamps a monthly mean at the END of its
averaging period, so record 0 of a YYYY-MM init carries time=YYYY-(MM+1) but IS the
YYYY-MM mean. `time_bnds[:, 0]` gives the true month. Lead month 1 is the init month,
i.e. records are 0-indexed and lead L is record L-1.

Grid: the model is coarsened onto the VERIFYING OBSERVATIONS' own grid, and the
observations are never resampled -- they are the ground truth, so their resolution is
the resolution the comparison can support. That is 5 deg for SLP (HadSLP2r), T62
Gaussian for TREFHT (NCEP R1) and 2.5 deg for PRECT (GPCP), so the three variables are
NOT on a common grid. `--grid n48` restores the old shared HadCM3 N48 grid for
comparability with `build_djf_cube.py`'s decadal cubes.

Interpolation is bilinear, not conservative. For a fine model field onto a coarse grid
that point-samples rather than area-averages, so it retains sub-grid variance an area
average would remove. Adequate here, wrong for a flux budget.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import argparse
import os
import warnings

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
SMYLE = os.path.join(ROOT, "data", "smyle")
OUTDIR = os.path.join(ROOT, "data", "smyle_cubes")
OBSDIR = os.path.join(ROOT, "data", "obs")

LAT48 = np.arange(-90.0, 90.0 + 1e-9, 2.5)          # 73
LON48 = np.arange(0.0, 360.0 - 1e-9, 3.75)          # 96
YEARS = range(1958, 2020)
INITS = ["02", "05", "08", "11"]
MEMBERS = range(1, 21)
NMONTH = 24                                         # every SMYLE run is 24 months

# This project calls sea-level pressure SLP; CESM calls it PSL, and the raw download
# tree under data/smyle/ keeps CESM's spelling because those are NCAR's filenames. The
# mapping is applied when reading raw files; everything downstream -- cube filenames,
# `--var`, the handles -- says SLP.
CESM_NAME = {"SLP": "PSL"}
VARS = ["SLP", "TREFHT", "PRECT"]

# Which observational field verifies each model variable: (filename in data/obs, varname).
# SLP comes from scripts/fetch_hadslp2r.py, the other two from scripts/fetch_obs.py.
# A variable absent here, or whose file is missing, builds a model-side cube with `obs`
# all-NaN -- enough for rho_m / lambda_m, not for anything with an _o in its name.
OBS_SPEC = {
    "SLP":    ("hadslp2r_monthly_1850_2019.nc", "psl"),      # 1850-2019, 5 deg
    "TREFHT": ("ncep_air2m_monthly.nc", "obs"),              # NCEP R1, 1948-, K
    "PRECT":  ("gpcp_precip_monthly.nc", "obs"),             # GPCP v2.3, 1979-, mm/day
}

# PRECT is stored in mm/day, not CESM's m/s. This is not cosmetic: infomeasure's KSG
# dithers by noise_level=1e-10 by default, and a PRECT anomaly in m/s is ~1e-9, so the
# dither is ~10% of the signal -- MI comes out ~17% low and, worse, scale-dependent.
# In mm/day the dither is negligible and m/s-vs-mm/day agree exactly.
SCALE = {"PRECT": 86400.0 * 1000.0}
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


def to_n48(da):
    return to_grid(da, LAT48, LON48)


def obs_grid(var):
    """The verifying observations' native grid. Models are coarsened onto THIS.

    The observations are the ground truth, so they are never resampled -- whatever
    resolution they have is the resolution the comparison can support. Each variable
    therefore has its own grid: HadSLP2r 5 deg for SLP, NCEP T62 for TREFHT, GPCP
    2.5 deg for PRECT. Fields of different variables are consequently NOT on a common
    grid and cannot be combined cell by cell.
    """
    spec = OBS_SPEC.get(var)
    if spec is None:
        return LAT48, LON48
    path = os.path.join(OBSDIR, spec[0])
    if not os.path.exists(path):
        raise SystemExit(f"need {path} to define the {var} grid; run scripts/fetch_obs.py")
    with xr.open_dataset(path, decode_times=False,
                         drop_variables=["time_bnds", "climatology_bounds"]) as d:
        return d["lat"].values, d["lon"].values


def span(year, mm):
    """The file's month range: init .. init+23, as YYYYMM-YYYYMM."""
    tot = year * 12 + int(mm) - 1 + NMONTH - 1
    return f"{year}{int(mm):02d}-{tot // 12}{tot % 12 + 1:02d}"


def parse_leads(text):
    """'1-3,2-4' -> [(1, 3), (2, 4)]. Inclusive, 1-based, lead 1 = the init month."""
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


def target_months(year, mm, lead):
    """The calendar months a lead window verifies into, as absolute month indices."""
    a, b = lead
    base = year * 12 + int(mm) - 1
    return [base + k - 1 for k in range(a, b + 1)]


def path_for(year, mm, n, var):
    cesm = CESM_NAME.get(var, var)               # SLP -> PSL on the raw tree
    case = f"b.e21.BSMYLE.f09_g17.{year}-{mm}.{n:03d}"
    return os.path.join(SMYLE, f"{year}-{mm}", cesm,
                        f"{case}.cam.h0.{cesm}.{span(year, mm)}.nc")


def init_list(inits):
    return [(y, mm) for y in YEARS for mm in inits]


def read_member(year, mm, n, var, leads, lat, lon):
    """One file -> {lead window: (lat, lon) mean on the target grid}. Asserts the lead
    convention."""
    p = path_for(year, mm, n, var)
    with xr.open_dataset(p, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as d:
        b0 = d.time_bnds.values[:, 0]                    # start of each averaging period
        assert b0[0].year == year and b0[0].month == int(mm), (p, b0[0])
        assert d.sizes["time"] == NMONTH, (p, d.sizes["time"])
        field = d[CESM_NAME.get(var, var)] * SCALE.get(var, 1.0)
        return {lead: to_grid(field.isel(time=slice(lead[0] - 1, lead[1])).mean("time"),
                              lat, lon).values
                for lead in leads}


def obs_windows(inits, leads, var, lat=None, lon=None):
    """The verifying observations on the same windows -> {lead: ((J,lat,lon), (J,) bool)}."""
    spec = OBS_SPEC.get(var)
    if spec is None:
        print(f"  no observational counterpart wired up for {var}; obs left as NaN")
        return None
    path = os.path.join(OBSDIR, spec[0])
    if not os.path.exists(path):
        print(f"  no observations at {path}; run scripts/fetch_obs.py "
              f"(or fetch_hadslp2r.py for SLP). obs left as NaN.")
        return None
    # time_bnds in the NOAA PSL files has a fill value that overflows the CF decoder
    d = xr.open_dataset(path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                        drop_variables=["time_bnds", "climatology_bounds"])[spec[1]]
    # on the observations' own grid nothing is resampled; only a legacy --grid n48 run
    # interpolates them, and then it is stated in the output
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="SLP", choices=VARS,
                    help="SLP (CESM's PSL), TREFHT or PRECT")
    ap.add_argument("--leads", default="1-3",
                    help="comma-separated inclusive lead-month windows, e.g. 1-3,2-4")
    ap.add_argument("--inits", default=",".join(INITS),
                    help=f"start months to include (default all of {','.join(INITS)})")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--grid", default="obs", choices=["obs", "n48"],
                    help="'obs' coarsens the model onto the verifying observations' own "
                         "grid (default); 'n48' is the legacy grid of the decadal cubes")
    ap.add_argument("--obs-only", action="store_true",
                    help="attach/refresh observations on cubes that already exist, "
                         "without re-reading the 4,960 raw files")
    a = ap.parse_args()

    leads = parse_leads(a.leads)
    inits = init_list([s.strip().zfill(2) for s in a.inits.split(",") if s.strip()])
    var, J, N = a.var, len(inits), len(MEMBERS)
    os.makedirs(a.outdir, exist_ok=True)
    LAT, LON = obs_grid(var) if a.grid == "obs" else (LAT48, LON48)

    print(f"{var}: {J} initializations x {N} members, "
          f"lead windows {', '.join(f'{x}-{y}' for x, y in leads)}, "
          f"grid {a.grid} {len(LAT)}x{len(LON)}", flush=True)

    if a.obs_only:
        obs = obs_windows(inits, leads, var, LAT, LON)
        if obs is None:
            raise SystemExit(f"nothing to attach for {var}")
        for lead in leads:
            out = os.path.join(a.outdir, f"{var}_lead{lead[0]}-{lead[1]}.npz")
            if not os.path.exists(out):
                raise SystemExit(f"{out} does not exist; build it without --obs-only first")
            z = dict(np.load(out))
            if not (np.array_equal(z["year"], [y for y, _ in inits])
                    and np.array_equal(z["month"], [int(m) for _, m in inits])):
                raise SystemExit(f"{out} has a different init list; rebuild it fully")
            g, ok = obs[lead]
            z["obs"], z["obs_ok"] = g, ok
            np.savez_compressed(out, **z)
            print(f"attached obs to {out}:  {int(ok.sum())}/{len(inits)} usable")
        return

    cubes = {lead: np.full((J, N, len(LAT), len(LON)), np.nan, np.float32)
             for lead in leads}
    for j, (y, mm) in enumerate(inits):
        for i, n in enumerate(MEMBERS):
            got = read_member(y, mm, n, var, leads, LAT, LON)
            for lead in leads:
                cubes[lead][j, i] = got[lead]
        if (j + 1) % 20 == 0 or j + 1 == J:
            print(f"  {j+1}/{J} inits", flush=True)

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
            month=np.array([int(m) for _, m in inits]),
            members=np.array([f"{n:03d}" for n in MEMBERS]),
            lat=LAT, lon=LON, lead=np.array(lead), var=np.array(var),
            units=np.array(UNITS.get(var, "")))
        print(f"wrote {out}")
        print(f"  cube {cube.shape}  mean {cube.mean():.4g} {UNITS.get(var, '')}   "
              f"obs {int(ok.sum())}/{J} usable")


if __name__ == "__main__":
    main()
