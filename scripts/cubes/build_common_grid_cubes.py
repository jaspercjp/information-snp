"""Re-grid every variable onto SLP's grid and smooth them all with one box.

    python build_common_grid_cubes.py                       # all of it
    python build_common_grid_cubes.py --vars TREFHT PRECT   # SLP is already there
    python build_common_grid_cubes.py --datasets monthly --starts 1961
    python build_common_grid_cubes.py --list                # what has been built

    import common_grid_handles as CG
    c = CG.get("dcpp", "TREFHT", "13-60")     # .F .G .lats .lons .members .model_of
    c.F.shape                                 # (N, T, 37, 72), same grid as SLP

Why this exists
---------------
As shipped, each variable lives on its verifying observations' native grid and gets
Eade's own per-variable smoothing box:

    SLP     HadSLP2r 5 deg    37x72     11.25 x 12.5 deg
    PRECT   GPCP 2.5 deg      72x144    11.25 x 12.5 deg
    TREFHT  NCEP T62          94x192    15 x 15 deg

so no two variables can be compared cell by cell, and TREFHT is additionally smoothed
over a box 1.6x larger in area than the others. This module removes both differences: one
grid (SLP's) and one box (11.25 x 12.5 deg by default) for all three.

Two departures you are buying, stated plainly
---------------------------------------------
**The 15x15 box for TREFHT is not a bug.** Eade et al. (2014) specify "regions of 11.25
deg latitude by 12.5 deg longitude (15 deg by 15 deg for SAT)". Forcing TREFHT onto the
smaller box is a deliberate break with Eade, and TREFHT results from these cubes are no
longer directly comparable with theirs. `--box 15 15` inverts the choice and puts
everything on the SAT box instead; `meta.json` records which was used.

**Observations get coarsened.** `build_smyle_cube.obs_grid` never resamples observations
on purpose -- they are the ground truth and their resolution is the resolution the
comparison can support. Going to SLP's 5 deg grid throws away real observed detail in
NCEP T62 (1.9 deg) and GPCP (2.5 deg). That is the price of a common grid; it is only
worth paying for genuinely cross-variable work, and the native-grid cubes remain the
right ones for anything single-variable.

Order of operations, and why
----------------------------
    handle pipeline (interp to obs grid, gm removal, anomalies, detrend)
      -> smooth with the COMMON box, on the NATIVE grid
      -> bilinear sample onto SLP's grid

Smoothing before re-gridding, not after. An 11.25 x 12.5 deg box mean band-limits the
field far below the 5 deg target grid's Nyquist scale, so sampling onto it afterwards is
well-resolved and bilinear interpolation is safe. Re-gridding first and smoothing after
would alias T62's fine structure into the coarse grid before anything removed it, and no
amount of subsequent smoothing recovers that.

The cost is that the box is discretised slightly differently on each native grid -- 2.25 x
2.5 cells on HadSLP2r, ~6 x 6.7 on T62. The physical footprint is identical and the
resulting fields are band-limited identically, which is what "smoothed identically" has to
mean across grids; only the quadrature differs, at second order. Smoothing is applied
inside the handle, in its proper place in the pipeline, so nothing else about the existing
pre-processing moves.

Caching
-------
One `.npz` per (dataset, variable, window) under `data/common_grid/`, with a `meta.json`
beside it. Re-runs skip what exists unless `--overwrite`. The arrays are float32 on disk
(the MI estimator promotes to float64 on load); at (112, 120, 37, 72) that is 143 MB
rather than 286.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import argparse
import json
import os
import sys
import time

import numpy as np
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
OUTROOT = os.path.join(ROOT, "data", "common_grid")
sys.path.insert(0, _HERE)

from build_smyle_cube import obs_grid, to_grid                       # noqa: E402

VARS = ("SLP", "TREFHT", "PRECT")
DATASETS = ("dcpp", "monthly")
DEFAULT_LEADS = ("13-60",)
DEFAULT_STARTS = (1961,)
DEFAULT_BOX = (11.25, 12.5)          # Eade's non-SAT box; SLP and PRECT already use it
TARGET_VAR = "SLP"                   # whose grid everything lands on
REGRID_CHUNK = 8                     # members per interp call, to bound peak memory


def target_grid():
    """(lat, lon) of the grid everything is put on -- SLP's verifying obs grid."""
    return obs_grid(TARGET_VAR)


def regrid(A, lats, lons, tlat, tlon, chunk=REGRID_CHUNK):
    """`(..., nlat, nlon)` -> `(..., len(tlat), len(tlon))`, bilinear, cyclic in longitude.

    A no-op when the source grid already is the target, so SLP passes through untouched
    rather than being interpolated onto itself.
    """
    A = np.asarray(A, dtype=float)
    if (A.shape[-2:] == (len(tlat), len(tlon))
            and np.allclose(lats, tlat) and np.allclose(lons, tlon)):
        return A

    lead = A.shape[:-2]
    flat = A.reshape((-1,) + A.shape[-2:]) if lead else A[None]
    out = np.empty((flat.shape[0], len(tlat), len(tlon)))
    for lo in range(0, flat.shape[0], chunk):     # interp copies; keep the copies small
        hi = min(lo + chunk, flat.shape[0])
        da = xr.DataArray(flat[lo:hi], dims=("n", "lat", "lon"),
                          coords={"lat": np.asarray(lats), "lon": np.asarray(lons)})
        out[lo:hi] = to_grid(da, np.asarray(tlat), np.asarray(tlon)).values
    return out.reshape(lead + (len(tlat), len(tlon))) if lead else out[0]


def tag_of(dataset, var, window):
    return (f"dcpp_{var}_lead{window}" if dataset == "dcpp"
            else f"monthly_{var}_s{window}")


def load_native(dataset, var, window, box):
    """The existing handle, with the COMMON smoothing box forced in place of the default."""
    if dataset == "dcpp":
        import dcpp_handles as G
        return G.get(lead=window, var=var, smooth=tuple(box))
    if dataset == "monthly":
        import dcpp_decadal_handles as D
        return D.get(int(window), var=var, smooth=tuple(box))
    raise ValueError(f"unknown dataset {dataset!r}")


def build_one(dataset, var, window, box, overwrite=False, dry_run=False):
    tag = tag_of(dataset, var, window)
    npz = os.path.join(OUTROOT, tag + ".npz")
    if dry_run:
        print(f"  would write {npz}")
        return "dry"
    if os.path.exists(npz) and not overwrite:
        print(f"  {tag}: present, skipping (--overwrite to redo)", flush=True)
        return "skipped"

    tlat, tlon = target_grid()
    t0 = time.perf_counter()
    c = load_native(dataset, var, window, box)
    t_load = time.perf_counter() - t0

    F_native, o_native = np.asarray(c.F, float), np.asarray(c.G, float)
    t0 = time.perf_counter()
    F = regrid(F_native, c.lats, c.lons, tlat, tlon)
    o = regrid(o_native, c.lats, c.lons, tlat, tlon)
    t_regrid = time.perf_counter() - t0

    os.makedirs(OUTROOT, exist_ok=True)
    np.savez_compressed(
        npz,
        F=F.astype(np.float32), o=o.astype(np.float32),
        lats=np.asarray(tlat, float), lons=np.asarray(tlon, float),
        members=np.asarray([str(m) for m in getattr(c, "members", [])]),
        model_of=np.asarray([str(m) for m in getattr(c, "model_of", [])]),
    )
    meta = {
        "dataset": dataset, "variable": var,
        "window": {"lead": window} if dataset == "dcpp" else {"start": int(window)},
        "label": getattr(c, "label", None), "units": getattr(c, "units", None),
        "smoothing_box_deg": list(box),
        "smoothing_note": ("common box forced for all variables; Eade et al. (2014) "
                           "specify 15x15 for SAT/TREFHT, so TREFHT departs from Eade"),
        "native_grid": [int(len(c.lats)), int(len(c.lons))],
        "common_grid": [int(len(tlat)), int(len(tlon))],
        "regrid": f"bilinear onto {TARGET_VAR}'s obs grid, after smoothing",
        "obs_were_coarsened": bool(len(c.lats) != len(tlat)),
        "shapes": {"F": list(F.shape), "o": list(o.shape)},
        "N": int(F.shape[0]), "T": int(F.shape[1]),
        "dtype_on_disk": "float32",
        "n_nan_F": int(np.isnan(F).sum()), "n_nan_o": int(np.isnan(o).sum()),
        "timing_s": {"load": round(t_load, 1), "regrid": round(t_regrid, 1)},
    }
    with open(os.path.join(OUTROOT, tag + ".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  {tag}: {tuple(F_native.shape)} -> {tuple(F.shape)}  "
          f"box {box[0]:g}x{box[1]:g}deg  "
          f"({os.path.getsize(npz) / 1e6:.0f} MB, load {t_load:.0f} s, "
          f"regrid {t_regrid:.0f} s)", flush=True)
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vars", nargs="+", default=list(VARS), choices=list(VARS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--leads", nargs="+", default=list(DEFAULT_LEADS))
    ap.add_argument("--starts", nargs="+", default=[str(s) for s in DEFAULT_STARTS])
    ap.add_argument("--box", nargs=2, type=float, default=list(DEFAULT_BOX),
                    metavar=("DLAT", "DLON"),
                    help="the one smoothing box for every variable, in degrees")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        if not os.path.isdir(OUTROOT):
            print("nothing built yet")
            return 0
        for f in sorted(os.listdir(OUTROOT)):
            if f.endswith(".npz"):
                print(f"  {f}  {os.path.getsize(os.path.join(OUTROOT, f)) / 1e6:7.0f} MB")
        return 0

    tlat, tlon = target_grid()
    print(f"target grid : {TARGET_VAR}'s obs grid, {len(tlat)}x{len(tlon)}")
    print(f"smoothing   : {args.box[0]:g} x {args.box[1]:g} deg for every variable")
    print(f"output      : {OUTROOT}\n")

    results = {}
    for dataset in args.datasets:
        windows = args.leads if dataset == "dcpp" else args.starts
        for var in args.vars:
            for w in windows:
                print(f"[{dataset} {var} {w}]", flush=True)
                try:
                    results[(dataset, var, w)] = build_one(
                        dataset, var, w, args.box, args.overwrite, args.dry_run)
                except Exception as e:                                # noqa: BLE001
                    results[(dataset, var, w)] = f"FAILED: {type(e).__name__}: {e}"
                    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

    print("\nsummary")
    for (d, v, w), r in results.items():
        print(f"  {d:<8s} {v:<7s} {str(w):<8s} {r}")
    bad = [k for k, v in results.items() if str(v).startswith("FAILED")]
    print(f"\n{len(results) - len(bad)}/{len(results)} succeeded")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
