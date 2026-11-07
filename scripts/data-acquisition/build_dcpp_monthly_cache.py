"""Build data/dcpp_monthly_cache/: one decade of raw monthly DCPP-A members per model.

    python build_dcpp_monthly_cache.py                       # the cases' decades
    python build_dcpp_monthly_cache.py --var SLP --start 1961 [--models MIROC6]

Writes, per (model, variable, start):

    data/dcpp_monthly_cache/<MODEL>/<VAR>_s<YYYY>_<YYYY+1>-01_120mo.npz
        cube    (N, 120, lat, lon) float32, on the observations' grid, raw (no anomaly,
                no smoothing); PRECT in mm/day
        members (N,), months (120,) absolute month index, lat, lon

Reads the raw monthly files indexed by data/dcpp_files.json ($SCRATCH/dcpp/...) and
data/depresys4_files.json (data/depresys4/...); every member that covers Jan(YYYY+1) ..
Dec(YYYY+10) is used. `dcpp_decadal_handles.get` builds the same file lazily on a cache
miss; this script only does it up front. Existing files are kept unless `--rebuild`.
"""
import argparse

import dcpp_decadal_handles as D
from obs_grid import obs_grid
from paths import MONTHLY_CACHE

# (var, start) of the monthly cases in make_cases.py
DEFAULT = [("SLP", 1961), ("TREFHT", 1961), ("PRECT", 1978)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--var", choices=list(D.CMIP_NAME))
    ap.add_argument("--start", type=int)
    ap.add_argument("--models", nargs="+", default=D.MODELS)
    ap.add_argument("--outdir", default=MONTHLY_CACHE)
    ap.add_argument("--rebuild", action="store_true",
                    help="re-read the raw files even if the cache file exists")
    a = ap.parse_args()
    todo = DEFAULT if a.var is None else [(a.var, a.start or 1961)]
    for var, start in todo:
        lat, lon = obs_grid(var)
        want = D.canonical_window(start)
        for m in a.models:
            cube, mems = D.model_cube(m, var, start, want, lat, lon, cache_dir=a.outdir,
                                      rebuild=a.rebuild)
            print(f"{D._cache_path(m, var, start, want, a.outdir)}  {cube.shape}",
                  flush=True)


if __name__ == "__main__":
    main()
