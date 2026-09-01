"""Compute and store the pairwise KSG mutual information for every DCPP cube and variable.

    python compute_and_store_pairwise_MI.py                  # all of it, defaults below
    python compute_and_store_pairwise_MI.py --dry-run        # what it would do, no compute
    python compute_and_store_pairwise_MI.py --vars SLP --datasets monthly
    python compute_and_store_pairwise_MI.py --leads 2-4 13-60 --starts 1961 1970
    sbatch compute_and_store_pairwise_MI.sbatch              # the whole sweep, 16 cores

For each (dataset, variable, window) it runs exactly

    MI_FG_pairwise = mi_vectorized.mi_member_vs_obs(F, o, k=4, copula=True)   # (N, lat, lon)
    MI_F_pairwise  = mi_vectorized.mi_member_vs_member(F, k=4, copula=True)   # (N, N, lat, lon)

and drops both, plus a `meta.json` describing exactly what they are, into its own
subfolder of `data/pairwise_MI/`.

The two cube kinds
------------------
`dcpp_handles` and `dcpp_decadal_handles` both present `(N, T, lat, lon)`, but T means
different things and neither has a single natural window, so each needs a coordinate:

    dcpp         lead window   T = 48-60 start dates, one lead-window mean each
    monthly      start year    T = 120 months within ONE hindcast

Defaults are the two the notebooks use -- lead `13-60` (yr 2-5) and start `1961`. Pass
`--leads` / `--starts` for more; every combination becomes its own subfolder.

Layout
------
    data/pairwise_MI/dcpp_SLP_lead13-60/MI_FG_pairwise.npy      (N, lat, lon)
                                       /MI_F_pairwise.npy       (N, N, lat, lon)
                                       /meta.json
    data/pairwise_MI/monthly_SLP_s1961/...

NATS, k=4, copula-transformed, dither 1e-10 at seed 0 -- the convention of
`smyle_metrics.calc_MI_sG(..., use_copula=True)`, so `smyle_metrics.lam_of` applies
directly to what is stored. `meta.json` records all of it; do not assume, read it.

Two things about the stored arrays
----------------------------------
`MI_F_pairwise` has `+inf` on its diagonal -- I(f_i; f_i) is infinite, not 1. `lam_of`
maps that to 1.0, but a raw `.mean()` over the member axis will not survive it. Mask the
diagonal or use `lam_of` first.

Cells that are NaN anywhere in their series come back NaN rather than a wrong number.
`meta.json` records how many, so a silently-masked variable is visible without reloading.

Cost, and why the monthly cubes dominate
----------------------------------------
The estimator is O(T^2) per problem, so the 120-month cubes cost 6x per problem what the
48-start ones do, and they have more members to pair up. Measured on a serc node at 16
threads: the grand ensemble (N=101, T=48) takes ~42 s per variable, the s1961 monthly cube
(N=92, T=120) ~3.5 min. The whole default sweep is ~15 min. Thread scaling saturates near
6x -- the kernel is memory-bandwidth-bound -- so asking for 64 cores buys little over 16.

Variables that are not available for a window (GPCP only starts in 1979, so PRECT has no
early start dates) raise inside the handle. Each combination is caught independently and
reported at the end; one missing variable does not lose the rest of the sweep.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
DATA = os.path.join(ROOT, "data")
OUTROOT = os.path.join(DATA, "pairwise_MI")
sys.path.insert(0, _HERE)

import mi_vectorized as V                                            # noqa: E402

VARS = ("SLP", "TREFHT", "PRECT")
DATASETS = ("dcpp", "monthly")
DEFAULT_LEADS = ("13-60",)
DEFAULT_STARTS = (1961,)

K_NN = 4
COPULA = True
NOISE_LEVEL = 1e-10
SEED = 0


def _obs_of(cube):
    """Observations off either handle. Both call it `G`; `o` is accepted for old code."""
    for name in ("G", "o"):
        if hasattr(cube, name):
            return getattr(cube, name)
    raise AttributeError("cube exposes neither .G nor .o for observations")


def tag_of(dataset, var, window, grid="native"):
    """The subfolder name. Carries the window, so two leads never overwrite each other,
    and the grid, so native and common-grid results never overwrite each other either."""
    base = (f"dcpp_{var}_lead{window}" if dataset == "dcpp"
            else f"monthly_{var}_s{window}")
    return base if grid == "native" else f"{base}_commongrid"


def load(dataset, var, window, grid="native"):
    """`window` is a lead string for the dcpp cube, a start year for the monthly one.

    grid="native" is each variable on its own obs grid with Eade's per-variable smoothing
    box. grid="common" is the re-gridded cubes from `build_common_grid_cubes.py`: every
    variable on SLP's grid, one smoothing box, so cells are comparable across variables.
    """
    if grid == "common":
        import common_grid_handles as CG
        return CG.get(dataset, var, window)
    if dataset == "dcpp":
        import dcpp_handles as G
        return G.get(lead=window, var=var)
    if dataset == "monthly":
        import dcpp_decadal_handles as D
        return D.get(int(window), var=var)
    raise ValueError(f"unknown dataset {dataset!r}")


def run_one(dataset, var, window, n_jobs, overwrite, dry_run, grid="native"):
    tag = tag_of(dataset, var, window, grid)
    outdir = os.path.join(OUTROOT, tag)
    if dry_run:
        print(f"  would write {outdir}")
        return "dry"

    fg_path = os.path.join(outdir, "MI_FG_pairwise.npy")
    f_path = os.path.join(outdir, "MI_F_pairwise.npy")
    if not overwrite and os.path.exists(fg_path) and os.path.exists(f_path):
        print(f"  {tag}: already present, skipping (--overwrite to redo)", flush=True)
        return "skipped"

    t_load = time.perf_counter()
    cube = load(dataset, var, window, grid)
    F = np.asarray(cube.F, dtype=float)
    o = np.asarray(_obs_of(cube), dtype=float)
    t_load = time.perf_counter() - t_load

    N, T = F.shape[:2]
    n_pairs = N * (N - 1) // 2
    n_cells = int(np.prod(F.shape[2:]))
    print(f"  {tag}: F{F.shape} o{o.shape}  N={N} T={T}  "
          f"{n_pairs * n_cells:,} member-pair problems  (load {t_load:.1f} s)", flush=True)

    t0 = time.perf_counter()
    MI_FG_pairwise = V.mi_member_vs_obs(F, o, k=K_NN, copula=COPULA,
                                        noise_level=NOISE_LEVEL, seed=SEED,
                                        n_jobs=n_jobs)
    t_fg = time.perf_counter() - t0

    t0 = time.perf_counter()
    MI_F_pairwise = V.mi_member_vs_member(F, k=K_NN, copula=COPULA,
                                          noise_level=NOISE_LEVEL, seed=SEED,
                                          n_jobs=n_jobs)
    t_f = time.perf_counter() - t0

    os.makedirs(outdir, exist_ok=True)
    np.save(fg_path, MI_FG_pairwise)
    np.save(f_path, MI_F_pairwise)

    off = ~np.eye(N, dtype=bool)
    meta = {
        "dataset": dataset,
        "variable": var,
        "grid": grid,
        "smoothing_box_deg": list(getattr(cube, "smooth_box_deg", ()) or ()),
        "window": {"lead": window} if dataset == "dcpp" else {"start": int(window)},
        "label": getattr(cube, "label", None),
        "units": getattr(cube, "units", None),
        "estimator": {
            "name": "KSG-1 (infomeasure approach='metric')",
            "implementation": "mi_vectorized",
            "k": K_NN,
            "copula": COPULA,
            "noise_level": NOISE_LEVEL,
            "seed": SEED,
            "units": "nats",
            "lam": "smyle_metrics.lam_of applies directly",
        },
        "shapes": {
            "F": list(F.shape),
            "obs": list(o.shape),
            "MI_FG_pairwise": list(MI_FG_pairwise.shape),
            "MI_F_pairwise": list(MI_F_pairwise.shape),
        },
        "N": int(N),
        "T": int(T),
        "members": [str(m) for m in getattr(cube, "members", [])],
        "model_of": [str(m) for m in getattr(cube, "model_of", [])],
        "by_model": {k: [int(v.start), int(v.stop)]
                     for k, v in getattr(cube, "by_model", {}).items()},
        "lats": np.asarray(cube.lats).tolist(),
        "lons": np.asarray(cube.lons).tolist(),
        "diagnostics": {
            "MI_F_diagonal": "+inf (self-information); mask it or use lam_of",
            "n_nan_cells_MI_FG": int(np.isnan(MI_FG_pairwise).sum()),
            "n_nan_cells_MI_F_offdiag": int(np.isnan(MI_F_pairwise[off]).sum()),
            "MI_F_offdiag_min": float(np.nanmin(MI_F_pairwise[off])),
            "MI_F_offdiag_max": float(np.nanmax(MI_F_pairwise[off])),
            "MI_F_offdiag_mean": float(np.nanmean(MI_F_pairwise[off])),
            "MI_F_offdiag_frac_negative": float(np.nanmean(MI_F_pairwise[off] < 0)),
            "MI_FG_min": float(np.nanmin(MI_FG_pairwise)),
            "MI_FG_max": float(np.nanmax(MI_FG_pairwise)),
            "MI_FG_mean": float(np.nanmean(MI_FG_pairwise)),
        },
        "timing_s": {"load": round(t_load, 2),
                     "MI_FG_pairwise": round(t_fg, 2),
                     "MI_F_pairwise": round(t_f, 2)},
        "n_jobs": n_jobs,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    mb = (MI_FG_pairwise.nbytes + MI_F_pairwise.nbytes) / 1e6
    print(f"    -> {outdir}  ({mb:.0f} MB)  "
          f"MI_FG {t_fg:.1f} s, MI_F {t_f:.1f} s, "
          f"off-diag mean {meta['diagnostics']['MI_F_offdiag_mean']:+.4f} nats",
          flush=True)
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vars", nargs="+", default=list(VARS), choices=list(VARS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--leads", nargs="+", default=list(DEFAULT_LEADS),
                    help="lead windows for the dcpp cube, e.g. 13-60 2-4")
    ap.add_argument("--starts", nargs="+", default=[str(s) for s in DEFAULT_STARTS],
                    help="start years for the monthly decadal cube, e.g. 1961 1970")
    ap.add_argument("--n-jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--grid", choices=("native", "common"), default="native",
                    help="native: each variable on its own obs grid with Eade's "
                         "per-variable box. common: build_common_grid_cubes.py output, "
                         "every variable on SLP's grid with one box.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"output root : {OUTROOT}")
    print(f"estimator   : KSG-1 k={K_NN}, copula={COPULA}, "
          f"noise={NOISE_LEVEL}, seed={SEED}, nats")
    print(f"grid        : {args.grid}")
    print(f"n_jobs      : {args.n_jobs}\n")
    os.makedirs(OUTROOT, exist_ok=True)

    jobs = []
    for dataset in args.datasets:
        windows = args.leads if dataset == "dcpp" else args.starts
        for var in args.vars:
            for w in windows:
                jobs.append((dataset, var, w))

    results, t_all = {}, time.perf_counter()
    for dataset, var, w in jobs:
        print(f"[{dataset} {var} {w}]", flush=True)
        try:
            results[(dataset, var, w)] = run_one(dataset, var, w, args.n_jobs,
                                                 args.overwrite, args.dry_run,
                                                 args.grid)
        except Exception as e:                                        # noqa: BLE001
            results[(dataset, var, w)] = f"FAILED: {type(e).__name__}: {e}"
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    print(f"\n{'=' * 70}\nsummary ({time.perf_counter() - t_all:.0f} s total)")
    for (dataset, var, w), r in results.items():
        print(f"  {dataset:<8s} {var:<7s} {str(w):<8s} {r}")
    bad = [k for k, v in results.items() if str(v).startswith("FAILED")]
    print(f"\n{len(results) - len(bad)}/{len(results)} succeeded")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
