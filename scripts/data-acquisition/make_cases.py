"""Write the per-case input arrays the analysis reads: data/cases/<VAR>_<case>.npz.

    python make_cases.py                       # all nine cases
    python make_cases.py SLP s1961 PRECT 2-4   # (VAR, case) pairs

Each file (uncompressed np.savez) holds
    F (N, T, ny, nx) float64   members
    o (T, ny, nx)    float64   observations
    s (T, ny, nx)    float64   ensemble mean, np.nanmean(F, axis=0) (cmip5: xarray's)
    lats (ny,), lons (nx,)

Cases:
    cmip5      the thesis CMIP5 decadal1961 SLP ensemble through calc_RPC cell 3
               (`cmip5_thesis.arrays`), T = 120 months, 64 x 128
    sYYYY      one DCPP-A decade, monthly (`dcpp_decadal_handles`, six models, Eade
               smoothing), thesis preprocessing: raw fields, a least-squares line
               (intercept AND slope) removed per member and cell (`thesis_detrend`),
               seasonal cycle kept
    sYYYYmoy   the same after removing the month-of-year climatology per model and from
               the obs, before `thesis_detrend`
    2-4        DJF means, one per start date, pooled over five November models
               (`dcpp_handles`, Eade smoothing, detrended per model over the common
               start dates; SLP also has the area-weighted global mean removed)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import CASES                                             # noqa: E402

ALL = [("SLP", "cmip5"), ("SLP", "s1961"), ("TREFHT", "s1961"), ("SLP", "s1961moy"),
       ("TREFHT", "s1961moy"), ("PRECT", "s1978"), ("SLP", "2-4"), ("PRECT", "2-4"),
       ("TREFHT", "2-4")]


def thesis_detrend(X):
    """Subtract a per-series least-squares line (intercept AND slope) along axis -3, NaN-aware.

    What calc_RPC cell 3 does with `polyfit(dim="time", deg=1)`, per member and cell.
    """
    X = np.asarray(X, dtype=float)          # read only, so no copy is needed
    T = X.shape[-3]
    t = np.arange(T, dtype=float).reshape((T, 1, 1))
    ok = ~np.isnan(X)
    n = ok.sum(-3, keepdims=True)
    tm = np.where(ok, t, 0).sum(-3, keepdims=True) / n
    xm = np.where(ok, X, 0).sum(-3, keepdims=True) / n
    dt = np.where(ok, t - tm, 0)
    slope = (dt * np.where(ok, X - xm, 0)).sum(-3, keepdims=True) / (dt ** 2).sum(-3, keepdims=True)
    return X - xm - slope * (t - tm)


def load(var, case):
    """(F, o, s, lats, lons) for one case."""
    if case == "cmip5":
        import cmip5_thesis
        return cmip5_thesis.arrays()
    if case.startswith("s"):
        import dcpp_decadal_handles
        c = dcpp_decadal_handles.get(int(case[1:5]), var=var, verbose=False,
                                     anom="moy" if case.endswith("moy") else "none")
        lats, lons = np.asarray(c.lats), np.asarray(c.lons)
        F, G = c.F, c.G
        del c                                     # so the raw F is freed by the rebind
        F = thesis_detrend(F)
        o = thesis_detrend(np.asarray(G, float)[None])[0]
        return F, o, np.nanmean(F, axis=0), lats, lons
    import dcpp_handles
    c = dcpp_handles.get(lead=case, var=var, remove_gm=var.upper() == "SLP")
    F, o = np.asarray(c.F, float), np.asarray(c.G, float)
    return F, o, np.nanmean(F, axis=0), np.asarray(c.lats), np.asarray(c.lons)


def write(var, case):
    t0 = time.time()
    F, o, s, lats, lons = load(var, case)
    os.makedirs(CASES, exist_ok=True)
    path = os.path.join(CASES, f"{var}_{case}.npz")
    np.savez(path, F=F, o=o, s=s, lats=lats, lons=lons)
    print(f"{path}  F{F.shape} o{o.shape}  {os.path.getsize(path) / 2**20:.0f} MB  "
          f"({time.time() - t0:.0f}s)", flush=True)


def main(argv):
    if argv[:1] in (["-h"], ["--help"]):
        return print(__doc__)
    if len(argv) % 2:
        raise SystemExit(__doc__)
    todo = list(zip(argv[::2], argv[1::2])) or ALL
    for var, case in todo:
        write(var, case)


if __name__ == "__main__":
    main(sys.argv[1:])
