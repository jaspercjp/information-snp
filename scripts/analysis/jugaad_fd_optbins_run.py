"""Thesis jugaad lambda on DCPP-A: Freedman-Diaconis vs Knuth OptBins, one npz per case.

    python scripts/analysis/jugaad_fd_optbins_run.py SLP 13-60 [--only fd_tiled] [--ny 3]
    python scripts/analysis/jugaad_fd_optbins_run.py SLP cmip5   # the thesis ensemble
    -> $SCRATCH/snp_jugaad_fd_optbins/<VAR>_lead<LEAD>.npz

`lead=cmip5` loads the rebuilt thesis CMIP5 decadal1961 ensemble through calc_RPC cell 3
(`cmip5_thesis_reproduce.arrays`), with s the xarray member mean exactly as the thesis.

Configurations (key -> rule, how o/s axes are sized, which s, alts):

    fd_tiled        FD on the tiled series        s    0.5/1/1.5   the thesis
    fd_tiled_loo    FD on the tiled series        s_-n 1
    knuth_untiled   Knuth on the T distinct values s    0.5/1/1.5   OptBins, main
    knuth_untiled_loo                              s_-n 1
    knuth_tiled80   Knuth on tiled, maxM=80       s    1           as jugaad.ipynb; degenerate
    fixed_B<b>      the same B on every axis      s    1           sensitivity sweep
    fixed_loo_B<b>                                s_-n 1
    null_fd / null_knuth  fd_tiled / knuth_untiled at alt 1 with the obs years permuted
                    (same permutation at every cell), --null draws

plus pooled Pearson |rho(f; o)| and |rho(f; s)|: lambda = |rho| exactly for Gaussian
dependence, so these are what lam_o and lam_m would be if the binning were unbiased and
the dependence linear.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import argparse
import os
import time

import numpy as np

import jugaad_thesis as JT                                           # noqa: E402

OUT = os.path.join(os.environ["SCRATCH"], "snp_jugaad_fd_optbins")
FIXED_B = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64)
KNUTH_MAX = 400


def configs(nulls):
    c = {"fd_tiled": dict(rule="fd", tiled=True, s="full"),
         "fd_tiled_loo": dict(rule="fd", tiled=True, s="loo", alts=(1.0,)),
         "knuth_untiled": dict(rule="knuth", tiled=False, s="full", maxM=KNUTH_MAX),
         "knuth_untiled_loo": dict(rule="knuth", tiled=False, s="loo", maxM=KNUTH_MAX,
                                   alts=(1.0,)),
         "knuth_tiled80": dict(rule="knuth", tiled=True, s="full", maxM=80, alts=(1.0,))}
    for b in FIXED_B:
        c[f"fixed_B{b}"] = dict(rule="fixed", B=b, s="full", alts=(1.0,))
        c[f"fixed_loo_B{b}"] = dict(rule="fixed", B=b, s="loo", alts=(1.0,))
    for d in range(nulls):
        c[f"null_fd_{d}"] = dict(rule="fd", tiled=True, s="full", alts=(1.0,), perm=d)
        c[f"null_knuth_{d}"] = dict(rule="knuth", tiled=False, s="full", maxM=KNUTH_MAX,
                                    alts=(1.0,), perm=d)
    return c


def pooled_rho(F, Y):
    """|corr| per cell between pooled f (N*T) and a pooled/tiled Y of the same layout."""
    X = np.where(np.isnan(F), 0.0, F).reshape(-1, *F.shape[2:])
    Y = Y.reshape(-1, *F.shape[2:])
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs((Xc * Yc).mean(0) / (Xc.std(0) * Yc.std(0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("var")
    ap.add_argument("lead")
    ap.add_argument("--only", nargs="*", help="config keys (default: all)")
    ap.add_argument("--ny", type=int, help="first ny rows only (timing tests)")
    ap.add_argument("--null", type=int, default=0, help="permutation-null draws")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    t0 = time.time()
    if a.lead == "cmip5":
        import cmip5_thesis_reproduce
        F, o, s_full, lats, lons = cmip5_thesis_reproduce.arrays()
    else:
        import dcpp_handles
        c = dcpp_handles.get(lead=a.lead, var=a.var, verbose=False,
                             remove_gm=a.var.upper() == "SLP")
        F, o = np.asarray(c.F, float), np.asarray(c.G, float)
        lats, lons = np.asarray(c.lats), np.asarray(c.lons)
        s_full = np.nanmean(F, axis=0)
    if a.ny:
        F, o, s_full, lats = F[:, :, :a.ny], o[:, :a.ny], s_full[:, :a.ny], lats[:a.ny]
    N, T = F.shape[:2]
    print(f"{a.var} lead {a.lead}: N={N} T={T} grid={F.shape[2:]} "
          f"(load {time.time() - t0:.0f}s)", flush=True)

    res = dict(lats=lats, lons=lons, N=N, T=T, fixed_B=np.array(FIXED_B),
               rho_o=pooled_rho(F, np.broadcast_to(o, F.shape)),
               rho_m=pooled_rho(F, np.broadcast_to(s_full, F.shape)),
               rho_m_loo=pooled_rho(F, JT.loo_means(F)))
    rng = np.random.default_rng(12345)
    perms = [rng.permutation(T) for _ in range(a.null)]
    todo = configs(a.null)
    for key, kw in todo.items():
        if a.only and not any(key.startswith(k) for k in a.only):
            continue
        kw = dict(kw)
        oo = o[perms[kw.pop("perm")]] if "perm" in kw else o
        if kw["s"] == "full":
            kw["s"] = s_full
        t1 = time.time()
        r = JT.jugaad_maps(F, oo, **kw)
        for k in ("I_o", "I_m", "lam_o", "lam_m", "lam", "bins", "alts"):
            res[f"{key}__{k}"] = r[k]
        if "cap_hit" in r:
            res[f"{key}__cap_hit"] = r["cap_hit"]
        b = r["bins"]
        print(f"  {key:<20} {time.time() - t1:6.1f}s  bins f/o/s median "
              f"{int(np.median(b[0]))}/{int(np.median(b[1]))}/{int(np.median(b[2]))}"
              f"  lam_o {np.nanmean(r['lam_o'][-2 if len(r['alts']) == 3 else 0]):.3f}",
              flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{a.var}_lead{a.lead}{a.tag}.npz")
    np.savez_compressed(path, **res)
    print(f"-> {path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
