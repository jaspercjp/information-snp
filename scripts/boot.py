"""Eade et al. (2014) bootstrap of the RPC maps; method in figures/notes/BOOTSTRAP.md.

    python scripts/boot.py SLP 2-4 fd [--boot 1000]     -> $SNP_OUT/boot_fd_SLP_2-4.npz

Draws resample time (whole years if monthly) with replacement and N-3 members without, one
seeded stream per case, so every mode sees the same draws. Modes:
    fd      rho RPC and the thesis FD lambda RPC, FD bins re-chosen in every draw
    knuth   Knuth lambda (T distinct values, untiled), bins re-chosen in every draw
    lmloo   Knuth lambda with lambda_m = lambda(I(f; s_-n)), s_-n sized on its T centres; each
            draw keeps the full-data bins (Knuth on exactly repeated years resolves the repeats)
    olmloo  lmloo with lambda_o = lambda(I(s_-n; o)), s_-n sized the same way
"""
import argparse
import time

import numpy as np

import jugaad as J
from data import load, out

KN = dict(rule="knuth", tiled=False)
LAM = {"knuth": KN, "lmloo": dict(KN, s="loo", loo_size="T"),
       "olmloo": dict(KN, s="loo", loo_size="T", o_vs="loo")}


def draws(var, case, N, T, n, seed=2014):
    """(members, times) index pairs for n draws; F[mi, ti] is one resampled ensemble."""
    rng = np.random.default_rng([seed, sum(map(ord, var + case))])
    for _ in range(n):
        if case == "cmip5" or case.startswith("s"):      # monthly: whole years
            years = T // 12
            ti = (rng.integers(0, years, years)[:, None] * 12 + np.arange(12)).ravel()
        else:
            ti = rng.integers(0, T, T)
        yield rng.choice(N, N - 3, replace=False)[:, None], ti


def fd_stats(F, o):
    """(RPC_rho, RPC_lam with FD bins at alt 1, rho_o) for one ensemble."""
    s = np.nanmean(F, axis=0)
    rho_o, rho_m, _ = J.rho_measures(F, o, s)
    with np.errstate(invalid="ignore", divide="ignore"):
        return rho_o / rho_m, J.maps(F, o, s=s, alts=(1.0,))["lam"][0], rho_o


def main(var, case, mode, boot, seed):
    t0 = time.time()
    F, o, _, lats, lons = load(var, case)
    N, T = F.shape[:2]
    res = (dict(zip(("rpc_rho", "rpc_lam", "rho_o"), fd_stats(F, o))) if mode == "fd"
           else J.maps(F, o, **LAM[mode]))                          # lambda: point + alt 0.5-1.5 band
    B = np.full((2 if mode == "fd" else 1, boot) + F.shape[2:], np.nan, np.float32)
    for b, (mi, ti) in enumerate(draws(var, case, N, T, boot, seed)):
        if mode == "fd":
            B[0, b], B[1, b], _ = fd_stats(F[mi, ti], o[ti])
        else:
            fix = None if mode == "knuth" else res["bins"]             # lmloo, olmloo: full-data bins
            B[0, b] = J.maps(F[mi, ti], o[ti], bins=fix, alts=(1.0,), **LAM[mode])["lam"][0]
        if (b + 1) % 100 == 0:
            print(f"  draw {b + 1}/{boot}  {(time.time() - t0) / (b + 1):.1f} s/draw", flush=True)
    res.update(B_lam=B[-1], **({"B_rho": B[0]} if mode == "fd" else {}))
    np.savez_compressed(out(f"boot_{mode}", var, case), lats=lats, lons=lons, N=N, T=T, **res)
    print(f"-> {out(f'boot_{mode}', var, case)}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a, kw in (("var", {}), ("case", {}), ("mode", dict(choices=("fd", *LAM))),
                  ("--boot", dict(type=int, default=1000)), ("--seed", dict(type=int, default=2014))):
        ap.add_argument(a, **kw)
    main(**vars(ap.parse_args()))
