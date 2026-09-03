"""Full-scale timing for mi_vectorized: the real DCPP shapes, and thread scaling."""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mi_vectorized as V  # noqa: E402


def timeit(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return time.perf_counter() - t0, out


def main():
    rng = np.random.default_rng(0)
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    print(f"cores available: {ncpu}\n")

    # throughput per core, both cube kinds, on a batch big enough to be representative
    print("kernel throughput, single thread")
    for T in (48, 60, 120):
        B = 200_000 if T == 48 else (150_000 if T == 60 else 40_000)
        X = rng.standard_normal((B, T))
        Y = rng.standard_normal((B, T))
        dt, _ = timeit(V.mi_batch, X, Y, k=4, copula=False, noise_level=0)
        print(f"  T={T:<4d} {B:>7d} problems in {dt:6.2f} s "
              f"-> {B / dt / 1e3:7.1f} k problems/s/core", flush=True)

    # thread scaling on one representative block
    print("\nthread scaling (T=48, 400k problems)")
    B, T = 400_000, 48
    X = rng.standard_normal((B, T))
    Y = rng.standard_normal((B, T))
    base = None
    for nj in (1, 2, 4, 8, 16, 32):
        if nj > ncpu:
            break
        dt, _ = timeit(V.mi_batch, X, Y, k=4, copula=False, noise_level=0,
                       n_jobs=nj)
        base = base or dt
        print(f"  n_jobs={nj:<3d} {dt:6.2f} s   speedup {base / dt:5.2f}x", flush=True)

    # the real thing: DCPP grand ensemble, all member pairs, whole grid
    print("\nfull DCPP grand ensemble: N=101, T=48, 37x72 grid")
    N, T, LA, LO = 101, 48, 37, 72
    F = rng.standard_normal((N, T, LA, LO))
    n_prob = N * (N - 1) // 2 * LA * LO
    print(f"  {n_prob:,} (pair, cell) problems")
    dt, out = timeit(V.mi_member_vs_member, F, k=4, n_jobs=ncpu)
    print(f"  n_jobs={ncpu}: {dt:.1f} s wall  ({dt * ncpu / 60:.1f} core-min), "
          f"out {out.shape}, {np.isfinite(out).sum():,} finite entries")

    # the decadal cube kind, same grid, T=120 months, fewer members
    print("\ndecadal monthly cube: N=46, T=120, 37x72 grid")
    N, T = 46, 120
    F = rng.standard_normal((N, T, LA, LO))
    n_prob = N * (N - 1) // 2 * LA * LO
    print(f"  {n_prob:,} (pair, cell) problems")
    dt, out = timeit(V.mi_member_vs_member, F, k=4, n_jobs=ncpu)
    print(f"  n_jobs={ncpu}: {dt:.1f} s wall  ({dt * ncpu / 60:.1f} core-min), "
          f"out {out.shape}")


if __name__ == "__main__":
    main()
