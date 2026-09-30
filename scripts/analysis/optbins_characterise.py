"""What `InfoTheoryMetrics.optbins` returns vs Knuth's optimum, and what tiling does to both.

    python scripts/analysis/optbins_characterise.py [maxM]

Gaussian series of T values, tiled N times as the jugaad layout does. Prints, per (T, N):
the original optbins(x, 80) (as jugaad.ipynb calls it), the corrected Knuth optimum over
1..79 and over 1..maxM, and FD (thesis rule) -- medians and ranges over 20 draws.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import types

import numpy as np

for _name, _attr, _obj in (("tqdm", "tqdm", lambda it, **kw: it),
                           ("sklearn.feature_selection", "mutual_info_regression", None)):
    try:
        __import__(_name)
    except ImportError:
        _sys.modules.setdefault(_name.split(".")[0], types.ModuleType(_name.split(".")[0]))
        setattr(_sys.modules.setdefault(_name, types.ModuleType(_name)), _attr, _obj)

import InfoTheoryMetrics as ITM                                      # noqa: E402
import jugaad_thesis as JT                                           # noqa: E402


def fmt(a):
    a = np.asarray(a)
    return f"{int(np.median(a)):>4} [{a.min():>3}-{a.max():>3}]"


def main(maxM=1000, draws=20):
    rng = np.random.default_rng(0)
    print(f"{'T':>4} {'N':>3} {'N*T':>6} | {'optbins(x,80)':>15} {'knuth<=79':>15} "
          f"{'knuth<=' + str(maxM):>15} {'FD thesis':>15}")
    for T in (51, 52, 120):
        for N in (1, 10, 48, 62):
            old, k79, kbig, fd = [], [], [], []
            for _ in range(draws):
                x = np.tile(rng.standard_normal(T), N)
                old.append(ITM.optbins(x, 80))
                k79.append(JT.knuth_bins(x, 79))
                kbig.append(JT.knuth_bins(x, maxM))
                fd.append(JT.fd_bins(x))
            print(f"{T:>4} {N:>3} {N * T:>6} | {fmt(old):>15} {fmt(k79):>15} "
                  f"{fmt(kbig):>15} {fmt(fd):>15}")
    # the pooled member series: N*T distinct values, i.e. what f_jugaad looks like
    print("\nuntiled N*T distinct Gaussian values (f_jugaad-like):")
    for M in (51 * 62, 120 * 48):
        k = [JT.knuth_bins(rng.standard_normal(M), maxM) for _ in range(draws)]
        f = [JT.fd_bins(rng.standard_normal(M)) for _ in range(draws)]
        print(f"  M={M:>5}: knuth {fmt(k)}  FD {fmt(f)}")


if __name__ == "__main__":
    main(int(_sys.argv[1]) if len(_sys.argv) > 1 else 1000)
