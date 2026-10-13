"""Bitwise checks of the core against the thesis code in thesis-old/.

    python scripts/tests/test_jugaad.py      (or pytest)
"""
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "thesis-old")]
import jugaad as J                                                   # noqa: E402
import InfoTheoryMetrics as ITM                                      # noqa: E402
import info_rpc_parallel as IRP                                      # noqa: E402


def cube(N=7, T=40, ny=3, nx=4, seed=0):
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal((T, ny, nx))
    F = 0.7 * sig + rng.standard_normal((N, T, ny, nx))
    F[1, 3, 0, 0] = np.nan                                           # a missing member value
    return F, sig + 0.5 * rng.standard_normal((T, ny, nx))


def datasets(F, o):
    dims = ("n", "time", "lat", "lon")
    f = xr.Dataset({"psl": (dims, F.copy())})
    return f, f.mean("n"), xr.Dataset({"psl": (dims[1:], o)})


def test_maps_is_info_rpc():
    F, o = cube()
    f, s, g = datasets(F, o)
    r = J.maps(F, o, s=s.psl.to_numpy())
    I_gf, I_sf, _, _ = ITM.info_RPC(*datasets(F, o), method="histogram")
    assert np.array_equal(r["I_o"], I_gf) and np.array_equal(r["I_m"], I_sf)


def test_parallel_info_rpc():
    F, o = cube()
    ref = ITM.info_RPC(*datasets(F, o), method="histogram")
    for n_jobs in (1, 2):
        got = IRP.info_RPC(*datasets(F, o), n_jobs=n_jobs)
        assert all(np.array_equal(a, b) for a, b in zip(got, ref)), n_jobs


def test_knuth():
    rng = np.random.default_rng(1)
    for x in (rng.standard_normal(97), rng.integers(0, 5, 60).astype(float), np.ones(10)):
        C = J.hist_counts_all(x, 40)
        assert all(np.array_equal(C[M - 1, :M], np.histogram(x, bins=M)[0]) for M in range(1, 41))
    for n in (30, 120, 500, 3000):                                  # optbins scans M < maxM and
        x = rng.standard_normal(n)                                  # returns the optimum + 1
        new = J.knuth_bins(x, 39)
        assert ITM.optbins(x, 40) == (new + 1 if new > 1 else 1)


def test_loo_and_rho():
    F, o = cube()
    L = J.loo_means(F)
    for n in range(F.shape[0]):
        assert np.allclose(L[n], np.nanmean(np.delete(F, n, 0), 0), equal_nan=True)
    G = np.nan_to_num(F)
    s = G.mean(0)
    rho_o, _, _ = J.rho_measures(G, o, s)
    assert np.allclose(rho_o[1, 2], np.corrcoef(s[:, 1, 2], o[:, 1, 2])[0, 1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
