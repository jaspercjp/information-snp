"""Checks `jugaad_thesis` against the code it reproduces: `InfoTheoryMetrics.info_RPC`.

    python scripts/tests/test_jugaad_thesis.py        # ~1 min

The claim is bitwise equality of I_o and I_m (and so of lam_o, lam_m, lam) on a synthetic
cube that exercises what could silently differ: NaNs in the members (thesis: NaN -> 0 in f,
but s is the NaN-skipping mean), skewed and heavy-tailed marginals, very different
per-cell FD bin counts, and all three `alt` scalings. The second block pins down what
`InfoTheoryMetrics.optbins` actually returns relative to Knuth's optimum.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import types

import numpy as np
import xarray as xr

# SNP_env has neither tqdm nor sklearn. InfoTheoryMetrics imports both, but on the histogram
# path tqdm is only a progress bar and sklearn only serves the unused Kraskov branch.
for _name, _attr, _obj in (("tqdm", "tqdm", lambda it, **kw: it),
                           ("sklearn.feature_selection", "mutual_info_regression", None)):
    try:
        __import__(_name)
    except ImportError:
        _sys.modules.setdefault(_name.split(".")[0], types.ModuleType(_name.split(".")[0]))
        _m = _sys.modules.setdefault(_name, types.ModuleType(_name))
        setattr(_m, _attr, _obj)

import InfoTheoryMetrics as ITM                                      # noqa: E402
import jugaad_thesis as JT                                           # noqa: E402


def synthetic_cube(N=7, T=40, ny=4, nx=5, seed=0):
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal((T, ny, nx))
    amp = rng.uniform(0.0, 1.5, (ny, nx))                           # predictability varies
    F = amp * sig + rng.standard_normal((N, T, ny, nx))
    F[..., 0, :] = np.exp(F[..., 0, :])                              # skewed row
    F[..., 1, :] = rng.standard_t(2, (N, T, nx))                     # heavy tails, no signal
    o = 0.7 * amp * sig + rng.standard_normal((T, ny, nx))
    F[rng.random(F.shape) < 0.03] = np.nan                           # scattered member NaNs
    F[2, :5, 3, 4] = np.nan
    return F, o


def as_datasets(F, o):
    N, T, ny, nx = F.shape
    coords = dict(n=np.arange(N), time=np.arange(T), lat=np.arange(ny), lon=np.arange(nx))
    f = xr.Dataset({"psl": (("n", "time", "lat", "lon"), F.copy())}, coords=coords)
    obs = xr.Dataset({"psl": (("time", "lat", "lon"), o.copy())},
                     coords={k: coords[k] for k in ("time", "lat", "lon")})
    return f, f.mean("n"), obs                                       # s exactly as calc_RPC


def test_info_rpc_bitwise():
    F, o = synthetic_cube()
    f, s, obs = as_datasets(F, o)
    I_fg, I_fs, _, _ = ITM.info_RPC(f, s, obs, method="histogram", quant_uncert=True)
    for n_jobs in (1, 4):
        r = JT.jugaad_maps(F, o, s="full", rule="fd", n_jobs=n_jobs)
        assert np.array_equal(r["I_o"], I_fg), np.abs(r["I_o"] - I_fg).max()
        assert np.array_equal(r["I_m"], I_fs), np.abs(r["I_m"] - I_fs).max()
        lam_o = np.sqrt(1 - np.power(2, -2 * I_fg))                  # calc_RPC cell 13
        lam_m = np.sqrt(1 - np.power(2, -2 * I_fs))
        assert np.array_equal(r["lam"], lam_o / lam_m)
    b = r["bins"]
    print(f"  info_RPC bitwise: OK  (FD bins f/o/s range {b[0].min()}-{b[0].max()}, "
          f"{b[1].min()}-{b[1].max()}, {b[2].min()}-{b[2].max()})")


def test_explicit_s_matches_full():
    F, o = synthetic_cube(seed=1)
    a = JT.jugaad_maps(F, o, s="full", n_jobs=1)
    b = JT.jugaad_maps(F, o, s=np.nanmean(F, axis=0), n_jobs=1)
    assert np.array_equal(a["I_m"], b["I_m"])
    print("  explicit s == 'full': OK")


def test_loo_means():
    F, _ = synthetic_cube(seed=2)
    L = JT.loo_means(F)
    for n in range(F.shape[0]):
        ref = np.nanmean(np.delete(F, n, axis=0), axis=0)
        assert np.allclose(L[n], ref, equal_nan=True)
    print("  loo_means == nanmean without member n: OK")


def test_optbins_is_knuth_plus_one():
    rng = np.random.default_rng(3)
    maxM = 80
    seen = {"shift": 0, "one": 0}
    for n in (30, 120, 500, 3000):
        for _ in range(10):
            x = rng.standard_normal(n)
            old = ITM.optbins(x, maxM)                               # evaluates M = 1..maxM-1
            new = JT.knuth_bins(x, maxM - 1)
            if new == 1:
                assert old == 1, (old, new)
                seen["one"] += 1
            else:
                assert old == new + 1, (n, old, new)
                seen["shift"] += 1
    # logp(M=1) is exactly 0, so the never-set logp[0] = 0 placeholder is harmless
    assert abs(JT.knuth_logpost(rng.standard_normal(50), 3)[0]) < 1e-9
    print(f"  optbins == knuth + 1 when knuth > 1 ({seen['shift']} cases), == 1 when knuth "
          f"== 1 ({seen['one']} cases): OK")


def test_hist_counts_all_matches_numpy():
    from scipy.special import gammaln
    rng = np.random.default_rng(4)
    cases = {"gauss": rng.standard_normal(3000),
             "tiled": np.tile(rng.standard_normal(52), 62),
             "integers": rng.integers(0, 7, 500).astype(float),
             "exp": np.exp(3 * rng.standard_normal(1000)),
             "constant": np.full(40, 2.5),
             "two": np.array([0.0, 1.0])}
    maxM = 150
    for name, x in cases.items():
        C = JT.hist_counts_all(x, maxM)
        for M in range(1, maxM + 1):
            ref, _ = np.histogram(x, bins=M)
            assert np.array_equal(C[M - 1, :M], ref), (name, M)
            assert not C[M - 1, M:].any(), (name, M)
        n = len(x)
        loop = np.array([n * np.log(M) + gammaln(M / 2) - gammaln(n + M / 2)
                         - M * gammaln(0.5) + gammaln(np.histogram(x, bins=M)[0] + 0.5).sum()
                         for M in range(1, maxM + 1)])
        assert np.allclose(JT.knuth_logpost(x, maxM), loop, rtol=0, atol=1e-6), name
    print(f"  hist_counts_all == np.histogram for M=1..{maxM} on {len(cases)} series; "
          "knuth_logpost == loop: OK")


def test_permuted_obs_reuse_bins():
    F, o = synthetic_cube(seed=5)
    perm = np.random.default_rng(6).permutation(o.shape[0])
    for kw in (dict(rule="fd"), dict(rule="knuth", tiled=False, maxM=120)):
        base = JT.jugaad_maps(F, o, n_jobs=1, **kw)
        fresh = JT.jugaad_maps(F, o[perm], n_jobs=1, **kw)
        reuse = JT.jugaad_maps(F, o[perm], n_jobs=1, bins=base["bins"], **kw)
        assert np.array_equal(fresh["bins"], base["bins"]), kw
        assert np.array_equal(fresh["I_o"], reuse["I_o"]), kw
    print("  permuted obs: bins unchanged, reuse == recompute: OK")


def test_fd_undefined_cell_is_nan():
    F, o = synthetic_cube(seed=7)
    F[:, :, 0, 0] = 0.0
    F[0, :3, 0, 0] = 1.0                                    # IQR 0: thesis FD would crash
    r = JT.jugaad_maps(F, o, n_jobs=1)
    assert np.isnan(r["I_o"][:, 0, 0]).all() and np.isfinite(r["I_o"][:, 1:, :]).all()
    print("  IQR = 0 cell -> NaN, others unaffected: OK")


def test_o_vs_loo():
    F, o = synthetic_cube(seed=8, N=5)
    r = JT.jugaad_maps(F, o, o_vs="loo", n_jobs=1)
    base = JT.jugaad_maps(F, o, n_jobs=1)
    assert np.array_equal(r["I_m"], base["I_m"])                 # denominator untouched
    L = JT.loo_means(F)
    j, i = 2, 3
    Lj = np.where(np.isnan(L[:, :, j, i]), 0.0, L[:, :, j, i]).flatten()
    oj = np.tile(o[:, j, i], F.shape[0])
    Bp, Bo = JT.fd_bins(Lj), JT.fd_bins(oj)
    for k, a in enumerate(JT.THESIS_ALTS):
        ref = ITM.I(oj, Lj, nbins_alt=(int(np.ceil(Bo * a)), int(np.ceil(Bp * a))))[0]
        assert r["I_o"][k, j, i] == ref
    assert r["bins"][3, j, i] == Bp
    print("  o_vs='loo': I_o == thesis I(tile(o); pooled s_-n), I_m unchanged: OK")


def test_lam_m_loo_sizing():
    F, o = synthetic_cube(seed=9, N=6)
    F = np.where(np.isnan(F), 0.0, F)
    r = JT.jugaad_maps(F, o, s="loo", rule="knuth", tiled=False, maxM=60, loo_size="T", n_jobs=1)
    L = JT.loo_means(F)
    j, i = 1, 2
    Bs = JT.knuth_bins(L[:, :, j, i].mean(0), min(60, F.shape[1]))
    assert r["bins"][2, j, i] == Bs
    Lj, fj = L[:, :, j, i].flatten(), F[:, :, j, i].flatten()
    ref = ITM.I(Lj, fj, nbins_alt=(Bs, int(r["bins"][0, j, i])))[0]
    assert r["I_m"][1, j, i] == ref                                   # alt = 1
    print("  lam_m with pooled s_-n, Knuth sized on the T cluster centres: OK")


if __name__ == "__main__":
    test_hist_counts_all_matches_numpy()
    test_info_rpc_bitwise()
    test_explicit_s_matches_full()
    test_loo_means()
    test_optbins_is_knuth_plus_one()
    test_permuted_obs_reuse_bins()
    test_fd_undefined_cell_is_nan()
    test_o_vs_loo()
    test_lam_m_loo_sizing()
    print("all passed")
