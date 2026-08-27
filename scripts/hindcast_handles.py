"""Handles onto the decadal hindcast cubes, so RPC can be computed without rebuilding.

    import sys; sys.path.insert(0, "../scripts")
    import hindcast_handles as H

Three verification windows, all on the same N48 grid, the same 45 start dates
(decadal1960..decadal2005, decadal1990 absent) and the same 29 members:

    "annual 2-5"  four-year mean over lead years 2-5   <- Eade et al.'s setup
    "DJF 2-5"     the four DJF seasons inside lead years 2-5, averaged
    "DJF 1"       the first complete winter, Dec/Jan/Feb at lead months 12-14

Loaded at import, no estimator calls:

    H.FIELDS[w]  -> (F, G)   F is (N, J, lat, lon), G is (J, lat, lon), both
                             linearly detrended along the start-date axis, in Pa
    H.MI_M[w]    -> (N, lat, lon)   I(f_n ; s_-n) in bits, KSG k=5, RAW
    H.MI_O[w]    -> (N, lat, lon)   I(f_n ; g)    in bits, KSG k=5, RAW
    H.MI_S_G[w]  -> (lat, lon)      I(s ; g)      in bits, KSG k=5, RAW
    H.NULL_BITS  -> scalar          pooled permutation-null floor

Apply H.excess() to any raw MI before H.lam(): lam() of the bare floor is ~0.26.
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

K = 5                                    # KSG neighbours, this project's default
WINDOWS = ["annual 2-5", "DJF 2-5", "DJF 1"]
_KEY = {"annual 2-5": "a25", "DJF 2-5": "lead2-5", "DJF 1": "lead1"}

_a = np.load(os.path.join(_DATA, "annual25_cube.npz"))    # lead-years-2-5 mean
_d = np.load(os.path.join(_DATA, "djf_cube.npz"))

lats, lons, members = _a["lat"], _a["lon"], _a["members"]
WINTER = 1962 + _a["j"]                  # DJF labelled by the January it ends in
START = 1960 + _a["j"]                   # the decadalYYYY the hindcast is named for
J, N = _a["cube"].shape[0], _a["cube"].shape[1]

RAW = {"annual 2-5": (_a["cube"], _a["obs"]),
       "DJF 2-5": (_d["cube25"], _d["obs25"]),
       "DJF 1": (_d["cube"], _d["obs"])}

W = np.cos(np.deg2rad(lats))[:, None] * np.ones(len(lons))   # N48 spans both poles


def detrend(X):
    """Remove the least-squares line along the start-date axis (axis 0), per grid cell."""
    t = np.arange(X.shape[0], dtype=float)
    t -= t.mean()
    flat = X.reshape(X.shape[0], -1)
    slope = (t[:, None] * flat).sum(axis=0) / (t ** 2).sum()
    return (flat - t[:, None] * slope - flat.mean(axis=0)).reshape(X.shape)


FIELDS = {w: (detrend(RAW[w][0].astype(float)).transpose(1, 0, 2, 3),
              detrend(RAW[w][1].astype(float)))
          for w in WINDOWS}


def gm(a):
    """Area-weighted global mean, NaNs skipped. Plain means over-weight the polar rows."""
    m = np.isfinite(a)
    return float((a[m] * W[m]).sum() / W[m].sum())


def loo_mean(Fx, n):
    """s_-n, the ensemble mean over the other N-1 members."""
    return (Fx.sum(axis=0) - Fx[n]) / (Fx.shape[0] - 1)


def pair(n, window):
    """(f_n, s_-n), both (J, lat, lon)."""
    Fx, _ = FIELDS[window]
    return Fx[n], loo_mean(Fx, n)


def pair_obs(n, window):
    """(f_n, g), both (J, lat, lon)."""
    Fx, Gx = FIELDS[window]
    return Fx[n], Gx


def rho_pair(x, y, axis=0):
    """Pearson along `axis`, vectorised over the trailing grid axes."""
    a = x - x.mean(axis=axis, keepdims=True)
    b = y - y.mean(axis=axis, keepdims=True)
    den = np.sqrt((a ** 2).sum(axis) * (b ** 2).sum(axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis) / np.where(den > 0, den, 1), np.nan)


def lam(I_bits):
    """sqrt(1 - 2^-2I), radicand clamped at 0. I must be in BITS."""
    return np.sqrt(np.maximum(0.0, 1.0 - 2.0 ** (-2.0 * np.asarray(I_bits, float))))


def _R():
    import sys
    sys.path.insert(0, _HERE)
    import infomeasure_rpc as R
    return R


def mi_field(x, y, k=K, n_jobs=-1):
    """I(x; y) in bits per grid cell for two (J, lat, lon) fields -> (lat, lon)."""
    from joblib import Parallel, delayed
    R = _R()

    def row(la):
        return [R.mi_bits(x[:, la, lo], y[:, la, lo], "ksg", param=k)
                for lo in range(x.shape[2])]

    return np.array(Parallel(n_jobs=n_jobs)(delayed(row)(la) for la in range(x.shape[1])))


def mi_grid(pairfn, window, k=K, n_jobs=-1):
    """I in bits for every member and grid cell -> (N, lat, lon)."""
    return np.stack([mi_field(*pairfn(n, window), k=k, n_jobs=n_jobs) for n in range(N)])


def mi_null_bits(x, y, n_tests=60, k=K):
    """Permutation-null mean in bits for one cell, from infomeasure's statistical_test."""
    import infomeasure as im
    R = _R()
    est = im.estimator(R.copula(x), R.copula(y), measure="mi", approach="ksg",
                       k=k, base=2, noise_level=0, minkowski_p=np.inf)
    return float(est.statistical_test(n_tests=n_tests, method="permutation_test").null_mean)


# ---------------------------------------------------------------------------
CACHE = os.path.join(_DATA, "hindcast_mi.npz")
NULL_POINTS = [(la, lo) for la in range(3, 73, 8) for lo in range(0, 96, 11)]
NULL_TESTS = 60


def build_cache(path=CACHE, windows=None, n_tests=NULL_TESTS):
    """Per-member MI, I(s;g) and the null floor for each window. ~2 min per window."""
    from joblib import Parallel, delayed
    windows = WINDOWS if windows is None else windows
    out = dict(np.load(path)) if os.path.exists(path) else {}
    nulls = list(out.get("null_all", np.array([])))
    for w in windows:
        kk = _KEY[w]
        F, G = FIELDS[w]
        out["m_" + kk] = mi_grid(pair, w)
        out["o_" + kk] = mi_grid(pair_obs, w)
        out["sg_" + kk] = mi_field(F.mean(axis=0), G)
        idx = [0, N // 3, 2 * N // 3, N - 1]
        for fn in (pair, pair_obs):
            jobs = []
            for i, (la, lo) in enumerate(NULL_POINTS):
                x, y = fn(idx[i % len(idx)], w)
                jobs.append((x[:, la, lo], y[:, la, lo]))
            nulls += Parallel(n_jobs=-1, batch_size=4)(
                delayed(mi_null_bits)(a, b, n_tests) for a, b in jobs)
    out["null_all"] = np.array(nulls)
    out["null_bits"] = np.array(float(np.mean(nulls)))
    np.savez_compressed(path, **out)
    return out


if os.path.exists(CACHE):
    _c = np.load(CACHE)
    MI_M = {w: _c["m_" + _KEY[w]] for w in WINDOWS if "m_" + _KEY[w] in _c}
    MI_O = {w: _c["o_" + _KEY[w]] for w in WINDOWS if "o_" + _KEY[w] in _c}
    MI_S_G = {w: _c["sg_" + _KEY[w]] for w in WINDOWS if "sg_" + _KEY[w] in _c}
    NULL_BITS = float(_c["null_bits"])
else:                                                    # run build_cache()
    MI_M, MI_O, MI_S_G, NULL_BITS = {}, {}, {}, None


def excess(I_bits):
    """Raw MI minus the pooled permutation-null floor, in bits."""
    return np.asarray(I_bits, float) - NULL_BITS


if __name__ == "__main__":
    print("%d members x %d start dates, decadal%d..decadal%d (decadal1990 absent), "
          "grid %d x %d, KSG k=%d"
          % (N, J, START[0], START[-1], len(lats), len(lons), K))
    print("null floor: %s bits\n" % ("not built" if NULL_BITS is None else "%.4f" % NULL_BITS))
    print("%-12s %9s %9s %9s %9s %9s" % ("window", "rho_o", "rho_m", "RPC_rho",
                                         "lam_o", "RPC_lam"))
    for w in WINDOWS:
        F, G = FIELDS[w]
        s = F.mean(axis=0)
        rho_o = rho_pair(s, G)
        rho_m = np.sqrt(s.var(axis=0) / F.var(axis=1).mean(axis=0))
        if w in MI_S_G:
            lo = lam(excess(MI_S_G[w]))
            lm = lam(excess(MI_M[w])).mean(axis=0)
            print("%-12s %9.4f %9.4f %9.3f %9.4f %9.3f"
                  % (w, gm(rho_o), gm(rho_m), gm(rho_o / rho_m), gm(lo), gm(lo / lm)))
        else:
            print("%-12s %9.4f %9.4f %9.3f %9s %9s"
                  % (w, gm(rho_o), gm(rho_m), gm(rho_o / rho_m), "-", "- (build_cache)"))
