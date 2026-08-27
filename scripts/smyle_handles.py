"""The CESM2 SMYLE hindcast cube. Data only -- no estimators, no caches.

    import sys; sys.path.insert(0, "../scripts")
    import smyle_handles as S

Lead months 1-3 from every quarterly initialization, 1958-2019, on the N48 grid:

    S.F       (N, J, lat, lon)   20 members x 247 initializations, Pa
    S.G       (J, lat, lon)      HadSLP2r over the same windows, Pa
    S.YEAR    (J,)               initialization year of each sample
    S.MONTH   (J,)               initialization month: 2, 5, 8 or 11

J=247, not 248: the 2019-11 init verifies into Jan 2020 and HadSLP2r stops at Dec 2019.

Detrending is done SEPARATELY WITHIN EACH INIT MONTH. The sample axis cycles
Feb/May/Aug/Nov, so one line over all 247 leaves the seasonal cycle in -- and model and
obs share it, which fakes skill: rho_o comes out 0.759 that way versus 0.450 done
properly. Per grid cell the four init-month climatologies differ by 199 Pa against a
127 Pa within-season anomaly, and model and obs seasonal cycles correlate 0.97.

Pooling seasons leaves the 247 samples heteroscedastic, since the four seasons have
different anomaly variances. Use S.MONTH to work one season at a time (~62 each).
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# The cube lives in the main checkout and is not duplicated into a worktree; look
# next to the script first, then fall back, so this works from either.
_MAIN = "/Users/jasperchen/Academics/Research/SNP/information-snp/data"
_DATA = next((d for d in (os.path.join(_HERE, "..", "data"), _MAIN)
              if os.path.exists(os.path.join(d, "smyle_cube.npz"))), _MAIN)

NLEAD = 3                                # lead months 1-3

_z = np.load(os.path.join(_DATA, "smyle_cube.npz"))
lats, lons, members = _z["lat"], _z["lon"], _z["members"]
YEAR, MONTH = _z["year"], _z["month"]
J, N = _z["cube"].shape[0], _z["cube"].shape[1]

W = np.cos(np.deg2rad(lats))[:, None] * np.ones(len(lons))   # N48 spans both poles


def _line(X):
    """Remove the least-squares line along axis 0, per grid cell. Mean goes with it."""
    t = np.arange(X.shape[0], dtype=float)
    t -= t.mean()
    flat = X.reshape(X.shape[0], -1)
    slope = (t[:, None] * flat).sum(axis=0) / (t ** 2).sum()
    return (flat - t[:, None] * slope - flat.mean(axis=0)).reshape(X.shape)


def detrend(X, month=MONTH):
    """`_line` within each init month, so the seasonal cycle goes with the trend."""
    out = np.empty_like(X, dtype=float)
    for m in np.unique(month):
        k = month == m
        out[k] = _line(X[k])
    return out


F = detrend(_z["cube"].astype(float)).transpose(1, 0, 2, 3)   # (N, J, lat, lon)
G = detrend(_z["obs"].astype(float))                          # (J, lat, lon)

# init label as a fractional year, handy for plotting against
TIME = YEAR + (MONTH - 1) / 12.0


def gm(a):
    """Area-weighted global mean, NaNs skipped. Plain means over-weight the polar rows."""
    m = np.isfinite(a)
    return float((a[m] * W[m]).sum() / W[m].sum())


if __name__ == "__main__":
    print(f"{N} members x {J} initializations (quarterly {YEAR[0]}-{YEAR[-1]}), "
          f"lead months 1-{NLEAD}, N48 {len(lats)}x{len(lons)}")
    print(f"  F {F.shape}  G {G.shape}   Pa, detrended within each init month")
    print(f"  inits per month: " + ", ".join(f"{m:02d}:{int((MONTH==m).sum())}"
                                             for m in np.unique(MONTH)))
