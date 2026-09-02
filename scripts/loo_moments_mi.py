"""Leave-one-out I(s;f), with the held-out ensemble represented by four moments.

The notebook's I_sf_loo represents the other m-1 members by their mean alone:

    I_sf_loo = < I( mean_-j ; f_j ) >_j

Here the rest of the ensemble is described by its leading moments per timestep,

    < I( [mean, std, skew, kurt]_-j ; f_j ) >_j

built up one moment at a time so the gain from each can be separated from the
cost of the extra dimension. KSG uses a max-norm, which is NOT scale invariant
across coordinates, so every column is z-scored before estimation.

LOO moments come from leave-one-out power sums (S1..S4 minus the held-out
member), which is O(m) rather than O(m^2). Checked against scipy below.
"""
import warnings

import numpy as np
import xarray as xr
import infomeasure as im
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
from joblib import Parallel, delayed

ROOT = "/Users/jasperchen/Academics/Research/SNP/information-snp"
LON, LAT = 60, 0
K = 5
NUM_SUBSETS = 20
SEED = 0
LABELS = ["mean", "+std", "+skew", "+kurt"]


def load_gridpoint():
    f = xr.open_dataset(f"{ROOT}/data/ensembles/large_psl_decadal_ensemble_no_extrapolate.nc").drop_dims(["bnds"])
    f["psl"] = f["psl"] - xr.polyval(f["time"], f.psl.polyfit(dim="time", skipna=True, deg=1).polyfit_coefficients)
    return f.sel(lon=LON, lat=LAT, method="nearest").psl.to_numpy()


def loo_moments(f_nt):
    """(mean, std, skew, kurt) of the other members, per held-out member and timestep.

    Returns an (m, time, 4) array. Uses leave-one-out power sums.
    """
    ok = ~np.isnan(f_nt)
    x = np.where(ok, f_nt, 0.0)
    n = ok.sum(axis=0) - ok                                  # (m, time) members excl. j
    S1 = x.sum(axis=0) - x
    S2 = (x ** 2).sum(axis=0) - x ** 2
    S3 = (x ** 3).sum(axis=0) - x ** 3
    S4 = (x ** 4).sum(axis=0) - x ** 4
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = np.where(n > 0, n, np.nan)
        mu = S1 / nn
        m2 = S2 / nn - mu ** 2
        m3 = S3 / nn - 3 * mu * S2 / nn + 2 * mu ** 3
        m4 = S4 / nn - 4 * mu * S3 / nn + 6 * mu ** 2 * S2 / nn - 3 * mu ** 4
        m2 = np.where(m2 > 0, m2, np.nan)
        out = np.stack([mu, np.sqrt(m2), m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0], axis=-1)
    # moments beyond the mean need enough members to be defined at all
    out[..., 1] = np.where(n >= 2, out[..., 1], np.nan)
    out[..., 2] = np.where(n >= 3, out[..., 2], np.nan)
    out[..., 3] = np.where(n >= 4, out[..., 3], np.nan)
    return out


def zscore(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def loo_moment_mi(f_pt, mem, n_moments, k=K):
    """< I( first n_moments LOO moments ; f_j ) >_j for one member subset."""
    f_nt = f_pt[mem]
    m = len(mem)
    ok = ~np.isnan(f_nt)
    M = loo_moments(f_nt)[..., :n_moments]                   # (m, time, d)
    vals = []
    for j in range(m):
        v = ok[j] & np.isfinite(M[j]).all(axis=-1)
        if v.sum() > k + n_moments:
            # KSG's max-norm mixes the X and Y coordinates, so BOTH sides must be
            # put on a common scale -- z-scoring X alone collapses the joint
            # distance onto Y and drives the estimate to zero.
            y = f_nt[j][v]
            vals.append(im.mutual_information(zscore(M[j][v]),
                                              (y - y.mean()) / y.std(),
                                              approach="metric", k=k))
    return float(np.mean(vals)) if vals else np.nan


def main():
    f_pt = load_gridpoint()
    n_mem = f_pt.shape[0]
    Ns = np.arange(n_mem)
    rng = np.random.default_rng(SEED)
    members = [rng.random((NUM_SUBSETS, n_mem)).argsort(axis=1)[:, : n + 1] for n in Ns]

    # --- correctness of the power-sum LOO moments -------------------------
    mem = members[48][0]
    M = loo_moments(f_pt[mem])
    j, t = 7, 30
    rest = np.delete(f_pt[mem][:, t], j)
    rest = rest[~np.isnan(rest)]
    ref = [rest.mean(), rest.std(), sp_skew(rest), sp_kurt(rest)]
    print("=== power-sum LOO moments vs scipy (member j=7, t=30, m=49) ===")
    for name, a, b in zip(["mean", "std", "skew", "kurt"], M[j, t], ref):
        print(f"  {name:5s} power-sum {a:12.6f}   scipy {b:12.6f}   diff {a-b:+.2e}")

    # --- dimensionality calibration on synthetic data ---------------------
    print(f"\n=== dimensionality penalty at N=120, k={K} (synthetic: y depends on col 0 only) ===")
    r = np.random.default_rng(3)
    X = r.normal(size=(120, 4))
    y = 0.8 * X[:, 0] + 0.5 * r.normal(size=120)
    yz = (y - y.mean()) / y.std()
    base = im.mutual_information(zscore(X[:, :1]), yz, approach="metric", k=K)
    for d in range(1, 5):
        v = im.mutual_information(zscore(X[:, :d]), yz, approach="metric", k=K)
        print(f"  d={d}  MI={v:.4f}   ({(v-base)/base:+6.1%} vs d=1)  <- extra columns are pure noise")
    print("  Any drop of this size in the real results is dimensionality, not lost information.")

    # --- the real calculation ---------------------------------------------
    print(f"\n=== < I( LOO moments ; f_j ) >_j   k={K}, {NUM_SUBSETS} subsets, "
          f"no replacement ===")
    print("   m  |" + "".join(f"  {lab:>7s}" for lab in LABELS) + "   |  best")
    MS = [5, 9, 13, 17, 25, 33, 41, 49]
    res = {}
    for m in MS:
        vals = Parallel(n_jobs=-1, batch_size=2)(
            delayed(loo_moment_mi)(f_pt, mm, d) for d in (1, 2, 3, 4) for mm in members[m - 1])
        vals = np.array(vals).reshape(4, NUM_SUBSETS)
        res[m] = np.nanmean(vals, axis=1)
        best = LABELS[int(np.nanargmax(res[m]))]
        print(f"  {m:3d} |" + "".join(f"  {v:7.4f}" for v in res[m]) + f"   |  {best}")

    print("\n  change relative to the mean-only estimator:")
    for m in MS:
        r0 = res[m][0]
        print(f"   m={m:3d}  " + "   ".join(
            f"{lab}: {(res[m][i]-r0)/r0:+6.1%}" for i, lab in enumerate(LABELS[1:], start=1)))


if __name__ == "__main__":
    main()
