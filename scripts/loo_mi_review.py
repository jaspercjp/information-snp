"""Review of the leave-one-out I(s;f) estimator: k sensitivity and significance.

Checks, in order:
  A  implementation guards -- does the minimum-sample guard ever fire, and is it
     correctly tied to k?
  B  sensitivity of I_sf_loo to the KSG neighbour count k
  C  infomeasure's permutation test (p-value against an independence null)
  D  whether a naive paired bootstrap can give a usable CI for a KSG estimate
"""
import warnings

import numpy as np
import xarray as xr
import infomeasure as im
from joblib import Parallel, delayed

ROOT = "/Users/jasperchen/Academics/Research/SNP/information-snp"
LON, LAT = 60, 0
NUM_SUBSETS = 20
SEED = 0


def load_gridpoint():
    f = xr.open_dataset(f"{ROOT}/data/ensembles/large_psl_decadal_ensemble_no_extrapolate.nc").drop_dims(["bnds"])
    f["psl"] = f["psl"] - xr.polyval(f["time"], f.psl.polyfit(dim="time", skipna=True, deg=1).polyfit_coefficients)
    return f.sel(lon=LON, lat=LAT, method="nearest").psl.to_numpy()


def loo_pairs(f_pt, mem):
    """The (s_-j , f_j) pairs the notebook's I_sf_loo averages over."""
    f_nt = f_pt[mem]
    m = len(mem)
    ok = ~np.isnan(f_nt)
    filled = np.where(ok, f_nt, 0.0)
    loo_sum = filled.sum(axis=0) - filled
    loo_cnt = ok.sum(axis=0) - ok
    with np.errstate(invalid="ignore"):
        loo_s = np.where(loo_cnt > 0, loo_sum / np.maximum(loo_cnt, 1), np.nan)
    out = []
    for j in range(m):
        v = ok[j] & ~np.isnan(loo_s[j])
        out.append((loo_s[j][v], f_nt[j][v]))
    return out


def i_sf_loo(f_pt, mem, k):
    pairs = loo_pairs(f_pt, mem)
    vals = [im.mutual_information(x, y, approach="metric", k=k)
            for x, y in pairs if len(x) > k]
    return float(np.mean(vals)) if vals else np.nan


def main():
    f_pt = load_gridpoint()
    n_mem = f_pt.shape[0]
    Ns = np.arange(n_mem)
    rng = np.random.default_rng(SEED)
    members = [rng.random((NUM_SUBSETS, n_mem)).argsort(axis=1)[:, : n + 1] for n in Ns]
    full = np.arange(n_mem)

    # ---- A: guards -------------------------------------------------------
    lens = [len(x) for n in Ns for mem in members[n] for x, _ in loo_pairs(f_pt, mem)]
    lens = np.array(lens)
    print("=== A. implementation guards ===")
    print(f"  paired sample length per LOO term: min={lens.min()}  median={int(np.median(lens))}  max={lens.max()}")
    print(f"  terms that would trip the notebook's `v.sum() > 3` guard: {int((lens <= 3).sum())} of {lens.size}")
    print("  NOTE the notebook hardcodes `> 3`, which is k+1 only while k==3.")
    print("       If k is raised the guard must become `> k`, or KSG gets fewer points than neighbours.")

    # ---- B: k sensitivity -------------------------------------------------
    print("\n=== B. sensitivity to KSG neighbour count k ===")
    KS = [4, 5, 8]
    MS = [2, 3, 5, 9, 13, 17, 25, 33, 41, 49]
    print("   m  |" + "".join(f"  k={k:<5d}" for k in KS))
    table = {}
    for m in MS:
        vals = Parallel(n_jobs=-1, batch_size=4)(
            delayed(i_sf_loo)(f_pt, mem, k) for k in KS for mem in members[m - 1])
        vals = np.array(vals).reshape(len(KS), NUM_SUBSETS)
        table[m] = vals.mean(axis=1)
        print(f"  {m:2d}  |" + "".join(f"  {v:6.4f}" for v in table[m]))
    print("\n  spread across k, as a fraction of the k=3 value:")
    for m in MS:
        row = table[m]
        k3 = row[KS.index(5)]   # k=5 as the mid reference
        print(f"    m={m:2d}  min={row.min():.4f}  max={row.max():.4f}  "
              f"(max-min)/k5 = {(row.max()-row.min())/k3:5.1%}")
    print("\n  does the rise with ensemble size survive? (I_sf_loo at m=49 minus at m=5)")
    for i, k in enumerate(KS):
        print(f"    k={k:<3d} m=5 {table[5][i]:.4f} -> m=49 {table[49][i]:.4f}   "
              f"delta={table[49][i]-table[5][i]:+.4f}")

    # ---- C: permutation test ---------------------------------------------
    print("\n=== C. infomeasure permutation test, full ensemble (m=49), each LOO term ===")
    pairs = loo_pairs(f_pt, full)

    def ptest(x, y, k):
        e = im.estimator(x, y, measure="mutual_information", approach="metric", k=k)
        r = e.statistical_test(n_tests=200, method="permutation_test")
        return e.result(), r.p_value, r.null_mean, r.null_std

    for k in KS:
        res = np.array(Parallel(n_jobs=-1, batch_size=4)(delayed(ptest)(x, y, k) for x, y in pairs))
        obs, p, nm, ns = res[:, 0], res[:, 1], res[:, 2], res[:, 3]
        print(f"  k={k:<3d} mean I={obs.mean():.4f}  null mean={nm.mean():+.4f} (sd {ns.mean():.4f})  "
              f"median p={np.median(p):.3f}  frac p<0.05: {(p < 0.05).mean():.2%}  frac p<0.01: {(p < 0.01).mean():.2%}")

    # ---- D: is a naive paired bootstrap usable here? ----------------------
    print("\n=== D. naive paired bootstrap (resample timesteps with replacement) ===")
    x0, y0 = pairs[0]
    obs = im.mutual_information(x0, y0, approach="metric", k=3)
    bs = np.random.default_rng(1)

    def draw(_):
        idx = bs.integers(0, len(x0), len(x0))
        return im.mutual_information(x0[idx], y0[idx], approach="metric", k=3)

    boot = np.array([draw(i) for i in range(200)])
    dup = np.mean([len(x0) - len(set(bs.integers(0, len(x0), len(x0)).tolist())) for _ in range(200)])
    print(f"  observed I = {obs:.4f}")
    print(f"  bootstrap mean = {boot.mean():.4f}  ->  bias {boot.mean()-obs:+.4f} ({(boot.mean()-obs)/obs:+.1%})")
    print(f"  bootstrap 5-95 pct = [{np.percentile(boot,5):.4f}, {np.percentile(boot,95):.4f}]")
    print(f"  avg duplicated timesteps per resample: {dup:.0f} of {len(x0)}")
    print("  -> resampling with replacement creates tied points, the same effect that")
    print("     inflated the with-replacement member sampling. Treat this CI as unusable.")


if __name__ == "__main__":
    main()
