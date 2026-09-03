"""rho_o computed two ways: against the ensemble mean, and against the flattened members.

  rho_o_mean   corr( g , s_m )                    obs vs the m-member ensemble mean
  rho_o_flat   corr( tile(g, m) , f_flat )        obs tiled against pooled members
                                                  (the same "jugaad" used for I_gf)

For complete data the two are related exactly. Writing SS_g, SS_s for the sums
of squares of the obs and ensemble-mean anomalies and SS_w for the within-
ensemble scatter sum_i sum_t (f_i - s)^2, pooling gives

    rho_o_flat = rho_o_mean * sqrt( m*SS_s / (SS_w + m*SS_s) )

so the flattening multiplies rho_o by a damping factor that tends to the square
root of the signal fraction, i.e. roughly rho_m. This script checks that
numerically alongside the measured values.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import warnings

import numpy as np
import xarray as xr

ROOT = "/Users/jasperchen/Academics/Research/SNP/information-snp"
LON, LAT = 60, 0
NUM_SAMPLES = 60
SEED = 0


def load_gridpoint():
    f = xr.open_dataset(f"{ROOT}/data/ensembles/large_psl_decadal_ensemble_no_extrapolate.nc").drop_dims(["bnds"])
    obs = xr.open_dataset(f"{ROOT}/data/obs/hadslp2/hadslp2_monthly_1850_2004.nc")
    obs = obs.where(obs["time.year"] >= 1962, drop=True).where(obs["time.year"] < 1972, drop=True).sortby("lon")
    with warnings.catch_warnings(action="ignore"):
        idx = obs.indexes["time"]
        obs["time"] = idx.to_datetimeindex() if hasattr(idx, "to_datetimeindex") else idx
    obs = obs.interp_like(f, kwargs={"fill_value": "extrapolate"})
    obs["psl"] = obs["psl"] - xr.polyval(obs["time"], obs.psl.polyfit(dim="time", deg=1).polyfit_coefficients)
    f["psl"] = f["psl"] - xr.polyval(f["time"], f.psl.polyfit(dim="time", skipna=True, deg=1).polyfit_coefficients)
    return (f.sel(lon=LON, lat=LAT, method="nearest").psl.to_numpy(),
            obs.sel(lon=LON, lat=LAT, method="nearest").psl.to_numpy())


def corr(a, b):
    """Pearson correlation over the entries where both are finite."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def one_subset(f_pt, g_t, mem):
    f_nt = f_pt[mem]
    m = len(mem)
    ok = ~np.isnan(f_nt)
    cnt = ok.sum(axis=0)
    with np.errstate(invalid="ignore"):
        s_t = np.where(cnt > 0, np.where(ok, f_nt, 0.0).sum(axis=0) / np.maximum(cnt, 1), np.nan)

    rho_mean = corr(s_t, g_t)
    rho_flat = corr(np.tile(g_t, m)[ok.ravel()], f_nt[ok])

    # rho_m as the notebook defines it: sqrt(var of the mean / mean member var)
    with np.errstate(invalid="ignore"):
        rho_m = np.sqrt(np.nanvar(s_t) / np.nanvar(f_nt, axis=1).mean())

    # the algebraic damping factor, computed on complete columns only
    full = ok.all(axis=0)
    if full.sum() > 3:
        F = f_nt[:, full]
        S = F.mean(axis=0)
        ss_s = ((S - S.mean()) ** 2).sum()
        ss_w = ((F - S) ** 2).sum()
        damping = np.sqrt(m * ss_s / (ss_w + m * ss_s))
    else:
        damping = np.nan
    return rho_mean, rho_flat, rho_m, damping


def main():
    f_pt, g_t = load_gridpoint()
    n_mem = f_pt.shape[0]
    Ns = np.arange(n_mem)
    rng = np.random.default_rng(SEED)
    members = [rng.random((NUM_SAMPLES, n_mem)).argsort(axis=1)[:, : n + 1] for n in Ns]

    print(f"gridpoint lon={LON} lat={LAT}   members={n_mem}  times={f_pt.shape[1]}  "
          f"NaN frac={np.isnan(f_pt).mean():.3f}\n")
    print("  m  | rho_o mean   rho_o flat |  rho_m   | mean*damping  (identity check)")
    rows = []
    for n in Ns:
        vals = np.array([one_subset(f_pt, g_t, mem) for mem in members[n]])
        rows.append((n + 1, *np.nanmean(vals, axis=0)))
    for m, rmean, rflat, rm, damp in rows:
        if m in (1, 2, 3, 5, 9, 13, 17, 25, 33, 41, 49):
            print(f"  {m:2d} | {rmean:10.4f}   {rflat:9.4f} | {rm:7.4f} | {rmean*damp:12.4f}")

    arr = np.array(rows)
    ms, rmean, rflat, rm, damp = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    print(f"\n  identity  rho_o_flat == rho_o_mean * damping :  max abs diff = "
          f"{np.nanmax(np.abs(rflat - rmean * damp)):.2e}")
    print(f"  damping factor:  m=1 {damp[0]:.4f}  ->  m=49 {damp[-1]:.4f}"
          f"    (rho_m at m=49: {rm[-1]:.4f})")
    print(f"\n  correlation with m:   rho_o mean {np.corrcoef(ms, rmean)[0,1]:+.3f}"
          f"   rho_o flat {np.corrcoef(ms, rflat)[0,1]:+.3f}")
    print(f"  ratio flat/mean:      m=2 {rflat[1]/rmean[1]:.4f}   m=49 {rflat[-1]/rmean[-1]:.4f}")


if __name__ == "__main__":
    main()
