"""Pairwise mutual information between ensemble members, summarised as a matrix norm.

Builds M[i, j] = I(f_i ; f_j) over every pair of ensemble members at one
gridpoint, then reports matrix norms of M (and of the m-member submatrices) as
an approximation to "how much the ensemble knows about itself".

The norms are compared against the two estimators of I(s ; f) used in
notebooks/jugaad_copy.ipynb:
    I_sf      MI(tile(s_m, m), f_flat)          ensemble mean vs pooled members
    I_sf_loo  <I(s_-j ; f_j)>_j                 leave-one-out, averaged
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import warnings

import numpy as np
import xarray as xr
import infomeasure as im
from joblib import Parallel, delayed

ROOT = "/Users/jasperchen/Academics/Research/SNP/information-snp"
LON, LAT = 60, 0
K = 3
NUM_SAMPLES = 60
SEED = 0


def load_gridpoint():
    """Same preprocessing as the notebook: detrended anomalies at one gridpoint."""
    f = xr.open_dataset(f"{ROOT}/data/ensembles/large_psl_decadal_ensemble_no_extrapolate.nc").drop_dims(["bnds"])
    obs = xr.open_dataset(f"{ROOT}/data/obs/hadslp2/hadslp2_monthly_1850_2004.nc")
    obs = obs.where(obs["time.year"] >= 1962, drop=True).where(obs["time.year"] < 1972, drop=True).sortby("lon")
    with warnings.catch_warnings(action="ignore"):
        idx = obs.indexes["time"]
        obs["time"] = idx.to_datetimeindex() if hasattr(idx, "to_datetimeindex") else idx
    obs = obs.interp_like(f, kwargs={"fill_value": "extrapolate"})
    f["psl"] = f["psl"] - xr.polyval(f["time"], f.psl.polyfit(dim="time", skipna=True, deg=1).polyfit_coefficients)
    return f.sel(lon=LON, lat=LAT, method="nearest").psl.to_numpy()   # (member, time), NaNs kept


def mi(a, b):
    return im.mutual_information(a, b, approach="metric", k=K)


def pairwise_matrix(f_pt):
    """M[i, j] = I(f_i ; f_j), NaNs dropped pairwise. Diagonal left at zero."""
    n = f_pt.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def one(i, j):
        ok = ~np.isnan(f_pt[i]) & ~np.isnan(f_pt[j])
        return mi(f_pt[i][ok], f_pt[j][ok])

    vals = Parallel(n_jobs=-1, batch_size=8)(delayed(one)(i, j) for i, j in pairs)
    M = np.zeros((n, n))
    for (i, j), v in zip(pairs, vals):
        M[i, j] = M[j, i] = v
    return M


def norms(M):
    """Matrix norms of a symmetric zero-diagonal MI matrix, plus per-pair rescalings.

    For the idealised case where every pair shares the same MI value c, the
    off-diagonal matrix is c(J - I) with top eigenvalue c(m-1) and Frobenius
    norm c*sqrt(m(m-1)). Dividing them out puts every column on the scale of a
    single pairwise MI, so they can sit next to I_sf.
    """
    m = M.shape[0]
    off = M[~np.eye(m, dtype=bool)]
    ev = np.linalg.eigvalsh(M)
    spectral = np.abs(ev).max()
    frob = np.linalg.norm(M, "fro")
    return dict(
        m=m,
        mean_off=off.mean() if off.size else np.nan,
        frob=frob,
        spectral=spectral,
        nuclear=np.abs(ev).sum(),
        frob_per_pair=frob / np.sqrt(m * (m - 1)) if m > 1 else np.nan,
        spectral_per_pair=spectral / (m - 1) if m > 1 else np.nan,
    )


def i_sf_both(f_pt, mem):
    """The notebook's two I(s;f) estimators, for one member subset."""
    f_nt = f_pt[mem]
    m = len(mem)
    ok = ~np.isnan(f_nt)
    cnt = ok.sum(axis=0)
    with np.errstate(invalid="ignore"):
        s_t = np.where(cnt > 0, np.where(ok, f_nt, 0.0).sum(axis=0) / np.maximum(cnt, 1), np.nan)
    tiled = mi(np.tile(s_t, m)[ok.ravel()], f_nt[ok])

    if m < 2:
        return tiled, np.nan
    filled = np.where(ok, f_nt, 0.0)
    loo_sum = filled.sum(axis=0) - filled
    loo_cnt = ok.sum(axis=0) - ok
    with np.errstate(invalid="ignore"):
        loo_s = np.where(loo_cnt > 0, loo_sum / np.maximum(loo_cnt, 1), np.nan)
    vals = []
    for j in range(m):
        v = ok[j] & ~np.isnan(loo_s[j])
        if v.sum() > 3:
            vals.append(mi(loo_s[j][v], f_nt[j][v]))
    return tiled, (float(np.mean(vals)) if vals else np.nan)


def main():
    f_pt = load_gridpoint()
    n_mem = f_pt.shape[0]
    print(f"gridpoint lon={LON} lat={LAT}   members={n_mem}  times={f_pt.shape[1]}  "
          f"NaN frac={np.isnan(f_pt).mean():.3f}   KSG k={K}\n")

    M = pairwise_matrix(f_pt)
    iu = np.triu_indices(n_mem, 1)
    off = M[iu]
    print(f"=== Pairwise MI matrix, {n_mem}x{n_mem}, {off.size} unique pairs (diagonal set to 0) ===")
    print(f"  off-diagonal I(f_i;f_j):  mean={off.mean():.4f}  median={np.median(off):.4f}  "
          f"sd={off.std():.4f}  min={off.min():.4f}  max={off.max():.4f}")
    print(f"  negative entries: {int((off < 0).sum())} of {off.size}")
    full = norms(M)
    print(f"\n  Frobenius   ||M||_F      = {full['frob']:.3f}     /sqrt(m(m-1)) = {full['frob_per_pair']:.4f}")
    print(f"  Spectral    |lambda|_max = {full['spectral']:.3f}     /(m-1)        = {full['spectral_per_pair']:.4f}")
    print(f"  Nuclear     sum|lambda|  = {full['nuclear']:.3f}")

    # Diagonal alternatives, since the choice dominates the norm.
    H = np.array([im.entropy(f_pt[i][~np.isnan(f_pt[i])], approach="metric", k=K) for i in range(n_mem)])
    for tag, D in [("diag = 0 (above)", 0.0), ("diag = H(f_i)", H)]:
        Md = M + np.diag(np.full(n_mem, D) if np.isscalar(D) else D)
        print(f"  [{tag:16s}]  ||M||_F = {np.linalg.norm(Md,'fro'):8.3f}   "
              f"|lambda|_max = {np.abs(np.linalg.eigvalsh(Md)).max():8.3f}")

    # Is the norm structure, or just noise? Shuffling the off-diagonal entries
    # destroys any member-to-member structure while preserving the value
    # distribution exactly, so it isolates what the norm owes to real coherence.
    rs = np.random.default_rng(SEED)
    lam_shuf, fro_shuf = [], []
    for _ in range(200):
        v = rs.permutation(off)
        S = np.zeros_like(M)
        S[iu] = v
        S = S + S.T
        lam_shuf.append(np.abs(np.linalg.eigvalsh(S)).max())
        fro_shuf.append(np.linalg.norm(S, "fro"))
    print(f"\n  shuffled null (200 draws, same entries, structure destroyed):")
    print(f"    |lambda|_max  observed {full['spectral']:.3f}  vs null {np.mean(lam_shuf):.3f} "
          f"+/- {np.std(lam_shuf):.3f}   z = {(full['spectral']-np.mean(lam_shuf))/np.std(lam_shuf):+.1f}")
    print(f"    ||M||_F       observed {full['frob']:.3f}  vs null {np.mean(fro_shuf):.3f} "
          f"+/- {np.std(fro_shuf):.3f}   z = {(full['frob']-np.mean(fro_shuf))/max(np.std(fro_shuf),1e-12):+.1f}")
    print(f"    RMS decomposition: sqrt(mean^2 + sd^2) = "
          f"{np.sqrt(off.mean()**2 + off.std()**2):.4f}  == frob/sqrt(m(m-1)) = {full['frob_per_pair']:.4f}")

    # --- vs ensemble size, on the SAME subsets the notebook uses -------------
    rng = np.random.default_rng(SEED)
    Ns = np.arange(n_mem)
    members = [rng.random((NUM_SAMPLES, n_mem)).argsort(axis=1)[:, : n + 1] for n in Ns]

    print(f"\n=== Against ensemble size (mean over {NUM_SAMPLES} subsets, without replacement) ===")
    print("  m  | spec/(m-1)  frob/sqrt(m(m-1))  mean_off | I_sf tiled   I_sf_loo")
    rows = []
    for n in Ns:
        sub = [norms(M[np.ix_(mem, mem)]) for mem in members[n]]
        both = Parallel(n_jobs=-1, batch_size=4)(delayed(i_sf_both)(f_pt, mem) for mem in members[n])
        both = np.array(both, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # m = 1 has no pairs
            rows.append((n + 1,
                         np.nanmean([s["spectral_per_pair"] for s in sub]),
                         np.nanmean([s["frob_per_pair"] for s in sub]),
                         np.nanmean([s["mean_off"] for s in sub]),
                         np.nanmean(both[:, 0]), np.nanmean(both[:, 1])))
    for m, sp, fr, mo, isf, iloo in rows:
        if m in (1, 2, 3, 5, 9, 13, 17, 25, 33, 41, 49):
            print(f"  {m:2d} | {sp:9.4f}  {fr:16.4f}  {mo:7.4f} | {isf:9.4f}   "
                  f"{'      n/a' if np.isnan(iloo) else f'{iloo:9.4f}'}")

    arr = np.array(rows)
    ms = arr[:, 0]
    print("\n  correlation with m (m>=2):")
    for j, nm in [(1, "spectral/(m-1)"), (2, "frob/sqrt(m(m-1))"), (3, "mean off-diag"),
                  (4, "I_sf tiled"), (5, "I_sf_loo")]:
        v = arr[1:, j]
        good = ~np.isnan(v)
        print(f"    {nm:20s} {np.corrcoef(ms[1:][good], v[good])[0,1]:+.3f}")


if __name__ == "__main__":
    main()
