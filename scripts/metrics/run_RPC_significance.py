"""T=J block bootstrap with blocks of 5 consecutive CALENDAR years, pooled ensembles.

Per-cell 5-95% CI on RPC from 2000 replicates; reports the fraction of grid cells whose
CI lies entirely above 1. One draw of M-3 members without replacement per replicate.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import sys
import numpy as np

sys.path.insert(0, "/oak/stanford/groups/cyaolai/JasperChen/Research/SNP/information-snp/scripts")
import dcpp_handles as G
import smyle_metrics as M
import significance_test_RPC as T


def blocks_by_year(Y, L, rng, B):
    """T=J cases drawn WITH replacement in blocks of L consecutive calendar years."""
    bl = T.year_windows(Y, L, full_only=True)
    J = len(Y)
    nb = -(-J // L)
    pick = rng.integers(len(bl), size=(B, nb))
    return np.stack([np.concatenate([bl[k] for k in r])[:J] for r in pick])


print("T=J cases, blocks of 5 consecutive CALENDAR years (non-circular), one M-3")
print("member subset per replicate, 2000 replicates.  SIG>1 = frac of grid cells whose")
print("per-cell 5-95% CI on RPC lies ENTIRELY above 1.   chance level = 5%\n")
print("%-7s %-9s %3s %3s %4s | %7s | %8s %8s | %6s" % (
    "var", "rho_m", "M", "J", "blk", "frac>1", "SIG>1", "SIG<1", "%bad"), flush=True)
print("-" * 78, flush=True)

for var in ("SLP", "PRECT", "TREFHT"):
    c = G.get(lead="13-60", var=var, verbose=False)
    ro = M.pearson_coeff(c.s, c.G)
    N, J = c.F.shape[:2]
    nbl = len(T.year_windows(c.YEAR, 5, True))
    F = np.asarray(c.F, float)
    Gg = np.asarray(c.G, float)
    for kind in ("eade", "debiased"):
        rm = M.ensemble_SNR(c.F) if kind == "eade" else M.ensemble_SNR_debiased(c.F)
        good = np.isfinite(ro) & np.isfinite(rm) & (rm > 0)
        mask = good & (ro > 0)
        pt = np.where(good, ro / rm, np.nan)
        rng = np.random.default_rng(0)
        Wt = T._weights(blocks_by_year(c.YEAR, 5, rng, 2000), J)
        Pm = np.zeros((2000, N))
        for i in range(2000):
            Pm[i, rng.choice(N, N - 3, replace=False)] = 1.0 / (N - 3)
        o, m_, rpc, _ = T._eade_boot_at(F, Gg, Wt, Pm, kind, 16)
        fr = rpc.copy()
        fin = np.isfinite(fr)
        fr[~fin] = np.nan
        ok = fin.mean(0) >= 0.95
        lo = np.full(fr.shape[1], np.nan)
        hi = lo.copy()
        lo[ok], hi[ok] = np.nanpercentile(fr[:, ok], [5, 95], axis=0)
        lo = lo.reshape(ro.shape); hi = hi.reshape(ro.shape)
        mk = mask & ok.reshape(ro.shape)
        f = lambda a, b: (100 * a.sum() / b.sum() if b.sum() else np.nan)
        print("%-7s %-9s %3d %3d %4d | %7.3f | %7.1f%% %7.1f%% | %5.1f%%" % (
            var, kind, N - 3, J, nbl, (pt[mask] > 1).mean(),
            f(np.isfinite(lo) & (lo > 1) & mk, mk),
            f(np.isfinite(hi) & (hi < 1) & mk, mk),
            100 * (1 - ok.mean())), flush=True)
