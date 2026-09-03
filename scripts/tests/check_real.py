"""End-to-end check on the two real DCPP cube kinds.

Loads a small subset of each and verifies that `mi_vectorized` reproduces
`mi_pairwise.MI_F_pairwise` exactly, then reports the |rho| vs lambda binned table that
`mi_pairwise`'s docstring says is the only thing worth reading off these estimates.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/oak/stanford/groups/cyaolai/JasperChen/Research/SNP/information-snp"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import mi_vectorized as V          # noqa: E402
import mi_pairwise                 # noqa: E402
import smyle_metrics as M          # noqa: E402


def pearson_F_pairwise(F):
    """The notebook's `pearson_coeff_F_pairwise`, verbatim."""
    c = F - np.mean(F, axis=1, keepdims=True)
    num = np.einsum("ntxy,mtxy->nmxy", c, c, optimize=True)
    ss = np.sum(c * c, axis=1)
    den = np.sqrt(ss[:, None] * ss[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def report(tag, F, n_mem, n_lat, n_lon):
    F = F[:n_mem, :, :n_lat, :n_lon]
    N, T = F.shape[:2]
    print(f"\n=== {tag}: F{F.shape}  (N={N}, T={T})", flush=True)

    t0 = time.perf_counter()
    got = V.mi_member_vs_member(F, k=4)
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = mi_pairwise.MI_F_pairwise(F, k=4, use_copula=True, noise_level=1e-10,
                                    seed=0, n_jobs=1)
    t_ref = time.perf_counter() - t0

    off = ~np.eye(N, dtype=bool)
    d = np.abs(got[off] - ref[off])
    print(f"  vs mi_pairwise: max|diff| = {d.max():.3e} nats   "
          f"({t_vec:.2f} s vs {t_ref:.2f} s, {t_ref / t_vec:.1f}x)")
    print(f"  MI off-diagonal: {got[off].min():+.3f} .. {got[off].max():+.3f} nats, "
          f"mean {got[off].mean():+.4f}, {100 * (got[off] < 0).mean():.0f}% negative")

    rho = np.abs(pearson_F_pairwise(F))
    lam = M.lam_of(got)
    r, l = rho[off].ravel(), lam[off].ravel()
    m = np.isfinite(r) & np.isfinite(l)
    r, l = r[m], l[m]
    edges = [0, .1, .2, .3, .4, .6, 1.01]
    print("  |rho| bin   " + "".join(f"{edges[i]:.1f}-{edges[i+1]:.1f}".rjust(9)
                                     for i in range(len(edges) - 1)))
    for name, v in (("mean |rho|", r), ("mean lam ", l)):
        row = []
        for i in range(len(edges) - 1):
            sel = (r >= edges[i]) & (r < edges[i + 1])
            row.append(f"{v[sel].mean():.3f}".rjust(9) if sel.any() else "     n/a")
        print(f"  {name}  " + "".join(row))
    print(f"  per-entry corr(lam, |rho|) = {np.corrcoef(r, l)[0, 1]:.3f}")


def main():
    try:
        import dcpp_handles as G
        c = G.get(lead="13-60", var="SLP")
        report("dcpp_handles (initialised, T = start dates)", c.F, 8, 6, 8)
    except Exception as e:                                        # noqa: BLE001
        print(f"[skip] dcpp_handles: {type(e).__name__}: {e}", flush=True)

    try:
        import dcpp_decadal_handles as D
        c = D.get(1961, var="SLP")
        report("dcpp_decadal_handles (monthly, T = months in one hindcast)",
               c.F, 6, 5, 6)
    except Exception as e:                                        # noqa: BLE001
        print(f"[skip] dcpp_decadal_handles: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
