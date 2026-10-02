"""lambda_m with the flattened leave-one-out means, Knuth OptBins, point estimates.

    python scripts/analysis/jugaad_lam_m_loo_point.py SLP 2-4
    -> $SCRATCH/snp_jugaad_fd_optbins/lamm_loo_knuth_<VAR>_lead<LEAD>.npz

    lambda_o = lambda(I(f_jugaad; o_jugaad))          unchanged (obs tiled)
    lambda_m = lambda(I(f_jugaad; s_-n_jugaad))       row (n, t) pairs f_n[t] with s_-n[t]
with the s_-n axis sized two ways: "pooled" (Knuth on all N*T rows) and "T" (Knuth on the
T cluster centres mean_n s_-n[n, t] = s[t], capped at T). alt 0.5/1/1.5 for both.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import os
import sys
import time

import numpy as np

import jugaad_thesis as JT                                           # noqa: E402
from jugaad_fd_optbins_run import KNUTH_MAX, OUT, load               # noqa: E402

KEYS = ("I_o", "I_m", "lam_o", "lam_m", "lam", "bins", "alts")


def main(var, lead):
    t0 = time.time()
    F, o, _, lats, lons = load(var, lead)
    out = dict(lats=lats, lons=lons, N=F.shape[0], T=F.shape[1])
    for size in ("pooled", "T"):
        r = JT.jugaad_maps(F, o, s="loo", rule="knuth", tiled=False, maxM=KNUTH_MAX,
                           loo_size=size)
        out.update({f"{size}__{k}": r[k] for k in KEYS})
    path = os.path.join(OUT, f"lamm_loo_knuth_{var}_lead{lead}.npz")
    np.savez_compressed(path, **out)
    print(f"-> {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
