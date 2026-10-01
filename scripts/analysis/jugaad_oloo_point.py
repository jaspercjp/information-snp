"""Point estimates of the jugaad lambda with lambda_o = lambda(I(s_-n_jugaad; o_jugaad)).

    python scripts/analysis/jugaad_oloo_point.py SLP 2-4
    -> $SCRATCH/snp_jugaad_fd_optbins/oloo_<VAR>_lead<LEAD>.npz

FD bins as the thesis, alt 0.5/1/1.5, full-mean s in lambda_m (identical to the fd_tiled
lambda_m of jugaad_fd_optbins_run.py). No bootstrap.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import os
import sys
import time

import numpy as np

import jugaad_thesis as JT                                           # noqa: E402
from jugaad_fd_optbins_run import OUT, load                          # noqa: E402


def main(var, lead):
    t0 = time.time()
    F, o, s, lats, lons = load(var, lead)
    r = JT.jugaad_maps(F, o, s=s, rule="fd", o_vs="loo")
    path = os.path.join(OUT, f"oloo_{var}_lead{lead}.npz")
    np.savez_compressed(path, lats=lats, lons=lons, N=F.shape[0], T=F.shape[1],
                        **{k: r[k] for k in ("I_o", "I_m", "lam_o", "lam_m", "lam", "bins", "alts")})
    print(f"-> {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
