"""Point estimates for one case, every configuration the figures and tables use.

    python scripts/run.py SLP 2-4          -> $SNP_OUT/point_SLP_2-4.npz

fd / knuth: FD on the tiled series (thesis) / Knuth on the T distinct o, s values, alt 0.5/1/1.5;
*_loo: s_-n in lambda_m; oloo_*: lambda_o = lambda(I(s_-n; o)); fixed_[loo_]B<b>: B on every
axis. Plus pooled |rho(f; o)|, |rho(f; s)|, |rho(f; s_-n)| and the time-axis rho measures.
"""
import sys
import time

import numpy as np

import jugaad as J
from data import load, out

FIXED_B = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64)
KNUTH = dict(rule="knuth", tiled=False)
ONE = dict(alts=(1.0,))
CONFIGS = {"fd": {}, "fd_loo": dict(ONE, s="loo"), "knuth": KNUTH,
           "knuth_loo": dict(KNUTH, s="loo", **ONE), "oloo_fd": dict(o_vs="loo"),
           "oloo_knuth": dict(KNUTH, o_vs="loo"),
           **{f"fixed_B{b}": dict(ONE, rule="fixed", B=b) for b in FIXED_B},
           **{f"fixed_loo_B{b}": dict(ONE, rule="fixed", B=b, s="loo") for b in FIXED_B}}


def main(var, case):
    t0 = time.time()
    F, o, s, lats, lons = load(var, case)
    rho_t = J.rho_measures(F, o, s)
    res = dict(lats=lats, lons=lons, N=F.shape[0], T=F.shape[1], fixed_B=np.array(FIXED_B),
               rho_o=J.pooled_rho(F, np.broadcast_to(o, F.shape)),
               rho_m=J.pooled_rho(F, np.broadcast_to(s, F.shape)),
               rho_m_loo=J.pooled_rho(F, J.loo_means(F)),
               t_rho_o=rho_t[0], t_rho_m_loo=rho_t[1], t_rho_m_mean=rho_t[2])
    for key, kw in CONFIGS.items():
        kw = dict(kw, s=kw.get("s", s))                             # "full": the loaded s
        r = J.maps(F, o, **kw)
        res.update({f"{key}__{k}": v for k, v in r.items()})
        print(f"  {key:<16} {time.time() - t0:6.0f}s", flush=True)
    np.savez_compressed(out("point", var, case), **res)
    print(f"-> {out('point', var, case)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
