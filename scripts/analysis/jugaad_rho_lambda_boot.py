"""Eade-style bootstrap of the two RPCs in rho_vs_lambda: Pearson rho and the jugaad lambda.

    python scripts/analysis/jugaad_rho_lambda_boot.py SLP 2-4 [--boot 1000] [--ny 4]
    -> $SCRATCH/snp_jugaad_fd_optbins/boot_<VAR>_lead<LEAD>.npz

Statistics, recomputed on every draw from the resampled data:
    RPC_rho = corr(s, o) / < corr(s_-n, f_n) >_n            (smyle_metrics measures)
    RPC_lam = lambda_o / lambda_m, thesis jugaad: FD bins (sized on the draw), alt = 1,
              s = mean of the draw's members, f_jugaad = the draw's members flattened

One draw (Eade et al. 2014 SI, the same draw for both statistics):
  1. resample the time axis WITH replacement -- single start dates for a lead window,
     whole 12-month calendar years for the monthly decade cases (cmip5, sYYYY), so the
     seasonal cycle and within-year autocorrelation stay intact;
  2. take N-3 members WITHOUT replacement; the ensemble mean, the leave-one-out means and
     the jugaad s all come from those members.
--o-vs loo: the lambda statistic only, with lambda_o = lambda(I(s_-n_jugaad; o_jugaad))
(the obs paired with the mean of the OTHER members of the draw), lambda_m unchanged.
Same seed and the same rng calls, so its draws are exactly the draws of the default run
and the rho column of that run pairs with it. Writes boot_oloo_<VAR>_lead<LEAD>.npz with
the point estimate's alt 0.5/1/1.5 band (for the binning rings) and, at alt = 1, the
variant whose lambda_m also uses s_-n.

"RPC not different from 1" stands where the 5-95% interval of the draws contains 1
(90%, two-sided). Draws are stored so any other interval can be read off later.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import argparse
import os
import time

import numpy as np

import jugaad_thesis as JT                                           # noqa: E402
from jugaad_fd_optbins_run import OUT, load                          # noqa: E402
from jugaad_rho_vs_lambda_figs import rho_measures                   # noqa: E402

MONTHLY = ("cmip5",)


def time_index(rng, T, lead):
    """Resampled time indices: calendar-year blocks for monthly cases, else single steps."""
    if lead in MONTHLY or lead.startswith("s"):
        years = T // 12
        pick = rng.integers(0, years, years)
        return (pick[:, None] * 12 + np.arange(12)[None]).ravel()
    return rng.integers(0, T, T)


def stats(F, o):
    """(RPC_rho, RPC_lam) maps for one ensemble F (N, T, ny, nx) and obs o (T, ny, nx)."""
    s = np.nanmean(F, axis=0)
    ro, rl, _ = rho_measures(F, o, s)
    lam = JT.jugaad_maps(F, o, s=s, rule="fd", alts=(1.0,))
    with np.errstate(invalid="ignore", divide="ignore"):
        rpc_rho = ro / rl
    return rpc_rho, lam["lam"][0], ro


def lam_stat(F, o, alts=(1.0,)):
    """Jugaad lambda with lambda_o = lambda(I(s_-n; o)); full-mean s in lambda_m."""
    return JT.jugaad_maps(F, o, s=np.nanmean(F, axis=0), rule="fd", alts=alts, o_vs="loo")


def main_oloo(a):
    t0 = time.time()
    F, o, _, lats, lons = load(a.var, a.lead)
    if a.ny:
        F, o, lats = F[:, :, :a.ny], o[:, :a.ny], lats[:a.ny]
    N, T = F.shape[:2]
    print(f"{a.var} {a.lead} (o_vs=loo): N={N} T={T} grid={F.shape[2:]} "
          f"(load {time.time() - t0:.0f}s)", flush=True)
    pt = lam_stat(F, o, alts=(0.5, 1.0, 1.5))
    both = JT.jugaad_maps(F, o, s="loo", rule="fd", alts=(1.0,), o_vs="loo")
    rng = np.random.default_rng([a.seed, sum(map(ord, a.var + a.lead))])
    B_lam = np.full((a.boot,) + F.shape[2:], np.nan, np.float32)
    t1 = time.time()
    for b in range(a.boot):
        ti = time_index(rng, T, a.lead)                       # identical rng calls to main()
        mi = rng.choice(N, N - 3, replace=False)
        B_lam[b] = lam_stat(F[mi][:, ti], o[ti])["lam"][0]
        if b in (0, 4) or (b + 1) % 100 == 0:
            print(f"  draw {b + 1}/{a.boot}  {(time.time() - t1) / (b + 1):.1f} s/draw",
                  flush=True)
    q = lambda X, p: np.nanpercentile(X, p, axis=0)                  # noqa: E731
    path = os.path.join(OUT, f"boot_oloo_{a.var}_lead{a.lead}{a.tag}.npz")
    np.savez_compressed(path, lats=lats, lons=lons, N=N, T=T, boot=a.boot,
                        alts=pt["alts"], lam=pt["lam"], lam_o=pt["lam_o"], lam_m=pt["lam_m"],
                        I_o=pt["I_o"], I_m=pt["I_m"], bins=pt["bins"],
                        lam_both_loo=both["lam"][0], lam_m_loo=both["lam_m"][0],
                        lam_q05=q(B_lam, 5), lam_q95=q(B_lam, 95), B_lam=B_lam)
    print(f"-> {path}  ({(time.time() - t0) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("var")
    ap.add_argument("lead")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2014)
    ap.add_argument("--ny", type=int, help="first ny rows only (timing tests)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--o-vs", default="f", choices=("f", "loo"),
                    help="lambda_o partner of the obs: members (thesis) or s_-n")
    a = ap.parse_args()
    if a.o_vs == "loo":
        return main_oloo(a)

    t0 = time.time()
    F, o, _, lats, lons = load(a.var, a.lead)
    if a.ny:
        F, o, lats = F[:, :, :a.ny], o[:, :a.ny], lats[:a.ny]
    N, T = F.shape[:2]
    print(f"{a.var} {a.lead}: N={N} T={T} grid={F.shape[2:]} (load {time.time() - t0:.0f}s)",
          flush=True)
    rpc_rho, rpc_lam, rho_o = stats(F, o)
    rng = np.random.default_rng([a.seed, sum(map(ord, a.var + a.lead))])
    B_rho = np.full((a.boot,) + F.shape[2:], np.nan, np.float32)
    B_lam = np.full_like(B_rho, np.nan)
    t1 = time.time()
    for b in range(a.boot):
        ti = time_index(rng, T, a.lead)
        mi = rng.choice(N, N - 3, replace=False)
        Fb = F[mi][:, ti]
        B_rho[b], B_lam[b], _ = stats(Fb, o[ti])
        if b in (0, 4) or (b + 1) % 100 == 0:
            print(f"  draw {b + 1}/{a.boot}  {(time.time() - t1) / (b + 1):.1f} s/draw",
                  flush=True)
    q = lambda X, p: np.nanpercentile(X, p, axis=0)                  # noqa: E731
    path = os.path.join(OUT, f"boot_{a.var}_lead{a.lead}{a.tag}.npz")
    np.savez_compressed(path, lats=lats, lons=lons, N=N, T=T, boot=a.boot,
                        rpc_rho=rpc_rho, rpc_lam=rpc_lam, rho_o=rho_o,
                        rho_q05=q(B_rho, 5), rho_q95=q(B_rho, 95),
                        lam_q05=q(B_lam, 5), lam_q95=q(B_lam, 95), B_rho=B_rho, B_lam=B_lam)
    print(f"-> {path}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
