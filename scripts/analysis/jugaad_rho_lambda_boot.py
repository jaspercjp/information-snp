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
--rule knuth: Knuth OptBins (sized on the T distinct obs/s values) instead of FD, lambda
only, -> boot_knuth_<VAR>_lead<LEAD>.npz (or boot_knuth_oloo_* with --o-vs loo).

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


def rule_kw(rule):
    """jugaad_maps binning kwargs: thesis FD, or Knuth sized on the T distinct obs/s values."""
    from jugaad_fd_optbins_run import KNUTH_MAX
    return dict(rule="fd") if rule == "fd" else dict(rule="knuth", tiled=False, maxM=KNUTH_MAX)


def lam_stat(F, o, alts=(1.0,), o_vs="loo", rule="fd", s_m="full", loo_size="T"):
    """Jugaad lambda; lambda_o partner of the obs per `o_vs`; lambda_m against the full mean
    s (tiled, s_m="full") or the flattened leave-one-out means (s_m="loo", s_-n axis sized
    per `loo_size`; "T" = on the T cluster centres)."""
    s = np.nanmean(F, axis=0) if s_m == "full" else "loo"
    return JT.jugaad_maps(F, o, s=s, alts=alts, o_vs=o_vs, loo_size=loo_size, **rule_kw(rule))


def lam_path(a):
    """boot_oloo_* (FD, s_-n numerator: the original name), else boot_<rule>[_oloo]_*."""
    if a.rule == "fd" and a.o_vs == "loo" and a.s_m == "full":
        pre = "boot_oloo"
    else:
        pre = (f"boot_{a.rule}" + ("_oloo" if a.o_vs == "loo" else "")
               + (f"_lmloo{a.loo_size}" if a.s_m == "loo" else ""))
    return os.path.join(OUT, f"{pre}_{a.var}_lead{a.lead}{a.tag}.npz")


def main_lam(a):
    """lambda only, same draws as main(); point estimate with its alt 0.5/1/1.5 band."""
    t0 = time.time()
    F, o, _, lats, lons = load(a.var, a.lead)
    if a.ny:
        F, o, lats = F[:, :, :a.ny], o[:, :a.ny], lats[:a.ny]
    N, T = F.shape[:2]
    print(f"{a.var} {a.lead} (lambda only, rule={a.rule}, o_vs={a.o_vs}): N={N} T={T} "
          f"grid={F.shape[2:]} (load {time.time() - t0:.0f}s)", flush=True)
    kws = dict(o_vs=a.o_vs, rule=a.rule, s_m=a.s_m, loo_size=a.loo_size)
    pt = lam_stat(F, o, alts=(0.5, 1.0, 1.5), **kws)
    extra = {}
    if a.o_vs == "loo" and a.s_m == "full":
        both = JT.jugaad_maps(F, o, s="loo", alts=(1.0,), o_vs="loo", **rule_kw(a.rule))
        extra = dict(lam_both_loo=both["lam"][0], lam_m_loo=both["lam_m"][0])
    rng = np.random.default_rng([a.seed, sum(map(ord, a.var + a.lead))])
    B_lam = np.full((a.boot,) + F.shape[2:], np.nan, np.float32)
    t1 = time.time()
    for b in range(a.boot):
        ti = time_index(rng, T, a.lead)                       # identical rng calls to main()
        mi = rng.choice(N, N - 3, replace=False)
        Fb = F[mi[:, None], ti]                       # one copy; same values as F[mi][:, ti]
        B_lam[b] = lam_stat(Fb, o[ti], **kws)["lam"][0]
        if b in (0, 4) or (b + 1) % 100 == 0:
            print(f"  draw {b + 1}/{a.boot}  {(time.time() - t1) / (b + 1):.1f} s/draw",
                  flush=True)
    q = lambda X, p: np.nanpercentile(X, p, axis=0)                  # noqa: E731
    path = lam_path(a)
    np.savez_compressed(path, lats=lats, lons=lons, N=N, T=T, boot=a.boot, rule=a.rule,
                        o_vs=a.o_vs, s_m=a.s_m, loo_size=a.loo_size, alts=pt["alts"], lam=pt["lam"], lam_o=pt["lam_o"],
                        lam_m=pt["lam_m"], I_o=pt["I_o"], I_m=pt["I_m"], bins=pt["bins"],
                        lam_q05=q(B_lam, 5), lam_q95=q(B_lam, 95), B_lam=B_lam, **extra)
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
    ap.add_argument("--rule", default="fd", choices=("fd", "knuth"),
                    help="binning rule; anything but the default fd/f runs lambda only")
    ap.add_argument("--lam-only", action="store_true", help="lambda only, even for fd/f")
    ap.add_argument("--s-m", default="full", choices=("full", "loo"),
                    help="lambda_m partner of f: tiled full mean s or flattened s_-n")
    ap.add_argument("--loo-size", default="T", choices=("T", "pooled"),
                    help="s_-n axis bin sizing when --s-m loo")
    a = ap.parse_args()
    if a.o_vs == "loo" or a.rule != "fd" or a.lam_only or a.s_m == "loo":
        return main_lam(a)

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
        Fb = F[mi[:, None], ti]                       # one copy; same values as F[mi][:, ti]
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
