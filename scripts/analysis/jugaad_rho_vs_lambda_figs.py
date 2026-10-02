"""Side by side: RPC from Pearson rho (left) and from the thesis jugaad lambda (right).

    python scripts/analysis/jugaad_rho_vs_lambda_figs.py
    -> $SCRATCH/snp_jugaad_fd_optbins/figs/{png,json}/rho_vs_lambda_<case>.{png,json}

Left, the project's current rho measures (`smyle_metrics`):
    rho_o = corr(s, o)                       ensemble mean vs obs   (`pearson_coeff`)
    rho_m = < corr(s_-n, f_n) >_n            leave-one-out           (`ensemble_SNR_loo`)
    RPC_rho = rho_o / rho_m
The ensemble-mean rho_m, sd(s)/<sd(f_n)> (`ensemble_SNR`, the thesis `var_RPC` form), is
recorded in the json beside it.

Right, the thesis jugaad lambda as specified: I(f_jugaad; o_jugaad) and I(f_jugaad; s_jugaad)
with full-mean s, FD bins at alt = 1, lambda = lambda_o / lambda_m (from
jugaad_fd_optbins_run.py).

Marks (with boot_<VAR>_lead<LEAD>.npz from jugaad_rho_lambda_boot.py):
  black dot   RPC not significantly different from 1: the 5-95% interval of 1000 Eade
              draws (years with replacement, N-3 members without) contains 1. Both columns.
  red ring    right column only, around a black dot, where in addition the binning range
              |lambda(1.5x bins) - lambda(0.5x bins)| exceeds lambda at 1.0x bins.
  grey        corr(s, o) <= 0, masked in both columns as Eade et al. (2014) Fig. 1.
Without boot files the right column falls back to dots where the alt band contains 1.

The CMIP5 ensemble has member NaNs (2.8%); the rho measures here skip them, as the thesis
`var_RPC` did (`skipna`). On NaN-free data they equal `smyle_metrics` exactly (checked at run).
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.colors import Normalize                              # noqa: E402

import mapaxes as MAP                                                # noqa: E402
import smyle_metrics as SM                                           # noqa: E402
import jugaad_thesis as JT                                           # noqa: E402
from jugaad_fd_optbins_run import OUT, load                          # noqa: E402
from jugaad_fd_optbins_figs import (CASES, DIV, FIG, INK, LEAD_LABEL, MUTED,  # noqa: E402
                                    draw_panel, load_case, save, stip_size, wfrac, wmean)

BOOT_TAG = os.environ.get("BOOT_TAG", "")
# LAM_RULE=knuth: the lambda column from Knuth OptBins bins (`knuth_untiled`, I(f; o)
# numerator), its own bootstrap (boot_knuth_*, the same draws as the rho column), red rings
# from Knuth's alt band; s1961 and lead 2-4 only; -> rho_vs_lambda_<lead>_knuth.
LAM_RULE = os.environ.get("LAM_RULE", "fd")
LAM_KEY = {"fd": "fd_tiled", "knuth": "knuth_untiled"}[LAM_RULE]
TITLES = ("Pearson $\\rho$", "Granger-Lin $\\lambda$ (jugaad"
          + (", Knuth)" if LAM_RULE == "knuth" else ")"))
KNUTH_LEADS = ("s1961", "2-4")
RING = "#d40000"


def boot_case(var, lead):
    path = os.path.join(OUT, f"boot_{var}_lead{lead}{BOOT_TAG}.npz")
    return dict(np.load(path)) if os.path.exists(path) else None


def lam_boot_case(var, lead):
    path = os.path.join(OUT, f"boot_{LAM_RULE}_{var}_lead{lead}{BOOT_TAG}.npz")
    return dict(np.load(path)) if os.path.exists(path) else None


def draw_boot(ax, d, r, q05, q95, keep, ring=None):
    """Map r (masked where not keep), black dots where 5-95% contains 1, red rings on `ring`."""
    lats, lons = d["lats"], d["lons"]
    valid = np.isfinite(r) & keep
    MAP.show(ax, np.where(valid, np.clip(r, 0, 2), np.nan), lats, lons, cmap=DIV,
             norm=Normalize(0.0, 2.0))
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    ns = valid & (q05 <= 1) & (q95 >= 1)
    s = stip_size(lats, lons)
    ax.scatter(LO[ns], LA[ns], s=s, c="#000000", marker="o", linewidths=0,
               transform=MAP.DATA, zorder=5)
    rg = ns & ring if ring is not None else np.zeros_like(ns)
    if rg.any():
        ax.scatter(LO[rg], LA[rg], s=6 * s, facecolors="none", edgecolors=RING,
                   linewidths=0.6, marker="o", transform=MAP.DATA, zorder=6)
    return ns, rg, valid


def _corr_t(a, b):
    """corr along axis -3 over times where both are finite."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(-3)
    a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    am, bm = a0.sum(-3) / n, b0.sum(-3) / n
    da, db = np.where(ok, a - am[..., None, :, :], 0), np.where(ok, b - bm[..., None, :, :], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (da * db).sum(-3) / np.sqrt((da ** 2).sum(-3) * (db ** 2).sum(-3))


def rho_measures(F, o, s):
    """rho_o = corr(s, o); rho_m_loo = <corr(s_-n, f_n)>_n; rho_m_mean = sd(s)/<sd(f_n)>."""
    rho_o = _corr_t(s, o)
    L = JT.loo_means(F)
    rho_m_loo = np.nanmean(np.stack([_corr_t(L[n], F[n]) for n in range(F.shape[0])]), 0)
    rho_m_mean = np.nanstd(s, axis=0) / np.nanmean(np.nanstd(F, axis=1), axis=0)
    return rho_o, rho_m_loo, rho_m_mean


def check_against_smyle(seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((9, 40, 3, 4)) + rng.standard_normal((40, 3, 4))
    o = rng.standard_normal((40, 3, 4)) + F.mean(0)
    s = F.mean(0)
    ro, rl, rm = rho_measures(F, o, s)
    assert np.allclose(ro, SM.pearson_coeff(s, o), atol=1e-12)
    assert np.allclose(rl, SM.ensemble_SNR_loo(F), atol=1e-12)
    assert np.allclose(rm, SM.ensemble_SNR(F), atol=1e-12)
    print("rho measures == smyle_metrics on NaN-free data: OK")


def rho_case(var, lead):
    path = os.path.join(OUT, f"{var}_lead{lead}_rho.npz")
    if not os.path.exists(path):
        F, o, s, _, _ = load(var, lead)
        ro, rl, rm = rho_measures(F, o, s)
        np.savez_compressed(path, rho_o=ro, rho_m_loo=rl, rho_m_mean=rm)
    return dict(np.load(path))


def figure(lead, ds):
    rows = [d for d in ds if d["lead"] == lead]
    fig = plt.figure(figsize=(9.6, 2.8 * len(rows) + 0.8), layout="compressed")
    axs = [[MAP.add_ax(fig, len(rows), 2, 2 * i + k + 1) for k in range(2)]
           for i in range(len(rows))]
    res = {"lead": lead, "left": "corr(s,o) / <corr(s_-n, f_n)>_n",
           "right": f"jugaad lambda_o/lambda_m, {LAM_RULE} bins, alt=1, full-mean s, "
                    "lambda_o = lambda(I(f; o))",
           "stipple": "5-95% of Eade bootstrap contains 1 (both columns); red ring (right): "
                      "|lam(1.5x) - lam(0.5x)| > lam(1x); grey: corr(s,o) <= 0, and (right) lambda "
                      "undefined (lambda_m <= 0.02, e.g. Knuth picks one bin for s)", "rows": {}}
    for i, d in enumerate(rows):
        r = rho_case(d["var"], lead)
        rpc_rho = r["rho_o"] / r["rho_m_loo"]
        b = d["rules"][LAM_KEY]
        bt = boot_case(d["var"], lead)
        if bt is not None and LAM_RULE == "knuth":
            bk = lam_boot_case(d["var"], lead)
            bt = None if bk is None else dict(bt, rpc_lam=bk["lam"][1], lam_q05=bk["lam_q05"],
                                              lam_q95=bk["lam_q95"], boot_lam=int(bk["boot"]))
        W = d["W"]
        marks = {}
        if bt is None:
            draw_panel(axs[i][0], d, rpc_rho)
            draw_panel(axs[i][1], d, b["lam1"], np.nanmin(b["lam"], 0), np.nanmax(b["lam"], 0))
        else:
            assert np.allclose(bt["rpc_rho"], rpc_rho, equal_nan=True)
            both = np.isfinite(bt["rpc_lam"]) & np.isfinite(b["lam1"])   # b["lam1"] is floored
            assert np.allclose(bt["rpc_lam"][both], b["lam1"][both])
            keep = r["rho_o"] > 0
            k1 = int(np.argmin(np.abs(b["alts_arr"] - 1.0)))
            lo_, hi_ = int(np.argmin(b["alts_arr"])), int(np.argmax(b["alts_arr"]))
            bin_range = np.abs(b["lam"][hi_] - b["lam"][lo_])
            ring = bin_range > b["lam"][k1]
            ns0, _, v0 = draw_boot(axs[i][0], d, rpc_rho, bt["rho_q05"], bt["rho_q95"], keep)
            ns1, rg1, v1 = draw_boot(axs[i][1], d, b["lam1"], bt["lam_q05"], bt["lam_q95"],
                                     keep, ring)
            marks = {"boot": int(bt["boot"]), "masked_rho_o_le0": wfrac(W, ~keep, np.isfinite(r["rho_o"])),
                     "rho_not_sig": wfrac(W, ns0, v0), "rho_sig_gt1": wfrac(W, v0 & (bt["rho_q05"] > 1), v0),
                     "rho_sig_lt1": wfrac(W, v0 & (bt["rho_q95"] < 1), v0),
                     "lam_not_sig": wfrac(W, ns1, v1), "lam_sig_gt1": wfrac(W, v1 & (bt["lam_q05"] > 1), v1),
                     "lam_sig_lt1": wfrac(W, v1 & (bt["lam_q95"] < 1), v1),
                     "lam_ring": wfrac(W, rg1, v1),
                     "lam_binrange_gt_lam_all_cells": wfrac(W, ring, v1)}
        v_r, v_l = np.isfinite(rpc_rho), np.isfinite(b["lam1"])
        res["rows"][d["var"]] = {
            "N": d["N"], "T": d["T"],
            "rho": {"rho_o": wmean(W, r["rho_o"]), "rho_m_loo": wmean(W, r["rho_m_loo"]),
                    "rho_m_ensmean": wmean(W, r["rho_m_mean"]),
                    "rpc_loo": wmean(W, rpc_rho), "area_rpc_loo_gt1": wfrac(W, rpc_rho > 1, v_r),
                    "rpc_ensmean": wmean(W, r["rho_o"] / r["rho_m_mean"]),
                    "area_rpc_ensmean_gt1": wfrac(W, r["rho_o"] / r["rho_m_mean"] > 1,
                                                  np.isfinite(r["rho_m_mean"]))},
            "lambda": {"lam_o": wmean(W, b["lam_o1"]), "lam_m": wmean(W, b["lam_m1"]),
                       "rpc": wmean(W, b["lam1"]), "area_gt1": wfrac(W, b["lam1"] > 1, v_l)},
            "marks": marks,
            "spatial_corr_rpc": float(np.corrcoef(rpc_rho[v_r & v_l], b["lam1"][v_r & v_l])[0, 1])}
        for k in range(2):
            axs[i][k].set_title(TITLES[k], fontsize=10, color=INK)
        axs[i][0].text(-0.03, 0.5, d["var"], rotation=90, ha="right", va="center",
                       fontsize=10, color=INK, transform=axs[i][0].transAxes)
        axs[i][0].text(-0.08, 0.5, f"T={d['T']}", rotation=90, ha="right", va="center",
                       fontsize=8, color=MUTED, transform=axs[i][0].transAxes)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 2), cmap=DIV)
    cb = fig.colorbar(sm, ax=[a for r in axs for a in r], orientation="horizontal",
                      shrink=0.45, pad=0.02, label="RPC")
    cb.set_ticks([0, 0.5, 1, 1.5, 2])
    cb.set_ticklabels(["0", "0.5", "1", "1.5", "≥2"])
    axs[0][0].text(0.0, 1.14, f"{LEAD_LABEL[lead]}  N={rows[0]['N']}", fontsize=8,
                   color=MUTED, transform=axs[0][0].transAxes)
    save(fig, res, f"rho_vs_lambda_{lead}{BOOT_TAG}" + ("_knuth" if LAM_RULE == "knuth" else ""))
    return res


def main():
    check_against_smyle()
    have = [c for c in CASES if os.path.exists(os.path.join(OUT, f"{c[0]}_lead{c[1]}.npz"))
            and (LAM_RULE == "fd" or c[1] in KNUTH_LEADS)]
    ds = [load_case(*c) for c in have]
    out = {lead: figure(lead, ds) for lead in dict.fromkeys(d["lead"] for d in ds)}
    print(f"\n{'case':<20} {'rho_o':>6} {'rho_m LOO':>9} {'RPC_rho':>8} {'%>1':>6} | "
          f"{'rho_m ens':>9} {'RPC ens':>8} {'%>1':>6} || {'lam_o':>6} {'lam_m':>6} "
          f"{'RPC_lam':>8} {'%>1':>6} | {'r(maps)':>7}")
    for lead, res in out.items():
        for var, v in res["rows"].items():
            p, l = v["rho"], v["lambda"]
            print(f"{var + ' ' + lead:<20} {p['rho_o']:>6.3f} {p['rho_m_loo']:>9.3f} "
                  f"{p['rpc_loo']:>8.3f} {100 * p['area_rpc_loo_gt1']:>6.1f} | "
                  f"{p['rho_m_ensmean']:>9.3f} {p['rpc_ensmean']:>8.3f} "
                  f"{100 * p['area_rpc_ensmean_gt1']:>6.1f} || {l['lam_o']:>6.3f} "
                  f"{l['lam_m']:>6.3f} {l['rpc']:>8.3f} {100 * l['area_gt1']:>6.1f} | "
                  f"{v['spatial_corr_rpc']:>7.2f}")


if __name__ == "__main__":
    main()
