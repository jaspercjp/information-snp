"""Every figure in figures/jugaad-fd-optbins, from the outputs of run.py and boot.py.

    python scripts/plots.py     -> figures/jugaad-fd-optbins/{png,pdf,json}/

    jugaad_lam_maps_<case>         FD lambda, Knuth lambda, pooled Pearson rho_o/rho_m; stippled
                                   where the alt 0.5-1.5 band of lambda contains 1
    jugaad_binning_uncertainty     dI against I per cell, FD and Knuth, with the 1:1 line
    jugaad_bins_sensitivity        area-mean I and lambda against a common bin count
    jugaad_knuth_lam_terms_<case>  lambda_o with I(f; o), with I(s_-n; o), and lambda_m, Knuth
    rho_vs_lambda_<case>           rho RPC | FD lambda RPC, Eade bootstrap dots (boot fd)
    rpc_rho_lambda[_oloo]_<fig>    production: rho | Knuth lambda with lambda_m from s_-n
                                   (boot lmloo) [| lambda_o from I(s_-n; o) (boot olmloo)]

Marks: black dot, the 5-95% interval of 1000 draws contains 1 (not significant); ring
around a dot, |lam(1.5x bins) - lam(0.5x bins)| > lam(1x bins); grey, corr(s, o) <= 0 or
lambda undefined (lambda_m <= 0.02). The json next to each png holds its area summaries.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize     # noqa: E402
import cartopy.crs as ccrs                                           # noqa: E402

from data import CASES, FIG, PRODUCTION, out                         # noqa: E402
from tables import FLOOR, RULES, SHORT, gauss_I, load_case, ratio, wfrac, wmean  # noqa: E402

PROJ, DATA = ccrs.PlateCarree(central_longitude=180), ccrs.PlateCarree()
BED, INK, MUTED, GRID = "0.75", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, RED, GREEN = "#2a78d6", "#eb6834", "#d40000", "#00a33a"
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#e34948"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9cc0ea", "#2a78d6", "#0e3f80"])
DIV.set_bad(BED)
SEQ.set_bad(BED)
LABEL = {"cmip5": "CMIP5 decadal1961 (thesis), monthly, linear detrend",
         "s1961": "DCPP-A s1961, monthly, linear detrend (thesis preprocessing)",
         "s1961moy": "DCPP-A s1961, monthly, month-of-year anomalies + linear detrend",
         "2-4": "DCPP-A lead 2-4 (DJF), across start dates"}
RPC_NORM = Normalize(0.0, 2.0)


# ---------------------------------------------------------------------- map panels

def _edges(x, lo=None, hi=None):
    x = np.asarray(x, float)
    mid = 0.5 * (x[1:] + x[:-1])
    e = np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])
    return e if lo is None else np.clip(e, lo, hi)


def add_ax(fig, nrows, ncols, idx):
    return fig.add_subplot(nrows, ncols, idx, projection=PROJ)


def show(ax, field, lats, lons, **kw):
    """field (nlat, nlon) on cell edges, haloed 110m coastlines on top."""
    ax.set_facecolor(BED)
    le, ae = _edges(lons), _edges(lats, -90.0, 90.0)
    im = ax.imshow(np.asarray(field, float), origin="lower", transform=DATA,
                   extent=[le[0], le[-1], ae[0], ae[-1]], **kw)
    ax.coastlines("110m", linewidth=0.45 * 1.5, color="white", alpha=0.75)
    ax.coastlines("110m", linewidth=0.45, color="0.10")
    ax.set_global()
    return im


def dots(ax, mask, lats, lons, s, **kw):
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    ax.scatter(LO[mask], LA[mask], s=s, marker="o", transform=DATA, **kw)


def stip_size(lats, lons):
    return 1.6 * np.median(np.abs(np.diff(lats))) * np.median(np.abs(np.diff(lons))) / (1.875 * 1.905)


def rpc_panel(ax, r, lats, lons, lo=None, hi=None, keep=True, ring=None, ring_color=RED, s=None):
    """RPC map, dots where lo <= 1 <= hi, rings on `ring` dots. Returns (dots, rings, valid)."""
    valid = np.isfinite(r) & keep
    show(ax, np.where(valid, np.clip(r, 0, 2), np.nan), lats, lons, cmap=DIV, norm=RPC_NORM)
    ns = valid & (lo <= 1) & (hi >= 1) if lo is not None else np.zeros_like(valid)
    dot = stip_size(lats, lons) if s is None else s
    dots(ax, ns, lats, lons, dot, c="#000000", linewidths=0, zorder=5)
    rg = ns & ring if ring is not None else np.zeros_like(ns)
    if rg.any():                         # production (s given): ring's inner edge on the dot
        lw = 0.6 if s is None else max(0.35 * np.sqrt(s), 0.3)
        dots(ax, rg, lats, lons, 6 * dot if s is None else (np.sqrt(s) + lw) ** 2,
             facecolors="none", edgecolors=ring_color, linewidths=lw, zorder=6)
    return ns, rg, valid


def row_labels(ax, var, sub):
    ax.text(-0.03, 0.5, var, rotation=90, ha="right", va="center", fontsize=10, color=INK,
            transform=ax.transAxes)
    ax.text(-0.08, 0.5, sub, rotation=90, ha="right", va="center", fontsize=8, color=MUTED,
            transform=ax.transAxes)


def colorbar(fig, axs, shrink, norm=RPC_NORM, cmap=DIV, label="RPC", ticks=(0, 0.5, 1, 1.5, 2),
             labels=("0", "0.5", "1", "1.5", "≥2")):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[a for r in axs for a in r],
                      orientation="horizontal", shrink=shrink, pad=0.02, label=label)
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    cb.solids.set_edgecolor("face")
    cb.solids.set_rasterized(True)


def grid_fig(nrows, ncols, size):
    fig = plt.figure(figsize=size, layout="compressed")
    return fig, [[add_ax(fig, nrows, ncols, ncols * i + k + 1) for k in range(ncols)]
                 for i in range(nrows)]


def save(fig, res, name, dpi=140, pdf=False):
    for sub in ("png", "json") + (("pdf",) if pdf else ()):
        os.makedirs(os.path.join(FIG, sub), exist_ok=True)
    fig.savefig(os.path.join(FIG, "png", name + ".png"), dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(FIG, "pdf", name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(FIG, "json", name + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"-> {name}.png", flush=True)


def q(B, p):
    return np.nanpercentile(B, p, axis=0)


# --------------------------------------------------------------- binning figures

def lam_maps(case, ds):
    rows = [d for d in ds if d["case"] == case]
    fig, axs = grid_fig(len(rows), 3, (14, 2.8 * len(rows) + 0.8))
    titles = ("FD (thesis) $\\lambda$", "Knuth OptBins $\\lambda$", "pooled Pearson $\\rho_o/\\rho_m$")
    for i, d in enumerate(rows):
        for k, (key, _) in enumerate(RULES):
            b = d["rules"][key]
            rpc_panel(axs[i][k], b["lam1"], d["lats"], d["lons"], np.nanmin(b["lam"], 0),
                      np.nanmax(b["lam"], 0))
        rpc_panel(axs[i][2], d["z"]["rho_o"] / d["z"]["rho_m"], d["lats"], d["lons"])
        for k in range(3):
            axs[i][k].set_title(titles[k], fontsize=10, color=INK)
        row_labels(axs[i][0], d["var"], f"T={d['T']}")
    colorbar(fig, axs, 0.3)
    axs[0][0].text(0.0, 1.14, f"{LABEL[case]}  N={rows[0]['N']}", fontsize=8, color=MUTED,
                   transform=axs[0][0].transAxes)
    save(fig, {"lead": case, "stipple": "alt 0.5-1.5 band of lambda contains 1"},
         f"jugaad_lam_maps_{case}")


def uncertainty(ds, n_show=3000, seed=0):
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(2, len(ds), figsize=(2.2 * len(ds), 4.8), layout="compressed")
    for j, d in enumerate(ds):
        for i, qq in enumerate(("I_o", "I_m")):
            ax, top = axs[i, j], 0.0
            for (key, _), col in zip(RULES, (BLUE, ORANGE)):
                x, y = d["rules"][key][f"{qq}1"].ravel(), d["rules"][key][f"d{qq}"].ravel()
                ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
                pick = rng.choice(ok, min(n_show, ok.size), replace=False)
                ax.scatter(x[pick], y[pick], s=3, color=col, alpha=0.35, linewidths=0, rasterized=True)
                top = max(top, np.nanquantile(x, 0.995), np.nanquantile(y, 0.995))
            ax.plot([0, top], [0, top], color=MUTED, lw=0.8)
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            style(ax)
            if j == 0:
                ax.set_ylabel(f"binning range $\\Delta {qq[0]}_{qq[-1]}$ (bits)", fontsize=8, color=MUTED)
            ax.set_xlabel(f"${qq[0]}_{qq[-1]}$ at alt=1 (bits)", fontsize=8, color=MUTED)
        axs[0, j].set_title(f"{d['var']}  {SHORT[d['case']]}", fontsize=9, color=INK)
    h = [plt.Line2D([], [], marker="o", color=c, ls="") for c in (BLUE, ORANGE)]
    fig.legend(h, [lab for _, lab in RULES], loc="outside lower center", ncol=2, fontsize=8, frameon=False)
    save(fig, {"x": "I at alt=1", "y": "max-min of I over alt 0.5/1/1.5",
               "points": f"random {n_show} cells per rule"}, "jugaad_binning_uncertainty")


def style(ax, grid=False):
    ax.tick_params(labelsize=7, colors=MUTED)
    if grid:
        ax.grid(color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def sensitivity(ds):
    fig, axs = plt.subplots(2, len(ds), figsize=(2.2 * len(ds), 5.0), sharex=True, layout="compressed")
    res = {}
    for j, d in enumerate(ds):
        z, W, B = d["z"], d["W"], d["z"]["fixed_B"]
        am = lambda key, k: [wmean(W, z[f"{key}{b}__{k}"][0]) for b in B]          # noqa: E731
        lr = lambda key: [wmean(W, ratio(z[f"{key}{b}__lam_o"], z[f"{key}{b}__lam_m"])[0]) for b in B]  # noqa: E731
        Io, Im, Iml, lam, laml = am("fixed_B", "I_o"), am("fixed_B", "I_m"), am("fixed_loo_B", "I_m"), \
            lr("fixed_B"), lr("fixed_loo_B")
        a, b_ = axs[0, j], axs[1, j]
        a.plot(B, Io, color=BLUE, lw=2)
        a.plot(B, Im, color=ORANGE, lw=2)
        a.plot(B, Iml, color=ORANGE, lw=2, ls="--")
        a.axhline(wmean(W, gauss_I(z["rho_o"])), color=BLUE, lw=1, ls=":")
        a.axhline(wmean(W, gauss_I(z["rho_m"])), color=ORANGE, lw=1, ls=":")
        b_.plot(B, lam, color=INK, lw=2)
        b_.plot(B, laml, color=INK, lw=2, ls="--")
        b_.axhline(wmean(W, z["rho_o"] / z["rho_m"]), color=INK, lw=1, ls=":")
        b_.axhline(1.0, color=MUTED, lw=0.6)
        row = {}
        for (key, _), mk in zip(RULES, ("o", "s")):
            r = d["rules"][key]
            xo, xm = (float(np.median(np.sqrt(r["bins"][0] * r["bins"][i]))) for i in (1, 2))
            row[{"fd": "fd_tiled", "knuth": "knuth_untiled"}[key]] = e = {}
            for ax, x, qq, col in ((a, xo, "I_o", BLUE), (a, xm, "I_m", ORANGE), (b_, np.sqrt(xo * xm), "lam", INK)):
                v = [wmean(W, r[qq][k]) for k in range(r[qq].shape[0])]
                ax.vlines(x, min(v), max(v), color=col, lw=1.5)
                ax.plot(x, wmean(W, r[f"{qq}1"]), mk, color=col, ms=7, mec="white", mew=1.2)
                e[qq] = {"alt_band": v, "x": x}
        res[f"{d['var']}_{d['case']}"] = dict(row, B=B.tolist(), I_o=Io, I_m=Im, I_m_loo=Iml, lam=lam, lam_loo=laml)
        a.set_title(f"{d['var']}  {SHORT[d['case']]}", fontsize=9, color=INK)
        for ax in (a, b_):
            ax.set_xscale("log")
            ax.set_xticks([2, 4, 8, 16, 32, 64])
            ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
            style(ax, grid=True)
        a.set_ylim(bottom=0)
        b_.set_ylim(0, 2)
        b_.set_xlim(1.7, 75)
        b_.set_xlabel("bins per axis", fontsize=8, color=MUTED)
    axs[0, 0].set_ylabel("area-mean MI (bits)", fontsize=8, color=MUTED)
    axs[1, 0].set_ylabel("area-mean ratio", fontsize=8, color=MUTED)
    L = plt.Line2D
    h = [L([], [], color=BLUE, lw=2), L([], [], color=ORANGE, lw=2), L([], [], color=ORANGE, lw=2, ls="--"),
         L([], [], color=INK, lw=2), L([], [], color=INK, lw=2, ls="--"), L([], [], color=MUTED, lw=1, ls=":"),
         L([], [], marker="o", color=MUTED, ls=""), L([], [], marker="s", color=MUTED, ls="")]
    fig.legend(h, ["$I_o$", "$I_m$, s", "$I_m$, $s_{-n}$", "$\\lambda$, s", "$\\lambda$, $s_{-n}$",
                   "Gaussian equivalent of pooled Pearson", "FD (thesis), bar = alt 0.5-1.5",
                   "Knuth, bar = alt 0.5-1.5"], loc="outside lower center", ncol=8, fontsize=8, frameon=False)
    save(fig, res, "jugaad_bins_sensitivity")


def knuth_terms(case, cases):
    rows = [load_case(v, c) for v, c in cases if c == case]
    fig, axs = grid_fig(len(rows), 3, (14, 2.8 * len(rows) + 0.8))
    titles = ("$\\lambda_o$: $I(f; o)$", "$\\lambda_o$: $I(s_{-n}; o)$", "$\\lambda_m$: $I(f; s)$")
    res = {"lead": case, "bins": "Knuth OptBins at 1.0x", "rows": {},
           "missing": [v for v in ("SLP", "PRECT", "TREFHT") if (v, case) not in cases]}
    for i, d in enumerate(rows):
        z, W, st = d["z"], d["W"], {}
        cols = (z["knuth__lam_o"][1], z["oloo_knuth__lam_o"][1], z["knuth__lam_m"][1])
        for k, (a, key) in enumerate(zip(cols, ("lam_o_f", "lam_o_sloo", "lam_m"))):
            show(axs[i][k], a, d["lats"], d["lons"], cmap=SEQ, norm=Normalize(0, 1))
            axs[i][k].set_title(titles[k], fontsize=10, color=INK)
            st[key] = {"area_mean": wmean(W, a), "p10": float(np.nanpercentile(a, 10)),
                       "p90": float(np.nanpercentile(a, 90)), "area_zero": wfrac(W, a == 0, np.isfinite(a))}
        bf = z["knuth__bins"]
        st.update(bins_median_f_o_s=[int(np.median(b)) for b in bf[:3]],
                  bins_median_sloo=int(np.median(z["oloo_knuth__bins"][3])),
                  share_one_bin_o_s=[float((bf[1] == 1).mean()), float((bf[2] == 1).mean())])
        res["rows"][d["var"]] = dict(N=d["N"], T=d["T"], **st)
        row_labels(axs[i][0], d["var"], f"T={d['T']}")
    colorbar(fig, axs, 0.3, Normalize(0, 1), SEQ, "$\\lambda$", (0, 0.25, 0.5, 0.75, 1),
             ("0.00", "0.25", "0.50", "0.75", "1.00"))
    fig.suptitle(f"{LABEL[case]}  N={rows[0]['N']}  Knuth OptBins", x=0.02, ha="left", fontsize=8, color=MUTED)
    save(fig, res, f"jugaad_knuth_lam_terms_{case}")


# ------------------------------------------------------------------ bootstrap maps

KNUTH_BOOT = ("2-4", "s1961")      # rho_vs_lambda: the lambda column from Knuth bins, boot.py knuth


def rho_vs_lambda(case, ds):
    rows = [d for d in ds if d["case"] == case]
    rule = "knuth" if case in KNUTH_BOOT else "fd"
    fig, axs = grid_fig(len(rows), 2, (9.6, 2.8 * len(rows) + 0.8))
    res = {"lead": case, "left": "corr(s,o) / <corr(s_-n, f_n)>_n",
           "right": f"jugaad lambda_o/lambda_m, {rule} bins, alt=1, full-mean s, lambda_o = lambda(I(f; o))",
           "stipple": "5-95% of Eade bootstrap contains 1 (both columns); red ring (right): "
                      "|lam(1.5x) - lam(0.5x)| > lam(1x); grey: corr(s,o) <= 0, and (right) lambda "
                      "undefined (lambda_m <= 0.02, e.g. Knuth picks one bin for s)", "rows": {}}
    for i, d in enumerate(rows):
        z, W, b = d["z"], d["W"], d["rules"][rule]
        bt, bl = (np.load(out(f"boot_{m}", d["var"], case)) for m in ("fd", rule))
        rpc_rho, keep = z["t_rho_o"] / z["t_rho_m_loo"], z["t_rho_o"] > 0
        assert np.allclose(bt["rpc_rho"], rpc_rho, equal_nan=True)
        ring = np.abs(b["lam"][2] - b["lam"][0]) > b["lam"][1]
        ns0, _, v0 = rpc_panel(axs[i][0], rpc_rho, d["lats"], d["lons"], q(bt["B_rho"], 5),
                               q(bt["B_rho"], 95), keep)
        lq05, lq95 = q(bl["B_lam"], 5), q(bl["B_lam"], 95)
        ns1, rg1, v1 = rpc_panel(axs[i][1], b["lam1"], d["lats"], d["lons"], lq05, lq95, keep, ring)
        rq05, rq95 = q(bt["B_rho"], 5), q(bt["B_rho"], 95)
        marks = {"boot": int(bt["B_rho"].shape[0]), "masked_rho_o_le0": wfrac(W, ~keep, np.isfinite(z["t_rho_o"])),
                 "rho_not_sig": wfrac(W, ns0, v0), "rho_sig_gt1": wfrac(W, v0 & (rq05 > 1), v0),
                 "rho_sig_lt1": wfrac(W, v0 & (rq95 < 1), v0), "lam_not_sig": wfrac(W, ns1, v1),
                 "lam_sig_gt1": wfrac(W, v1 & (lq05 > 1), v1), "lam_sig_lt1": wfrac(W, v1 & (lq95 < 1), v1),
                 "lam_ring": wfrac(W, rg1, v1), "lam_binrange_gt_lam_all_cells": wfrac(W, ring, v1)}
        rm = z["t_rho_o"] / z["t_rho_m_mean"]
        v_r, v_l = np.isfinite(rpc_rho), np.isfinite(b["lam1"])
        res["rows"][d["var"]] = {
            "N": d["N"], "T": d["T"],
            "rho": {"rho_o": wmean(W, z["t_rho_o"]), "rho_m_loo": wmean(W, z["t_rho_m_loo"]),
                    "rho_m_ensmean": wmean(W, z["t_rho_m_mean"]), "rpc_loo": wmean(W, rpc_rho),
                    "area_rpc_loo_gt1": wfrac(W, rpc_rho > 1, v_r), "rpc_ensmean": wmean(W, rm),
                    "area_rpc_ensmean_gt1": wfrac(W, rm > 1, np.isfinite(z["t_rho_m_mean"]))},
            "lambda": {"lam_o": wmean(W, b["lam_o1"]), "lam_m": wmean(W, b["lam_m1"]),
                       "rpc": wmean(W, b["lam1"]), "area_gt1": wfrac(W, b["lam1"] > 1, v_l)},
            "marks": marks,
            "spatial_corr_rpc": float(np.corrcoef(rpc_rho[v_r & v_l], b["lam1"][v_r & v_l])[0, 1])}
        for k, t in enumerate(("Pearson $\\rho$", "Granger-Lin $\\lambda$ (jugaad)")):
            axs[i][k].set_title(t, fontsize=10, color=INK)
        row_labels(axs[i][0], d["var"], f"T={d['T']}")
    colorbar(fig, axs, 0.45)
    axs[0][0].text(0.0, 1.14, f"{LABEL[case]}  N={rows[0]['N']}", fontsize=8, color=MUTED,
                   transform=axs[0][0].transAxes)
    save(fig, res, f"rho_vs_lambda_{case}")


def production(name, oloo=False):
    """3 x 2 (or 3 x 3 with oloo) production map: rho | Knuth lambda [| lambda_o from s_-n]."""
    nc = 3 if oloo else 2
    fig, axs = grid_fig(3, nc, (14.4 if oloo else 9.6, 8.6))
    res, letters = {"figure": name + ("_oloo" if oloo else ""), "rows": {}}, iter("abcdefghi")
    titles = ("Pearson $\\rho$", "Granger-Lin $\\lambda$") if not oloo else (
        "Pearson $\\rho$", "Granger-Lin $\\lambda$, $\\lambda_o$ from $I(\\tilde{f}; \\tilde{o})$",
        "Granger-Lin $\\lambda$, $\\lambda_o$ from $I(\\tilde{s}_{-n}; \\tilde{o})$")
    for i, (var, case) in enumerate(PRODUCTION[name]):
        bt = np.load(out("boot_fd", var, case))
        lats, lons, N, T = bt["lats"], bt["lons"], int(bt["N"]), int(bt["T"])
        W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
        w = np.sqrt(np.median(np.abs(np.diff(lats))) * np.median(np.abs(np.diff(lons))))
        s = (np.sqrt(1.6) * np.sqrt(w / 1.875)) ** 2               # dot area ~ sqrt(cell width)
        keep = bt["rho_o"] > 0
        cols = [("rho", bt["rpc_rho"], bt["B_rho"], None, None)]
        for key, mode, col in (("lambda", "lmloo", RED), ("lambda_oloo", "olmloo", GREEN))[:nc - 1]:
            k = np.load(out(f"boot_{mode}", var, case))
            band = ratio(k["lam_o"], k["lam_m"])
            cols.append((key, band[1], k["B_lam"], np.abs(band[2] - band[0]) > band[1], col))
        row = res["rows"][f"{var} {case}"] = {"N": N, "T": T}
        for c, (key, r, B, ring, rc) in enumerate(cols):
            lo, hi = q(B, 5), q(B, 95)
            ns, rg, valid = rpc_panel(axs[i][c], r, lats, lons, lo, hi, keep, ring, rc, s)
            row[key] = {"area_not_sig": wfrac(W, ns, valid), "area_sig_gt1": wfrac(W, valid & (lo > 1), valid),
                        "area_sig_lt1": wfrac(W, valid & (hi < 1), valid), "area_ring": wfrac(W, rg, valid),
                        "area_grey": 1 - float((W * valid).sum() / W.sum()),
                        "area_mean": float((W * np.where(valid, r, 0)).sum() / (W * valid).sum()),
                        "dot_diameter_pt": float(np.sqrt(s))}
            ax = axs[i][c]
            if i == 0:
                ax.set_title(titles[c], fontsize=11, color=INK)
            ax.text(0.01, 0.97, f"({next(letters)})", transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, color=INK, zorder=20,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
        lab = var + ("" if case == "2-4" else f" {case}")
        axs[i][0].text(-0.03, 0.5, lab, rotation=90, ha="right", va="center", fontsize=10, color=INK,
                       transform=axs[i][0].transAxes)
        axs[i][0].text(-0.085, 0.5, f"N={N}, T={T}", rotation=90, ha="right", va="center", fontsize=8,
                       color=MUTED, transform=axs[i][0].transAxes)
    colorbar(fig, axs, 0.45)
    save(fig, res, f"rpc_rho_lambda_{'oloo_' if oloo else ''}{name}", dpi=300, pdf=True)


def main():
    ds = [load_case(*c) for c in CASES]
    for case in dict.fromkeys(d["case"] for d in ds):
        lam_maps(case, ds)
        rho_vs_lambda(case, ds)
    uncertainty(ds)
    sensitivity(ds)
    for case in ("s1961", "2-4"):
        knuth_terms(case, CASES)
    for name in PRODUCTION:
        production(name)
        production(name, oloo=True)


if __name__ == "__main__":
    main()
