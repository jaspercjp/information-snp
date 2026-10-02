"""Production figures: RPC from Pearson rho (left) and the jugaad lambda (right), 3 x 2.

    python scripts/analysis/rpc_rho_lambda_production.py
    -> $SCRATCH/snp_jugaad_fd_optbins/figs/{png,pdf,json}/rpc_rho_lambda_{lead2-4,decadal}.*

Figures (rows SLP, PRECT, TREFHT):
    lead2-4   DCPP-A DJF lead 2-4 across start dates
    decadal   DCPP-A monthly decade: SLP s1961, PRECT s1978 (GPCP starts 1979), TREFHT s1961

Left:  RPC_rho = corr(s, o) / < corr(s_-n, f_n) >_n             (boot_<case>.npz)
Right: RPC_lam = lambda_o / lambda_m, jugaad layout, Knuth OptBins bins:
       lambda_o = lambda(I(f_jugaad; o_jugaad)), obs tiled
       lambda_m = lambda(I(f_jugaad; s_-n_jugaad)), each member's rows against the mean of
                  the other members (flattened), s_-n axis sized on its T cluster centres
       point + alt band: lamm_loo_knuth_<case>.npz (T__*);
       draws: boot_knuth_lmlooT_fixbins_<case>.npz -- each draw keeps the cell's full-data
       Knuth bin counts (re-choosing Knuth on resampled years, which repeat exactly, would
       give 42-44 bins instead of 4: a different estimator from the one mapped)

Marks (method: figures/jugaad-fd-optbins/BOOTSTRAP.md):
    black dot   5-95% of 1000 Eade draws contains 1 (not significantly different from 1)
    red ring    right column, around a black dot, where |lam(1.5x bins) - lam(0.5x bins)| >
                lam(1.0x bins); the ring's inner edge sits on the dot's edge
    grey        corr(s, o) <= 0 (both columns); right column also lambda undefined
                (lambda_m <= 0.02)
Dot size scales with the SQUARE ROOT of the grid-cell width (1.6 pt^2 on the 1.875 deg
TREFHT grid), so coarse grids do not get oversized dots.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize     # noqa: E402

import mapaxes as MAP                                                # noqa: E402

OUT = os.path.join(os.environ["SCRATCH"], "snp_jugaad_fd_optbins")
FIG = os.path.join(OUT, "figs")
FIGS = {"lead2-4": [("SLP", "2-4"), ("PRECT", "2-4"), ("TREFHT", "2-4")],
        "decadal": [("SLP", "s1961"), ("PRECT", "s1978"), ("TREFHT", "s1961")]}
ROWLAB = {"2-4": "", "s1961": "s1961", "s1978": "s1978"}
INK, MUTED = "#0b0b0b", "#52514e"
RING = "#d40000"
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#e34948"])
DIV.set_bad(MAP.BED)
TITLES = ("Pearson $\\rho$", "Granger-Lin $\\lambda$")
FLOOR = 0.02
D_REF, W_REF = np.sqrt(1.6), 1.875          # dot diameter (pt) on the 1.875-deg grid
RING_LW_FRAC, RING_LW_MIN = 0.35, 0.3      # ring stroke as a fraction of the dot diameter


def dot_diameter(lats, lons):
    w = np.sqrt(np.median(np.abs(np.diff(lats))) * np.median(np.abs(np.diff(lons))))
    return D_REF * np.sqrt(w / W_REF)


def load(var, lead):
    b = np.load(os.path.join(OUT, f"boot_{var}_lead{lead}.npz"))
    k = np.load(os.path.join(OUT, f"boot_knuth_lmlooT_fixbins_{var}_lead{lead}.npz"))
    assert bool(k["fixed_bins"]) and str(k["s_m"]) == "loo" and str(k["loo_size"]) == "T"
    q = np.load(os.path.join(OUT, f"lamm_loo_knuth_{var}_lead{lead}.npz"))
    alts = q["T__alts"]
    i05, i1, i15 = (int(np.argmin(np.abs(alts - a))) for a in (0.5, 1.0, 1.5))
    lam_o, lam_m = q["T__lam_o"], q["T__lam_m"]
    with np.errstate(invalid="ignore", divide="ignore"):
        band = np.where(lam_m > FLOOR, lam_o / lam_m, np.nan)        # (n_alt, ny, nx)
    # the bootstrap file's own full-data point estimate must be the mapped one
    ok = np.isfinite(band[i1]) & np.isfinite(k["lam"][1])
    assert np.allclose(k["lam"][1][ok], band[i1][ok]), f"{var} {lead}: point mismatch"
    assert int(k["boot"]) == int(b["boot"]) == 1000
    return dict(var=var, lead=lead, lats=b["lats"], lons=b["lons"], N=int(b["N"]), T=int(b["T"]),
                rho=b["rpc_rho"], rho_q05=b["rho_q05"], rho_q95=b["rho_q95"], rho_o=b["rho_o"],
                lam=band[i1], lam_q05=k["lam_q05"], lam_q95=k["lam_q95"],
                binrange=np.abs(band[i15] - band[i05]))


def wfrac(W, m, v):
    return float((W * (m & v)).sum() / (W * v).sum())


def panel(ax, d, r, q05, q95, keep, ring=None):
    lats, lons = d["lats"], d["lons"]
    valid = np.isfinite(r) & keep
    im = MAP.show(ax, np.where(valid, np.clip(r, 0, 2), np.nan), lats, lons, cmap=DIV,
                  norm=Normalize(0.0, 2.0))
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    ns = valid & (q05 <= 1) & (q95 >= 1)
    dd = dot_diameter(lats, lons)
    ax.scatter(LO[ns], LA[ns], s=dd ** 2, c="#000000", marker="o", linewidths=0,
               transform=MAP.DATA, zorder=5)
    rg = np.zeros_like(ns) if ring is None else ns & ring
    if rg.any():
        lw = max(RING_LW_FRAC * dd, RING_LW_MIN)
        # stroke is centred on the marker path: diameter d + lw puts its inner edge on the dot
        ax.scatter(LO[rg], LA[rg], s=(dd + lw) ** 2, facecolors="none", edgecolors=RING,
                   linewidths=lw, marker="o", transform=MAP.DATA, zorder=6)
    W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
    return im, {"area_not_sig": wfrac(W, ns, valid),
                "area_sig_gt1": wfrac(W, valid & (q05 > 1), valid),
                "area_sig_lt1": wfrac(W, valid & (q95 < 1), valid),
                "area_ring": wfrac(W, rg, valid), "area_grey": 1 - float((W * valid).sum() / W.sum()),
                "area_mean": float((W * np.where(valid, r, 0)).sum() / (W * valid).sum()),
                "dot_diameter_pt": float(dd)}


def figure(name):
    rows = [load(v, l) for v, l in FIGS[name]]
    fig = plt.figure(figsize=(9.6, 8.6), layout="compressed")
    axs = [[MAP.add_ax(fig, 3, 2, 2 * i + k + 1) for k in range(2)] for i in range(3)]
    res = {"figure": name, "rows": {}}
    letters = iter("abcdef")
    for i, d in enumerate(rows):
        keep = d["rho_o"] > 0
        _, st_r = panel(axs[i][0], d, d["rho"], d["rho_q05"], d["rho_q95"], keep)
        im, st_l = panel(axs[i][1], d, d["lam"], d["lam_q05"], d["lam_q95"], keep,
                         ring=d["binrange"] > d["lam"])
        res["rows"][f"{d['var']} {d['lead']}"] = {"N": d["N"], "T": d["T"], "rho": st_r,
                                                  "lambda": st_l}
        for k in range(2):
            ax = axs[i][k]
            if i == 0:
                ax.set_title(TITLES[k], fontsize=11, color=INK)
            ax.text(0.01, 0.97, f"({next(letters)})", transform=ax.transAxes, ha="left",
                    va="top", fontsize=9, color=INK, zorder=20,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
        lab = d["var"] + (f" {ROWLAB[d['lead']]}" if ROWLAB[d["lead"]] else "")
        axs[i][0].text(-0.03, 0.5, lab, rotation=90, ha="right", va="center", fontsize=10,
                       color=INK, transform=axs[i][0].transAxes)
        axs[i][0].text(-0.085, 0.5, f"N={d['N']}, T={d['T']}", rotation=90, ha="right",
                       va="center", fontsize=8, color=MUTED, transform=axs[i][0].transAxes)
    sm = plt.cm.ScalarMappable(norm=Normalize(0.0, 2.0), cmap=DIV)    # continuous bar
    cb = fig.colorbar(sm, ax=[a for r in axs for a in r], orientation="horizontal",
                      shrink=0.45, pad=0.02, label="RPC")
    cb.set_ticks([0, 0.5, 1, 1.5, 2])
    cb.set_ticklabels(["0", "0.5", "1", "1.5", "≥2"])
    cb.solids.set_edgecolor("face")                  # no hairline seams between the strips
    cb.solids.set_rasterized(True)
    for sub in ("png", "pdf", "json"):
        os.makedirs(os.path.join(FIG, sub), exist_ok=True)
    base = f"rpc_rho_lambda_{name}"
    fig.savefig(os.path.join(FIG, "png", base + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "pdf", base + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(FIG, "json", base + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"-> {base}")
    return res


def main(names=None):
    for name in names or FIGS:
        res = figure(name)
        for case, v in res["rows"].items():
            r, l = v["rho"], v["lambda"]
            print(f"  {case:<14} rho: not sig {100 * r['area_not_sig']:5.1f}%  >1 {100 * r['area_sig_gt1']:5.1f}%"
                  f"  <1 {100 * r['area_sig_lt1']:5.1f}% | lam: not sig {100 * l['area_not_sig']:5.1f}%"
                  f"  >1 {100 * l['area_sig_gt1']:5.1f}%  <1 {100 * l['area_sig_lt1']:5.1f}%"
                  f"  rings {100 * l['area_ring']:5.1f}%  grey {100 * l['area_grey']:4.1f}%"
                  f" | dot {l['dot_diameter_pt']:.2f} pt")


if __name__ == "__main__":
    main(_sys.argv[1:] or None)
