"""Binning uncertainty of the thesis jugaad MI and lambda, FD vs Knuth, from jugaad_fd_optbins_run.py.

    python scripts/analysis/jugaad_fd_optbins_figs.py
    -> $SCRATCH/snp_jugaad_fd_optbins/figs/{png,json}/ and figs/jugaad_fd_optbins_table.md

No permutation null: on the jugaad layout the obs are tiled N times, so permuting years
breaks the row alignment in a way that does not isolate "no skill". What is quantified
instead is how much the answer moves with the binning, set against the answer itself:

* the thesis band: each rule's bin counts scaled by alt = 0.5, 1, 1.5 (calc_RPC cell 19);
  dI = max - min of I over the band, compared with I at alt = 1;
* rule to rule: FD (thesis; sized on the tiled series) vs Knuth (sized on the T distinct
  obs/s values and the N*T pooled members);
* a common B on every axis, B = 2..64.

Figures
* `jugaad_lam_maps_<case>`: rows = variables, columns = FD lambda, Knuth lambda, pooled
  Pearson rho_o/rho_m (no binning). Stippled: the alt band of lambda contains 1, i.e. the
  binning decides which side of 1 the cell falls.
* `jugaad_binning_uncertainty`: dI against I per cell, for I_o (top) and I_m (bottom),
  FD and Knuth, with the 1:1 line (above it: the band is wider than the estimate).
* `jugaad_bins_sensitivity`: area-mean I_o, I_m in bits (top) and lambda (bottom) against a
  common B; FD and Knuth at the geometric-mean bin count of their two axes, with their
  alt band as a vertical bar; dotted: the Gaussian-equivalent -log2(1-rho^2)/2 of the
  pooled Pearson rho (and rho_o/rho_m below).
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
from jugaad_fd_optbins_run import OUT                                # noqa: E402

FIG = os.path.join(OUT, "figs")
# PRECT has no s1961: GPCP starts in 1979.
CASES = [("SLP", "cmip5"), ("SLP", "s1961"), ("TREFHT", "s1961"),
         ("SLP", "s1961moy"), ("TREFHT", "s1961moy"),
         ("SLP", "2-4"), ("PRECT", "2-4"), ("TREFHT", "2-4")]
LEAD_LABEL = {"cmip5": "CMIP5 decadal1961 (thesis), monthly, linear detrend",
              "s1961": "DCPP-A s1961, monthly, linear detrend (thesis preprocessing)",
              "s1961moy": "DCPP-A s1961, monthly, month-of-year anomalies + linear detrend",
              "2-4": "DCPP-A lead 2-4 (DJF), across start dates"}
SHORT = {"cmip5": "CMIP5", "s1961": "s1961", "s1961moy": "s1961 moy", "2-4": "lead 2-4"}
RULES = (("fd_tiled", "FD (thesis)"), ("knuth_untiled", "Knuth"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#e34948"])
DIV.set_bad(MAP.BED)
STIP_S, STIP_REF_CELL = 1.6, 1.875 * 1.905                           # as the figN figures
FLOOR = 0.02        # lambda_m at or below this -> ratio undefined, as jugaad_metrics.metrics
TITLES = ("FD (thesis) $\\lambda$", "Knuth OptBins $\\lambda$", "pooled Pearson $\\rho_o/\\rho_m$")


def stip_size(lats, lons):
    cell = np.median(np.abs(np.diff(lats))) * np.median(np.abs(np.diff(lons)))
    return STIP_S * cell / STIP_REF_CELL


def ratio(lam_o, lam_m):
    """lambda_o / lambda_m, NaN where lambda_m <= FLOOR: when a rule gives s one or two bins,
    I_m is float32 round-off (~1e-9 bits) and the raw ratio runs to 1e5."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lam_m > FLOOR, lam_o / lam_m, np.nan)


def zlam(z, key):
    return ratio(z[f"{key}__lam_o"], z[f"{key}__lam_m"])


def band(z, key):
    """Per-cell band over alts: dict of (n_alt, ny, nx) arrays plus alt=1 slices."""
    alts = z[f"{key}__alts"]
    k1 = int(np.argmin(np.abs(alts - 1.0)))
    d = {k: z[f"{key}__{k}"] for k in ("I_o", "I_m", "lam_o", "lam_m")}
    d["lam"] = zlam(z, key)
    d.update({f"{k}1": d[k][k1] for k in list(d)})
    for k in ("I_o", "I_m", "lam"):
        d[f"d{k}"] = np.nanmax(d[k], 0) - np.nanmin(d[k], 0)
    d["bins"] = z[f"{key}__bins"]
    d["cap_hit"] = z[f"{key}__cap_hit"] if f"{key}__cap_hit" in z.files else None
    return d


def gauss_I(rho):
    return -0.5 * np.log2(1 - np.minimum(rho, 0.999999) ** 2)


def load_case(var, lead):
    z = np.load(os.path.join(OUT, f"{var}_lead{lead}.npz"))
    lats, lons = z["lats"], z["lons"]
    W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
    return dict(z=z, var=var, lead=lead, lats=lats, lons=lons, W=W,
                N=int(z["N"]), T=int(z["T"]), rules={k: band(z, k) for k, _ in RULES})


def wmean(W, a, m=None):
    ok = np.isfinite(a) if m is None else (np.isfinite(a) & m)
    return float((W * np.where(ok, a, 0)).sum() / (W * ok).sum()) if ok.any() else np.nan


def wfrac(W, m, valid):
    return float((W * (m & valid)).sum() / (W * valid).sum())


# ------------------------------------------------------------------------------ maps

def draw_panel(ax, d, r, lo=None, hi=None):
    lats, lons = d["lats"], d["lons"]
    valid = np.isfinite(r)
    im = MAP.show(ax, np.where(valid, np.clip(r, 0, 2), np.nan), lats, lons, cmap=DIV,
                  norm=Normalize(0.0, 2.0))
    if lo is not None:
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        amb = valid & (lo <= 1) & (hi >= 1)
        ax.scatter(LO[amb], LA[amb], s=stip_size(lats, lons), c="#000000", marker="o",
                   linewidths=0, transform=MAP.DATA, zorder=5)


def maps(lead, ds):
    rows = [d for d in ds if d["lead"] == lead]
    fig = plt.figure(figsize=(14, 2.8 * len(rows) + 0.8), layout="compressed")
    axs = [[MAP.add_ax(fig, len(rows), 3, 3 * i + k + 1) for k in range(3)]
           for i in range(len(rows))]
    for i, d in enumerate(rows):
        for k, (key, _) in enumerate(RULES):
            b = d["rules"][key]
            draw_panel(axs[i][k], d, b["lam1"], np.nanmin(b["lam"], 0), np.nanmax(b["lam"], 0))
        draw_panel(axs[i][2], d, d["z"]["rho_o"] / d["z"]["rho_m"])
        for k in range(3):
            axs[i][k].set_title(TITLES[k], fontsize=10, color=INK)
        axs[i][0].text(-0.03, 0.5, d["var"], rotation=90, ha="right", va="center",
                       fontsize=10, color=INK, transform=axs[i][0].transAxes)
        axs[i][0].text(-0.08, 0.5, f"T={d['T']}", rotation=90, ha="right", va="center",
                       fontsize=8, color=MUTED, transform=axs[i][0].transAxes)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 2), cmap=DIV)
    cb = fig.colorbar(sm, ax=[a for r in axs for a in r], orientation="horizontal",
                      shrink=0.3, pad=0.02, label="RPC")
    cb.set_ticks([0, 0.5, 1, 1.5, 2])
    cb.set_ticklabels(["0", "0.5", "1", "1.5", "≥2"])
    axs[0][0].text(0.0, 1.14, f"{LEAD_LABEL[lead]}  N={rows[0]['N']}", fontsize=8,
                   color=MUTED, transform=axs[0][0].transAxes)
    save(fig, {"lead": lead, "stipple": "alt 0.5-1.5 band of lambda contains 1"},
         f"jugaad_lam_maps_{lead}")


# ------------------------------------------------------------------ uncertainty vs I

def uncertainty(ds, n_show=3000, seed=0):
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(2, len(ds), figsize=(2.2 * len(ds), 4.8), layout="compressed")
    for j, d in enumerate(ds):
        for i, q in enumerate(("I_o", "I_m")):
            ax = axs[i, j]
            top = 0.0
            for (key, _), col in zip(RULES, (BLUE, ORANGE)):
                b = d["rules"][key]
                x, y = b[f"{q}1"].ravel(), b[f"d{q}"].ravel()
                ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
                pick = rng.choice(ok, min(n_show, ok.size), replace=False)
                ax.scatter(x[pick], y[pick], s=3, color=col, alpha=0.35, linewidths=0,
                           rasterized=True)
                top = max(top, np.nanquantile(x, 0.995), np.nanquantile(y, 0.995))
            ax.plot([0, top], [0, top], color=MUTED, lw=0.8)
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.tick_params(labelsize=7, colors=MUTED)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"binning range $\\Delta {q[0]}_{q[-1]}$ (bits)", fontsize=8,
                              color=MUTED)
            ax.set_xlabel(f"${q[0]}_{q[-1]}$ at alt=1 (bits)", fontsize=8, color=MUTED)
        axs[0, j].set_title(f"{d['var']}  {SHORT[d['lead']]}", fontsize=9, color=INK)
    h = [plt.Line2D([], [], marker="o", color=c, ls="") for c in (BLUE, ORANGE)]
    fig.legend(h, [lab for _, lab in RULES], loc="outside lower center", ncol=2, fontsize=8,
               frameon=False)
    save(fig, {"x": "I at alt=1", "y": "max-min of I over alt 0.5/1/1.5",
               "points": f"random {n_show} cells per rule"}, "jugaad_binning_uncertainty")


# ---------------------------------------------------------------------- sensitivity

def gmean_bins(b, i):
    """Median over cells of sqrt(B_f * B_axis): where a rule sits on the common-B axis."""
    return float(np.median(np.sqrt(b[0] * b[i])))


def sensitivity(ds):
    fig, axs = plt.subplots(2, len(ds), figsize=(2.2 * len(ds), 5.0), sharex=True,
                            layout="compressed")
    res = {}
    for j, d in enumerate(ds):
        z, W = d["z"], d["W"]
        B = z["fixed_B"]
        Io = [wmean(W, z[f"fixed_B{b}__I_o"][0]) for b in B]
        Im = [wmean(W, z[f"fixed_B{b}__I_m"][0]) for b in B]
        Iml = [wmean(W, z[f"fixed_loo_B{b}__I_m"][0]) for b in B]
        lr = [wmean(W, zlam(z, f"fixed_B{b}")[0]) for b in B]
        lrl = [wmean(W, zlam(z, f"fixed_loo_B{b}")[0]) for b in B]
        a, b_ = axs[0, j], axs[1, j]
        a.plot(B, Io, color=BLUE, lw=2)
        a.plot(B, Im, color=ORANGE, lw=2)
        a.plot(B, Iml, color=ORANGE, lw=2, ls="--")
        a.axhline(wmean(W, gauss_I(z["rho_o"])), color=BLUE, lw=1, ls=":")
        a.axhline(wmean(W, gauss_I(z["rho_m"])), color=ORANGE, lw=1, ls=":")
        b_.plot(B, lr, color=INK, lw=2)
        b_.plot(B, lrl, color=INK, lw=2, ls="--")
        b_.axhline(wmean(W, z["rho_o"] / z["rho_m"]), color=INK, lw=1, ls=":")
        b_.axhline(1.0, color=MUTED, lw=0.6)
        row = {}
        for (key, _), mk in zip(RULES, ("o", "s")):
            r = d["rules"][key]
            xo, xm = gmean_bins(r["bins"], 1), gmean_bins(r["bins"], 2)
            e = {}
            for ax, x, q, col in ((a, xo, "I_o", BLUE), (a, xm, "I_m", ORANGE),
                                  (b_, np.sqrt(xo * xm), "lam", INK)):
                v = [wmean(W, r[q][k]) for k in range(r[q].shape[0])]
                ax.vlines(x, min(v), max(v), color=col, lw=1.5)
                ax.plot(x, wmean(W, r[f"{q}1"]), mk, color=col, ms=7, mec="white", mew=1.2)
                e[q] = {"alt_band": v, "x": x}
            row[key] = e
        res[f"{d['var']}_{d['lead']}"] = dict(row, B=B.tolist(), I_o=Io, I_m=Im,
                                              I_m_loo=Iml, lam=lr, lam_loo=lrl)
        a.set_title(f"{d['var']}  {SHORT[d['lead']]}", fontsize=9, color=INK)
        for ax in (a, b_):
            ax.set_xscale("log")
            ax.set_xticks([2, 4, 8, 16, 32, 64])
            ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.grid(color=GRID, lw=0.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        a.set_ylim(bottom=0)
        b_.set_ylim(0, 2)
        b_.set_xlim(1.7, 75)
        b_.set_xlabel("bins per axis", fontsize=8, color=MUTED)
    axs[0, 0].set_ylabel("area-mean MI (bits)", fontsize=8, color=MUTED)
    axs[1, 0].set_ylabel("area-mean ratio", fontsize=8, color=MUTED)
    h = [plt.Line2D([], [], color=BLUE, lw=2), plt.Line2D([], [], color=ORANGE, lw=2),
         plt.Line2D([], [], color=ORANGE, lw=2, ls="--"),
         plt.Line2D([], [], color=INK, lw=2), plt.Line2D([], [], color=INK, lw=2, ls="--"),
         plt.Line2D([], [], color=MUTED, lw=1, ls=":"),
         plt.Line2D([], [], marker="o", color=MUTED, ls=""),
         plt.Line2D([], [], marker="s", color=MUTED, ls="")]
    fig.legend(h, ["$I_o$", "$I_m$, s", "$I_m$, $s_{-n}$", "$\\lambda$, s",
                   "$\\lambda$, $s_{-n}$", "Gaussian equivalent of pooled Pearson",
                   "FD (thesis), bar = alt 0.5-1.5", "Knuth, bar = alt 0.5-1.5"],
               loc="outside lower center", ncol=8, fontsize=8, frameon=False)
    save(fig, res, "jugaad_bins_sensitivity")


# ---------------------------------------------------------------------------- table

def table(ds):
    out = {}
    md = ["Binning uncertainty vs estimate. I in bits at alt=1; dI = max-min over alt "
          "0.5/1/1.5; medians and shares over cells (cos-lat area-weighted shares).", "",
          "| case | rule | I_o | dI_o | med dI_o/I_o | % dI_o>I_o | I_m | dI_m | med dI_m/I_m "
          "| λ | dλ | % λ>1 | % band >1 | % band <1 | % band ∋1 | % λ undef. | bins f/o/s "
          "| % 1 bin f/o/s | cap f/o/s % |",
          "|" + "---|" * 19]
    for d in ds:
        z, W = d["z"], d["W"]
        cid = f"{d['var']} {SHORT[d['lead']]}"
        out[cid] = {"N": d["N"], "T": d["T"], "rules": {}}
        for key, lab in RULES:
            r = d["rules"][key]
            valid = np.isfinite(r["lam1"])
            lo, hi = np.nanmin(r["lam"], 0), np.nanmax(r["lam"], 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_o, rel_m = r["dI_o"] / r["I_o1"], r["dI_m"] / r["I_m1"]
            e = {"I_o": wmean(W, r["I_o1"]), "dI_o": wmean(W, r["dI_o"]),
                 "med_rel_dI_o": float(np.nanmedian(rel_o)),
                 "area_dI_o_gt_I_o": wfrac(W, r["dI_o"] > r["I_o1"], np.isfinite(r["I_o1"])),
                 "I_m": wmean(W, r["I_m1"]), "dI_m": wmean(W, r["dI_m"]),
                 "med_rel_dI_m": float(np.nanmedian(rel_m)),
                 "area_dI_m_gt_I_m": wfrac(W, r["dI_m"] > r["I_m1"], np.isfinite(r["I_m1"])),
                 "lam_o": wmean(W, r["lam_o1"]), "lam_m": wmean(W, r["lam_m1"]),
                 "lam": wmean(W, r["lam1"]), "dlam": wmean(W, r["dlam"]),
                 "area_lam_gt1": wfrac(W, r["lam1"] > 1, valid),
                 "area_band_gt1": wfrac(W, lo > 1, valid),
                 "area_band_lt1": wfrac(W, hi < 1, valid),
                 "area_band_contains1": wfrac(W, (lo <= 1) & (hi >= 1), valid),
                 "area_lam_undefined": 1 - wfrac(W, valid, np.isfinite(r["lam_o1"])),
                 "bins_median": [int(np.median(x)) for x in r["bins"]],
                 "bins_range": [[int(x.min()), int(x.max())] for x in r["bins"]],
                 "alt_I_o": [wmean(W, x) for x in r["I_o"]],
                 "alt_I_m": [wmean(W, x) for x in r["I_m"]]}
            e["bins_eq1_frac"] = [float((x == 1).mean()) for x in r["bins"]]
            if r["cap_hit"] is not None:
                e["cap_hit_frac"] = [float(x.mean()) for x in r["cap_hit"]]
            loo = f"{key}_loo"
            if f"{loo}__lam" in z.files:
                ll = zlam(z, loo)[0]
                e["loo"] = {"I_m": wmean(W, z[f"{loo}__I_m"][0]),
                            "lam_m": wmean(W, z[f"{loo}__lam_m"][0]),
                            "lam": wmean(W, ll), "area_lam_gt1": wfrac(W, ll > 1, np.isfinite(ll)),
                            "bins_s_median": int(np.median(z[f"{loo}__bins"][2]))}
            out[cid]["rules"][lab] = e
            cap = "/".join(f"{100 * c:.0f}" for c in e["cap_hit_frac"]) if "cap_hit_frac" in e else ""
            md.append(f"| {cid} | {lab} | {e['I_o']:.3f} | {e['dI_o']:.3f} | "
                      f"{e['med_rel_dI_o']:.2f} | {100 * e['area_dI_o_gt_I_o']:.0f} | "
                      f"{e['I_m']:.3f} | {e['dI_m']:.3f} | {e['med_rel_dI_m']:.2f} | "
                      f"{e['lam']:.3f} | {e['dlam']:.3f} | {100 * e['area_lam_gt1']:.1f} | "
                      f"{100 * e['area_band_gt1']:.1f} | {100 * e['area_band_lt1']:.1f} | "
                      f"{100 * e['area_band_contains1']:.1f} | "
                      f"{100 * e['area_lam_undefined']:.1f} | "
                      f"{'/'.join(map(str, e['bins_median']))} | "
                      f"{'/'.join(f'{100 * x:.0f}' for x in e['bins_eq1_frac'])} | {cap} |")
        fd, kn = d["rules"]["fd_tiled"], d["rules"]["knuth_untiled"]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[cid]["fd_minus_knuth"] = {
                "I_o": wmean(W, fd["I_o1"] - kn["I_o1"]), "I_m": wmean(W, fd["I_m1"] - kn["I_m1"]),
                "lam": wmean(W, fd["lam1"] - kn["lam1"]),
                "med_rel_I_o": float(np.nanmedian((fd["I_o1"] - kn["I_o1"]) / fd["I_o1"]))}
        B = z["fixed_B"]
        Io = [wmean(W, z[f"fixed_B{b}__I_o"][0]) for b in B]
        out[cid]["fixed_B"] = {"B": B.tolist(), "I_o": Io,
                               "I_m": [wmean(W, z[f"fixed_B{b}__I_m"][0]) for b in B],
                               "lam": [wmean(W, zlam(z, f"fixed_B{b}")[0]) for b in B]}
        out[cid]["pooled_pearson"] = {
            "rho_o": wmean(W, z["rho_o"]), "rho_m": wmean(W, z["rho_m"]),
            "rho_m_loo": wmean(W, z["rho_m_loo"]), "ratio": wmean(W, z["rho_o"] / z["rho_m"]),
            "gauss_I_o": wmean(W, gauss_I(z["rho_o"])), "gauss_I_m": wmean(W, gauss_I(z["rho_m"])),
            "area_ratio_gt1": wfrac(W, z["rho_o"] > z["rho_m"], np.isfinite(z["rho_o"]))}
    md += ["", "| case | FD−Knuth I_o | FD−Knuth I_m | FD−Knuth λ | I_o at B=2 → 64 | "
           "Gaussian-equiv. I_o, I_m | pooled ρ_o/ρ_m | % ρ_o/ρ_m>1 |", "|" + "---|" * 8]
    for cid, v in out.items():
        f, p, fx = v["fd_minus_knuth"], v["pooled_pearson"], v["fixed_B"]
        md.append(f"| {cid} | {f['I_o']:+.3f} | {f['I_m']:+.3f} | {f['lam']:+.3f} | "
                  f"{fx['I_o'][0]:.3f} → {fx['I_o'][-1]:.3f} | {p['gauss_I_o']:.3f}, "
                  f"{p['gauss_I_m']:.3f} | {p['ratio']:.3f} | {100 * p['area_ratio_gt1']:.1f} |")
    os.makedirs(os.path.join(FIG, "json"), exist_ok=True)
    with open(os.path.join(FIG, "json", "jugaad_fd_optbins_table.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(FIG, "jugaad_fd_optbins_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


def save(fig, res, name):
    for sub in ("png", "json"):
        os.makedirs(os.path.join(FIG, sub), exist_ok=True)
    fig.savefig(os.path.join(FIG, "png", name + ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(FIG, "json", name + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"-> {name}.png", flush=True)


def main():
    have = [c for c in CASES if os.path.exists(os.path.join(OUT, f"{c[0]}_lead{c[1]}.npz"))]
    ds = [load_case(*c) for c in have]
    for lead in dict.fromkeys(d["lead"] for d in ds):
        maps(lead, ds)
    uncertainty(ds)
    sensitivity(ds)
    table(ds)


if __name__ == "__main__":
    main()
