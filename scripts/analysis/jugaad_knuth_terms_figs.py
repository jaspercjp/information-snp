"""Maps of the three jugaad lambda terms with Knuth OptBins bins, s1961 and lead 2-4.

    python scripts/analysis/jugaad_knuth_terms_figs.py
    -> $SCRATCH/snp_jugaad_fd_optbins/figs/{png,json}/jugaad_knuth_lam_terms_<lead>.{png,json}

Columns, all at 1.0x Knuth bins (sized on the T distinct obs / s values and on the N*T
pooled f / s_-n, cap 200):
    lambda_o = lambda(I(f_jugaad; o_jugaad))       <case>.npz, knuth_untiled
    lambda_o = lambda(I(s_-n_jugaad; o_jugaad))    oloo_knuth_<case>.npz
    lambda_m = lambda(I(f_jugaad; s_jugaad))       full-mean s (identical in both files)
Rows: SLP, PRECT, TREFHT where obs exist (PRECT has none for s1961: GPCP starts in 1979).
Where Knuth picks one bin for an axis the plug-in I is exactly 0, so lambda is 0 there;
those shares are in the json. In s1961 (seasonal cycle kept) that happens for the obs axis
in 33% of TREFHT cells: a near-sinusoidal series has a flat, U-shaped distribution, which
Knuth's density posterior describes as well with one bin (best M > 1 scores -0.03 to -2).
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize     # noqa: E402

import mapaxes as MAP                                                # noqa: E402
from jugaad_fd_optbins_figs import INK, MUTED, save, wfrac, wmean    # noqa: E402
from jugaad_fd_optbins_run import OUT                                # noqa: E402

LEADS = {"s1961": ("SLP", "PRECT", "TREFHT"), "2-4": ("SLP", "PRECT", "TREFHT")}
LABEL = {"s1961": "DCPP-A s1961, monthly, linear detrend (thesis preprocessing)",
         "2-4": "DCPP-A lead 2-4 (DJF), across start dates"}
TITLES = ("$\\lambda_o$: $I(f; o)$", "$\\lambda_o$: $I(s_{-n}; o)$", "$\\lambda_m$: $I(f; s)$")
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9cc0ea", "#2a78d6", "#0e3f80"])
SEQ.set_bad(MAP.BED)


def fields(var, lead):
    pf, po = (os.path.join(OUT, f"{var}_lead{lead}.npz"),
              os.path.join(OUT, f"oloo_knuth_{var}_lead{lead}.npz"))
    if not (os.path.exists(pf) and os.path.exists(po)):
        return None
    z, q = np.load(pf), np.load(po)
    k1 = int(np.argmin(np.abs(z["knuth_untiled__alts"] - 1.0)))
    lm = z["knuth_untiled__lam_m"][k1]
    assert np.allclose(q["lam_m"][int(np.argmin(np.abs(q["alts"] - 1.0)))], lm, equal_nan=True)
    return dict(lats=z["lats"], lons=z["lons"], N=int(z["N"]), T=int(z["T"]),
                cols=(z["knuth_untiled__lam_o"][k1],
                      q["lam_o"][int(np.argmin(np.abs(q["alts"] - 1.0)))], lm),
                bins_f=z["knuth_untiled__bins"], bins_p=q["bins"])


def figure(lead):
    rows = [(v, fields(v, lead)) for v in LEADS[lead]]
    missing = [v for v, d in rows if d is None]
    rows = [(v, d) for v, d in rows if d is not None]
    fig = plt.figure(figsize=(14, 2.8 * len(rows) + 0.8), layout="compressed")
    axs = [[MAP.add_ax(fig, len(rows), 3, 3 * i + k + 1) for k in range(3)]
           for i in range(len(rows))]
    res = {"lead": lead, "bins": "Knuth OptBins at 1.0x", "missing": missing, "rows": {}}
    for i, (var, d) in enumerate(rows):
        W = np.cos(np.deg2rad(d["lats"]))[:, None] * np.ones(d["lons"].size)
        st = {}
        for k, (a, lab) in enumerate(zip(d["cols"], TITLES)):
            MAP.show(axs[i][k], a, d["lats"], d["lons"], cmap=SEQ, norm=Normalize(0, 1))
            axs[i][k].set_title(lab, fontsize=10, color=INK)
            v = np.isfinite(a)
            st[["lam_o_f", "lam_o_sloo", "lam_m"][k]] = {
                "area_mean": wmean(W, a), "p10": float(np.nanpercentile(a, 10)),
                "p90": float(np.nanpercentile(a, 90)), "area_zero": wfrac(W, a == 0, v)}
        bf, bp = d["bins_f"], d["bins_p"]
        st["bins_median_f_o_s"] = [int(np.median(b)) for b in bf[:3]]
        st["bins_median_sloo"] = int(np.median(bp[3]))
        st["share_one_bin_o_s"] = [float((bf[1] == 1).mean()), float((bf[2] == 1).mean())]
        res["rows"][var] = dict(N=d["N"], T=d["T"], **st)
        axs[i][0].text(-0.03, 0.5, var, rotation=90, ha="right", va="center", fontsize=10,
                       color=INK, transform=axs[i][0].transAxes)
        axs[i][0].text(-0.08, 0.5, f"T={d['T']}", rotation=90, ha="right", va="center",
                       fontsize=8, color=MUTED, transform=axs[i][0].transAxes)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=SEQ)
    cb = fig.colorbar(sm, ax=[a for r in axs for a in r], orientation="horizontal",
                      shrink=0.3, pad=0.02, label="$\\lambda$")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    fig.suptitle(f"{LABEL[lead]}  N={rows[0][1]['N']}  Knuth OptBins", x=0.02, ha="left",
                 fontsize=8, color=MUTED)       # figure-level: does not widen column 1
    save(fig, res, f"jugaad_knuth_lam_terms_{lead}")
    return res


def main():
    for lead in LEADS:
        res = figure(lead)
        for var, st in res["rows"].items():
            print(f"{lead:6s} {var:6s} " + "  ".join(
                f"{k} {st[k]['area_mean']:.3f} ({st[k]['p10']:.2f}-{st[k]['p90']:.2f}, "
                f"zero {100 * st[k]['area_zero']:.0f}%)" for k in ("lam_o_f", "lam_o_sloo", "lam_m")))
        if res["missing"]:
            print(f"{lead}: no data for {res['missing']}")


if __name__ == "__main__":
    main()
