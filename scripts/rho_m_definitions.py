"""Two definitions of rho_m on the six production cases: signal-to-total ratio vs leave-one-out.

    python scripts/rho_m_definitions.py [--recompute]
        -> $SNP_OUT/rho_m_def1_<VAR>_<case>.npz, figures/data/rho-m-definitions.json,
           figures/supp-info/{png,pdf}/rho-m-definitions[-lead2-4]
    (needs member_rho_m.py's $SNP_OUT/member_rho_m_all_<VAR>_<case>.npz for rho_o and rho_m^(2))

    rho_m^(1) = sd(s) / sqrt(<var(f_n)>_n)     signal-to-total ratio, s the ensemble mean
    rho_m^(2) = <corr(f_n, s_-n)>_n            leave-one-out, production rho_m
For homogeneous members with pairwise correlation r, rho_m^(1) = sqrt(r + (1 - r)/N) (each member's
own share of s keeps it above 0 at r = 0) and rho_m^(2) = r / sqrt(r + (1 - r)/(N - 1)), so
rho_m^(1) >= rho_m^(2) and the rho RPC = rho_o / rho_m is smaller with definition 1. Also stored:
sd(s) / <sd(f_n)>_n (jugaad.rho_measures' third value), which is not definition 1 when member
variances differ. Variances over time per cell (ddof 0), NaN-skipping; area shares cos-lat
weighted over rho_o > 0, as production.
"""
import argparse
import json
import os

import numpy as np

import plots as P                                                    # sets the figure style
import style
from data import OUT, PRODUCTION, load
from tables import wfrac, wmean

NAME = "rho-m-definitions"
BANDS = 6
DIFF = P.Normalize(-0.1, 0.1)
RHO = P.Normalize(0.0, 1.0)


def band(F, s):
    vf = np.nanvar(F, axis=1)                                       # (N, ny, nx)
    vs = np.nanvar(s, axis=0)
    return np.sqrt(vs / np.nanmean(vf, axis=0)), np.sqrt(vs) / np.nanmean(np.sqrt(vf), axis=0)


def compute(var, case, path):
    F, o, s, lats, lons = load(var, case)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = [band(F[:, :, r], s[:, r]) for r in np.array_split(np.arange(F.shape[2]), BANDS)]
    np.savez_compressed(path, rho_m_1=np.concatenate([b[0] for b in out], axis=0),
                        sd_ratio=np.concatenate([b[1] for b in out], axis=0))


def summary(z):
    W = np.cos(np.deg2rad(z["lats"]))[:, None] * np.ones(z["lons"].size)
    keep = z["rho_o"] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        r1, r2 = z["rho_o"] / z["rho_m_1"], z["rho_o"] / z["rho_m"]
    v1, v2 = keep & np.isfinite(r1), keep & np.isfinite(r2)
    return {"N": int(z["N"]), "T": int(z["T"]),
            "area_mean_rho_m_1": wmean(W, z["rho_m_1"], keep),
            "area_mean_rho_m_2": wmean(W, z["rho_m"], keep),
            "area_mean_rho_m_1_minus_2": wmean(W, z["rho_m_1"] - z["rho_m"], keep),
            "area_mean_sd_ratio_minus_rho_m_1": wmean(W, z["sd_ratio"] - z["rho_m_1"], keep),
            "share_rho_m_1_gt_rho_m_2": wfrac(W, z["rho_m_1"] > z["rho_m"], keep),
            "area_rpc_gt1_def1": wfrac(W, r1 > 1, v1),
            "area_rpc_gt1_def2": wfrac(W, r2 > 1, v2)}


def figure(cases, Z, S, name):
    fig, axs = P.grid_fig(len(cases), 3, (14.4, 2.9 * len(cases) + 1.2))
    titles = (r"$\rho_m^{(1)} = \mathrm{sd}(s) / \sqrt{\langle \mathrm{var}(f_n) \rangle_n}$",
              r"$\rho_m^{(2)} = \langle \mathrm{corr}(f_n, s_{-n}) \rangle_n$ (production)",
              r"$\rho_m^{(1)} - \rho_m^{(2)}$")
    for i, (var, case) in enumerate(cases):
        z, e = Z[var, case], S[f"{var} {case}"]
        P.show(axs[i][0], z["rho_m_1"], z["lats"], z["lons"], cmap=P.SEQ, norm=RHO)
        P.show(axs[i][1], z["rho_m"], z["lats"], z["lons"], cmap=P.SEQ, norm=RHO)
        P.show(axs[i][2], z["rho_m_1"] - z["rho_m"], z["lats"], z["lons"], cmap=P.DIV, norm=DIFF)
        notes = (f"mean {e['area_mean_rho_m_1']:.2f} · RPC>1 {e['area_rpc_gt1_def1']:.0%}",
                 f"mean {e['area_mean_rho_m_2']:.2f} · RPC>1 {e['area_rpc_gt1_def2']:.0%}",
                 f"mean {e['area_mean_rho_m_1_minus_2']:+.2f}")
        for c in range(3):
            axs[i][c].text(0.01, -0.03, notes[c], transform=axs[i][c].transAxes, ha="left",
                           va="top", fontsize=8, color=P.MUTED)
            if i == 0:
                axs[i][c].set_title(titles[c])
        P.row_labels(axs[i][0], f"{var} {case}", f"N={int(z['N'])}, T={int(z['T'])}")
    P.colorbar(fig, [r[:2] for r in axs], 0.6, norm=RHO, cmap=P.SEQ, label=r"$\rho_m$",
               ticks=(0, 0.25, 0.5, 0.75, 1), labels=("≤0", "0.25", "0.5", "0.75", "1"))
    P.colorbar(fig, [r[2:] for r in axs], 0.9, norm=DIFF, label=r"$\rho_m^{(1)} - \rho_m^{(2)}$",
               ticks=(-0.1, -0.05, 0, 0.05, 0.1), labels=("≤−0.1", "−0.05", "0", "0.05", "≥0.1"))
    style.save(fig, name, dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    Z, S = {}, {}
    for var, case in [c for cs in PRODUCTION.values() for c in cs]:
        path = os.path.join(OUT, f"rho_m_def1_{var}_{case}.npz")
        if args.recompute or not os.path.exists(path):
            compute(var, case, path)
        z = dict(np.load(os.path.join(OUT, f"member_rho_m_all_{var}_{case}.npz")))
        z.update(np.load(path))
        Z[var, case] = z
        e = S[f"{var} {case}"] = summary(z)
        print(f"{var:6s} {case:6s} rho_m^(1) {e['area_mean_rho_m_1']:.3f} rho_m^(2) "
              f"{e['area_mean_rho_m_2']:.3f} (1>2 on {e['share_rho_m_1_gt_rho_m_2']:.0%}) | "
              f"rho RPC>1: def1 {e['area_rpc_gt1_def1']:.3f} def2 {e['area_rpc_gt1_def2']:.3f} | "
              f"sd-ratio - def1 {e['area_mean_sd_ratio_minus_rho_m_1']:+.4f}", flush=True)
    with open(style.data_path("rho-m-definitions.json"), "w") as fh:
        json.dump(S, fh, indent=1)
    figure(PRODUCTION["decadal"], Z, S, NAME)
    figure(PRODUCTION["lead2-4"], Z, S, NAME + "-lead2-4")


if __name__ == "__main__":
    main()
