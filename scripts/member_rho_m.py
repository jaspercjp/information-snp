"""rho_m from each member as pseudo-observation: how far does the rho RPC > 1 area move?

    python scripts/member_rho_m.py [--recompute]
        -> $SNP_OUT/member_rho_m_all_<VAR>_<case>.npz, figures/data/rho-m-member-all.json,
           figures/supp-info/{png,pdf}/rho-m-member-rpc-all[-lead2-4]
    (~1.5 min for all six cases on one node: run it in an allocation, no job array needed)

The rho counterpart of member_lam_m.py, every member of the six production cases, unpooled as
production. Production rho RPC (boot.py fd, plots.py production) = rho_o / rho_m with
    rho_o = corr(s, o),      rho_m = <rho_m^(k)>_k,      rho_m^(k) = corr(s_-k, f_k),
s_-k the mean of the other N - 1 members: production rho_m is already the average over members of
each member taken as the observation. Here each rho_m^(k) is kept, with RPC_k = rho_o / rho_m^(k):
the spread over k is what using one member instead of the average does. Signed correlations
(jugaad.corr_t), area shares cos-lat weighted over rho_o > 0.
"""
import argparse
import json
import os

import numpy as np

import jugaad as J
import plots as P                                                    # sets the figure style
import style
from data import OUT, PRODUCTION, load
from tables import wfrac, wmean

NAME = "rho-m-member-rpc-all"
BANDS = 6                   # latitude bands: per-cell quantities, so this is exact


def band(F, o, s):
    """rho_o = corr(s, o) and rho_m^(k) = corr(s_-k, f_k) for every member k."""
    L = J.loo_means(F)
    return J.corr_t(s, o), np.stack([J.corr_t(L[k], F[k]) for k in range(F.shape[0])])


def compute(var, case, path):
    F, o, s, lats, lons = load(var, case)
    out = [band(F[:, :, r], o[:, r], s[:, r]) for r in np.array_split(np.arange(F.shape[2]), BANDS)]
    rho_o = np.concatenate([b[0] for b in out], axis=0)
    rho_m_k = np.concatenate([b[1] for b in out], axis=1)
    rho_m = J.rho_measures(F, o, s)[1]                              # production, for the check
    np.savez_compressed(path, lats=lats, lons=lons, N=F.shape[0], T=F.shape[1], rho_o=rho_o,
                        rho_m=rho_m, rho_m_k=rho_m_k)


def summary(z):
    W = np.cos(np.deg2rad(z["lats"]))[:, None] * np.ones(z["lons"].size)
    keep = z["rho_o"] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        r0 = z["rho_o"] / z["rho_m"]
        rk = z["rho_o"][None] / z["rho_m_k"]
    valid = keep & np.isfinite(r0)
    pm = np.array([wfrac(W, np.isfinite(r) & (r > 1), valid) for r in rk])
    frac = np.mean(rk > 1, axis=0)                                   # share of members with RPC_k > 1
    return {"N": int(z["N"]), "T": int(z["T"]), "area_valid": float((W * valid).sum() / W.sum()),
            "check_mean_k_minus_production_rho_m_maxabs": float(np.nanmax(np.abs(
                np.nanmean(z["rho_m_k"], axis=0) - z["rho_m"]))),
            "area_rpc_gt1_production": wfrac(W, r0 > 1, valid),
            "area_rpc_gt1_member": pm.tolist(),
            "area_rpc_gt1_member_quantiles_5_50_95": np.percentile(pm, [5, 50, 95]).tolist(),
            "area_mean_share_of_members_rpc_gt1": wmean(W, frac, valid),
            "area_mean_rho_m": wmean(W, z["rho_m"], valid),
            "area_mean_rho_m_k_sd": wmean(W, np.nanstd(z["rho_m_k"], axis=0), valid)}


def figure(cases, Z, name):
    fig, axs = P.grid_fig(len(cases), 3, (14.4, 2.9 * len(cases) + 1.2))
    titles = (r"$\rho_o / \rho_m$ (production, $\rho_m = \langle\rho_m^{(k)}\rangle_k$)",
              r"share of members with $\rho_o / \rho_m^{(k)} > 1$",
              r"spread of $\rho_m^{(k)}$: sd over $k$ / $\rho_m$")
    SH = P.Normalize(0.0, 1.0)
    for i, (var, case) in enumerate(cases):
        z = Z[var, case]
        keep = z["rho_o"] > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            P.rpc_panel(axs[i][0], z["rho_o"] / z["rho_m"], z["lats"], z["lons"], keep=keep)
            share = np.where(keep, np.mean(z["rho_o"][None] / z["rho_m_k"] > 1, axis=0), np.nan)
            P.show(axs[i][1], share, z["lats"], z["lons"], cmap=P.DIV, norm=SH)
            cv = np.where(z["rho_m"] > 0, np.nanstd(z["rho_m_k"], axis=0) / z["rho_m"], np.nan)
            P.show(axs[i][2], cv, z["lats"], z["lons"], cmap=P.SEQ, norm=P.Normalize(0, 0.5))
        for c in range(3):
            if i == 0:
                axs[i][c].set_title(titles[c])
        P.row_labels(axs[i][0], f"{var} {case}", f"N={int(z['N'])}, T={int(z['T'])}")
    P.colorbar(fig, [r[:1] for r in axs], 0.9)
    P.colorbar(fig, [r[1:2] for r in axs], 0.9, norm=SH, label="share of members",
               ticks=(0, 0.25, 0.5, 0.75, 1), labels=("0", "0.25", "0.5", "0.75", "1"))
    P.colorbar(fig, [r[2:] for r in axs], 0.9, norm=P.Normalize(0, 0.5), cmap=P.SEQ,
               label=r"sd$_k$ / $\rho_m$", ticks=(0, 0.25, 0.5), labels=("0", "0.25", "≥0.5"))
    style.save(fig, name, dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    path = lambda var, case: os.path.join(OUT, f"member_rho_m_all_{var}_{case}.npz")  # noqa: E731
    os.makedirs(OUT, exist_ok=True)
    Z, res = {}, {}
    for var, case in [c for cs in PRODUCTION.values() for c in cs]:
        if args.recompute or not os.path.exists(path(var, case)):
            compute(var, case, path(var, case))
        Z[var, case] = dict(np.load(path(var, case)))
        e = res[f"{var} {case}"] = summary(Z[var, case])
        q = e["area_rpc_gt1_member_quantiles_5_50_95"]
        print(f"{var:6s} {case:6s} area {e['area_valid']:.2f} | rho RPC>1: production "
              f"{e['area_rpc_gt1_production']:.3f}; single members 5/50/95% {q[0]:.2f}/{q[1]:.2f}/"
              f"{q[2]:.2f} | members with RPC_k>1 {e['area_mean_share_of_members_rpc_gt1']:.2f} | "
              f"rho_m {e['area_mean_rho_m']:.3f} sd_k {e['area_mean_rho_m_k_sd']:.3f} | "
              f"check {e['check_mean_k_minus_production_rho_m_maxabs']:.1e}", flush=True)
    with open(style.data_path("rho-m-member-all.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    figure(PRODUCTION["decadal"], Z, NAME)
    figure(PRODUCTION["lead2-4"], Z, NAME + "-lead2-4")


if __name__ == "__main__":
    main()
