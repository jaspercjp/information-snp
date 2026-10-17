"""Two estimators of lambda_m on the six production cases, bias corrected: per-member vs pooled.

    python scripts/lam_m_definitions.py --case VAR:case     -> $SNP_OUT/lam_m_def_<VAR>_<case>.npz
    python scripts/lam_m_definitions.py --summarize         -> figures/data/lambda-m-definitions.json,
                                     figures/supp-info/{png,pdf}/lambda-m-definitions[-lead2-4]
    bash scripts/run_lam_m_definitions.sh                   all six cases + summary, on Slurm

    lambda_m^avg  = <lambda(I(s_-n; f_n) - null_n)>_n      each member on its T times (cf. rho_m^(2))
    lambda_m^pool = lambda(I(s_-n~; f~) - null)            the jugaad pooled N T pairs (production)
    lambda_o      = lambda(I(o~; s_-n~) - null)            production oloo, for the RPC
Knuth bins: tiled or one-series axes (o, s_-n via its T centres) on their T values, the pooled
members on all N T values (cap KNUTH_MAX), and per member both axes on their T values. The bias
correction subtracts the mean of P_NULL time-permutation nulls (gaussian_demo.py), permuting the
first series in time (one permutation for every member row); lambda = sqrt(1 - 2^(-2 I)) with I
clipped at 0. A one-bin axis gives I = 0 exactly: undefined (NaN), and lambda_m^avg averages over
the members with more than one bin. Area shares cos-lat weighted over rho_o > 0, as production.
"""
import argparse
import json
import os
import warnings
from multiprocessing import get_context

import numpy as np

from data import OUT, PRODUCTION, load
from jugaad import KNUTH_MAX, knuth_bins, lam_bits, loo_means, mi_bits

NAME = "lambda-m-definitions"
P_NULL = 4
SEED = 20261017
_G = {}


def knuth(v, cap):
    return knuth_bins(v, min(cap, v.size))


def info(X, Y, bins, rng):
    """I(X; Y) - mean of P_NULL nulls permuting X in time; X, Y (rows, T); NaN if one bin."""
    if min(bins) < 2:
        return np.nan
    x, y = X.ravel(), Y.ravel()
    null = np.mean([mi_bits(X[:, rng.permutation(X.shape[1])].ravel(), y, bins)
                    for _ in range(P_NULL)])
    return mi_bits(x, y, bins) - null


def cell(f, o, rng):
    f = np.nan_to_num(np.asarray(f, float))
    N, T = f.shape
    L = loo_means(f)
    c = L.mean(0)
    I_o = info(np.tile(o, (N, 1)), L, (knuth(o, T), knuth(c, T)), rng)
    I_p = info(L, f, (knuth(c, T), knuth(f.ravel(), KNUTH_MAX)), rng)
    I_n = [info(L[n][None], f[n][None], (knuth(L[n], T), knuth(f[n], T)), rng) for n in range(N)]
    return I_o, I_p, np.array(I_n)


def _row(j):
    F, o = _G["F"], _G["o"]
    N, nx = F.shape[0], F.shape[-1]
    rng = np.random.default_rng([SEED, j])
    Io, Ip, In = np.full(nx, np.nan), np.full(nx, np.nan), np.full((N, nx), np.nan)
    for i in range(nx):
        if not np.all(np.isnan(o[:, j, i])):
            Io[i], Ip[i], In[:, i] = cell(F[:, :, j, i], o[:, j, i], rng)
    return Io, Ip, In


def compute(var, case):
    F, o, s, lats, lons = load(var, case)
    _G.update(F=np.asarray(F, float), o=np.asarray(o, float))
    with get_context("fork").Pool(int(os.environ.get("SLURM_CPUS_ON_NODE", 1))) as pool:
        rows = pool.map(_row, range(F.shape[2]))
    _G.clear()
    I_o, I_p, I_n = (np.stack([r[k] for r in rows], axis=-2) for k in range(3))
    path = os.path.join(OUT, f"lam_m_def_{var}_{case}.npz")
    np.savez_compressed(path, lats=lats, lons=lons, N=F.shape[0], T=F.shape[1], I_o=I_o,
                        I_m_pool=I_p, I_m_n=I_n.astype(np.float32))
    print(f"{var} {case} -> {path}", flush=True)


def lam(I):
    return np.where(np.isfinite(I), lam_bits(np.nan_to_num(I)), np.nan)


def fields(var, case):
    z = dict(np.load(os.path.join(OUT, f"lam_m_def_{var}_{case}.npz")))
    z["rho_o"] = np.load(os.path.join(OUT, f"member_rho_m_all_{var}_{case}.npz"))["rho_o"]
    z["lam_o"], z["lam_m_pool"] = lam(z["I_o"]), lam(z["I_m_pool"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z["lam_m_avg"] = np.nanmean(lam(z["I_m_n"].astype(float)), axis=0)
    z["onebin_n"] = np.mean(np.isnan(z["I_m_n"]), axis=0)
    return z


def summary(z):
    from tables import wfrac, wmean
    W = np.cos(np.deg2rad(z["lats"]))[:, None] * np.ones(z["lons"].size)
    keep = z["rho_o"] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ra, rp = z["lam_o"] / z["lam_m_avg"], z["lam_o"] / z["lam_m_pool"]
    both = keep & np.isfinite(z["lam_m_avg"]) & np.isfinite(z["lam_m_pool"])
    va, vp = keep & np.isfinite(ra) & (z["lam_m_avg"] > 0), keep & np.isfinite(rp) & (z["lam_m_pool"] > 0)

    def above(r, v):
        m = v & (r > 1)
        return {"area": wfrac(W, m, v), "mean": wmean(W, r, m),
                "median": float(np.median(r[m])) if m.any() else np.nan}
    return {"N": int(z["N"]), "T": int(z["T"]),
            "area_lam_o_undefined": wfrac(W, ~np.isfinite(z["lam_o"]), keep),
            "area_lam_m_pool_undefined": wfrac(W, ~np.isfinite(z["lam_m_pool"]), keep),
            "area_mean_share_members_onebin": wmean(W, z["onebin_n"], keep),
            "area_mean_lam_m_avg": wmean(W, z["lam_m_avg"], both),
            "area_mean_lam_m_pool": wmean(W, z["lam_m_pool"], both),
            "area_mean_avg_minus_pool": wmean(W, z["lam_m_avg"] - z["lam_m_pool"], both),
            "share_avg_gt_pool": wfrac(W, z["lam_m_avg"] > z["lam_m_pool"], both),
            "rpc_gt1_avg": above(ra, va), "rpc_gt1_pool": above(rp, vp)}


def figure(cases, Z, S, name):
    import plots as P                                                # sets the figure style
    import style
    RHO, DIFF = P.Normalize(0.0, 1.0), P.Normalize(-0.2, 0.2)
    fig, axs = P.grid_fig(len(cases), 3, (14.4, 2.9 * len(cases) + 1.2))
    titles = (r"$\lambda_m^{\mathrm{avg}} = \langle \lambda(s_{-n}; f_n) \rangle_n$",
              r"$\lambda_m^{\mathrm{pool}} = \lambda(\tilde s_{-n}; \tilde f)$ (production)",
              r"$\lambda_m^{\mathrm{avg}} - \lambda_m^{\mathrm{pool}}$")
    for i, (var, case) in enumerate(cases):
        z, e = Z[var, case], S[f"{var} {case}"]
        P.show(axs[i][0], z["lam_m_avg"], z["lats"], z["lons"], cmap=P.SEQ, norm=RHO)
        P.show(axs[i][1], z["lam_m_pool"], z["lats"], z["lons"], cmap=P.SEQ, norm=RHO)
        P.show(axs[i][2], z["lam_m_avg"] - z["lam_m_pool"], z["lats"], z["lons"], cmap=P.DIV,
               norm=DIFF)
        notes = (f"mean {e['area_mean_lam_m_avg']:.2f} · RPC>1 {e['rpc_gt1_avg']['area']:.0%}",
                 f"mean {e['area_mean_lam_m_pool']:.2f} · RPC>1 {e['rpc_gt1_pool']['area']:.0%}",
                 f"mean {e['area_mean_avg_minus_pool']:+.2f}")
        for c in range(3):
            axs[i][c].text(0.01, -0.03, notes[c], transform=axs[i][c].transAxes, ha="left",
                           va="top", fontsize=8, color=P.MUTED)
            if i == 0:
                axs[i][c].set_title(titles[c])
        P.row_labels(axs[i][0], f"{var} {case}", f"N={int(z['N'])}, T={int(z['T'])}")
    P.colorbar(fig, [r[:2] for r in axs], 0.6, norm=RHO, cmap=P.SEQ,
               label=r"$\lambda_m$ (bias corrected)", ticks=(0, 0.25, 0.5, 0.75, 1),
               labels=("0", "0.25", "0.5", "0.75", "1"))
    P.colorbar(fig, [r[2:] for r in axs], 0.9, norm=DIFF,
               label=r"$\lambda_m^{\mathrm{avg}} - \lambda_m^{\mathrm{pool}}$",
               ticks=(-0.2, -0.1, 0, 0.1, 0.2), labels=("≤−0.2", "−0.1", "0", "0.1", "≥0.2"))
    style.save(fig, name, dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", help="VAR:case, compute one case")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.case:
        compute(*args.case.split(":"))
    if not args.summarize:
        return
    import style
    Z, S = {}, {}
    for var, case in [c for cs in PRODUCTION.values() for c in cs]:
        Z[var, case] = fields(var, case)
        e = S[f"{var} {case}"] = summary(Z[var, case])
        a, p = e["rpc_gt1_avg"], e["rpc_gt1_pool"]
        print(f"{var:6s} {case:6s} lam_m avg {e['area_mean_lam_m_avg']:.3f} pool "
              f"{e['area_mean_lam_m_pool']:.3f} (avg>pool on {e['share_avg_gt_pool']:.0%}) | "
              f"RPC>1 avg {a['area']:.2f} (median {a['median']:.2f}) pool {p['area']:.2f} "
              f"(median {p['median']:.2f}) | one-bin members {e['area_mean_share_members_onebin']:.2f}",
              flush=True)
    with open(style.data_path("lambda-m-definitions.json"), "w") as fh:
        json.dump(S, fh, indent=1)
    figure(PRODUCTION["decadal"], Z, S, NAME)
    figure(PRODUCTION["lead2-4"], Z, S, NAME + "-lead2-4")


if __name__ == "__main__":
    main()
