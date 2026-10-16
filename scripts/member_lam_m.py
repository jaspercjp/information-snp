"""lambda_m from one member as pseudo-observation, against the production lambda_m: does RPC > 1 move?

    python scripts/member_lam_m.py [--members 5|all] [--recompute]
        -> $SNP_OUT/member_lam_m[_all]_<VAR>_<case>.npz, figures/data/lambda-m-member[-all].json,
           figures/supp-info/{png,pdf}/lambda-m-member-rpc[-all][-lead2-4]
    python scripts/member_lam_m.py --members all --case TREFHT:s1961     one case, npz only
    python scripts/member_lam_m.py --members all --summarize             json + figures only
    bash scripts/run_member_lam_m.sh       all members, one Slurm array task per case + summary

The production oloo RPC (boot.py olmloo point estimate) is lambda_o / lambda_m with
    lambda_o = lambda(I(s_-n~; o~))      the N leave-one-out means against o, tiled (jugaad)
    lambda_m = lambda(I(s_-n~; f~))      each s_-n against its own member.
The variant treats member k exactly as o is treated: drop it from the ensemble, take the other
N - 1 members' leave-one-out means s'_-n (each over N - 2), and pair them with f_k tiled,
    lambda_m^(k) = lambda(I(s'_-n~; f_k~)),
the same jugaad.maps call as lambda_o with o -> f_k, F -> F without k (Knuth, untiled, s_-n sized
on its T centres, alt 1). f_k is not in its own predictor, as o is not. K members, evenly
spaced over the (multi-model) member axis; RPC_k = lambda_o / lambda_m^(k), and RPC_mean uses the
mean of lambda_m^(k) over k. Point estimates only. Area shares are cos-lat weighted over the
cells production keeps (rho_o > 0, lambda_m > FLOOR).

One-bin cells: Knuth on the T = 120 values of a single series (o, f_k, or the s_-n centres)
sometimes picks M = 1, and one bin gives I = 0 exactly, so lambda = 0 (production lambda_o is 0
there and RPC = 0; SLP s1961: 469 of 2664 cells, every one a one-bin cell). The json reports
the shares with every cell ("all") and with lambda = 0 treated as undefined ("defined"); the
figure shows the defined version, one-bin cells grey, and averages lambda_m^(k) over the members
that have more than one bin.
"""
import argparse
import json
import os
import warnings

import numpy as np

import jugaad as J
import plots as P                                                    # sets the figure style
import style
from data import OUT, PRODUCTION, load
from tables import FLOOR, ratio, wfrac, wmean

NAME = "lambda-m-member-rpc"
OLOO = dict(rule="knuth", tiled=False, s="loo", loo_size="T", o_vs="loo", alts=(1.0,))
BANDS = 6                   # latitude bands per maps call: bins are per cell, so this is exact


def maps(F, o):
    """(lam_o, lam_m) of jugaad.maps(F, o, **OLOO) at alt 1, in latitude bands (the largest
    case is ~10 GB at once otherwise)."""
    rows = np.array_split(np.arange(F.shape[2]), BANDS)
    out = [J.maps(F[:, :, r], o[:, r], **OLOO) for r in rows]
    return tuple(np.concatenate([m[k][0] for m in out], axis=0) for k in ("lam_o", "lam_m"))


def compute(var, case, K, path):
    """K evenly spaced members (K = None: all of them). Writes path; resumes from path + '.part'
    (saved every 10 members) if a previous run was cut off."""
    F, o, s, lats, lons = load(var, case)
    N = F.shape[0]
    ks = np.arange(N) if K is None else np.unique(np.round(np.linspace(0, N - 1, K)).astype(int))
    part = path + ".part.npz"
    if os.path.exists(part):
        z = dict(np.load(part))
        lam_o, lam_m, rho_o, done = z["lam_o"], z["lam_m"], z["rho_o"], list(z["lam_m_k"])
        print(f"{var} {case}: resuming at member {len(done)}/{len(ks)}", flush=True)
    else:
        (lam_o, lam_m), rho_o, done = maps(F, o), J.rho_measures(F, o, s)[0], []
    for i in range(len(done), len(ks)):
        k = ks[i]
        done.append(maps(F[np.arange(N) != k], F[k])[0])
        if (i + 1) % 10 == 0 and i + 1 < len(ks):
            np.savez(part, lam_o=lam_o, lam_m=lam_m, rho_o=rho_o, lam_m_k=np.stack(done))
            print(f"{var} {case}: {i + 1}/{len(ks)} members", flush=True)
    np.savez_compressed(path, lats=lats, lons=lons, N=N, T=F.shape[1], members=ks, lam_o=lam_o,
                        lam_m=lam_m, lam_m_k=np.stack(done), rho_o=rho_o)
    if os.path.exists(part):
        os.remove(part)


def defined(z):
    """lambda = 0 (a one-bin axis, I = 0 exactly) -> NaN in lam_o, lam_m, lam_m_k."""
    nan0 = lambda a: np.where(a == 0, np.nan, a)                    # noqa: E731
    return dict(z, lam_o=nan0(z["lam_o"]), lam_m=nan0(z["lam_m"]), lam_m_k=nan0(z["lam_m_k"]))


def shares(z):
    W = np.cos(np.deg2rad(z["lats"]))[:, None] * np.ones(z["lons"].size)
    keep = z["rho_o"] > 0
    r0 = ratio(z["lam_o"], z["lam_m"])
    with warnings.catch_warnings():                                # all-NaN cells: undefined
        warnings.simplefilter("ignore", RuntimeWarning)
        lm_mean = np.nanmean(z["lam_m_k"], axis=0)
    rk = [ratio(z["lam_o"], l) for l in z["lam_m_k"]]
    rmean = ratio(z["lam_o"], lm_mean)
    valid = np.isfinite(r0) & np.isfinite(rmean) & keep            # both RPCs defined
    gt1 = lambda r: wfrac(W, np.isfinite(r) & (r > 1), valid & np.isfinite(r))   # noqa: E731
    return {"area_valid": float((W * valid).sum() / W.sum()),
            "area_rpc_gt1_production": gt1(r0),
            "area_rpc_gt1_member": [gt1(r) for r in rk],
            "area_rpc_gt1_member_mean": gt1(rmean),
            "area_mean_lam_m_production": wmean(W, z["lam_m"], valid),
            "area_mean_lam_m_member_mean": wmean(W, lm_mean, valid),
            "area_mean_rpc_production": wmean(W, r0, valid),
            "area_mean_rpc_member_mean": wmean(W, rmean, valid)}


def summary(z):
    W = np.cos(np.deg2rad(z["lats"]))[:, None] * np.ones(z["lons"].size)
    keep = z["rho_o"] > 0
    one = lambda m: float((W * (m & keep)).sum() / (W * keep).sum())   # noqa: E731
    return {"N": int(z["N"]), "T": int(z["T"]), "members": [int(k) for k in z["members"]],
            "area_lam_o_onebin": one(z["lam_o"] == 0), "area_lam_m_onebin": one(z["lam_m"] == 0),
            "area_lam_m_k_onebin": [one(l == 0) for l in z["lam_m_k"]],
            "all": shares(z), "defined": shares(defined(z))}


def figure(cases, Z, name):
    fig, axs = P.grid_fig(len(cases), 3, (14.4, 2.9 * len(cases) + 1.2))
    titles = (r"$\lambda_o / \lambda_m$ (production)",
              r"$\lambda_o / \langle\lambda_m^{(k)}\rangle_k$ (member as observation)",
              r"$\langle\lambda_m^{(k)}\rangle_k / \lambda_m$")
    LMN = P.Normalize(0.5, 1.5)
    for i, (var, case) in enumerate(cases):
        z = defined(Z[var, case])
        keep = z["rho_o"] > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lm_mean = np.nanmean(z["lam_m_k"], axis=0)
        P.rpc_panel(axs[i][0], ratio(z["lam_o"], z["lam_m"]), z["lats"], z["lons"], keep=keep)
        P.rpc_panel(axs[i][1], ratio(z["lam_o"], lm_mean), z["lats"], z["lons"], keep=keep)
        q = np.where(z["lam_m"] > FLOOR, lm_mean / z["lam_m"], np.nan)
        P.show(axs[i][2], q, z["lats"], z["lons"], cmap=P.DIV, norm=LMN)
        for c in range(3):
            if i == 0:
                axs[i][c].set_title(titles[c])
        P.row_labels(axs[i][0], f"{var} {case}", f"N={int(z['N'])}, T={int(z['T'])}, K={len(z['members'])}")
    P.colorbar(fig, [r[:2] for r in axs], 0.6)
    P.colorbar(fig, [r[2:] for r in axs], 0.9, norm=LMN, label=r"$\lambda_m$ ratio",
               ticks=(0.5, 0.75, 1, 1.25, 1.5), labels=("≤0.5", "0.75", "1", "1.25", "≥1.5"))
    style.save(fig, name, dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--members", default="5", help="number of evenly spaced members, or 'all'")
    ap.add_argument("--case", help="VAR:case, compute this case only (no json, no figure)")
    ap.add_argument("--summarize", action="store_true", help="json and figures from existing npz only")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    K = None if args.members == "all" else int(args.members)
    tag = "_all" if K is None else ""
    path = lambda var, case: os.path.join(OUT, f"member_lam_m{tag}_{var}_{case}.npz")  # noqa: E731
    os.makedirs(OUT, exist_ok=True)
    cases = [c for cs in PRODUCTION.values() for c in cs]
    if args.case:
        cases = [tuple(args.case.split(":"))]
    for var, case in cases:
        if not args.summarize and (args.recompute or not os.path.exists(path(var, case))):
            compute(var, case, K, path(var, case))
    if args.case:
        return
    Z, res = {}, {}
    for var, case in cases:
        Z[var, case] = dict(np.load(path(var, case)))
        e = res[f"{var} {case}"] = summary(Z[var, case])
        for kind in ("all", "defined"):
            d, pm = e[kind], np.array(e[kind]["area_rpc_gt1_member"])
            print(f"{var:6s} {case:6s} {kind:7s} area {d['area_valid']:.2f} | RPC>1: production "
                  f"{d['area_rpc_gt1_production']:.3f}, member mean {d['area_rpc_gt1_member_mean']:.3f}"
                  f" (single members {pm.min():.2f}-{np.median(pm):.2f}-{pm.max():.2f}, K={pm.size}) | "
                  f"lam_m {d['area_mean_lam_m_production']:.3f} -> {d['area_mean_lam_m_member_mean']:.3f}",
                  flush=True)
    with open(style.data_path(f"lambda-m-member{tag.replace('_', '-')}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    figure(PRODUCTION["decadal"], Z, NAME + tag.replace("_", "-"))
    if K is None:
        figure(PRODUCTION["lead2-4"], Z, NAME + "-all-lead2-4")


if __name__ == "__main__":
    main()
