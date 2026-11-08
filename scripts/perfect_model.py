"""Perfect-model test of RPC = 1: each member in turn plays the observations.

    python scripts/perfect_model.py run VAR CASE     -> $SNP_OUT/pm_VAR_CASE.npz
    python scripts/perfect_model.py figure           -> figures/slides/{png,pdf}/perfect-model-<set>
                                                        (+ -nostipple), figures/data/perfect-model-null.json

Per grid cell, F (N, T) members and o (T,) observations. For each member j, the remaining N - 1
members are the ensemble and f_j the "observations": the RPC computed exactly as for the real o
(lambda: ksg_lambda.cell, KSG on normal scores, lambda_o from the T pairs and lambda_m from the
member-averaged I; rho: rho_o = corr(s, o), rho_m = <corr(f_n, s_-n)>_n as jugaad.rho_measures).
Those N values are the null distribution of the RPC when the observations are just another member
(perfect model): it carries the estimator's bias and noise at this T, so no Gaussian or bootstrap
assumption is needed.
  One model (CanESM5): leave one MEMBER out; the members are exchangeable, so the null is exact.
  Pooled models: leave the pseudo-observation's whole MODEL out (its siblings would otherwise stay
  in the ensemble and flatter it), and pair each null value with the observed RPC against the SAME
  reduced ensemble, ref_j = RPC(o; members not in model(j)), so ensemble size and make-up cancel.
  The pooled null is approximate: the models differ, so the pseudo-observations are not one
  distribution (see ExchangeabilityMaps).
The maps show the observed RPC against all members.

Per cell: p = min(1, 2 min(F_lo, F_hi)), F_lo = (1 + #{j: null_j <= ref_j}) / (N + 1), F_hi the
same with >= (ref_j = the observed RPC against pseudo-observation j's ensemble); the smallest p is 2 / (N + 1) (0.095 for N = 20), so the per-cell level is ALPHA = 0.10.
Field: each pseudo-observation j's p-map against the other N - 1 null values gives a null area
share with p <= ALPHA (spatial correlation kept); field p = (1 + #{null area >= observed}) / (N + 1),
separately for RPC above the null (the paradox direction) and below it.
"""
import argparse
import json
import os
from multiprocessing import get_context

import numpy as np

import ksg_lambda as K
from data import OUT, load


def owners(var, case, N):
    """Each member's model: CanESM5_* cases are one model; the rest from member_models.json
    (a moy case uses its base case's labels), checked against the member count."""
    if case.startswith("CanESM5"):
        return np.array(["CanESM5"] * N)
    base = case[:-3] if case.endswith("moy") else case
    with open(os.path.join(OUT, "member_models.json")) as fh:
        lab = json.load(fh)[f"{var} {base}"]
    owner = np.array(lab["model_of"])
    if not lab["matches_case_file"] or owner.size != N:
        raise ValueError(f"{var} {case}: {owner.size} labels for {N} members (or unchecked)")
    return owner

ALPHA = 0.10
FLOOR = K.FLOOR


def rho_rpc(f, o):
    """rho_o / rho_m: rho_o = corr(s, o), rho_m = <corr(f_n, s_-n)>_n; NaN where undefined."""
    N = f.shape[0]
    s = f.mean(0)
    s_loo = (f.sum(0, keepdims=True) - f) / (N - 1)
    c = lambda a, b: np.corrcoef(a, b)[0, 1]                         # noqa: E731
    ro = c(s, o)
    rm = np.mean([c(f[n], s_loo[n]) for n in range(N)])
    return ro / rm if (ro > 0 and rm > 0) else np.nan


def lam_rpc(f, o, rng):
    lo, lm = K.cell(f, o, rng)
    return lo / lm if lm > FLOOR else np.nan


_G = {}


def _row(j):
    F, o, owner = _G["F"], _G["o"], _G["owner"]
    N, nx = F.shape[0], F.shape[-1]
    models = list(dict.fromkeys(owner))
    single = len(models) == 1
    out = {k: np.full(nx, np.nan) for k in ("lam_obs", "rho_obs")}
    out |= {k: np.full((N, nx), np.nan, np.float32) for k in
            ("lam_null", "rho_null", "lam_ref", "rho_ref")}
    for i in range(nx):
        f, g = F[:, :, j, i], o[:, j, i]
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(g))):
            continue
        rng = np.random.default_rng([2026, j, i])
        out["lam_obs"][i], out["rho_obs"][i] = lam_rpc(f, g, rng), rho_rpc(f, g)
        if single:                                       # leave one member out, ref = the obs RPC
            for m in range(N):
                rest = np.delete(f, m, axis=0)
                out["lam_null"][m, i] = lam_rpc(rest, f[m], rng)
                out["rho_null"][m, i] = rho_rpc(rest, f[m])
            out["lam_ref"][:, i], out["rho_ref"][:, i] = out["lam_obs"][i], out["rho_obs"][i]
            continue
        for a in models:                                 # leave the whole model out, paired ref
            inA = owner == a
            rest = f[~inA]
            ref_l, ref_r = lam_rpc(rest, g, rng), rho_rpc(rest, g)
            for m in np.flatnonzero(inA):
                out["lam_null"][m, i] = lam_rpc(rest, f[m], rng)
                out["rho_null"][m, i] = rho_rpc(rest, f[m])
                out["lam_ref"][m, i], out["rho_ref"][m, i] = ref_l, ref_r
    return out


def run(var, case):
    F, o, _, lats, lons = load(var, case)
    owner = owners(var, case, F.shape[0])
    _G.update(F=F, o=o, owner=owner)
    with get_context("fork").Pool(int(os.environ.get("SLURM_CPUS_ON_NODE", 1))) as pool:
        rows = pool.map(_row, range(len(lats)), chunksize=1)
    res = {k: np.stack([r[k] for r in rows], axis=-2) for k in rows[0]}   # obs (ny, nx); null (N, ny, nx)
    path = os.path.join(OUT, f"pm_{var}_{case}.npz")
    np.savez_compressed(path, lats=lats, lons=lons, N=F.shape[0], T=F.shape[1], model_of=owner,
                        n_models=len(set(owner)), **res)
    print(f"-> {path}", flush=True)


def pvals(ref, null):
    """Two-sided p of the observed RPC against the null, and the direction (+1 above, -1 below).
    ref: (ny, nx), or (N, ny, nx) paired with each null value (the observed RPC against that null
    member's ensemble); null (N, ny, nx)."""
    ok = np.isfinite(null) & np.isfinite(ref)
    n = ok.sum(0)
    lo = (1 + np.sum(ok & (null <= ref), 0)) / (n + 1)
    hi = (1 + np.sum(ok & (null >= ref), 0)) / (n + 1)
    p = np.minimum(1.0, 2 * np.minimum(lo, hi))
    p = np.where(n >= 5, p, np.nan)
    return p, np.where(hi < lo, 1.0, -1.0)


def field(ref, null, W):
    """Observed and null area shares with p <= ALPHA, above / below the null, and field p's. The
    null areas: each pseudo-observation j against the other N - 1 null values (unpaired, so
    approximate for the pooled leave-one-model-out null)."""
    p, d = pvals(ref, null)
    valid = np.isfinite(p)
    w = W / (W * valid).sum()
    area = {s: float((w * (valid & (p <= ALPHA) & (d == sgn))).sum()) for s, sgn in
            (("above", 1), ("below", -1))}
    N = null.shape[0]
    nul = {"above": [], "below": []}
    for j in range(N):
        pj, dj = pvals(null[j], np.delete(null, j, axis=0))
        vj = np.isfinite(pj)
        wj = W / max((W * vj).sum(), 1e-12)
        for s, sgn in (("above", 1), ("below", -1)):
            nul[s].append(float((wj * (vj & (pj <= ALPHA) & (dj == sgn))).sum()))
    out = {}
    for s in ("above", "below"):
        a = np.array(nul[s])
        out[s] = {"area_obs": area[s], "null_area_median": float(np.median(a)),
                  "null_area_95pct": float(np.percentile(a, 95)),
                  "p_field": float((1 + np.sum(a >= area[s])) / (N + 1))}
    return p, d, out


def figure():
    """One slide per case set (plots.SLIDE_CASES), rho RPC (top) and lambda RPC (bottom), dots where
    the perfect-model test does NOT reject (p > ALPHA); under each map the share above and below
    the null and the field p; plus a -nostipple version."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plots as P
    import style as sty
    summary = {}
    rc = {"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13}
    for name, cases in P.SLIDE_CASES.items():
        for stipple in (True, False):
            with plt.rc_context(rc):
                fig = plt.figure(figsize=(13.33, 6.8), layout="constrained")
                gs = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.16))
                axs = []
                sizes = {int(np.load(os.path.join(OUT, f"pm_{v}_{c}.npz"))["N"]) for v, c in cases}
                for k, (var, case) in enumerate(cases):
                    z = np.load(os.path.join(OUT, f"pm_{var}_{case}.npz"))
                    lats, lons, N, T = z["lats"], z["lons"], int(z["N"]), int(z["T"])
                    W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
                    w = np.sqrt(np.median(np.abs(np.diff(lats))) * np.median(np.abs(np.diff(lons))))
                    s = (np.sqrt(1.6) * np.sqrt(w / 1.875)) ** 2
                    for i, key in enumerate(("rho", "lam")):
                        obs, null, ref = z[f"{key}_obs"], z[f"{key}_null"], z[f"{key}_ref"]
                        p, d, fl = field(ref, null, W)
                        if stipple:
                            summary[f"{key} {var} {case}"] = dict(
                                fl, N=N, T=T, alpha=ALPHA, n_models=int(z["n_models"]),
                                null="leave one member out" if int(z["n_models"]) == 1 else
                                "leave one model out, paired")
                        ax = fig.add_subplot(gs[i, k], projection=P.PROJ)
                        axs.append(ax)
                        valid = np.isfinite(p)
                        P.show(ax, np.where(valid, np.clip(obs, 0, 2), np.nan), lats, lons,
                               cmap=P.DIV, norm=P.RPC_NORM)
                        if stipple:
                            P.dots(ax, valid & (p > ALPHA), lats, lons, s, c="#000000",
                                   linewidths=0, zorder=5)
                        ax.text(0.5, -0.04, f"above null {100 * fl['above']['area_obs']:.0f}% "
                                f"(field p {fl['above']['p_field']:.2f})   below "
                                f"{100 * fl['below']['area_obs']:.0f}% (p {fl['below']['p_field']:.2f})",
                                transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
                                color=P.MUTED)
                        if i == 0:
                            ax.set_title(f"{var}   (N={N}, T={T})" if k == 0 or len(sizes) > 1
                                         else f"{var}   (T={T})")
                        if k == 0:
                            ax.text(-0.04, 0.5, ("ρ RPC", "λ RPC")[i], transform=ax.transAxes,
                                    rotation=90, ha="right", va="center", fontsize=14, color=P.INK)
                cax = fig.add_subplot(gs[2, :2]).inset_axes([0.15, 0.55, 0.7, 0.3])
                for a_ in fig.axes:
                    if a_ is not cax and a_.get_subplotspec() is not None and \
                            a_.get_subplotspec().rowspan.start == 2:
                        a_.set_axis_off()
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=P.RPC_NORM, cmap=P.DIV), cax=cax,
                                  orientation="horizontal", label="RPC")
                cb.set_ticks((0, 0.5, 1, 1.5, 2))
                cb.set_ticklabels(("0", "0.5", "1", "1.5", "≥2"))
                cb.solids.set_rasterized(True)
                leg_ax = fig.add_subplot(gs[2, 2])
                leg_ax.set_axis_off()
                handles = ([Line2D([], [], color=P.INK, marker="o", ms=5, lw=0,
                                   label=f"not significant: within the perfect-model null "
                                         f"(p > {ALPHA:.2f})")] if stipple else []) + \
                    [Line2D([], [], marker="s", ms=8, lw=0, color=P.BED,
                            label="RPC undefined (ρ$_\\mathregular{o}$ ≤ 0 or λ$_\\mathregular{m}$ ≈ 0)")]
                leg_ax.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.02, 0.5),
                              frameon=True, fancybox=False, edgecolor=P.MUTED, borderpad=0.6,
                              fontsize=10).set_in_layout(False)
                sty.save(fig, f"perfect-model-{name}" + ("" if stipple else "-nostipple"), dpi=200,
                         folder="slides")
                plt.close(fig)
    with open(sty.data_path("perfect-model-null.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    for k, e in summary.items():
        print(f"{k:24s} above null {100 * e['above']['area_obs']:4.0f}% (null 95% "
              f"{100 * e['above']['null_area_95pct']:3.0f}%, p {e['above']['p_field']:.3f})   below "
              f"{100 * e['below']['area_obs']:4.0f}% (null 95% {100 * e['below']['null_area_95pct']:3.0f}%,"
              f" p {e['below']['p_field']:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("run", "figure"))
    ap.add_argument("var", nargs="?")
    ap.add_argument("case", nargs="?")
    a = ap.parse_args()
    run(a.var, a.case) if a.what == "run" else figure()


if __name__ == "__main__":
    main()
