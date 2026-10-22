"""Validate exchangeability.py on synthetic ensembles, then run it on the DJF NAO (SLP lead 2-4).

    python scripts/exchangeability_check.py synthetic [--draws 200] [--n-perm 1000]
        -> figures/data/exchangeability-synthetic.json
    python scripts/exchangeability_check.py synthetic --exchangeable-only --draws 1000
        -> figures/data/exchangeability-synthetic-exchangeable-1000.json   (tighter size check)
    python scripts/exchangeability_check.py nao [--n-perm 1000]
        -> figures/data/exchangeability-nao.json   (all members and each model alone)
    python scripts/exchangeability_check.py grid [--case SLP:2-4] [--n-perm 200]
        -> figures/data/exchangeability-grid-<VAR>-<case>.json   (every cell, field significance)
           $SNP_OUT/exchangeability_grid_<VAR>_<case>.npz       (per-cell maps, null area shares)
    python scripts/exchangeability_check.py figure      (after grid for SLP, PRECT, TREFHT 2-4)
        -> figures/supp-info/{png,pdf}/exchangeability-lead2-4
    python scripts/exchangeability_check.py figure-main (same inputs)
        -> figures/main-text/{png,pdf}/members-not-exchangeable

Synthetic: x[r, t] = mu[r] + beta[r] s[t] + sig[r] z[r, t], s, z iid N(0, 1), with R, T as the
real index (R = 62, T = 51, B = 4). Exchangeable: mu = 0, beta = 0.5, sig = 1. Heterogeneous
(power), at full and half spread: beta proportional to sig (sig over [0.5, 1.5], beta = 0.5 sig:
the same signal fraction, different amplitude), spread in beta only (over [0, 1]), spread in mu
only (over [-0.5, 0.5] sig). Rejection rate at alpha = 0.05 over independent draws, each with its
own seed; Wilson 95% intervals.

NAO: Dunstone et al. (2016) boxes, cos-lat weighted means of the case SLP (DJF, global mean removed,
Eade-smoothed): Azores 36-40N, 28-20W minus Iceland 63-70N, 25-16W, for the members and the
observations (T values only).
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np

import exchangeability as E
import style
from data import OUT, load

ALPHA = 0.05
R_, T_ = 62, 51
CASES = {"exchangeable": {}}
for lvl, f in (("full", 1.0), ("half", 0.5)):
    CASES[f"beta_prop_sig_{lvl}"] = dict(sig=(1 - 0.5 * f, 1 + 0.5 * f), beta_per_sig=0.5)
    CASES[f"beta_spread_{lvl}"] = dict(beta=(0.5 - 0.5 * f, 0.5 + 0.5 * f))
    CASES[f"mu_spread_{lvl}"] = dict(mu=(-0.5 * f, 0.5 * f))
RUNS = [("exchangeable", None, "bins", 1)] + \
       [("exchangeable", b, "bins", 1) for b in E.bin_choices(T_)[::2]] + \
       [("exchangeable", None, "copula", 1), ("exchangeable", None, "bins", 3)] + \
       [(c, None, "bins", 1) for c in CASES if c != "exchangeable"]


def members(case, R, rng):
    c = CASES[case]
    lin = lambda ab: np.linspace(ab[0], ab[1], R)                    # noqa: E731
    sig = lin(c["sig"]) if "sig" in c else np.ones(R)
    beta = c["beta_per_sig"] * sig if "beta_per_sig" in c else \
        (lin(c["beta"]) if "beta" in c else np.full(R, 0.5))
    mu = lin(c["mu"]) if "mu" in c else np.zeros(R)
    return [rng.permutation(v) for v in (mu, beta, sig)]


def draw(args):
    case, B, est, block, d, n_perm = args
    rng = np.random.default_rng([20261018, list(CASES).index(case), d])
    mu, beta, sig = members(case, R_, rng)
    s = rng.standard_normal(T_)
    x = mu[:, None] + beta[:, None] * s + sig[:, None] * rng.standard_normal((R_, T_))
    res = E.exchangeability_test(x, B=B, n_perm=n_perm, block=block, estimator=est,
                                 seed=int(rng.integers(2 ** 31)))
    return {f"{k}/{q}": float(res[k][q]) for k in E.STATS for q in res[k]} | {"B": res["meta"]["B"]}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, c - h), min(1.0, c + h)]


def synthetic(draws, n_perm, exchangeable_only=False):
    t0 = time.time()
    runs = [r for r in RUNS if r[0] == "exchangeable"] if exchangeable_only else RUNS
    jobs = [(c, B, e, L, d, n_perm) for c, B, e, L in runs for d in range(draws)]
    with get_context("fork").Pool(int(os.environ.get("SLURM_CPUS_ON_NODE", 1))) as pool:
        out = pool.map(draw, jobs, chunksize=4)
    res = {}
    for i, (c, B, e, L) in enumerate(runs):
        rs = out[i * draws:(i + 1) * draws]
        get = lambda k: np.array([r[k] for r in rs])                 # noqa: E731
        name = f"{c} | B={rs[0]['B']} | {e} | block={L}"
        e_ = res[name] = {"draws": draws, "n_perm": n_perm}
        for s in ("Delta", "I_MXY"):
            k = int((get(f"{s}/p") <= ALPHA).sum())
            e_[f"reject_{s}"] = k / draws
            e_[f"reject_{s}_wilson95"] = wilson(k, draws)
        for s in ("Delta", "I_pool_minus_cond", "I_MXY", "I_cond", "I_pool_T"):
            for q in ("obs", "null_mean", "corrected"):
                v = get(f"{s}/{q}")
                e_[f"{s}_{q}_mean"] = float(v.mean())
                e_[f"{s}_{q}_se"] = float(v.std(ddof=1) / np.sqrt(draws))
        print(f"{name:52s} reject Delta {e_['reject_Delta']:.3f} "
              f"[{e_['reject_Delta_wilson95'][0]:.3f}, {e_['reject_Delta_wilson95'][1]:.3f}] "
              f"I_MXY {e_['reject_I_MXY']:.3f} | I_pool-I_cond raw "
              f"{e_['I_pool_minus_cond_obs_mean']:+.4f} +- {e_['I_pool_minus_cond_obs_se']:.4f}, "
              f"corrected {e_['I_pool_minus_cond_corrected_mean']:+.4f} | Delta raw "
              f"{e_['Delta_obs_mean']:+.4f}, corrected {e_['Delta_corrected_mean']:+.4f} bits",
              flush=True)
    res["_runtime_s"] = time.time() - t0
    name = "exchangeability-synthetic" + (f"-exchangeable-{draws}" if exchangeable_only else "")
    with open(style.data_path(name + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"runtime {res['_runtime_s']:.0f} s for {len(jobs)} draws x {n_perm} permutations")


BOXES = {"south": ((36, 40), (-28, -20)), "north": ((63, 70), (-25, -16))}


def box_mean(A, lats, lons, box):
    (la0, la1), (lo0, lo1) = box
    lw = ((lons + 180) % 360) - 180
    i = (lats >= la0) & (lats <= la1)
    k = (lw >= lo0) & (lw <= lo1)
    w = np.cos(np.deg2rad(lats[i]))[:, None] * np.ones(k.sum())
    sub = A[..., i, :][..., k]
    return (sub * w).sum((-2, -1)) / w.sum(), int(i.sum() * k.sum())


def nao_index():
    F, o, s, lats, lons = load("SLP", "2-4")
    (fs, ns), (fn, nn) = (box_mean(F, lats, lons, BOXES[b]) for b in ("south", "north"))
    go = box_mean(o, lats, lons, BOXES["south"])[0] - box_mean(o, lats, lons, BOXES["north"])[0]
    return fs - fn, go, (ns, nn)


def summarize(res):
    out = {"meta": res["meta"]}
    for s in E.STATS:
        out[s] = {k: float(v) for k, v in res[s].items()}
    return out


def nao(n_perm):
    x, g, cells = nao_index()
    with open(os.path.join(OUT, "member_models.json")) as fh:
        lab = json.load(fh)["SLP 2-4"]
    assert lab["matches_case_file"] and len(lab["model_of"]) == x.shape[0]
    owner = np.array(lab["model_of"])
    sets = {"all members": np.ones(len(owner), bool)} | {m: owner == m for m in dict.fromkeys(owner)}
    out = {"index": "DJF NAO, SLP lead 2-4, Dunstone et al. (2016) boxes",
           "box_cells_south_north": cells, "R": int(x.shape[0]), "T": int(x.shape[1])}
    for name, sel in sets.items():
        t0 = time.time()
        e = out[name] = {}
        for B in E.bin_choices(x.shape[1]):
            e[f"B={B}"] = summarize(E.exchangeability_test(x[sel], B=B, n_perm=n_perm, seed=1))
        e["copula"] = summarize(E.exchangeability_test(x[sel], n_perm=n_perm, seed=1,
                                                       estimator="copula"))
        e["obs_mi"] = {k: float(v) for k, v in E.obs_mi(g, x[sel], n_perm=n_perm, seed=2).items()}
        e["runtime_s"] = time.time() - t0
        b = e[f"B={E.n_bins(x.shape[1])}"]
        print(f"{name:16s} R={sel.sum():2d} | Delta {b['Delta']['obs']:+.4f} (null "
              f"{b['Delta']['null_mean']:+.4f} +- {b['Delta']['null_sd']:.4f}) p {b['Delta']['p']:.3f}"
              f" | I_MXY {b['I_MXY']['obs']:.4f} (null {b['I_MXY']['null_mean']:.4f}) p "
              f"{b['I_MXY']['p']:.3f} | copula p {e['copula']['Delta']['p']:.3f}/"
              f"{e['copula']['I_MXY']['p']:.3f} | I_o {e['obs_mi']['obs']:.4f} corrected "
              f"{e['obs_mi']['corrected']:+.4f} p {e['obs_mi']['p']:.3f} | {e['runtime_s']:.1f} s",
              flush=True)
    with open(style.data_path("exchangeability-nao.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def field_test(res, stat, W):
    """Field significance of the area with p <= ALPHA: each permutation's own p map (against
    the other permutations, the same rule as the observed p) gives a null area share; the
    permutation tables are shared by every cell, so the spatial correlation is kept."""
    nl = res["null"][stat].reshape(res["null"][stat].shape[0], -1)
    n = nl.shape[0]
    a = np.abs(nl - nl.mean(0)) if stat in E.TWO_SIDED else nl
    pk = (a[None] >= a[:, None]).sum(1) / n                          # (1 + #others) / n
    w = W.ravel() / W.sum()
    Fk = (w * (pk <= ALPHA)).sum(1)
    Fo = float((w * (res[stat]["p"].ravel() <= ALPHA)).sum())
    return {"area_obs": Fo, "null_area_mean": float(Fk.mean()),
            "null_area_95pct": float(np.percentile(Fk, 95)), "null_area_max": float(Fk.max()),
            "p_field": float((1 + (Fk >= Fo).sum()) / (n + 1)), "null_area": Fk}


def grid(n_perm, case="SLP:2-4"):
    from tables import wfrac
    var, cs = case.split(":")
    F, o, s, lats, lons = load(var, cs)
    t0 = time.time()
    res = E.exchangeability_test(F, n_perm=n_perm, seed=3, return_null=True)
    dt = time.time() - t0
    W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
    ok = np.isfinite(res["Delta"]["p"])
    field = {s: field_test(res, s, W) for s in ("Delta", "I_MXY")}
    out = {"case": f"{var} {cs}", "shape": list(F.shape), "n_perm": n_perm, "runtime_s": dt,
           "area_p_Delta_le_0.05": wfrac(W, res["Delta"]["p"] <= ALPHA, ok),
           "area_p_I_MXY_le_0.05": wfrac(W, res["I_MXY"]["p"] <= ALPHA, ok),
           "area_mean_Delta_corrected": float((W * res["Delta"]["corrected"]).sum() / W.sum()),
           "area_mean_I_pool_minus_cond_raw": float((W * res["I_pool_minus_cond"]["obs"]).sum()
                                                    / W.sum()),
           "field_significance": {s: {k: v for k, v in f.items() if k != "null_area"}
                                  for s, f in field.items()}}
    np.savez_compressed(os.path.join(OUT, f"exchangeability_grid_{var}_{cs}.npz"), lats=lats,
                        lons=lons, R=F.shape[0], T=F.shape[1],
                        **{f"{k}_{q}": v for k in E.STATS for q, v in res[k].items()},
                        **{f"field_null_area_{s}": f["null_area"] for s, f in field.items()})
    with open(style.data_path(f"exchangeability-grid-{var}-{cs}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


FIG_CASES = (("SLP", "2-4"), ("PRECT", "2-4"), ("TREFHT", "2-4"))
FIG_STATS = (("I_MXY", r"$I(M; X, Y)$", 0), ("Delta", r"$\Delta = I_{\mathrm{pool},T} - I_{\mathrm{cond}}$", 1))


SHORT_LABEL = {"I_MXY": r"$I(M; X, Y)$", "Delta": r"$\Delta$"}


def figure(cases=FIG_CASES, name="exchangeability-lead2-4"):
    """Per field: maps of the bias-corrected I(M; X, Y) and Delta (observed - null mean, bits),
    dotted where p <= ALPHA, and the permutation distribution of the area share with p <= ALPHA
    (spatial correlation kept) against the observed share."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import plots as P                                                # map helpers, figure style
    Z = {c: dict(np.load(os.path.join(OUT, f"exchangeability_grid_{c[0]}_{c[1]}.npz")))
         for c in cases}
    lim = {s: float(np.nanpercentile(np.abs(np.concatenate(
        [Z[c][f"{s}_corrected"].ravel() for c in cases])), 98)) for s, _, _ in FIG_STATS}
    fig = plt.figure(figsize=(15.0, 3.0 * len(cases) + 1.4), layout="constrained")
    gs = fig.add_gridspec(len(cases), 3, width_ratios=(1, 1, 0.62))
    maps = [[], []]
    for i, c in enumerate(cases):
        z = Z[c]
        lats, lons = z["lats"], z["lons"]
        W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
        for s, label, k in FIG_STATS:
            ax = fig.add_subplot(gs[i, k], projection=P.PROJ)
            maps[k].append(ax)
            P.show(ax, z[f"{s}_corrected"], lats, lons, cmap=P.DIV,
                   norm=Normalize(-lim[s], lim[s]))
            sig = z[f"{s}_p"] <= ALPHA
            P.dots(ax, sig, lats, lons, P.stip_size(lats, lons), color=P.INK, lw=0, alpha=0.7)
            ax.text(0.01, -0.03, f"p $\\leq$ {ALPHA} on {(W * sig).sum() / W.sum():.1%} of the area",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8, color=P.MUTED)
            if i == 0:
                ax.set_title(label)
        P.row_labels(maps[0][-1], f"{c[0]} {c[1]}", f"R={int(z['R'])}, T={int(z['T'])}")
        ax = fig.add_subplot(gs[i, 2])
        area = {s: (W * (z[f"{s}_p"] <= ALPHA)).sum() / W.sum() for s, _, _ in FIG_STATS}
        hi = max(max(z[f"field_null_area_{s}"].max(), area[s]) for s, _, _ in FIG_STATS)
        bins = np.linspace(0, 1.08 * hi, 40)
        top = 0
        for s, label, k in FIG_STATS:
            null = z[f"field_null_area_{s}"]
            top = max(top, ax.hist(null, bins=bins, color=style.SERIES[k], alpha=0.45, lw=0)[0].max())
        right = max(area.values())
        for s, label, k in FIG_STATS:
            null = z[f"field_null_area_{s}"]
            pf = (1 + (null >= area[s]).sum()) / (null.size + 1)
            ax.axvline(area[s], color=style.SERIES[k], lw=2)
            ax.text(area[s], 0.97, f" {area[s]:.1%} \n field p {'≤' if pf <= 1 / null.size else '='} "
                    f"{pf:.3f} ", transform=ax.get_xaxis_transform(), va="top", fontsize=8,
                    ha="left" if area[s] == right else "right", color=style.SERIES[k])
        ax.axvline(ALPHA, color="0.35", lw=0.8, ls=":")
        ax.set_xlim(0, 1.45 * right)
        ax.set_ylim(0, 1.7 * top)
        ax.set_yticks([])
        ax.set_xlabel(f"area share with p $\\leq$ {ALPHA}")
        style.tidy(ax, grid=False)
        if i == 0:
            ax.set_title(f"rejecting area: {null.size} permutations vs observed")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [h for s, label, k in FIG_STATS for h in
               (Patch(color=style.SERIES[k], alpha=0.45, label=SHORT_LABEL[s] + ": null"),
                Line2D([], [], color=style.SERIES[k], lw=2, label="observed"))]
    handles.append(Line2D([], [], color="0.35", lw=0.8, ls=":", label=f"nominal {ALPHA:.0%}"))
    fig.legend(handles=handles, loc="lower right", ncol=1, frameon=False, fontsize=8)
    for k, (s, label, _) in enumerate(FIG_STATS):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-lim[s], lim[s]), cmap=P.DIV),
                          ax=maps[k], orientation="horizontal", shrink=0.8, pad=0.02,
                          label=f"{label}, observed $-$ null mean (bits)")
        cb.solids.set_rasterized(True)
    style.save(fig, name, dpi=200)
    plt.close(fig)


def figure_main(cases=FIG_CASES, name="members-not-exchangeable"):
    """Main text: (a-c) bias-corrected I(M; X, Y) where p <= ALPHA (other cells flat grey), one
    map per field; (d) the share of area with p <= ALPHA, observed against the permutation null
    (5-95% and full range), with the nominal ALPHA marked. Everything else is in the caption."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import plots as P                                                # map helpers, figure style
    Z = {c: dict(np.load(os.path.join(OUT, f"exchangeability_grid_{c[0]}_{c[1]}.npz")))
         for c in cases}
    sig = {c: Z[c]["I_MXY_p"] <= ALPHA for c in cases}
    vals = np.concatenate([Z[c]["I_MXY_corrected"][sig[c]] for c in cases])
    vmax = float(np.ceil(np.percentile(vals, 98) * 100) / 100)       # bits, rounded up to 0.01
    cmap = LinearSegmentedColormap.from_list("seq_sig", P.SEQ(np.linspace(0.22, 1.0, 256)))
    cmap.set_bad(style.GRID)                                         # not significant: flat grey
    norm = Normalize(0.0, vmax)
    fig = plt.figure(figsize=(7.2, 4.15), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=(1, 1))
    letters = iter("abcd")
    for i, c in enumerate(cases):
        z = Z[c]
        ax = fig.add_subplot(gs[i // 2, i % 2], projection=P.PROJ)
        P.show(ax, np.where(sig[c], z["I_MXY_corrected"], np.nan), z["lats"], z["lons"],
               cmap=cmap, norm=norm)
        for sp in ax.spines.values():
            sp.set(color=P.MUTED, lw=0.6)
        style.panel_label(ax, next(letters), f"{c[0]}")
        if i == 2:
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                              orientation="horizontal", shrink=0.62, pad=0.025, aspect=38,
                              extend="max", label="excess $I(M; X, Y)$ (bits)")
            cb.outline.set(color=P.MUTED, lw=0.6)
            cb.ax.tick_params(length=2.5, width=0.6, labelsize=8)
            cb.solids.set_rasterized(True)
    ax = fig.add_subplot(gs[1, 1])
    y = np.arange(len(cases))[::-1]
    area = {}
    for yy, c in zip(y, cases):
        z = Z[c]
        W = np.cos(np.deg2rad(z["lats"]))[:, None] * np.ones(z["lons"].size)
        null = 100 * z["field_null_area_I_MXY"]
        area[c] = 100 * float((W * sig[c]).sum() / W.sum())
        ax.hlines(yy, null.min(), null.max(), color="0.62", lw=1.0)
        ax.hlines(yy, *np.percentile(null, [5, 95]), color="0.62", lw=4.0)
        ax.plot(area[c], yy, "o", ms=6.5, color=style.INK, zorder=3)
        ax.text(area[c], yy + 0.2, f"{area[c]:.1f}%", ha="center", va="bottom", fontsize=8,
                color=style.INK)
    ax.axvline(100 * ALPHA, color="0.45", lw=0.7, zorder=0)
    ax.set_yticks(y, [c[0] for c in cases])
    ax.set_ylim(-0.6, len(cases) - 0.3)
    ax.set_xlim(0, 5 * np.ceil(1.12 * max(area.values()) / 5))
    ax.set_xlabel("area with $p \\leq 0.05$ (%)")
    style.tidy(ax, grid=False)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.spines["left"].set_visible(False)
    style.panel_label(ax, next(letters), "area rejecting exchangeability")
    style.save(fig, name, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("synthetic", "nao", "grid", "figure", "figure-main"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--exchangeable-only", action="store_true")
    ap.add_argument("--case", default="SLP:2-4", help="VAR:case for grid")
    a = ap.parse_args()
    n = a.n_perm or (200 if a.what == "grid" else 1000)
    {"synthetic": lambda: synthetic(a.draws, n, a.exchangeable_only), "nao": lambda: nao(n),
     "grid": lambda: grid(n, a.case), "figure": figure,
     "figure-main": figure_main}[a.what]()


if __name__ == "__main__":
    main()
