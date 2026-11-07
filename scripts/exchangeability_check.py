"""Validate exchangeability.py on synthetic ensembles, then run it on the DJF NAO (SLP lead 2-4).

    python scripts/exchangeability_check.py synthetic [--draws 200] [--n-perm 1000]
        -> figures/data/exchangeability-synthetic.json
    python scripts/exchangeability_check.py synthetic --exchangeable-only --draws 1000
        -> figures/data/exchangeability-synthetic-exchangeable-1000.json   (tighter size check)
    python scripts/exchangeability_check.py nao [--n-perm 1000]
        -> figures/data/exchangeability-nao.json   (all members and each model alone)
    python scripts/exchangeability_check.py synthetic-ar [--draws 400] [--n-perm 1000]
        -> figures/data/exchangeability-synthetic-ar1.json   (size of the block null, AR(1) members)
    python scripts/exchangeability_check.py grid [--case SLP:2-4] [--n-perm 200]
                                                 [--model CanESM5] [--block 12]
        -> figures/data/exchangeability-grid-<VAR>-<case>[-<model>][-block<L>].json
           $SNP_OUT/exchangeability_grid_<VAR>_<case>[_<model>][_block<L>].npz
           (every cell, field significance; per-cell maps, null area shares)
    python scripts/exchangeability_check.py figure-slides   (after the exact-null grids below)
        -> figures/slides/{png,pdf}/exchangeability-{all-models,canesm5}, copied by the manifest
           into figures/Presentation/ExchangeabilityMaps/: all members and CanESM5 alone, SLP,
           PRECT, TREFHT, DJF per start date (2-4; CanESM5 12-14) and the moy decade
    python scripts/exchangeability_check.py figure-model [--model CanESM5] [--block 12]
        -> figures/supp-info/{png,pdf}/members-exchangeable-<model>   (decadal cases)
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

Synthetic AR(1) (`synthetic-ar`), for the decadal cases, where each member is ONE continuous run:
the same model with z replaced by a stationary unit-variance AR(1) in time with lag-1 correlation
phi, then a least-squares line removed per member (as `make_cases.thesis_detrend`), R = 20 or 40,
T = 120 months, B = 5. Exchangeable members, phi over 0-0.9, permutation blocks of 1, 6, 12, 24
steps; the same draws for every block (paired). Power: beta_prop_sig_half and beta_spread_half at
phi = 0.5, R = 40. --redetrend: the per-member detrend inside the test (observed: detrend(x);
null: detrend(block-permuted x)), plus exchangeable lognormal members (exp of the Gaussian ones).

--model: only that model's members, chosen from $SNP_OUT/member_models.json (labels checked
against the case file and the member count; a `moy` case uses its base case's labels after a check
that the two differ by the same series for every selected member).

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


def ar1(z, phi):
    """Stationary unit-variance AR(1) along the last axis from iid N(0, 1) innovations z; phi = 0
    returns z itself."""
    e = np.array(z, dtype=float)
    a = np.sqrt(1 - phi * phi)
    for t in range(1, e.shape[-1]):
        e[..., t] = phi * e[..., t - 1] + a * e[..., t]
    return e


def detrend(x):
    """Remove a least-squares line (intercept and slope) per member along the last axis."""
    t = np.arange(x.shape[-1], dtype=float)
    A = np.stack([np.ones_like(t), t - t.mean()], 1)
    coef = np.linalg.lstsq(A, x.reshape(-1, t.size).T, rcond=None)[0]
    return x - (A @ coef).T.reshape(x.shape)


AR_T, AR_R, AR_PHI, AR_BLOCKS = 120, (20, 40), (0.0, 0.3, 0.5, 0.75, 0.9), (1, 6, 12, 24)
AR_RUNS = [("exchangeable", R, phi, L) for R in AR_R for phi in AR_PHI for L in AR_BLOCKS] + \
          [(c, 40, 0.5, L) for c in ("beta_prop_sig_half", "beta_spread_half") for L in (1, 12, 24)]
AR_RUNS_REDETREND = [r for r in AR_RUNS if r[3] != 6] + \
    [("exchangeable+lognormal", 40, phi, L) for phi in (0.0, 0.5) for L in (1, 12)]


def draw_ar(args):
    case, R, phi, block, d, n_perm, redetrend = args
    base, *mods = case.split("+")                                    # "+lognormal": exp(x)
    rng = np.random.default_rng([20261104, list(CASES).index(base), R, int(round(100 * phi)), d])
    mu, beta, sig = members(base, R, rng)
    s = rng.standard_normal(AR_T)
    x = mu[:, None] + beta[:, None] * s + sig[:, None] * ar1(rng.standard_normal((R, AR_T)), phi)
    if "lognormal" in mods:
        x = np.exp(x)
    # redetrend: the raw members, detrended inside the test (observed and every permutation);
    # otherwise the detrended members, as in the case files
    res = E.exchangeability_test(x if redetrend else detrend(x), n_perm=n_perm, block=block,
                                 seed=int(rng.integers(2 ** 31)), detrend=redetrend)
    return {f"{k}/{q}": float(res[k][q]) for k in ("Delta", "I_MXY") for q in res[k]}


def synthetic_ar(draws, n_perm, redetrend=False):
    t0 = time.time()
    runs = AR_RUNS_REDETREND if redetrend else AR_RUNS
    jobs = [(c, R, phi, L, d, n_perm, redetrend)
            for c, R, phi, L in runs for d in range(draws if c == "exchangeable" else draws // 2)]
    with get_context("fork").Pool(int(os.environ.get("SLURM_CPUS_ON_NODE", 1))) as pool:
        out = pool.map(draw_ar, jobs, chunksize=8)
    res, i = {}, 0
    for c, R, phi, L in runs:
        n = draws if c == "exchangeable" else draws // 2
        rs, i = out[i:i + n], i + n
        get = lambda k: np.array([r[k] for r in rs])                 # noqa: E731
        name = f"{c} | R={R} | T={AR_T} | phi={phi} | block={L}"
        e_ = res[name] = {"case": c, "R": R, "T": AR_T, "phi": phi, "block": L, "draws": n,
                          "n_perm": n_perm, "redetrend": redetrend}
        for s in ("Delta", "I_MXY"):
            k = int((get(f"{s}/p") <= ALPHA).sum())
            e_[f"reject_{s}"] = k / n
            e_[f"reject_{s}_wilson95"] = wilson(k, n)
            e_[f"{s}_corrected_mean"] = float(get(f"{s}/corrected").mean())
        print(f"{name:58s} reject I_MXY {e_['reject_I_MXY']:.3f} [{e_['reject_I_MXY_wilson95'][0]:.3f}, "
              f"{e_['reject_I_MXY_wilson95'][1]:.3f}] Delta {e_['reject_Delta']:.3f}", flush=True)
    res["_runtime_s"] = time.time() - t0
    name = "exchangeability-synthetic-ar1" + ("-redetrend" if redetrend else "")
    with open(style.data_path(name + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"runtime {res['_runtime_s']:.0f} s for {len(jobs)} draws x {n_perm} permutations")


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


def model_mask(lab, model, n):
    """Members of `model` from one member_models.json entry, checked: labels verified against
    the case file, one label per member (n), and the model present."""
    if not lab["matches_case_file"]:
        raise ValueError("member labels were not checked against the case file")
    owner = np.array(lab["model_of"])
    if owner.size != n:
        raise ValueError(f"{owner.size} labels for {n} members")
    if not (owner == model).any():
        raise ValueError(f"no members of {model}")
    return owner == model


def model_members(var, cs, model, F):
    """F restricted to `model`'s members (a copy). A `moy` case has no labels of its own: it takes
    its base case's and checks that F_moy - F_base is one series shared by every selected member
    (the per-model climatology), which holds only if the member order is the same."""
    with open(os.path.join(OUT, "member_models.json")) as fh:
        labels = json.load(fh)
    base = cs[:-3] if cs.endswith("moy") else cs
    sel = model_mask(labels[f"{var} {base}"], model, F.shape[0])
    x = np.array(F[sel])
    if base != cs:
        d = x - load(var, base)[0][sel]
        spread = np.abs(d - d[:1]).max()
        if spread > 1e-6 * max(1.0, np.abs(x).max()):
            raise ValueError(f"{var} {cs}: members do not line up with {base} ({spread:.3g})")
    return x


PHI_OK = 0.5     # lag-1 autocorrelation up to which the 12-month re-detrended null holds its size


def lag1(F):
    """Lag-1 autocorrelation in time of the member deviations from the ensemble mean, pooled
    over members: (R, T, *space) -> (*space,)."""
    d = F - F.mean(0)
    return (d[:, 1:] * d[:, :-1]).sum((0, 1)) / (d * d).sum((0, 1))


def raw_members(var, cs, model, x):
    """`model`'s (None: every model's) members before their per-member detrend, checked
    against x, the same members from the case file: detrending raw must reproduce x. Decadal
    (sYYYY[moy]): dcpp_decadal_handles, smoothed (`moy` anomalies for a moy case), then
    make_cases.thesis_detrend over the 120 months. Lead window (2-4, MODEL_a-b): dcpp_handles
    with detrend_data=False (global mean removed for SLP, smoothed), then a line per member
    over the start dates (as _cube_io.detrend). Used for the exact null of a detrended case."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data-acquisition"))
    if cs.startswith("s"):                                           # decadal: sYYYY[moy]
        import dcpp_decadal_handles as H
        import make_cases as MC
        c = H.get(int(cs[1:5]), var=var, models=[model] if model else None, verbose=False,
                  anom="moy" if cs.endswith("moy") else "none")
        raw = np.asarray(c.F, float)
        detrended = MC.thesis_detrend
    else:                                                            # lead window: one value
        import dcpp_handles as H                                     # per start date
        owner, _, lead = cs.rpartition("_")
        c = H.get(lead=lead, var=var, models=[owner] if owner else None,
                  remove_gm=var.upper() == "SLP", detrend_data=False)
        raw = np.asarray(c.F, float)
        if model and not owner:
            raw = raw[np.asarray(c.model_of) == model]
        detrended = E._detrend_members
    if raw.shape != x.shape:
        raise ValueError(f"{var} {cs} {model}: raw {raw.shape} vs case file {x.shape}")
    err = np.abs(detrended(raw) - x).max()
    if err > 1e-9 * np.abs(x).max():
        raise ValueError(f"{var} {cs} {model}: detrended raw differs from the case file ({err:.3g})")
    return raw


def grid_tag(model, block, redetrend):
    """File-name suffix: '' for all members, per-step null (the lead 2-4 runs)."""
    return ((f"_{model}" if model else "") + (f"_block{block}" if block != 1 else "")
            + ("_redetrend" if redetrend else ""))


def grid(n_perm, case="SLP:2-4", model=None, block=1, redetrend=False):
    from tables import wfrac
    var, cs = case.split(":")
    F, o, s, lats, lons = load(var, cs)
    if model:
        F = model_members(var, cs, model, F)
    tag = grid_tag(model, block, redetrend)
    phi1 = lag1(F)
    shape = F.shape
    if redetrend:                                                    # exact null: from the raw
        x = raw_members(var, cs, model, F)
        del F                                                        # one copy of the members
    else:
        x = F
    t0 = time.time()
    res = E.exchangeability_test(x, n_perm=n_perm, block=block, seed=3, return_null=True,
                                 detrend=redetrend)
    del x
    dt = time.time() - t0
    W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
    ok = np.isfinite(res["Delta"]["p"])
    field = {s: field_test(res, s, W) for s in ("Delta", "I_MXY")}
    okphi = phi1 <= PHI_OK
    field_okphi = {s: field_test(res, s, W * okphi) for s in ("Delta", "I_MXY")}
    out = {"case": f"{var} {cs}", "model": model or "all members", "block": block,
           "redetrend": redetrend,
           "shape": list(shape), "n_perm": n_perm, "runtime_s": dt,
           "area_p_Delta_le_0.05": wfrac(W, res["Delta"]["p"] <= ALPHA, ok),
           "area_p_I_MXY_le_0.05": wfrac(W, res["I_MXY"]["p"] <= ALPHA, ok),
           "area_mean_Delta_corrected": float((W * res["Delta"]["corrected"]).sum() / W.sum()),
           "area_mean_I_pool_minus_cond_raw": float((W * res["I_pool_minus_cond"]["obs"]).sum()
                                                    / W.sum()),
           "field_significance": {s: {k: v for k, v in f.items() if k != "null_area"}
                                  for s, f in field.items()},
           f"area_lag1_le_{PHI_OK}": wfrac(W, okphi, ok),
           f"field_significance_lag1_le_{PHI_OK}": {
               s: {k: v for k, v in f.items() if k != "null_area"} for s, f in field_okphi.items()}}
    np.savez_compressed(os.path.join(OUT, f"exchangeability_grid_{var}_{cs}{tag}.npz"), lats=lats,
                        lons=lons, R=shape[0], T=shape[1],
                        **{f"{k}_{q}": v for k in E.STATS for q, v in res[k].items()},
                        **{f"field_null_area_{s}": f["null_area"] for s, f in field.items()},
                        **{f"field_null_area_lag1ok_{s}": f["null_area"]
                           for s, f in field_okphi.items()}, lag1=phi1)
    with open(style.data_path(f"exchangeability-grid-{var}-{cs}{tag.replace('_', '-')}.json"),
              "w") as fh:
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


def figure_main(cases=FIG_CASES, name="members-not-exchangeable", tag=""):
    """Main text: (a-c) bias-corrected I(M; X, Y) where p <= ALPHA (other cells flat grey), one
    map per field; (d) the share of area with p <= ALPHA, observed against the permutation null
    (5-95% and full range), with the nominal ALPHA marked. Everything else is in the caption."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import plots as P                                                # map helpers, figure style
    Z = {c: dict(np.load(os.path.join(OUT, f"exchangeability_grid_{c[0]}_{c[1]}{tag}.npz")))
         for c in cases}
    sig = {c: Z[c]["I_MXY_p"] <= ALPHA for c in cases}
    vals = np.concatenate([Z[c]["I_MXY_corrected"][sig[c]] for c in cases])
    vmax = float(np.ceil(np.percentile(vals, 98) * 100) / 100) if vals.size else 0.01  # bits
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
        style.panel_label(ax, next(letters), f"{c[0]}" if c[1] == "2-4" else f"{c[0]} {c[1]}")
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
    ax.set_yticks(y, [c[0] if c[1] == "2-4" else f"{c[0]} {c[1]}" for c in cases])
    ax.set_ylim(-0.6, len(cases) - 0.3)
    ax.set_xlim(0, 5 * np.ceil(1.12 * max(max(area.values()),
                                        max(100 * Z[c]["field_null_area_I_MXY"].max()
                                            for c in cases)) / 5))
    ax.set_xlabel("area with $p \\leq 0.05$ (%)")
    style.tidy(ax, grid=False)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.spines["left"].set_visible(False)
    style.panel_label(ax, next(letters), "area rejecting exchangeability")
    style.save(fig, name, dpi=300)
    plt.close(fig)


MODEL_CASES = (("SLP", "s1961"), ("PRECT", "s1978"), ("TREFHT", "s1961"))


def figure_model(model="CanESM5", block=12, redetrend=True):
    """The members-not-exchangeable figure for one model's members on the decadal cases, with
    the label permutation in blocks of `block` months (the members are continuous runs)."""
    figure_main(MODEL_CASES, f"members-exchangeable-{model}",
                grid_tag(model, block, redetrend))


SLIDE_ROWS = {
    "all-models": (("2-4", "_redetrend", "DJF, lead months 2-4"),
                   ("moy", "_block12_redetrend", "Decade, seasonal cycle removed")),
    "canesm5": (("CanESM5_12-14", "_redetrend", "DJF, lead months 12-14"),
                ("moy", "_CanESM5_block12_redetrend", "Decade, seasonal cycle removed")),
}
SLIDE_TITLE = {"all-models": "All models pooled: members are not exchangeable",
               "canesm5": "CanESM5 alone: its members compared with each other"}
DECADAL_MOY = {"SLP": "s1961moy", "PRECT": "s1978moy", "TREFHT": "s1961moy"}
SLIDE_VARS = ("SLP", "PRECT", "TREFHT")


def _slide_grids(which):
    """{(row, var): npz dict} for one slide; row 0 the start-date rows, row 1 the moy decade."""
    out = {}
    for i, (case, tag, _) in enumerate(SLIDE_ROWS[which]):
        for v in SLIDE_VARS:
            cs = DECADAL_MOY[v] if case == "moy" else case
            out[i, v] = dict(np.load(os.path.join(OUT, f"exchangeability_grid_{v}_{cs}{tag}.npz")))
            out[i, v]["case"] = cs
    return out


def figure_slides(which=("all-models", "canesm5")):
    """Talk slides (16:9 width), one per member set, maps only: excess I(M; X, Y) where
    p <= ALPHA (flat grey elsewhere) for SLP, PRECT, TREFHT (columns) in two designs (rows), the
    share of area with p <= ALPHA under each map. The colour scale is shared by both slides, so
    they compare directly. Exact nulls only: per start date for the DJF rows, 12-month blocks for
    the decade, the per-member detrend inside the test. No title: what the slides show, and each
    case's rejecting area against its permutation null, go to figures/slides/exchangeability-
    README.txt (rewritten here). -> figures/slides/{png,pdf}/exchangeability-<which>, copied
    into Presentation/ExchangeabilityMaps/ by the manifest."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import plots as P
    Z = {w: _slide_grids(w) for w in ("all-models", "canesm5")}
    sig_vals = np.concatenate([z["I_MXY_corrected"][z["I_MXY_p"] <= ALPHA]
                               for zz in Z.values() for z in zz.values()])
    vmax = float(np.ceil(np.percentile(sig_vals, 98) * 100) / 100)
    area = {(w, k): 100 * _area(z) for w, zz in Z.items() for k, z in zz.items()}
    nulls = {(w, k): 100 * z["field_null_area_I_MXY"] for w, zz in Z.items() for k, z in zz.items()}
    cmap = LinearSegmentedColormap.from_list("seq_sig", P.SEQ(np.linspace(0.22, 1.0, 256)))
    cmap.set_bad(style.GRID)
    norm = Normalize(0.0, vmax)
    rc = {"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 12,
          "ytick.labelsize": 12}
    for w in ([which] if isinstance(which, str) else which):
        with plt.rc_context(rc):
            fig = plt.figure(figsize=(13.33, 6.0), layout="constrained")
            gs = fig.add_gridspec(2, 3)
            maps = []
            for (i, v), z in Z[w].items():
                ax = fig.add_subplot(gs[i, SLIDE_VARS.index(v)], projection=P.PROJ)
                maps.append(ax)
                sig = z["I_MXY_p"] <= ALPHA
                P.show(ax, np.where(sig, z["I_MXY_corrected"], np.nan), z["lats"], z["lons"],
                       cmap=cmap, norm=norm)
                for sp in ax.spines.values():
                    sp.set(color=P.MUTED, lw=0.6)
                if i == 0:
                    ax.set_title(v)
                ax.text(0.5, -0.04, f"{area[w, (i, v)]:.0f}% of area rejects    R={int(z['R'])}, "
                        f"T={int(z['T'])}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=11, color=P.MUTED)
                if SLIDE_VARS.index(v) == 0:
                    ax.text(-0.04, 0.5, SLIDE_ROWS[w][i][2], transform=ax.transAxes, ha="right",
                            va="center", rotation=90, fontsize=13, color=style.INK)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=maps,
                              orientation="horizontal", shrink=0.42, pad=0.03, aspect=38,
                              extend="max", label="excess $I(M; X, Y)$ where $p \\leq 0.05$ (bits)")
            cb.outline.set(color=P.MUTED, lw=0.6)
            cb.solids.set_rasterized(True)
            style.save(fig, f"exchangeability-{w}", dpi=200, folder="slides")
            plt.close(fig)
    _slides_readme(Z, area, nulls)


def _slides_readme(Z, area, nulls):
    """figures/slides/exchangeability-README.txt: what the two ExchangeabilityMaps slides show and
    each case's rejecting area against its permutation null (the numbers the slides leave out)."""
    lines = ["ExchangeabilityMaps: are the ensemble members exchangeable?",
             "===========================================================", "",
             "Two slides, same layout and colour scale: columns SLP, PRECT, TREFHT; colour is the",
             "excess I(M; X, Y) (observed minus its permutation-null mean, bits) where p <= 0.05,",
             "flat grey elsewhere. M is the member label, X the mean of the other members, Y the",
             "member. Under exchangeable members about 5% of the area rejects by chance.", "",
             "  exchangeability-all-models  all members pooled: SLP_TITLE_ALL",
             "  exchangeability-canesm5     CanESM5 alone: SLP_TITLE_CAN", "",
             "Rows", "  DJF          one value per start date. All models: lead months 2-4, the five",
             "               November models (CanESM5 starts in January and is not in that pool).",
             "               CanESM5: lead months 12-14, its first full DJF (one year longer lead).",
             "  Decade       the monthly decade s1961 (PRECT s1978), each model's month-of-year",
             "               climatology removed: one continuous 120-month run per member.", "",
             "Test: I(M; X, Y) with a member-label permutation null, 200 permutations, the per-member",
             "detrend done inside the test (on raw members, so every permuted member is detrended",
             "like the observed ones). DJF: labels permuted per start date (exact). Decade: in",
             "12-month blocks (validated for lag-1 autocorrelation up to about 0.5; above that, as in",
             "tropical TREFHT, the null over-rejects). Field significance: the permutations' own",
             "area shares (spatial correlation kept).", "",
             "Rejecting area (%), observed | permutation null 5-95% | field p", ""]
    for w, label in (("all-models", "all members"), ("canesm5", "CanESM5")):
        for (i, v), z in Z[w].items():
            n = nulls[w, (i, v)]
            pf = (1 + (n >= area[w, (i, v)]).sum()) / (n.size + 1)
            row = SLIDE_ROWS[w][i][2]
            lines.append(f"  {label:12s} {row:32s} {v:7s} {area[w, (i, v)]:5.1f} | "
                         f"{np.percentile(n, 5):4.1f}-{np.percentile(n, 95):4.1f} | "
                         f"{pf:.3f}   R={int(z['R'])}, T={int(z['T'])}")
    lines += ["", "Regenerate (compute node, repository root):",
              "  python scripts/exchangeability_check.py grid --case VAR:CASE [--model CanESM5] "
              "--block {1,12} --redetrend",
              "  python scripts/exchangeability_check.py figure-slides",
              "  python scripts/figure_set.py Presentation refresh && ... verify",
              "Inputs: $SNP_OUT/exchangeability_grid_*_redetrend.npz; per-case JSON in figures/data.", ""]
    text = "\n".join(lines).replace("SLP_TITLE_ALL", SLIDE_TITLE["all-models"].split(": ", 1)[1]) \
        .replace("SLP_TITLE_CAN", SLIDE_TITLE["canesm5"].split(": ", 1)[1])
    import figure_sets
    path = figure_sets.figures_root() / "slides" / "exchangeability-README.txt"
    path.write_text(text)
    print(f"-> slides/{path.name}")


def _area(z):
    """Area share (cos-lat weighted) with p <= ALPHA for I_MXY."""
    W = np.cos(np.deg2rad(z["lats"]))[:, None] * np.ones(z["lons"].size)
    return float((W * (z["I_MXY_p"] <= ALPHA)).sum() / W.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("synthetic", "synthetic-ar", "nao", "grid", "figure",
                                     "figure-main", "figure-model", "figure-slides"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--exchangeable-only", action="store_true")
    ap.add_argument("--case", default="SLP:2-4", help="VAR:case for grid")
    ap.add_argument("--model", default=None, help="only this model's members (grid)")
    ap.add_argument("--block", type=int, default=1, help="permutation block length, time steps")
    ap.add_argument("--redetrend", action="store_true",
                    help="re-detrend every permuted pseudo-member (decadal cases)")
    a = ap.parse_args()
    n = a.n_perm or (200 if a.what == "grid" else 1000)
    {"synthetic": lambda: synthetic(a.draws, n, a.exchangeable_only), "nao": lambda: nao(n),
     "synthetic-ar": lambda: synthetic_ar(a.draws, n, a.redetrend),
     "grid": lambda: grid(n, a.case, a.model, a.block, a.redetrend), "figure": figure,
     "figure-model": lambda: figure_model(a.model or "CanESM5", a.block, a.redetrend),
     "figure-slides": figure_slides,
     "figure-main": figure_main}[a.what]()


if __name__ == "__main__":
    main()
