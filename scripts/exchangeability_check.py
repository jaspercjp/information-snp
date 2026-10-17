"""Validate exchangeability.py on synthetic ensembles, then run it on the DJF NAO (SLP lead 2-4).

    python scripts/exchangeability_check.py synthetic [--draws 200] [--n-perm 1000]
        -> figures/data/exchangeability-synthetic.json
    python scripts/exchangeability_check.py synthetic --exchangeable-only --draws 1000
        -> figures/data/exchangeability-synthetic-exchangeable-1000.json   (tighter size check)
    python scripts/exchangeability_check.py nao [--n-perm 1000]
        -> figures/data/exchangeability-nao.json   (all members and each model alone)
    python scripts/exchangeability_check.py grid [--n-perm 200]
        -> figures/data/exchangeability-grid-SLP-2-4.json   (every cell, for runtime and maps)

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


def grid(n_perm):
    from tables import wfrac
    F, o, s, lats, lons = load("SLP", "2-4")
    t0 = time.time()
    res = E.exchangeability_test(F, n_perm=n_perm, seed=3)
    dt = time.time() - t0
    W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
    ok = np.isfinite(res["Delta"]["p"])
    out = {"case": "SLP 2-4", "shape": list(F.shape), "n_perm": n_perm, "runtime_s": dt,
           "area_p_Delta_le_0.05": wfrac(W, res["Delta"]["p"] <= ALPHA, ok),
           "area_p_I_MXY_le_0.05": wfrac(W, res["I_MXY"]["p"] <= ALPHA, ok),
           "area_mean_Delta_corrected": float((W * res["Delta"]["corrected"]).sum() / W.sum()),
           "area_mean_I_pool_minus_cond_raw": float((W * res["I_pool_minus_cond"]["obs"]).sum()
                                                    / W.sum())}
    np.savez_compressed(os.path.join(OUT, "exchangeability_grid_SLP_2-4.npz"), lats=lats,
                        lons=lons, **{f"{k}_{q}": v for k in E.STATS for q, v in res[k].items()})
    with open(style.data_path("exchangeability-grid-SLP-2-4.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("synthetic", "nao", "grid"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--exchangeable-only", action="store_true")
    a = ap.parse_args()
    n = a.n_perm or (200 if a.what == "grid" else 1000)
    {"synthetic": lambda: synthetic(a.draws, n, a.exchangeable_only), "nao": lambda: nao(n),
     "grid": lambda: grid(n)}[a.what]()


if __name__ == "__main__":
    main()
