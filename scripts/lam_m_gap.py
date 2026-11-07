"""Is lambda_m^pool > lambda_m^avg an estimator artefact? Gaussian ensembles with a known answer.

    python scripts/lam_m_gap.py [--reps 100] [--recompute]
        -> figures/data/lambda-m-gap-gaussian.npz, figures/supp-info/{png,pdf}/lambda-m-gap-gaussian
    (~5 min on 16 cores)

Exchangeable members f_n = sqrt(r) x + sqrt(1 - r) z_n, x and z_n iid N(0, 1) over T times. Every
(s_-n, f_n) pair, in one member or pooled over members, has the same bivariate Gaussian law with
    rho = r / sqrt(r + (1 - r)/(N - 1)),      and lambda = rho for a Gaussian,
so both estimators of lam_m_definitions.py target the same number. Variants that swap one
ingredient at a time (all bias corrected as there, unless "raw"):
    pool           lambda(I(s_-n~; f~))                  bins: s_-n centres on T, f on N T (production)
    avg            <lambda(I(s_-n; f_n))>_n              bins: each member's T values
    avg_poolbins   avg with the pool's bin counts        -> how much is resolution
    pool_avgbins   pool with avg's (median) bin counts   -> how much is pooling the samples
    lam_of_avg_I   lambda(<I(s_-n; f_n)>_n)              -> averaging after the concave lambda()
    pool_raw, avg_raw                                     -> no time-permutation null
and the RPC with each, for an observation that is one more exchangeable member (true RPC = 1):
    lam_o = lambda(I(o~; s_-n~))      as production (o and the s_-n centres sized on T)
    rpc_pool = lam_o / pool,  rpc_avg = lam_o / avg
(N, T) as the production cases: lead2-4 single models (10, 33), (16, 51), all (62, 51); decadal
single models (10, 120), (20, 120), all (112, 120).
"""
import argparse
import os
import warnings
from multiprocessing import get_context

import numpy as np

import style
from jugaad import KNUTH_MAX, lam_bits, loo_means, mi_bits
from lam_m_definitions import P_NULL, knuth

NAME = "lambda-m-gap-gaussian"
CONFIGS = [(10, 33), (16, 51), (62, 51), (10, 120), (20, 120), (112, 120)]
RHOS = np.round(np.arange(0.2, 0.95, 0.1), 2)
KEYS = ("pool", "avg", "avg_poolbins", "pool_avgbins", "lam_of_avg_I", "pool_raw", "avg_raw",
        "lam_o")
SEED = 20261018


def r_for(rho, N):
    """Pairwise member correlation r that gives corr(s_-n, f_n) = rho."""
    a = rho ** 2 * (1 - 1 / (N - 1))
    return (a + np.sqrt(a ** 2 + 4 * rho ** 2 / (N - 1))) / 2


def mi(X, Y, bins, rng):
    """(plug-in I, mean of P_NULL time-permutation nulls of X); NaN if an axis has one bin."""
    if min(bins) < 2:
        return np.nan, np.nan
    y = Y.ravel()
    null = np.mean([mi_bits(X[:, rng.permutation(X.shape[1])].ravel(), y, bins)
                    for _ in range(P_NULL)])
    return mi_bits(X.ravel(), y, bins), null


def lam(I):
    return float(lam_bits(I)) if np.isfinite(I) else np.nan


def estimates(f, o, rng):
    N, T = f.shape
    L = loo_means(f)
    Io, No = mi(np.tile(o, (N, 1)), L, (knuth(o, T), knuth(L.mean(0), T)), rng)
    pb = (knuth(L.mean(0), T), knuth(f.ravel(), KNUTH_MAX))
    mb = [(knuth(L[n], T), knuth(f[n], T)) for n in range(N)]
    ab = tuple(int(np.median([b[k] for b in mb])) for k in range(2))
    Ip, Np = mi(L, f, pb, rng)
    Ia, Na = np.array([mi(L[n][None], f[n][None], mb[n], rng) for n in range(N)]).T
    Iq, Nq = np.array([mi(L[n][None], f[n][None], pb, rng) for n in range(N)]).T
    Ir, Nr = mi(L, f, ab, rng)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # all members one-bin
        return dict(lam_o=lam(Io - No), pool=lam(Ip - Np), avg=np.nanmean([lam(d) for d in Ia - Na]),
                    avg_poolbins=np.nanmean([lam(d) for d in Iq - Nq]),
                    pool_avgbins=lam(Ir - Nr), lam_of_avg_I=lam(np.nanmean(Ia - Na)),
                    pool_raw=lam(Ip), avg_raw=np.nanmean([lam(d) for d in Ia]),
                    B_s_pool=pb[0], B_f_pool=pb[1], B_s_avg=ab[0], B_f_avg=ab[1])


def point(args):
    c, k, reps = args
    N, T = CONFIGS[c]
    rho = RHOS[k]
    r = r_for(rho, N)
    rng = np.random.default_rng([SEED, c, k])
    out = []
    for _ in range(reps):
        x = rng.standard_normal(T)
        f = np.sqrt(r) * x + np.sqrt(1 - r) * rng.standard_normal((N, T))
        o = np.sqrt(r) * x + np.sqrt(1 - r) * rng.standard_normal(T)   # one more member
        out.append(estimates(f, o, rng))
    return c, k, {key: np.array([o[key] for o in out]) for key in out[0]}


def compute(reps):
    jobs = [(c, k, reps) for c in range(len(CONFIGS)) for k in range(len(RHOS))]
    with get_context("fork").Pool(int(os.environ.get("SLURM_CPUS_ON_NODE", 1))) as pool:
        res = pool.map(point, jobs)
    keys = list(res[0][2])
    D = {key: np.full((len(CONFIGS), len(RHOS), reps), np.nan) for key in keys}
    for c, k, d in res:
        for key in keys:
            D[key][c, k] = d[key]
    np.savez_compressed(style.data_path(NAME + ".npz"), configs=np.array(CONFIGS), rhos=RHOS,
                        **D)


def figure(z):
    """Top: estimate - truth for lambda_m (band: +-1 sd over reps for the two estimators).
    Bottom: median lambda RPC with each, for an observation that is one more member (truth 1)."""
    import matplotlib.pyplot as plt
    show = (("pool", r"$\lambda_m^{\mathrm{pool}}$"), ("avg", r"$\lambda_m^{\mathrm{avg}}$"),
            ("pool_avgbins", "pooled, per-member bin counts"),
            ("avg_poolbins", "per member, pooled bin counts"))
    C = style.SERIES
    nc = len(z["configs"])
    fig, axs = plt.subplots(2, nc, figsize=(16.0, 6.0), layout="constrained", sharex=True,
                            sharey="row", height_ratios=(1.4, 1))
    rho = z["rhos"]
    style_of = lambda i: (dict(color=C[i // 2 * 2], lw=1.5, ls="-" if i < 2 else "--",  # noqa: E731
                               marker="o", ms=5, mfc=C[i // 2 * 2] if i % 2 == 0 else "white",
                               mec=C[i // 2 * 2], mew=1.3))
    for c, (N, T) in enumerate(z["configs"]):
        ax, R = axs[0][c], axs[1][c]
        ax.axhline(0, color="0.2", lw=0.8)
        for i, (key, label) in enumerate(show):
            m = np.nanmean(z[key][c], axis=-1) - rho
            ax.plot(rho, m, label=label, **style_of(i))
            if i < 2:
                sd = np.nanstd(z[key][c], axis=-1)
                ax.fill_between(rho, m - sd, m + sd, color=C[0], alpha=0.10, lw=0)
        ax.set_title(f"N = {N}, T = {T}")
        R.axhline(1, color="0.2", lw=0.8)
        for i, key in enumerate(("pool", "avg")):
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.nanmedian(z["lam_o"][c] / z[key][c], axis=-1)
            R.plot(rho, q, **style_of(i))
        R.set_ylim(0.4, 1.5)
        R.set_xlabel(r"true $\lambda_m = \rho(s_{-n}, f_n)$")
        style.tidy(ax)
        style.tidy(R)
    axs[0][0].set_ylabel(r"$\lambda_m$ estimate $-$ truth")
    axs[1][0].set_ylabel(r"median $\lambda_o / \lambda_m$ (truth 1)")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, frameon=False)
    style.save(fig, NAME, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    path = style.data_path(NAME + ".npz")
    if args.recompute or not os.path.exists(path):
        compute(args.reps)
    z = dict(np.load(path))
    print("estimate - truth (mean over reps), rows = true lambda", RHOS.tolist())
    for c, (N, T) in enumerate(z["configs"]):
        print(f"N={N} T={T}: bins pool s/f {np.nanmedian(z['B_s_pool'][c]):.0f}/"
              f"{np.nanmedian(z['B_f_pool'][c]):.0f}, avg s/f {np.nanmedian(z['B_s_avg'][c]):.0f}/"
              f"{np.nanmedian(z['B_f_avg'][c]):.0f}")
        for key in KEYS:
            d = np.nanmean(z[key][c], axis=-1) - z["rhos"]
            print(f"   {key:13s} " + " ".join(f"{v:+.3f}" for v in d))
        with np.errstate(divide="ignore", invalid="ignore"):
            for key in ("pool", "avg"):
                q = z["lam_o"][c] / z[key][c]
                print(f"   RPC {key:9s} median " + " ".join(f"{v:5.2f}" for v in np.nanmedian(q, -1))
                      + " | share>1 " + " ".join(f"{v:4.2f}" for v in np.mean(q > 1, -1)))
    figure(z)


if __name__ == "__main__":
    main()
