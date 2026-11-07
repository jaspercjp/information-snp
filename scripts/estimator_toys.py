"""Which lambda estimator for N ~ 50, T ~ 30? Seven estimators on four toy problems with a known
exact lambda, at the sample sizes of the maps.

    python scripts/estimator_toys.py [--recompute] [--draws 500] [--seed S]
        -> figures/supp-info/{png,pdf}/estimator-toys, numbers in figures/data/estimator-toys.json
           (raw draws cached in figures/data/estimator-toys.npz)

Cases (x ~ N(0, 1), z ~ N(0, 1) independent, n pairs per draw; lambda = sqrt(1 - exp(-2 I)), I in nats):
    independent    y = z                                     exact lambda 0 (the false signal)
    gaussian       y = 0.8 x + 0.6 z                         exact lambda 0.8 (= |rho|)
    skewed         y = (exp(2 u) - 1) / 2, u = 0.8 x + 0.6 z  exact lambda 0.8 (monotonic, rho 0.22)
    non-monotonic  y = x^2 + 0.75 z                          exact lambda 0.825 (numerical; rho 0)
Sample sizes n = 33 (PRECT lead 2-4), 51 (SLP lead 2-4), 120 (the monthly decade).

Estimators (all on the n pairs of one draw):
    knuth            production: Knuth's count for x and for y on their own n raw values,
                     equal-width bins, plug-in MI in bits (jugaad.knuth_bins, jugaad.mi_bits)
    knuth-scores-3   the same on normal scores of x and y, at least 3 bins per axis
    rank-bins        equal-count bins, B = max(3, round(n^(1/3))) per axis, plug-in MI
    rank-bins-bc     rank-bins minus the mean of P = 200 permutation nulls (y shuffled), clipped at 0
    gauss-copula     |corr| of the normal scores: lambda for a Gaussian dependence (monotonic only)
    ksg-scores       Kraskov-Stoegbauer-Grassberger, k = 4, on normal scores (infomeasure)
    ksg-raw          the same on the raw values
Normal scores: Phi^-1((rank - 1/2)/n) (jugaad.normal_scores), no assumption on the distribution.
"""
import argparse
import json
import logging
import os
from multiprocessing import get_context

import numpy as np

import exchangeability as E
import jugaad as J
import style

CASES = {"independent": 0.0, "gaussian": 0.8, "skewed": 0.8, "non-monotonic": None}
SIGMA_SQ = 0.75
NS = (33, 51, 120)
ESTIMATORS = ("knuth", "knuth-scores-3", "rank-bins", "rank-bins-bc", "gauss-copula",
              "ksg-scores", "ksg-raw")
P_NULL = 200
NAME = "estimator-toys"


def exact_nonmonotonic(s=SIGMA_SQ):
    """lambda of (x, x^2 + s z): I = h(y) - h(s z), h(y) by numerical integration of the density."""
    from scipy.stats import norm
    v = np.linspace(-6 * s - 1, 40, 40001)
    x = np.linspace(-8, 8, 8001)
    p = (norm.pdf(x)[None, :] * norm.pdf(v[:, None] - x[None, :] ** 2, scale=s)).sum(1) * (x[1] - x[0])
    dv = v[1] - v[0]
    p /= p.sum() * dv
    I = -(p[p > 0] * np.log(p[p > 0])).sum() * dv - 0.5 * np.log(2 * np.pi * np.e * s * s)
    return float(np.sqrt(1 - np.exp(-2 * I)))


def sample(case, n, rng):
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    if case == "independent":
        return x, z
    if case == "gaussian":
        return x, 0.8 * x + 0.6 * z
    if case == "skewed":
        return x, np.expm1(2 * (0.8 * x + 0.6 * z)) / 2
    return x, x ** 2 + SIGMA_SQ * z


def lam_bits(I):
    return float(J.lam_bits(I))


def plugin_rank(x, y, B):
    cx = E.codes(x[:, None], E.pooled_edges(x[None, :, None], B))[:, 0]
    cy = E.codes(y[:, None], E.pooled_edges(y[None, :, None], B))[:, 0]
    c = np.zeros((B, B))
    np.add.at(c, (cx, cy), 1)
    return float(E.mi_counts(c[None])[0])                           # bits


def ksg(x, y):
    import infomeasure as im
    logging.getLogger("infomeasure").setLevel(logging.WARNING)
    return float(np.sqrt(1 - np.exp(-2 * max(im.mutual_information(x, y, approach="metric", k=4),
                                              0.0))))


def estimates(x, y, rng):
    n = x.size
    cap = min(J.KNUTH_MAX, n)
    xs, ys = J.normal_scores(x), J.normal_scores(y)
    B = max(3, E.n_bins(n))
    I_rank = plugin_rank(x, y, B)
    null = np.mean([plugin_rank(x, rng.permutation(y), B) for _ in range(P_NULL)])
    return {
        "knuth": lam_bits(J.mi_bits(x, y, (J.knuth_bins(x, cap), J.knuth_bins(y, cap)))),
        "knuth-scores-3": lam_bits(J.mi_bits(xs, ys, (max(3, J.knuth_bins(xs, cap)),
                                                       max(3, J.knuth_bins(ys, cap))))),
        "rank-bins": lam_bits(I_rank),
        "rank-bins-bc": lam_bits(max(I_rank - null, 0.0)),
        "gauss-copula": float(abs(np.corrcoef(xs, ys)[0, 1])),
        "ksg-scores": ksg(xs, ys),
        "ksg-raw": ksg(x, y),
    }


def one(args):
    case, n, d, seed = args
    rng = np.random.default_rng([seed, list(CASES).index(case), n, d])
    x, y = sample(case, n, rng)
    e = estimates(x, y, rng)
    return [e[k] for k in ESTIMATORS]


def compute(draws, seed):
    jobs = [(c, n, d, seed) for c in CASES for n in NS for d in range(draws)]
    with get_context("fork").Pool(int(os.environ.get("SLURM_CPUS_ON_NODE", 1))) as pool:
        out = np.array(pool.map(one, jobs, chunksize=20))
    return out.reshape(len(CASES), len(NS), draws, len(ESTIMATORS))


def summarize(V, exact):
    res = {}
    for i, c in enumerate(CASES):
        for j, n in enumerate(NS):
            for k, e in enumerate(ESTIMATORS):
                v = V[i, j, :, k]
                res[f"{c} | n={n} | {e}"] = {
                    "exact": exact[c], "median": float(np.median(v)),
                    "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                    "bias_median": float(np.median(v) - exact[c]),
                    "rmse": float(np.sqrt(np.mean((v - exact[c]) ** 2))),
                    "share_zero": float(np.mean(v <= 1e-9))}
    return res


def figure(V, exact):
    import matplotlib.pyplot as plt
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    cols = dict(zip(NS, style.SERIES[:3]))
    fig, axs = plt.subplots(1, len(CASES), figsize=(13.33, 4.6), sharey=True, layout="constrained")
    y0 = np.arange(len(ESTIMATORS))[::-1]
    for i, (c, ax) in enumerate(zip(CASES, axs)):
        for j, n in enumerate(NS):
            yy = y0 + (1 - j) * 0.22
            v = V[i, j]
            lo, md, hi = np.percentile(v, [5, 50, 95], axis=0)
            ax.hlines(yy, lo, hi, color=cols[n], lw=1.6)
            ax.plot(md, yy, "o", color=cols[n], ms=5, label=f"n = {n}")
        ax.axvline(exact[c], color=style.INK, lw=1.2)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(f"{c}  (exact λ = {exact[c]:.2f})", fontsize=11)
        ax.set_xlabel("λ estimate (median, 5-95%)")
        style.tidy(ax)
    axs[0].set_yticks(y0, ESTIMATORS)
    axs[0].tick_params(axis="y", length=0)
    axs[1].legend(loc="lower left", frameon=True, fancybox=False, fontsize=9)
    style.save(fig, NAME, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    exact = dict(CASES, **{"non-monotonic": exact_nonmonotonic()})
    path = style.data_path(NAME + ".npz")
    if os.path.exists(path) and not a.recompute:
        z = np.load(path)
        V = z["V"] if int(z["draws"]) == a.draws and int(z["seed"]) == a.seed else None
    else:
        V = None
    if V is None:
        V = compute(a.draws, a.seed)
        np.savez_compressed(path, V=V, draws=a.draws, seed=a.seed, ns=np.array(NS),
                            cases=np.array(list(CASES)), estimators=np.array(ESTIMATORS))
    res = summarize(V, exact)
    with open(style.data_path(NAME + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    figure(V, exact)
    for c in CASES:
        print(f"\n{c} (exact {exact[c]:.3f})   median [5-95%]  zero-share, for n = {NS}")
        for e in ESTIMATORS:
            cells = [res[f"{c} | n={n} | {e}"] for n in NS]
            print(f"  {e:15s} " + "   ".join(f"{r['median']:.2f} [{r['p5']:.2f},{r['p95']:.2f}] "
                                               f"{100 * r['share_zero']:3.0f}%" for r in cells))


if __name__ == "__main__":
    main()
