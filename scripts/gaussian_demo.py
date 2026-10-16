"""Main-text figures gaussian-array-lambda-rho and fixed-ensemble-lambda-o: lambda against rho.

    python scripts/gaussian_demo.py [--recompute] [--seed S]
        -> figures/main-text/{png,pdf}/{gaussian-array-lambda-rho,fixed-ensemble-lambda-o},
           curves in figures/data/*.npz

gaussian-array-lambda-rho: one generative model for every panel. A predictable signal x ~ N(0, 1), T = 600 times; with
g_hat = g(x) standardised, N = 500 members and the observations share the signal fraction a:
    f_n = sqrt(a) g_hat(x) + sqrt(1 - a) z_n,    o = sqrt(a) g_hat(x) + sqrt(1 - a) eta,
z_n, eta ~ N(0, 1) independent, s the member mean, s_-n the mean of the other N - 1. So any two
members, and o with any member, correlate at a: the observations are one more member (a
perfect model), and the RPC should be 1. Columns: g(x) = x (Gaussian) and g(x) = exp(x) (skewed).
Top row, the ensemble: lambda_m from I(s_-n; f), pooled (the jugaad flattening with s_-n, as
    the production oloo estimator), Knuth bins; rho_m^(1) = sqrt(var s / <var f_n>),
    rho_m^(2) = <corr(f_n, s_-n)>.
Bottom row, the observations: lambda_o from I(s; o) and from I(f; o) (o tiled against every
    member); rho_o = corr(s, o) and the pooled corr(f, o).
Bins: Knuth's rule, sized as in the production estimator (o and s_-n on their T values, the
members on all N T). Every I is the plug-in MI minus the mean of P_NULL time-permutation nulls,
lambda = sqrt(1 - 2^(-2 I)). For Gaussian pairs lambda = |rho|, so the left column checks the
estimators.

fixed-ensemble-lambda-o: the members do not change with a, f_n = sqrt(R_M) x + sqrt(1 - R_M) z_n
(R_M = 0.5, linear in x), only the observations do, o = sqrt(a) g_hat(x) + sqrt(1 - a) eta, for
g(x) = x and g(x) = exp(x). Same N, T, bins and estimators as the bottom row above. A member is at
most sqrt(R_M) correlated with x, so the (f; o) curves stop near sqrt(0.5) at a = 1; with g = exp
the observations are a nonlinear function of the members' signal, which lambda_o sees and
rho_o, a linear measure, does not fully.
Both run in under a minute on 16 cores (multiprocessing over the curve points).
"""
import argparse
import os
from multiprocessing import get_context

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

import style                                                         # noqa: E402
from jugaad import KNUTH_MAX, knuth_bins, lam_bits, loo_means, mi_bits   # noqa: E402

NAME, NAME_FIXED = "gaussian-array-lambda-rho", "fixed-ensemble-lambda-o"
N, T, A = 500, 600, np.linspace(0, 1, 51)
R_M = 0.5                                                    # fixed-ensemble members' signal share
P_NULL = 4
G = {"linear": lambda x: x, "exp": np.exp}
TITLES = {"linear": r"Signal $\mathbf{g(x) = x}$", "exp": r"Signal $\mathbf{g(x) = e^{x}}$"}
LAM_M, RHO_M = style.SERIES[0], style.SERIES[1]             # top row: lambda_m, rho_m
LAM_O, RHO_O = "tomato", "darkseagreen"                      # bottom row: lambda_o, rho_o
DASH = (0, (4, 2.5))
KEYS_O = ("I_s", "I_s_null", "I_f", "I_f_null", "rho_s", "rho_f")
KEYS = ("I_m", "I_m_null", "rho_m_1", "rho_m_2") + KEYS_O


def knuth(v):
    return knuth_bins(v, min(KNUTH_MAX, len(v)))


def info(X, Y, rng, size_x=None):
    """Plug-in I(X; Y) in bits on the pooled (N, T) arrays, and the mean of P_NULL nulls that
    permute X along time (the same permutation for every row). Knuth bins as the production
    estimator (run.py KNUTH, boot.py lmloo): an axis that repeats or nearly repeats one series
    over the members (tiled o, s_-n) is sized on its T values `size_x`, capped at T; the members'
    axis Y on all N T values."""
    x, y = X.ravel(), Y.ravel()
    bins = (knuth(x if size_x is None else size_x), knuth(y))
    I = mi_bits(x, y, bins)
    null = np.mean([mi_bits(X[:, rng.permutation(X.shape[1])].ravel(), y, bins)
                    for _ in range(P_NULL)])
    return I, null


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))


def point(args):
    """Both rows at one (g, a): the ensemble measures and the skill against o."""
    g, a, seed = args
    rng = np.random.default_rng(seed)
    gx = G[g](rng.standard_normal(T))
    gx = (gx - gx.mean()) / gx.std()
    F = np.sqrt(a) * gx + np.sqrt(1 - a) * rng.standard_normal((N, T))
    o = np.sqrt(a) * gx + np.sqrt(1 - a) * rng.standard_normal(T)
    s, L = F.mean(0), loo_means(F)
    I_m, null_m = info(L, F, rng, size_x=L.mean(0))                 # s_-n on its T centres
    rho1 = np.sqrt(np.var(s) / np.mean(np.var(F, axis=1)))
    rho2 = np.mean([corr(F[n], L[n]) for n in range(N)])
    return (I_m, null_m, rho1, rho2) + skill(F, s, o, rng)


def skill(F, s, o, rng):
    """lambda_o terms and rho_o, from the ensemble mean s and from the members F."""
    I_s, null_s = info(s[None], o[None], rng)
    I_f, null_f = info(np.tile(o, (N, 1)), F, rng, size_x=o)        # null permutes o in time
    return I_s, null_s, I_f, null_f, corr(s, o), corr(F.ravel(), np.tile(o, N))


def point_fixed(args):
    """The skill at one (g, a) with the members fixed at signal share R_M, linear in x."""
    g, a, seed = args
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    F = np.sqrt(R_M) * x + np.sqrt(1 - R_M) * rng.standard_normal((N, T))
    gx = G[g](x)
    o = np.sqrt(a) * (gx - gx.mean()) / gx.std() + np.sqrt(1 - a) * rng.standard_normal(T)
    return skill(F, F.mean(0), o, rng)


def compute(seed, fn, keys, tag):
    res = dict(a=A, seed=seed)
    with get_context("fork").Pool(int(os.environ.get("SLURM_CPUS_ON_NODE", 1))) as pool:
        for j, g in enumerate(G):
            out = np.array(pool.map(fn, [(g, a, [seed, tag, j, k]) for k, a in enumerate(A)])).T
            res.update({f"{g}_{k}": v for k, v in zip(keys, out)})
    return res


def lam(d, k):
    """Debiased lambda from I and its null mean."""
    return lam_bits(d[k] - d[k + "_null"])


def skill_curves(ax, d, g):
    a = d["a"]
    ax.plot(a, lam(d, f"{g}_I_s"), color=LAM_O, lw=1.6, label=r"$\lambda_o$, $I(s;o)$")
    ax.plot(a, lam(d, f"{g}_I_f"), color=LAM_O, lw=1.6, ls=DASH, label=r"$\lambda_o$, $I(f;o)$")
    ax.plot(a, np.abs(d[f"{g}_rho_s"]), color=RHO_O, lw=1.6, label=r"$\rho_o$, corr$(s,o)$")
    ax.plot(a, np.abs(d[f"{g}_rho_f"]), color=RHO_O, lw=1.6, ls=DASH, label=r"$\rho_o$, corr$(f,o)$")


def finish(ax):
    style.tidy(ax)
    ax.set(xlim=(-0.015, 1.015), ylim=(0, 1.03), xticks=np.arange(0, 1.01, 0.25),
           yticks=np.arange(0, 1.01, 0.25))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7, handlelength=2.0, borderpad=0.4,
              labelspacing=0.3, handletextpad=0.5)


def plot(d):
    fig, axs = plt.subplots(2, 2, figsize=(5.0, 5.0), layout="constrained", sharex=True, sharey=True)
    a = d["a"]
    for c, g in enumerate(G):
        top, bot = axs[0, c], axs[1, c]
        top.plot(a, lam(d, f"{g}_I_m"), color=LAM_M, lw=1.6, label=r"$\lambda_m$")
        top.plot(a, d[f"{g}_rho_m_1"], color=RHO_M, lw=1.6, label=r"$\rho_m^{(1)}$")
        top.plot(a, d[f"{g}_rho_m_2"], color=RHO_M, lw=1.6, ls=DASH, label=r"$\rho_m^{(2)}$")
        skill_curves(bot, d, g)
        style.panel_label(top, "ab"[c], TITLES[g], inside=True)
        style.panel_label(bot, "cd"[c], inside=True)
        bot.set_xlabel("Signal fraction of variance, $a$")
    for ax in axs.flat:
        finish(ax)
    for ax in axs[:, 0]:
        ax.set_ylabel("Predictable component")
    style.save(fig, NAME)


def plot_fixed(d):
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.75), layout="constrained", sharey=True)
    for ax, letter, g in zip(axs, "ab", G):
        skill_curves(ax, d, g)
        style.panel_label(ax, letter, TITLES[g], inside=True)
        ax.set_xlabel("Signal fraction of var($o$), $a$")
        finish(ax)
    axs[0].set_ylabel("Predictable component")
    style.save(fig, NAME_FIXED)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="redo the arrays even if the npz exists")
    args = ap.parse_args()
    for name, fn, keys, tag, draw in ((NAME, point, KEYS, 0, plot),
                                      (NAME_FIXED, point_fixed, KEYS_O, 1, plot_fixed)):
        npz = style.data_path(name + ".npz")
        if args.recompute or not os.path.exists(npz):
            np.savez(npz, **compute(args.seed, fn, keys, tag))
        draw(dict(np.load(npz)))


if __name__ == "__main__":
    main()
