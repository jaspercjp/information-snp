"""Main-text figure gaussian-array-lambda-rho: lambda against rho on synthetic ensembles.

    python scripts/gaussian_demo.py [--recompute] [--seed S]
        -> figures/main-text/{png,pdf}/gaussian-array-lambda-rho, curves in figures/data/*.npz

N = 500 members of T = 40000 standard-normal draws in every panel; z_0, z_n, x, eta ~ N(0, 1)
independent, s the member mean, s_-n the mean of the other N - 1.
(a) Gaussian members f_n = sqrt(r) z_0 + sqrt(1 - r) z_n, pairwise correlation r (the notebook's
    multivariate draw, in distribution). lambda_m from I(s_-n; f), pooled (the jugaad flattening
    with s_-n, as the production oloo estimator); rho_m^(1) = sqrt(var s / <var f_n>),
    rho_m^(2) = <corr(f_n, s_-n)>. For Gaussian pairs lambda = |rho|, so lambda_m = rho_m^(2).
(b, c) Skill. Members fixed, f_n = sqrt(R_M) x + sqrt(1 - R_M) z_n (R_M = 0.5, linear in x); only the
    observations vary, o = sqrt(a) g_hat(x) + sqrt(1 - a) eta, g_hat = g standardised, for
    (b) g(x) = x and (c) g(x) = exp(x), against a, the signal fraction of var(o).
    lambda_o from I(s; o) and from I(f; o) (o tiled against every member); rho_o = corr(s, o) and
    the pooled corr(f, o). A member is at most sqrt(R_M) correlated with x, so the (f; o) curves
    stop near sqrt(0.5). The ensemble's own lambda_m and rho_m (= rho_m^(2)) are drawn for
    reference, solid black and grey; they do not depend on a.
Bins: Knuth's rule, sized as in the production estimator (o and s_-n on their T values, the
members on all N T). Every I is the plug-in MI minus the mean of P_NULL time-permutation nulls,
lambda = sqrt(1 - 2^(-2 I)). Runs in ~12 min with 7 worker processes over the curve points
($SNP_DEMO_JOBS; each holds ~1.6 GB at T = 40000, so the default stays inside a 16 GB allocation).
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

NAME = "gaussian-array-lambda-rho"
N, T = 500, 40000
R, A = np.linspace(0, 1, 100), np.linspace(0, 1, 51)        # panel (a) r; panels (b, c) a
R_M = 0.5                                                    # (b, c): the members' signal share
P_NULL = 4
G = {"linear": lambda x: x, "exp": np.exp}
TITLES = {"linear": r"Signal $\mathbf{g(x) = x}$", "exp": r"Signal $\mathbf{g(x) = e^{x}}$"}
LAM_M, RHO_M = style.SERIES[0], style.SERIES[1]             # (a): lambda_m, rho_m
LAM_O, RHO_O = "tomato", "darkseagreen"                      # (b, c): lambda_o, rho_o
DASH = (0, (4, 2.5))
KEYS_M = ("I_m", "I_m_null", "rho_m_1", "rho_m_2")
KEYS = KEYS_M + ("I_s", "I_s_null", "I_f", "I_f_null", "rho_s", "rho_f")
LEGEND = dict(fontsize=7, handlelength=2.0, borderpad=0.4, labelspacing=0.3, handletextpad=0.5)


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


def ensemble(F, rng):
    """lambda_m terms from I(s_-n; f) pooled, rho_m^(1) and rho_m^(2)."""
    s, L = F.mean(0), loo_means(F)
    I_m, null_m = info(L, F, rng, size_x=L.mean(0))                 # s_-n on its T centres
    rho1 = np.sqrt(np.var(s) / np.mean(np.var(F, axis=1)))
    rho2 = np.mean([corr(F[n], L[n]) for n in range(N)])
    return I_m, null_m, rho1, rho2


def skill(F, s, o, rng):
    """lambda_o terms and rho_o, from the ensemble mean s and from the members F."""
    I_s, null_s = info(s[None], o[None], rng)
    I_f, null_f = info(np.tile(o, (N, 1)), F, rng, size_x=o)        # null permutes o in time
    return I_s, null_s, I_f, null_f, corr(s, o), corr(F.ravel(), np.tile(o, N))


def point_m(args):
    """Panel (a) at one r."""
    r, seed = args
    rng = np.random.default_rng(seed)
    F = np.sqrt(r) * rng.standard_normal(T) + np.sqrt(1 - r) * rng.standard_normal((N, T))
    return ensemble(F, rng)


def point_o(args):
    """Panels (b, c) at one (g, a): the fixed ensemble and its skill against o."""
    g, a, seed = args
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    F = np.sqrt(R_M) * x + np.sqrt(1 - R_M) * rng.standard_normal((N, T))
    gx = G[g](x)
    o = np.sqrt(a) * (gx - gx.mean()) / gx.std() + np.sqrt(1 - a) * rng.standard_normal(T)
    return ensemble(F, rng) + skill(F, F.mean(0), o, rng)


def compute(seed):
    jobs = int(os.environ.get("SNP_DEMO_JOBS", min(7, int(os.environ.get("SLURM_CPUS_ON_NODE", 1)))))
    with get_context("fork").Pool(jobs) as pool:
        m = np.array(pool.map(point_m, [(r, [seed, 0, k]) for k, r in enumerate(R)])).T
        res = dict(r=R, a=A, seed=seed, **dict(zip(KEYS_M, m)))
        for j, g in enumerate(G):
            out = np.array(pool.map(point_o, [(g, a, [seed, j + 1, k]) for k, a in enumerate(A)])).T
            res.update({f"{g}_{k}": v for k, v in zip(KEYS, out)})
    return res


def lam(d, k):
    """Debiased lambda from I and its null mean."""
    return lam_bits(d[k] - d[k + "_null"])


def finish(ax):
    style.tidy(ax)
    ax.set(xlim=(-0.015, 1.015), ylim=(0, 1.03), xticks=np.arange(0, 1.01, 0.25),
           yticks=np.arange(0, 1.01, 0.25))
    ax.set_aspect("equal")


def plot(d):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75), layout="constrained")
    ax = axs[0]
    ax.plot(d["r"], lam(d, "I_m"), color=LAM_M, lw=1.6, label=r"$\lambda_m$")
    ax.plot(d["r"], d["rho_m_1"], color=RHO_M, lw=1.6, label=r"$\rho_m^{(1)}$")
    ax.plot(d["r"], d["rho_m_2"], color=RHO_M, lw=1.6, ls=DASH, label=r"$\rho_m^{(2)}$")
    ax.set_xlabel("Correlation between members, $r$")
    ax.set_ylabel("Predictable component")
    style.panel_label(ax, "a", "Gaussian Members", inside=True)
    finish(ax)
    ax.legend(loc="lower right", handlelength=2.2, borderpad=0.5, labelspacing=0.35)
    for ax, letter, g in zip(axs[1:], "bc", G):
        a = d["a"]
        ax.plot(a, lam(d, f"{g}_I_s"), color=LAM_O, lw=1.6, label=r"$\lambda_o$, $I(s;o)$")
        ax.plot(a, lam(d, f"{g}_I_f"), color=LAM_O, lw=1.6, ls=DASH, label=r"$\lambda_o$, $I(f;o)$")
        ax.plot(a, np.abs(d[f"{g}_rho_s"]), color=RHO_O, lw=1.6, label=r"$\rho_o$, corr$(s,o)$")
        ax.plot(a, np.abs(d[f"{g}_rho_f"]), color=RHO_O, lw=1.6, ls=DASH, label=r"$\rho_o$, corr$(f,o)$")
        ax.plot(a, lam(d, f"{g}_I_m"), color="black", lw=1.2, label=r"$\lambda_m$")
        ax.plot(a, d[f"{g}_rho_m_2"], color="0.55", lw=1.2, label=r"$\rho_m$")
        ax.set_xlabel("Signal fraction of var($o$), $a$")
        style.panel_label(ax, letter, TITLES[g], inside=True)
        finish(ax)
        h, lab = ax.get_legend_handles_labels()                    # skill lower right,
        ax.add_artist(ax.legend(h[:4], lab[:4], loc="lower right", **LEGEND))
        ax.legend(h[4:], lab[4:], loc="upper left", bbox_to_anchor=(0.11, 1.0), **LEGEND)
    style.save(fig, NAME)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="redo the arrays even if the npz exists")
    args = ap.parse_args()
    npz = style.data_path(NAME + ".npz")
    if args.recompute or not os.path.exists(npz):
        np.savez(npz, **compute(args.seed))
    plot(dict(np.load(npz)))


if __name__ == "__main__":
    main()
