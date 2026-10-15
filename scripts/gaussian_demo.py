"""Main-text figure gaussian-array-lambda-rho: lambda against rho on synthetic ensembles.

    python scripts/gaussian_demo.py [--recompute] [--seed S]
        -> figures/main-text/{png,pdf}/gaussian-array-lambda-rho, curves in figures/data/*.npz

(a) lambda_m. N = 500 members of M = 10000 standard normals with pairwise correlation r,
    f_n = sqrt(r) z_0 + sqrt(1 - r) z_n (the notebook's multivariate draw, in distribution).
    lambda_m from I(s_-n; f), pooled, Knuth bins: the production oloo estimator. Plug-in MI is
    biased up by ~(B-1)^2 / (2 n ln 2) bits, which lambda ~ sqrt(2 ln2 I) turns into a floor of
    ~0.1 at r = 0 (~0.2 with the thesis FD bins); subtracting the mean of P time-permutation nulls (lam-o-hist-null's
    "debiased I") removes it, so lambda_m follows rho_m down to 0.
    rho_m^(1) = sqrt(var s / <var f_n>), rho_m^(2) = <corr(f_n, s_-n)>.
(b, c) lambda_o. A predictable signal x ~ N(0, 1); N = 40 members f_n = sqrt(0.5) x + sqrt(0.5) e_n,
    s their mean; observations o = sqrt(a) g(x) + sqrt(1 - a) eta, g standardised, for
    (b) g(x) = x and (c) g(x) = exp(x), against a, the predictable fraction of var(o).
    lambda_o from I(s; o) and from I(f; o) (jugaad: o tiled against every member), debiased
    as in (a); rho_o = corr(s, o) and the pooled corr(f, o). For Gaussian pairs lambda = |rho|, so
    (b) checks the estimator; in (c) MI is unchanged by the monotone g but corr is not.
Runs in ~4 min on 16 cores (multiprocessing over the curve points).
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
N_A, M_A, R = 500, 10000, np.linspace(0, 1, 100)                    # panel (a)
N_O, T_O, R_M, A = 40, 40000, 0.5, np.linspace(0, 1, 51)            # panels (b, c)
P_NULL = 4
G = {"linear": lambda x: x, "exp": np.exp}
BLUE, ORANGE = style.SERIES[0], style.SERIES[1]             # (a): lambda_m, rho_m
AQUA, VIOLET = style.SERIES[2], "#4a3aa7"                    # (b, c): lambda_o, rho_o; validated with (a)


def info(X, Y, rng):
    """Plug-in I(X; Y) in bits on the pooled (N, T) arrays, Knuth bins per axis, and the mean
    of P_NULL nulls that permute X along time (the same permutation for every row)."""
    x, y = X.ravel(), Y.ravel()
    bins = (knuth_bins(x, KNUTH_MAX), knuth_bins(y, KNUTH_MAX))
    I = mi_bits(x, y, bins)
    null = np.mean([mi_bits(X[:, rng.permutation(X.shape[1])].ravel(), y, bins)
                    for _ in range(P_NULL)])
    return I, null


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))


def point_m(args):
    r, seed = args
    rng = np.random.default_rng(seed)
    f = np.sqrt(r) * rng.standard_normal(M_A) + np.sqrt(1 - r) * rng.standard_normal((N_A, M_A))
    s, L = f.mean(0), loo_means(f)
    I, null = info(L, f, rng)
    rho1 = np.sqrt(np.var(s) / np.mean(np.var(f, axis=1)))
    rho2 = np.mean([corr(f[n], L[n]) for n in range(N_A)])
    return I, null, rho1, rho2


def point_o(args):
    g, a, seed = args
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T_O)
    F = np.sqrt(R_M) * x + np.sqrt(1 - R_M) * rng.standard_normal((N_O, T_O))
    s = F.mean(0)
    gx = G[g](x)
    o = np.sqrt(a) * (gx - gx.mean()) / gx.std() + np.sqrt(1 - a) * rng.standard_normal(T_O)
    I_s, null_s = info(s[None], o[None], rng)
    I_f, null_f = info(np.tile(o, (N_O, 1)), F, rng)                 # null permutes o in time
    return I_s, null_s, I_f, null_f, corr(s, o), corr(F.ravel(), np.tile(o, N_O))


def compute(seed):
    with get_context("fork").Pool(int(os.environ.get("SLURM_CPUS_ON_NODE", 1))) as pool:
        m = np.array(pool.map(point_m, [(r, [seed, 0, k]) for k, r in enumerate(R)])).T
        res = dict(r=R, I_m=m[0], I_m_null=m[1], rho_m_1=m[2], rho_m_2=m[3], a=A, seed=seed)
        for j, g in enumerate(G):
            o = np.array(pool.map(point_o, [(g, a, [seed, j + 1, k]) for k, a in enumerate(A)])).T
            res.update({f"{g}_{k}": v for k, v in
                        zip(("I_s", "I_s_null", "I_f", "I_f_null", "rho_s", "rho_f"), o)})
    return res


def lam(d, k):
    """(plug-in, debiased) lambda from I and its null mean."""
    return lam_bits(d[k]), lam_bits(d[k] - d[k + "_null"])


def plot(d):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.05), layout="constrained")
    _, lam_m = lam(d, "I_m")
    ax = axs[0]
    ax.plot(d["r"], lam_m, color=BLUE, lw=1.6, label=r"$\lambda_m$")
    ax.plot(d["r"], d["rho_m_1"], color=ORANGE, lw=1.6, label=r"$\rho_m^{(1)}$")
    ax.plot(d["r"], d["rho_m_2"], color=ORANGE, lw=1.6, ls=(0, (4, 2.5)), label=r"$\rho_m^{(2)}$")
    ax.set_xlabel("Correlation between members, $r$")
    ax.set_ylabel("Predictable component")
    style.panel_label(ax, "a", "Gaussian Members", inside=True)
    titles = {"linear": r"Signal $\mathbfit{g(x) = x}$", "exp": r"$\mathbfit{g(x) = e^{x}}$"}
    for ax, (letter, g) in zip(axs[1:], (("b", "linear"), ("c", "exp"))):
        _, ls_ = lam(d, f"{g}_I_s")
        _, lf_ = lam(d, f"{g}_I_f")
        ax.plot(d["a"], ls_, color=AQUA, lw=1.6, label=r"$\lambda_o$, $I(s;o)$")
        ax.plot(d["a"], lf_, color=AQUA, lw=1.6, ls=(0, (4, 2.5)), label=r"$\lambda_o$, $I(f;o)$")
        ax.plot(d["a"], np.abs(d[f"{g}_rho_s"]), color=VIOLET, lw=1.6, label=r"$\rho_o$, corr$(s,o)$")
        ax.plot(d["a"], np.abs(d[f"{g}_rho_f"]), color=VIOLET, lw=1.6, ls=(0, (4, 2.5)),
                label=r"$\rho_o$, corr$(f,o)$")
        ax.set_xlabel("Signal fraction of var($o$), $a$")
        style.panel_label(ax, letter, titles[g], inside=True)
    for ax in axs:
        style.tidy(ax)
        ax.set(xlim=(-0.015, 1.015), ylim=(-0.015, 1.03), xticks=np.arange(0, 1.01, 0.25),
               yticks=np.arange(0, 1.01, 0.25))
        ax.set_aspect("equal")
    axs[0].legend(loc="lower right", handlelength=2.2, borderpad=0.5, labelspacing=0.35)
    fig.legend(*axs[1].get_legend_handles_labels(), loc="outside lower center", ncol=4,
               handlelength=2.2, borderpad=0.5, columnspacing=1.4)
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
