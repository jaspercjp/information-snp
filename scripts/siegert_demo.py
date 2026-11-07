"""Supp-info figure siegert-model-ensemble-size: lambda and rho RPC under Siegert et al. (2016).

    python scripts/siegert_demo.py [--recompute] [--seed S]
        -> figures/supp-info/{png,pdf}/siegert-model-ensemble-size, curves in figures/data/*.npz

Siegert et al. (2016): observations y_t = mu_y + s_t + eps_t, members x_{t,r} = mu_x + beta s_t +
eta_{t,r}, with s, eps, eta independent N(0, sigma^2); here mu_x = mu_y = 0.5 and every sigma = 1
(the means do not change a correlation or an MI). Weak (beta = 0.1) and strong (beta = 0.9) signal,
against ensemble size N, T = 40000 times. One draw of s, eps and N_MAX members' eta for both
regimes; an ensemble of N is the first N members, so the curves are smooth in N.
    lambda_o = lambda(I(s_-n~; y~))   production oloo (jugaad, s_-n against y tiled)
    rho_o    = corr(s, y)             s the N-member mean
    lambda_m = lambda(I(s_-n~; x~))   production lmloo
    rho_m    = <corr(x_r, s_-r)>_r
Bins and bias correction as gaussian_demo.py (production Knuth sizing, time-permutation nulls).
Analytic, sigma = 1: rho_o = beta / sqrt(2 (beta^2 + 1/N)), rho_m = beta^2 / sqrt((1 + beta^2)
(beta^2 + 1/(N-1))); lambda_o (from s_-n) is rho_o at N - 1. For Gaussians lambda = rho, so the
lambda RPC is lambda_o(N - 1) / rho_m(N) = sqrt(1 + beta^2) / (sqrt(2) beta) at every N (the s_-n
construction cancels the finite-N term), while the rho RPC from the full mean s is biased high at
small N. In the weak regime lambda_m ~ 0.01-0.03 (~1e-4 bits) is below what the estimator resolves at
small N, so the lambda RPC there is noise until N ~ 35. For N -> inf, RPC -> sqrt(1 +
beta^2) / (sqrt(2) beta): 7.1 at beta = 0.1, 1.05 at beta = 0.9.
"""
import argparse
import os
from multiprocessing import get_context

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

import style                                                         # noqa: E402
from gaussian_demo import LEGEND, corr, info                         # noqa: E402
from jugaad import lam_bits, loo_means                               # noqa: E402

NAME = "siegert-model-ensemble-size"
T, MU = 40000, 0.5
BETAS = (0.1, 0.9)
NS = np.array([2, 3, 5, 8, 12, 20, 35, 60, 100, 170, 300, 500])
N_MAX = NS.max()
KEYS = ("I_o", "I_o_null", "rho_o", "I_m", "I_m_null", "rho_m")
LAM_O, RHO_O = "tomato", "darkseagreen"
LAM_M, RHO_M = style.SERIES[0], style.SERIES[1]
_D = {}


def point(args):
    """All six terms for one (beta, N) from the shared draw."""
    beta, n, seed = args
    rng = np.random.default_rng(seed)
    X = MU + beta * _D["s"] + _D["eta"][:n]                          # (n, T) members
    y = MU + _D["s"] + _D["eps"]
    L = loo_means(X)
    I_o, null_o = info(np.broadcast_to(y, X.shape), L, rng, size_x=y)   # o tiled vs s_-n
    I_m, null_m = info(L, X, rng, size_x=L.mean(0))
    rho_m = np.mean([corr(X[r], L[r]) for r in range(n)])
    return I_o, null_o, corr(X.mean(0), y), I_m, null_m, rho_m


def compute(seed):
    rng = np.random.default_rng([seed, 5])
    _D.update(s=rng.standard_normal(T), eps=rng.standard_normal(T),
              eta=rng.standard_normal((N_MAX, T)))
    jobs = int(os.environ.get("SNP_DEMO_JOBS", min(7, int(os.environ.get("SLURM_CPUS_ON_NODE", 1)))))
    res = dict(N=NS, betas=np.array(BETAS), seed=seed)
    with get_context("fork").Pool(jobs) as pool:
        for j, beta in enumerate(BETAS):
            out = np.array(pool.map(point, [(beta, n, [seed, 6, j, k]) for k, n in enumerate(NS)])).T
            res.update({f"b{j}_{k}": v for k, v in zip(KEYS, out)})
    _D.clear()
    return res


def lam(d, k):
    return lam_bits(d[k] - d[k + "_null"])


def theory(beta, n):
    rho_o = beta / np.sqrt(2 * (beta ** 2 + 1 / n))
    rho_m = beta ** 2 / np.sqrt((1 + beta ** 2) * (beta ** 2 + 1 / (n - 1)))
    lam_o = beta / np.sqrt(2 * (beta ** 2 + 1 / (n - 1)))
    return rho_o, rho_m, lam_o


def plot(d):
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.4), layout="constrained", sharex=True)
    nn = np.geomspace(2, N_MAX, 200)
    for j, beta in enumerate(BETAS):
        top, bot = axs[0, j], axs[1, j]
        lo, lm = lam(d, f"b{j}_I_o"), lam(d, f"b{j}_I_m")
        ro, rm = d[f"b{j}_rho_o"], d[f"b{j}_rho_m"]
        t_ro, t_rm, t_lo = theory(beta, nn)
        for ax, x, c, lab in ((top, lo, LAM_O, r"$\lambda_o$"), (top, ro, RHO_O, r"$\rho_o$"),
                              (top, lm, LAM_M, r"$\lambda_m$"), (top, rm, RHO_M, r"$\rho_m$")):
            ax.plot(d["N"], x, color=c, lw=1.6, marker="o", ms=3, label=lab)
        for x in (t_ro, t_rm):
            top.plot(nn, x, color="0.35", lw=0.8, ls=(0, (1, 1.5)))
        top.plot([], [], color="0.35", lw=0.8, ls=(0, (1, 1.5)), label="analytic")
        with np.errstate(divide="ignore", invalid="ignore"):
            bot.plot(d["N"], lo / lm, color="black", lw=1.6, marker="o", ms=3, label=r"$\lambda_o / \lambda_m$")
            bot.plot(d["N"], ro / rm, color="0.55", lw=1.6, marker="o", ms=3, label=r"$\rho_o / \rho_m$")
        bot.plot(nn, t_ro / t_rm, color="0.35", lw=0.8, ls=(0, (1, 1.5)), label=r"analytic, $\rho$")
        bot.plot(nn, t_lo / t_rm, color="0.35", lw=0.8, ls=(0, (4, 2)), label=r"analytic, $\lambda$")
        bot.axhline(1, color=style.MUTED, lw=0.8, zorder=0)
        regime = "Weak" if beta < 0.5 else "Strong"
        style.panel_label(top, "ab"[j], rf"{regime} signal, $\mathbf{{\beta = {beta}}}$", inside=True)
        style.panel_label(bot, "cd"[j], inside=True)
        for ax in (top, bot):
            style.tidy(ax)
            ax.set_xscale("log")
        top.set_ylim(0, 1.03)
        bot.set_ylim(0, 12 if beta < 0.5 else 1.6)
        bot.set_xlabel("Ensemble size, $N$")
        if beta < 0.5:                                             # curves stay below 0.65
            top.legend(loc="upper left", bbox_to_anchor=(0.11, 1.0), ncol=2, **LEGEND)
        else:
            top.legend(loc="lower right", **LEGEND)
        bot.legend(loc="upper right" if beta < 0.5 else "lower right", **LEGEND)
    axs[0, 0].set_ylabel("Predictable component")
    axs[1, 0].set_ylabel("RPC")
    style.save(fig, NAME)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    npz = style.data_path(NAME + ".npz")
    if args.recompute or not os.path.exists(npz):
        np.savez(npz, **compute(args.seed))
    plot(dict(np.load(npz)))


if __name__ == "__main__":
    main()
