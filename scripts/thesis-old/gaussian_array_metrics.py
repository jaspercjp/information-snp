"""The thesis Gaussian-array figure: gamma_m, lambda_m, rho_m^(1) and rho_m^(2) against r.

    python scripts/thesis-old/gaussian_array_metrics.py [--recompute] [--seed S]
        -> figures/supp-info/{png,pdf}/thesis-gaussian-array, curves in figures/data/*.npz

The npz is reused unless --recompute is given or it is missing. The main-text version, with
debiased lambda and the lambda_o demonstration, is scripts/gaussian_demo.py.

A script version of notebooks/gaussian_arrays.ipynb, cells 0-5. For each r in linspace(0, 1, 100)
an N = 500 member "ensemble" f of m = 10000 standard normals with pairwise correlation r is drawn,
and with s the ensemble mean:
    gamma_m    I(s; f) / H(f), jugaad flattening, Freedman-Diaconis bins
    lambda_m   sqrt(1 - 2^(-2 I(s; f)))
    rho_m^(1)  sqrt(var(s) / mean_n var(f_n))
    rho_m^(2)  mean_n corr(f_n, s_-n), s_-n the mean of the other N - 1 members
The notebook was unseeded and has the rho_m^(2) loop commented out (the png was made with it
in); here it is seeded and vectorised. gamma_m, lambda_m and rho_m^(1) match the original png up
to sampling noise. rho_m^(2) here lies on rho_m^(1), as it should for N = 500
(r / sqrt(r + (1-r)/(N-1)) against sqrt(r + (1-r)/N)); in the original it sits ~0.02 lower
for r < 0.3, from a run the notebook does not record.
Takes about 6 min on 16 cores. Text in the repo figure font (scripts/style.py).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style                                                         # noqa: E402
import InfoTheoryMetrics                                             # noqa: E402
from InfoTheoryMetrics import freedman_diaconis_bins                 # noqa: E402

NAME = "thesis-gaussian-array"
N, M = 500, 10000                    # ensemble members, samples per member
R = np.linspace(0, 1, 100)
SERIES = [("gamma_m", r"$\gamma_m$", style.SERIES[3], "-"),
          ("lambda_m", r"$\lambda_m$", style.SERIES[0], "-"),
          ("rho_m_1", r"$\rho_m^{(1)}$", style.SERIES[1], "-"),
          ("rho_m_2", r"$\rho_m^{(2)}$", style.SERIES[1], (0, (4, 2.5)))]


def rowcorr(a, b):
    """Pearson correlation of each row of a with the same row of b."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return np.sum(a * b, axis=1) / np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))


def metrics(r, rng):
    """gamma_m, lambda_m, rho_m^(1), rho_m^(2) for one equicorrelated Gaussian array."""
    sigma = np.full((N, N), r)
    np.fill_diagonal(sigma, 1)
    f = rng.multivariate_normal(np.zeros(N), sigma, size=M).T        # N x M
    s = f.mean(axis=0)
    I_sf, _, H_f = InfoTheoryMetrics.I_jugaad(
        s, f, nbins_alt=(freedman_diaconis_bins(np.tile(s, N)), freedman_diaconis_bins(f.ravel())))
    rho1 = np.sqrt(np.var(s) / np.mean(np.var(f, axis=1)))
    rho2 = np.mean(rowcorr(f, (N * s - f) / (N - 1)))
    return I_sf / H_f, np.sqrt(1 - 2.0 ** (-2 * I_sf)), rho1, rho2


def plot(d):
    fig, ax = plt.subplots(figsize=(3.6, 3.3), layout="constrained")
    for key, label, color, ls in SERIES:
        ax.plot(d["r"], d[key], color=color, ls=ls, lw=1.6, label=label)
    ax.set(xlim=(-0.015, 1.015), ylim=(-0.015, 1.03), xticks=np.arange(0, 1.01, 0.25),
           yticks=np.arange(0, 1.01, 0.25))
    ax.set_aspect("equal")
    style.tidy(ax)
    ax.set_xlabel("Correlation between members, $r$")
    ax.set_ylabel("Predictable component")
    ax.legend(loc="center", bbox_to_anchor=(0.72, 0.42), handlelength=2.2, borderpad=0.5, labelspacing=0.35)
    style.save(fig, NAME)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="redo the arrays even if the npz exists")
    args = ap.parse_args()
    npz = style.data_path(NAME + ".npz")
    if args.recompute or not os.path.exists(npz):
        rng = np.random.default_rng(args.seed)
        res = np.array([metrics(r, rng) for r in InfoTheoryMetrics.tqdm(R)]).T
        np.savez(npz, r=R, gamma_m=res[0], lambda_m=res[1], rho_m_1=res[2], rho_m_2=res[3],
                 seed=args.seed)
    plot(np.load(npz))


if __name__ == "__main__":
    main()
