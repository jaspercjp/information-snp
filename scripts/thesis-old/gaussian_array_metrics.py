"""The thesis Gaussian-array figure: gamma_m, lambda_m, rho_m^(1) and rho_m^(2) against r.

    python scripts/thesis-old/gaussian_array_metrics.py [--seed S] [--out PATH]
        -> figures/thesis-old/from-notebooks/gaussian-array.png

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
Takes about 6 min on 16 cores. Uses LaTeX text like the original when `latex` is on PATH
(`ml system texlive`), otherwise matplotlib's Computer Modern mathtext.
"""
import argparse
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

import InfoTheoryMetrics                                             # noqa: E402
from InfoTheoryMetrics import freedman_diaconis_bins                 # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "figures", "thesis-old", "from-notebooks", "gaussian-array.png")
N, M = 500, 10000                    # ensemble members, samples per member
R = np.linspace(0, 1, 100)


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


def plot(gam, lam, rho1, rho2, out):
    if shutil.which("latex"):
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif", serif=["Computer Modern"])
    else:
        plt.rc("mathtext", fontset="cm")
        plt.rc("font", family="serif", serif=["cmr10"])
        plt.rc("axes.formatter", use_mathtext=True)
    fig = plt.figure(figsize=(5, 4.8))
    plt.plot(R, gam, label=r"$\gamma_m$")
    plt.plot(R, lam, label=r"$\lambda_m$")
    plt.plot(R, rho1, label=r"$\rho_{m}^{(1)}$")
    plt.plot(R, rho2, "--", label=r"$\rho_{m}^{(2)}$")
    plt.legend()
    plt.xlabel("$r$")
    plt.ylabel("Predictable Component")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = np.array([metrics(r, rng) for r in InfoTheoryMetrics.tqdm(R)]).T
    plot(*res, args.out)
    np.savez(os.path.splitext(args.out)[0] + ".npz", r=R, gamma_m=res[0], lambda_m=res[1],
             rho_m_1=res[2], rho_m_2=res[3], seed=args.seed)
    print(args.out)


if __name__ == "__main__":
    main()
