"""The thesis Gaussian-array figure, redrawn: lambda_m, rho_m^(1) and rho_m^(2) against r.

    python scripts/thesis-old/gaussian_array_metrics.py [--recompute] [--seed S] [--out PATH]
        -> figures/thesis-old/from-notebooks/gaussian-array-lambda-rho.{png,pdf,npz}

The npz holds the curves (gamma_m too); it is reused unless --recompute is given or it is missing,
so restyling does not redo the arrays. The original png (gaussian-array.png, with gamma_m) is kept.

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
OUT = os.path.join(REPO, "figures", "thesis-old", "from-notebooks", "gaussian-array-lambda-rho.png")
N, M = 500, 10000                    # ensemble members, samples per member
R = np.linspace(0, 1, 100)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"     # as scripts/plots.py
SERIES = [("lambda_m", r"$\lambda_m$", "#2a78d6", "-"),  # categorical slots 1-3, validated
          ("rho_m_1", r"$\rho_m^{(1)}$", "#eb6834", "-"),
          ("rho_m_2", r"$\rho_m^{(2)}$", "#1baf7a", (0, (4, 2.5)))]


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


def plot(d, out):
    if shutil.which("latex"):
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif", serif=["Computer Modern"])
    else:
        plt.rc("mathtext", fontset="cm")
        plt.rc("font", family="serif", serif=["cmr10"])
        plt.rc("axes.formatter", use_mathtext=True)
    plt.rc("font", size=10)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for key, label, color, ls in SERIES:
        ax.plot(d["r"], d[key], color=color, ls=ls, lw=1.6, solid_capstyle="round",
                dash_capstyle="round", label=label)
    ax.set(xlim=(-0.015, 1.015), ylim=(-0.015, 1.03), xticks=np.arange(0, 1.01, 0.2), yticks=np.arange(0, 1.01, 0.2))
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set(color=MUTED, lw=0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3, width=0.8)
    ax.set_xlabel("Correlation between members, $r$", color=INK)
    ax.set_ylabel("Predictable component", color=INK)
    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, handlelength=2.4,
                    borderpad=0.6, labelspacing=0.5, labelcolor=INK)
    leg.get_frame().set(edgecolor=MUTED, facecolor="white", linewidth=0.8, alpha=1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stem = os.path.splitext(out)[0]
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--recompute", action="store_true", help="redo the arrays even if the npz exists")
    args = ap.parse_args()
    npz = os.path.splitext(args.out)[0] + ".npz"
    if args.recompute or not os.path.exists(npz):
        rng = np.random.default_rng(args.seed)
        res = np.array([metrics(r, rng) for r in InfoTheoryMetrics.tqdm(R)]).T
        os.makedirs(os.path.dirname(npz), exist_ok=True)
        np.savez(npz, r=R, gamma_m=res[0], lambda_m=res[1], rho_m_1=res[2], rho_m_2=res[3],
                 seed=args.seed)
    plot(np.load(npz), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
