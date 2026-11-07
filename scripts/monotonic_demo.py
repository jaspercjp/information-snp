"""Toy problem: a monotonic non-linearity lowers Pearson rho but leaves lambda unchanged.

    python scripts/monotonic_demo.py [--recompute] [--seed S] [--draws 20] [--n 1000000] [--r 0.8]
        -> figures/slides/{png,pdf}/monotonic-lambda-rho, curves in figures/data/monotonic-lambda-rho.npz
           (the npz is reused unless --recompute or the settings change)

Model, per draw: x, z ~ N(0, 1) independent, N samples each;
    u = r x + sqrt(1 - r^2) z            (x, u) bivariate Gaussian, correlation r
    y = g_a(u) = (exp(a u) - 1) / a      g_0(u) = u; increasing for every a, skewed for a > 0
The noise enters BEFORE the transform, so I(x; y) = I(x; u) = -1/2 ln(1 - r^2) for every a and
the exact lambda = sqrt(1 - exp(-2 I)) = r, flat in a. Pearson rho is not invariant:
    rho(x, y) = a r / sqrt(exp(a^2) - 1)     (r at a = 0; Stein's lemma)
x plays the predictable signal (the leave-one-out ensemble mean), y the observations.

Estimates on each draw of N pairs. N = 1,000,000 by default (20 draws): large enough that the
sample rho matches the exact curve even at a = 2, where y is lognormal with sigma = 2 and its
correlation converges slowly (median 0.30 at N = 1e4, 0.25 at 1e5, 0.23 at 1e6 vs exact 0.22).
--n 51 reproduces the small-sample version (the T of SLP lead 2-4), where Knuth's rule picks a
single bin for a variable in 14-28% of draws and lambda is then exactly 0.
    rho          sample Pearson correlation
    lambda rank  plug-in MI on equal-count bins (B = exchangeability.n_bins(N) per axis, 100 at
                 N = 1e6): it sees
                 only ranks, which g_a keeps, so it is the same number at every a (exactly flat)
    lambda Knuth the production estimator (jugaad, boot olmloo): plug-in MI in bits on equal-width
                 bins, Knuth's bin count for each variable on its own N values (cap min(KNUTH_MAX, N))
    lambda Knuth, normal scores
                 the production estimator after a rank transform of each variable to normal scores
                 (Phi^-1 of (rank - 1/2)/N): Gaussian marginals, so Knuth's count is sensible and
                 the estimate depends only on ranks (exactly the same at every a). The proposed fix.
                 (Knuth on the plain ranks fails: a uniform histogram is best fitted by ONE bin, so
                 lambda = 0 in every draw.)
    lambda KSG   infomeasure's Kraskov-Stoegbauer-Grassberger estimator, k = 4 (nats)
No bias correction. Common random numbers: every a reuses the same x and z of a draw (seeded per
draw, [seed, d]), so the curves move only with a. Points: medians over the draws; bands: 25-75%.
Panel (a): 100,000 points of one draw per a, y on its own scale, clipped to its 0.1-99.9%
range. Panel (b) uses the rank lambda (invariant by construction); panel (c) compares the three
estimators. Equal-width (Knuth) bins are not invariant: as y skews, most of it falls in the
lowest bins and the production lambda collapses (0.24 at a = 2, N = 1e6, vs the exact 0.8).
The (draw, a) jobs run on $SLURM_CPUS_ON_NODE processes (KSG takes ~25 s per call at N = 1e6).
"""
import argparse
import logging
import os
from multiprocessing import get_context

import numpy as np

import exchangeability as E
import jugaad as J
import style

A = np.linspace(0.0, 2.0, 9)
NAME = "monotonic-lambda-rho"


def g(u, a):
    """(exp(a u) - 1) / a, and u at a = 0."""
    return u if a == 0 else np.expm1(a * u) / a


def rho_exact(a, r):
    a = np.asarray(a, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = a * r / np.sqrt(np.expm1(a * a))
    return np.where(a == 0, r, out)


def lam_rank(x, y, B):
    """lambda from plug-in MI (bits) on B x B equal-count bins."""
    cx = E.codes(x[:, None], E.pooled_edges(x[None, :, None], B))[:, 0]
    cy = E.codes(y[:, None], E.pooled_edges(y[None, :, None], B))[:, 0]
    c = np.zeros((B, B))
    np.add.at(c, (cx, cy), 1)
    return float(J.lam_bits(E.mi_counts(c[None])[0]))


def lam_knuth(x, y, Bx):
    cap = min(J.KNUTH_MAX, x.size)
    return float(J.lam_bits(J.mi_bits(x, y, (Bx, J.knuth_bins(y, cap)))))


def normal_scores(v):
    """Phi^-1((rank - 1/2) / N): the rank transform to a standard normal marginal."""
    from scipy.special import ndtri
    from scipy.stats import rankdata
    return ndtri((rankdata(v) - 0.5) / v.size)


def lam_knuth_ns(x, y):
    """The production Knuth lambda on normal scores of x and y."""
    xs, ys = normal_scores(x), normal_scores(y)
    return lam_knuth(xs, ys, J.knuth_bins(xs, min(J.KNUTH_MAX, x.size)))


def lam_ksg(x, y):
    import infomeasure as im
    I = im.mutual_information(x, y, approach="metric", k=4)          # nats
    return float(np.sqrt(1 - np.exp(-2 * max(I, 0.0))))


KEYS = ("rho", "lam_rank", "lam_knuth", "lam_knuth_ns", "lam_ksg")


def draw_xu(n, r, seed, d):
    """x and u of draw d: seeded per draw, so every a (and every process) sees the same pair."""
    rng = np.random.default_rng([seed, d])
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    return x, r * x + np.sqrt(1 - r * r) * z


def one(args):
    """All four estimates for one (draw, a)."""
    n, r, seed, d, a = args
    x, u = draw_xu(n, r, seed, d)
    y = g(u, a)
    return (np.corrcoef(x, y)[0, 1], lam_rank(x, y, E.n_bins(n)),
            lam_knuth(x, y, J.knuth_bins(x, min(J.KNUTH_MAX, n))), lam_knuth_ns(x, y), lam_ksg(x, y))


def compute(draws, n, r, seed):
    jobs = [(n, r, seed, d, a) for d in range(draws) for a in A]
    procs = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    with get_context("fork").Pool(procs) as pool:
        vals = np.array(pool.map(one, jobs, chunksize=1)).reshape(draws, A.size, len(KEYS))
    return {k: vals[..., i] for i, k in enumerate(KEYS)}


def figure(res, n, r, seed):
    import matplotlib.pyplot as plt
    rc = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
          "ytick.labelsize": 12, "legend.fontsize": 11}
    blue, orange, green, yellow = style.SERIES
    q = lambda k: np.percentile(res[k], [25, 50, 75], axis=0)          # noqa: E731
    logging.getLogger("fontTools").setLevel(logging.WARNING)       # infomeasure turns on INFO
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(13.33, 4.8), layout="constrained")
        gs = fig.add_gridspec(3, 3, width_ratios=(1, 1.15, 1.15))
        x, u = draw_xu(n, r, seed, 0)                              # one draw, three values of a
        m = min(n, 100_000)
        for i, (a, c) in enumerate(((0.0, blue), (1.0, orange), (2.0, green))):
            ax = fig.add_subplot(gs[i, 0])
            y = g(u[:m], a)
            ax.plot(x[:m], y, ",", color=c, alpha=0.25, rasterized=True)
            ax.set_ylim(*np.percentile(y, [0.1, 99.9]))
            ax.set_xlim(-4, 4)
            ax.text(0.03, 0.9, f"$a$ = {a:g}", transform=ax.transAxes, va="top", fontsize=12,
                    color=style.INK)
            ax.set_yticks([])
            style.tidy(ax)
            if i == 0:
                style.panel_label(ax, "a", f"$y = g_a(u)$ against $x$, {m:,} samples")
            if i < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("$x$ (signal)")
        axs = [None, fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])]

        ax = axs[1]
        aa = np.linspace(0, 2, 200)
        ax.plot(aa, rho_exact(aa, r), color=blue, lw=2, label="ρ, exact")
        lo, md, hi = q("rho")
        ax.fill_between(A, lo, hi, color=blue, alpha=0.18, lw=0)
        ax.plot(A, md, "o", color=blue, ms=6, label=f"ρ, sample of $N$ = {n:,} (median, IQR)")
        ax.axhline(r, color=green, lw=2, label="λ, exact = $r$")
        lo, md, hi = q("lam_rank")
        ax.fill_between(A, lo, hi, color=green, alpha=0.18, lw=0)
        ax.plot(A, md, "D", color=green, ms=6, label="λ, rank bins (median, IQR)")
        ax.set_xlabel("strength of the monotonic non-linearity $a$")
        ax.set_ylabel("ρ or λ")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fancybox=False, loc="lower left")
        style.tidy(ax)
        style.panel_label(ax, "b", "ρ falls, λ should not")

        ax = axs[2]
        ax.axhline(r, color=style.INK, lw=2, label="exact = $r$")
        for k, c, m, ls, lab in (("lam_rank", green, "D", "-", "rank (equal-count) bins"),
                                 ("lam_knuth", orange, "s", "-", "Knuth bins (production)"),
                                 ("lam_knuth_ns", yellow, "o", "-",
                                  "Knuth bins on normal scores (fix)"),
                                 ("lam_ksg", style.MUTED, "^", "--", "KSG, $k$ = 4")):
            lo, md, hi = q(k)
            ax.fill_between(A, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(A, md, m + ls, color=c, ms=6, lw=1.2, label=lab, mfc=c if k != "lam_knuth_ns"
                    else "none", mew=1.4)
        zero = 100 * (res["lam_knuth"] <= 1e-9).mean(0)
        if zero.max() > 0:
            ax.text(0.98, 0.97, f"Knuth: λ = 0 in {zero.min():.0f}-{zero.max():.0f}% of "
                    "draws\n(one bin chosen for a variable)", transform=ax.transAxes, ha="right",
                    va="top", fontsize=10.5, color=style.MUTED)
        ax.set_xlabel("strength of the monotonic non-linearity $a$")
        ax.set_ylabel("λ")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fancybox=False, loc="lower left")
        style.tidy(ax)
        style.panel_label(ax, "c", "λ estimators")
        style.save(fig, NAME, dpi=200, folder="slides")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--r", type=float, default=0.8)
    a = ap.parse_args()
    path = style.data_path(NAME + ".npz")
    settings = dict(seed=a.seed, draws=a.draws, n=a.n, r=a.r)
    old = dict(np.load(path)) if os.path.exists(path) and not a.recompute else None
    if old is not None and all(float(old[k]) == float(v) for k, v in settings.items()):
        res = {k: old[k] for k in KEYS}
    else:
        res = compute(a.draws, a.n, a.r, a.seed)
        np.savez_compressed(path, a=A, rho_exact=rho_exact(A, a.r), **res, **settings)
    figure(res, a.n, a.r, a.seed)
    for k in res:
        md = np.median(res[k], axis=0)
        print(f"{k:10s} median at a = 0, 1, 2: {md[0]:.3f} {md[4]:.3f} {md[8]:.3f}")
    print(f"rho exact at a = 0, 1, 2: {rho_exact(np.array([0, 1, 2.0]), a.r).round(3)}")


if __name__ == "__main__":
    main()
