"""Toy problem: a monotonic non-linearity lowers Pearson rho but leaves lambda unchanged.

    python scripts/monotonic_demo.py [--recompute] [--seed S] [--draws D] [--n N] [--r R]
        -> figures/slides/{png,pdf}/monotonic-lambda-rho, curves in figures/data/monotonic-lambda-rho.npz
           (the npz is reused unless --recompute or the settings change)

Model, per draw: x, z ~ N(0, 1) independent, N samples each;
    u = r x + sqrt(1 - r^2) z            (x, u) bivariate Gaussian, correlation r
    y = g_a(u) = (exp(a u) - 1) / a      g_0(u) = u; increasing for every a, skewed for a > 0
The noise enters BEFORE the transform, so I(x; y) = I(x; u) = -1/2 ln(1 - r^2) for every a and
the exact lambda = sqrt(1 - exp(-2 I)) = r, flat in a. Pearson rho is not invariant:
    rho(x, y) = a r / sqrt(exp(a^2) - 1)     (r at a = 0; Stein's lemma)
x plays the predictable signal (the leave-one-out ensemble mean), y the observations.

Estimates on each draw of N pairs (N = 51 by default, the T of SLP lead 2-4):
    rho          sample Pearson correlation
    lambda rank  plug-in MI on equal-count bins (B = exchangeability.n_bins(N) per axis): it sees
                 only ranks, which g_a keeps, so it is the same number at every a (exactly flat)
    lambda Knuth the production estimator (jugaad, boot olmloo): plug-in MI in bits on equal-width
                 bins, Knuth's bin count for each variable on its own N values (cap min(KNUTH_MAX, N))
    lambda KSG   infomeasure's Kraskov-Stoegbauer-Grassberger estimator, k = 4 (nats)
No bias correction: plug-in binned MI is biased at small N, so the level of a binned lambda is
not r; what matters is whether it stays flat as a grows. Common random numbers: every a reuses
the same x and z of a draw, so the curves move only with a. Points: medians over the draws;
bands: 25-75%. Panel (b) uses the rank lambda (invariant by construction); panel (c) compares the
three estimators. At N = 51 Knuth's rule often picks a single bin for one variable, which makes
the plug-in MI, and lambda, exactly 0: the share of such draws is printed on panel (c).
Sample rho at small N falls far less than the exact rho: with a skewed y the population value is
set by rare extremes that a sample of N seldom contains.
"""
import argparse
import logging
import os

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


def lam_ksg(x, y):
    import infomeasure as im
    I = im.mutual_information(x, y, approach="metric", k=4)          # nats
    return float(np.sqrt(1 - np.exp(-2 * max(I, 0.0))))


def compute(draws, n, r, seed):
    rng = np.random.default_rng(seed)
    B = E.n_bins(n)
    out = {k: np.empty((draws, A.size)) for k in ("rho", "lam_rank", "lam_knuth", "lam_ksg")}
    for d in range(draws):
        x, z = rng.standard_normal(n), rng.standard_normal(n)
        u = r * x + np.sqrt(1 - r * r) * z
        Bx = J.knuth_bins(x, min(J.KNUTH_MAX, n))
        for i, a in enumerate(A):
            y = g(u, a)
            out["rho"][d, i] = np.corrcoef(x, y)[0, 1]
            out["lam_rank"][d, i] = lam_rank(x, y, B)
            out["lam_knuth"][d, i] = lam_knuth(x, y, Bx)
            out["lam_ksg"][d, i] = lam_ksg(x, y)
    return out


def figure(res, n, r, seed):
    import matplotlib.pyplot as plt
    rc = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
          "ytick.labelsize": 12, "legend.fontsize": 11}
    blue, orange, green, yellow = style.SERIES
    q = lambda k: np.percentile(res[k], [25, 50, 75], axis=0)          # noqa: E731
    logging.getLogger("fontTools").setLevel(logging.WARNING)       # infomeasure turns on INFO
    with plt.rc_context(rc):
        fig, axs = plt.subplots(1, 3, figsize=(13.33, 4.4), layout="constrained",
                                gridspec_kw=dict(width_ratios=(1, 1.15, 1.15)))
        ax = axs[0]                                               # one draw, three values of a
        rng = np.random.default_rng(seed)
        x, z = rng.standard_normal(n), rng.standard_normal(n)
        u = r * x + np.sqrt(1 - r * r) * z
        for a, c in ((0.0, blue), (1.0, orange), (2.0, green)):
            y = g(u, a)
            ax.plot(x, (y - y.mean()) / y.std(), "o", ms=5, color=c, alpha=0.85, mec="white",
                    mew=0.5, label=f"$a$ = {a:g}")
        ax.set_xlabel("$x$ (signal)")
        ax.set_ylabel("$y = g_a(u)$, standardised")
        ax.legend(frameon=True, fancybox=False, loc="upper left")
        style.tidy(ax)
        style.panel_label(ax, "a", "samples")

        ax = axs[1]
        aa = np.linspace(0, 2, 200)
        ax.plot(aa, rho_exact(aa, r), color=blue, lw=2, label="$\\rho$, exact")
        lo, md, hi = q("rho")
        ax.fill_between(A, lo, hi, color=blue, alpha=0.18, lw=0)
        ax.plot(A, md, "o", color=blue, ms=6, label=f"$\\rho$, sample of $N$ = {n} (median, IQR)")
        ax.axhline(r, color=green, lw=2, label="$\\lambda$, exact $= r$")
        lo, md, hi = q("lam_rank")
        ax.fill_between(A, lo, hi, color=green, alpha=0.18, lw=0)
        ax.plot(A, md, "D", color=green, ms=6, label="$\\lambda$, rank bins (median, IQR)")
        ax.set_xlabel("strength of the monotonic non-linearity $a$")
        ax.set_ylabel("$\\rho$ or $\\lambda$")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fancybox=False, loc="lower left")
        style.tidy(ax)
        style.panel_label(ax, "b", "$\\rho$ falls, $\\lambda$ should not")

        ax = axs[2]
        ax.axhline(r, color=style.INK, lw=2, label="exact $= r$")
        for k, c, m, lab in (("lam_rank", green, "D", "rank (equal-count) bins"),
                             ("lam_knuth", orange, "s", "Knuth bins (production)"),
                             ("lam_ksg", yellow, "^", "KSG, $k$ = 4")):
            lo, md, hi = q(k)
            ax.fill_between(A, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(A, md, m + "-", color=c, ms=6, lw=1.2, label=lab)
        zero = 100 * (res["lam_knuth"] <= 1e-9).mean(0)
        ax.text(0.98, 0.97, f"Knuth: $\\lambda$ = 0 in {zero.min():.0f}-{zero.max():.0f}% of draws\n"
                "(one bin chosen for a variable)", transform=ax.transAxes, ha="right", va="top",
                fontsize=10.5, color=style.MUTED)
        ax.set_xlabel("strength of the monotonic non-linearity $a$")
        ax.set_ylabel("$\\lambda$")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fancybox=False, loc="lower left")
        style.tidy(ax)
        style.panel_label(ax, "c", "$\\lambda$ estimators")
        style.save(fig, NAME, dpi=200, folder="slides")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--n", type=int, default=51)
    ap.add_argument("--r", type=float, default=0.8)
    a = ap.parse_args()
    path = style.data_path(NAME + ".npz")
    settings = dict(seed=a.seed, draws=a.draws, n=a.n, r=a.r)
    old = dict(np.load(path)) if os.path.exists(path) and not a.recompute else None
    if old is not None and all(float(old[k]) == float(v) for k, v in settings.items()):
        res = {k: old[k] for k in ("rho", "lam_rank", "lam_knuth", "lam_ksg")}
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
