"""Main-text figure flattened-histograms: pooled distributions of s, o and f against Gaussians.

    python scripts/flattened_hist.py [--recompute]
        -> figures/main-text/{png,pdf}/flattened-histograms, histograms in figures/data/*.npz

The decadal rows of rpc_rho_lambda_oloo_decadal (data.PRODUCTION["decadal"]: SLP s1961, PRECT
s1978, TREFHT s1961), as the thesis figure (notebooks/calc_RPC.ipynb cells 7-8): every value
of the ensemble mean s, the observations o and the members f, pooled over grid cells and months
(unweighted), as densities on one set of edges. Each is drawn against the normal density with its
own mean and variance (dashed), on a log axis so the tails show; the legend gives the excess
kurtosis (kappa_f, kappa_o, kappa_s). Obs: HadSLP2r, GPCP, NCEP R1; N = 92, 112, 112, T = 120.
Note the cases keep their seasonal cycle (thesis_detrend removes a line per member and
cell only), so the pooled spread mixes seasons and places as well as anomalies.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

import style                                                         # noqa: E402
from data import PRODUCTION, load                                    # noqa: E402

NAME = "flattened-histograms"
NBINS = 160
OBS = {"SLP": "HadSLP2r", "PRECT": "GPCP", "TREFHT": "NCEP R1"}
XLABEL = {"SLP": "Sea-level pressure (Pa)", "PRECT": "Precipitation (mm day$^{-1}$)",
          "TREFHT": "2 m temperature (K)"}
SERIES = (("f", "Ensemble members $f$"), ("o", "Observations $o$"), ("s", "Ensemble mean $s$"))
COLOR = dict(zip("fos", style.SERIES))


def moments(x):
    """mean, sd and excess kurtosis of all finite values, in float64, chunked along axis 0."""
    n = s1 = 0.0
    for c in x:
        c = c[np.isfinite(c)].astype(float)
        n, s1 = n + c.size, s1 + c.sum()
    mu = s1 / n
    m2 = m4 = 0.0
    for c in x:
        d = c[np.isfinite(c)].astype(float) - mu
        d2 = d * d
        m2, m4 = m2 + d2.sum(), m4 + (d2 * d2).sum()
    m2, m4 = m2 / n, m4 / n
    return mu, np.sqrt(m2), m4 / m2 ** 2 - 3


def compute():
    res = {}
    for var, case in PRODUCTION["decadal"]:
        F, o, s, _, _ = load(var, case)
        arrs = {"f": F, "o": o, "s": s}
        half = max(np.abs(np.nanpercentile(a[::4].ravel() if a.ndim == 4 else a.ravel(), (0.001, 99.999))).max()
                   for a in arrs.values())                           # all three inside the edges
        edges = np.linspace(-half, half, NBINS + 1)
        res[f"{var}_edges"] = edges
        res[f"{var}_case"] = case
        for k, a in arrs.items():
            res[f"{var}_{k}_dens"] = sum(np.histogram(c[np.isfinite(c)], bins=edges)[0] for c in a)
            res[f"{var}_{k}_dens"] = res[f"{var}_{k}_dens"] / (np.sum(np.isfinite(a)) * np.diff(edges))
            res[f"{var}_{k}_moments"] = np.array(moments(a))
        res[f"{var}_N"], res[f"{var}_T"] = F.shape[0], F.shape[1]
        print(var, case, {k: np.round(res[f"{var}_{k}_moments"], 3).tolist() for k in arrs}, flush=True)
        del F, o, s, arrs
    return res


def plot(d):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.85), layout="constrained")
    for ax, letter, (var, case) in zip(axs, "abc", PRODUCTION["decadal"]):
        e = d[f"{var}_edges"]
        x = np.linspace(e[0], e[-1], 400)
        peak = max(d[f"{var}_{k}_dens"].max() for k, _ in SERIES)
        kappa = []
        for k, label in SERIES:
            mu, sd, kurt = d[f"{var}_{k}_moments"]
            dens = d[f"{var}_{k}_dens"]
            line, = ax.stairs(np.where(dens > 0, dens, np.nan), e, color=COLOR[k], lw=1.3,
                              label=label),
            gauss = np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            ax.plot(x, gauss, color=COLOR[k], lw=0.9, ls=(0, (3, 2)), alpha=0.9)
            kappa.append((line, f"$\\kappa_{k}$ = {kurt:.1f}"))
        ax.set_yscale("log")
        ax.set_ylim(1e-5 * peak, 40 * peak)                            # headroom for the key
        ax.set_xlim(e[0], e[-1])
        ax.set_xlabel(XLABEL[var])
        style.tidy(ax)
        ax.grid(axis="x", visible=False)
        style.panel_label(ax, letter, f"{var}, DCPP-A {case}")
        ax.legend(*zip(*kappa), loc="upper left", fontsize=7, handlelength=1.4, borderpad=0.4,
                  labelspacing=0.25, handletextpad=0.5)
    axs[0].set_ylabel("Density")
    h = [plt.Line2D([], [], color=COLOR[k], lw=1.3) for k, _ in SERIES]
    h.append(plt.Line2D([], [], color=style.MUTED, lw=0.9, ls=(0, (3, 2))))
    fig.legend(h, [lab for _, lab in SERIES] + ["Gaussian with the same mean and sd"],
               loc="outside lower center", ncol=4, handlelength=2.0, columnspacing=1.6)
    style.save(fig, NAME)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true", help="redo the histograms even if the npz exists")
    args = ap.parse_args()
    npz = style.data_path(NAME + ".npz")
    if args.recompute or not os.path.exists(npz):
        np.savez(npz, **compute())
    plot(dict(np.load(npz)))


if __name__ == "__main__":
    main()
