"""Main-text figure gaussian-array-lambda-rho: lambda against rho on synthetic ensembles; and
supp-info gaussian-array-rm-sensitivity, its panel (b) at other member signal shares R_M, and
gaussian-array-heterogeneous-rm, panel (b) with a different signal share for every member.

    python scripts/gaussian_demo.py [--recompute] [--seed S]
        -> figures/{main-text,supp-info}/{png,pdf}/..., curves in figures/data/*.npz
           (each npz is reused unless --recompute)

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
    reference, solid black and grey.
Common random numbers: (a) reuses one z_0 and z_n at every r; (b) and (c) share one ensemble
(x, z_n) and one eta at every a, so lambda_m and rho_m there are one value each and the curves
move only with r or a, not with a fresh draw per point.
gaussian-array-rm-sensitivity: panel (b) (g = x) for R_M in RMS, with one x, z_n and eta for
every R_M and a (common random numbers), so the panels differ only through R_M. Expected
rho(s; o) = sqrt(R_M a) / sqrt(R_M + (1 - R_M)/N), rho(f; o) = sqrt(R_M a), rho_m ~ sqrt(R_M).
gaussian-array-heterogeneous-rm: panels (b) (top row) and (c) (bottom row, o from exp(x)) with
f_n = sqrt(R_n) x + sqrt(1 - R_n) z_n, R_n per member,
mean 0.5 in each panel: (a) R_n = 0.5 for all, (b) R_n evenly spaced over [0.1, 0.9], (c) half the
members at 0.1, half at 0.9. rho_m = <corr(f_n, s_-n)> ~ <sqrt(R_n)> (0.707, 0.685, 0.632; Jensen), so
rho RPC = 1 near a = <sqrt(R_n)>^2. The pooled (s_-n, f_n) pairs are a mixture of Gaussians with
different correlations, not Gaussian, so lambda_m need not equal rho_m.
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

NAME, NAME_RM = "gaussian-array-lambda-rho", "gaussian-array-rm-sensitivity"
NAME_HET = "gaussian-array-heterogeneous-rm"
RMS = (0.2, 0.4, 0.6, 0.8)                                   # supp-info: members' signal shares
N, T = 500, 40000
HET = {"homogeneous": np.full(N, 0.5), "uniform": np.linspace(0.1, 0.9, N),
       "two groups": np.where(np.arange(N) % 2 == 0, 0.1, 0.9)}          # per-member R_n, mean 0.5
HET_TITLES = (r"$\mathbf{R_n = 0.5}$", r"$\mathbf{R_n}$ from 0.1 to 0.9",
              r"$\mathbf{R_n \in \{0.1, 0.9\}}$")
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


_D = {}                     # one draw per panel, set before the pool forks (workers share it)


def point_m(args):
    """Panel (a) at one r, from the panel's one z_0 and z_n."""
    r, seed = args
    F = np.sqrt(r) * _D["z0"] + np.sqrt(1 - r) * _D["Z"]
    return ensemble(F, np.random.default_rng(seed))


def point_o(args):
    """Panels (b, c) at one (g, a): the one fixed ensemble's skill against o."""
    g, a, seed = args
    o = np.sqrt(a) * _D["gx"][g] + np.sqrt(1 - a) * _D["eta"]
    return skill(_D["F"], _D["s"], o, np.random.default_rng(seed))


def compute(seed):
    """Common random numbers: panel (a) reuses one z_0, z_n at every r; panels (b, c) share one
    ensemble (x, z_n) and one eta at every a and for both g. Only the nulls' permutations vary."""
    jobs = int(os.environ.get("SNP_DEMO_JOBS", min(7, int(os.environ.get("SLURM_CPUS_ON_NODE", 1)))))
    rng = np.random.default_rng([seed, 0])
    _D.update(z0=rng.standard_normal(T), Z=rng.standard_normal((N, T)))
    with get_context("fork").Pool(jobs) as pool:
        m = np.array(pool.map(point_m, [(r, [seed, 0, k]) for k, r in enumerate(R)])).T
    res = dict(r=R, a=A, seed=seed, **dict(zip(KEYS_M, m)))
    rng = np.random.default_rng([seed, 1])
    x = rng.standard_normal(T)
    F = np.sqrt(R_M) * x + np.sqrt(1 - R_M) * rng.standard_normal((N, T))
    gx = {g: (G[g](x) - G[g](x).mean()) / G[g](x).std() for g in G}
    _D.clear()
    _D.update(F=F, s=F.mean(0), gx=gx, eta=rng.standard_normal(T))
    ens = ensemble(F, rng)                                         # one value: F is fixed
    with get_context("fork").Pool(jobs) as pool:
        for j, g in enumerate(G):
            out = np.array(pool.map(point_o, [(g, a, [seed, j + 1, k]) for k, a in enumerate(A)])).T
            res.update({f"{g}_{k}": v for k, v in zip(KEYS[len(KEYS_M):], out)})
            res.update({f"{g}_{k}": np.full(A.size, v) for k, v in zip(KEYS_M, ens)})
    _D.clear()
    return res


def skill_runs(seed, tag, shares, g="linear", prefix="rm"):
    """Panel (b) (g = x) or (c) (g = exp) for each per-member signal-share array in shares, one x,
    z_n, eta for all of them -> keys <prefix><i>_*."""
    jobs = int(os.environ.get("SNP_DEMO_JOBS", min(7, int(os.environ.get("SLURM_CPUS_ON_NODE", 1)))))
    rng = np.random.default_rng([seed, tag])
    x, Z, eta = rng.standard_normal(T), rng.standard_normal((N, T)), rng.standard_normal(T)
    res = dict(a=A, seed=seed)
    for i, R in enumerate(shares):
        R = np.broadcast_to(np.asarray(R, float), (N,))[:, None]
        F = np.sqrt(R) * x + np.sqrt(1 - R) * Z
        _D.clear()
        gx = G[g](x)
        _D.update(F=F, s=F.mean(0), gx={g: (gx - gx.mean()) / gx.std()}, eta=eta)
        ens = ensemble(F, np.random.default_rng([seed, tag, i]))
        with get_context("fork").Pool(jobs) as pool:
            out = np.array(pool.map(point_o, [(g, a, [seed, tag + 1, i, k]) for k, a in enumerate(A)])).T
        res.update({f"{prefix}{i}_{k}": v for k, v in zip(KEYS[len(KEYS_M):], out)})
        res.update({f"{prefix}{i}_{k}": np.full(A.size, v) for k, v in zip(KEYS_M, ens)})
        print(f"shares {i} (mean {R.mean():.2f}, <sqrt R> {np.sqrt(R).mean():.3f}): lambda_m "
              f"{lam_bits(ens[0] - ens[1]):.3f}, rho_m {ens[3]:.3f}", flush=True)
    _D.clear()
    return res


def compute_rm(seed):
    """Panel (b) at each R_M in RMS."""
    return dict(skill_runs(seed, 2, RMS), rm=np.array(RMS))


def compute_het(seed, have=None):
    """Panels (b) (g = x, keys rm<i>) and (c) (g = exp, keys ex<i>) for each per-member R_n design
    in HET; reuses the g = x half of `have` (an older npz) if it is there."""
    res = dict(have) if have is not None and "rm0_I_s" in have else dict(
        skill_runs(seed, 4, list(HET.values())), shares=np.stack(list(HET.values())))
    if "ex0_I_s" not in res:
        res.update({k: v for k, v in skill_runs(seed, 7, list(HET.values()), g="exp", prefix="ex").items()
                    if k.startswith("ex")})
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
        skill_panel(ax, d, g, letter, TITLES[g])
    style.save(fig, NAME)


def skill_panel(ax, d, k, letter, title, xlabel=True, legends=True):
    """lambda_o, rho_o from s and f, and the ensemble's lambda_m, rho_m; d keys prefixed k_."""
    a = d["a"]
    ax.plot(a, lam(d, f"{k}_I_s"), color=LAM_O, lw=1.6, label=r"$\lambda_o$, $I(s;o)$")
    ax.plot(a, lam(d, f"{k}_I_f"), color=LAM_O, lw=1.6, ls=DASH, label=r"$\lambda_o$, $I(f;o)$")
    ax.plot(a, np.abs(d[f"{k}_rho_s"]), color=RHO_O, lw=1.6, label=r"$\rho_o$, corr$(s,o)$")
    ax.plot(a, np.abs(d[f"{k}_rho_f"]), color=RHO_O, lw=1.6, ls=DASH, label=r"$\rho_o$, corr$(f,o)$")
    ax.plot(a, lam(d, f"{k}_I_m"), color="black", lw=1.2, label=r"$\lambda_m$")
    ax.plot(a, d[f"{k}_rho_m_2"], color="0.55", lw=1.2, label=r"$\rho_m$")
    if xlabel:
        ax.set_xlabel("Signal fraction of var($o$), $a$")
    style.panel_label(ax, letter, title, inside=True)
    finish(ax)
    if not legends:
        return
    h, lab = ax.get_legend_handles_labels()                        # skill lower right,
    ax.add_artist(ax.legend(h[:4], lab[:4], loc="lower right", **LEGEND))
    ax.legend(h[4:], lab[4:], loc="upper left", bbox_to_anchor=(0.11, 1.0), **LEGEND)


def plot_rm(d):
    fig, axs = plt.subplots(2, 2, figsize=(5.0, 5.5), layout="constrained", sharex=True, sharey=True)
    for i, (ax, letter) in enumerate(zip(axs.flat, "abcd")):
        skill_panel(ax, d, f"rm{i}", letter, rf"$\mathbf{{R_n = {RMS[i]}}}$", xlabel=i >= 2,
                    legends=False)
    for ax in axs[:, 0]:
        ax.set_ylabel("Predictable component")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3,
               **dict(LEGEND, fontsize=8))                         # one key for the four panels
    style.save(fig, NAME_RM)


def plot_het(d):
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.6), layout="constrained", sharex=True, sharey=True)
    for r, (g, prefix, letters) in enumerate(zip(G, ("rm", "ex"), ("abc", "def"))):
        for i, (ax, letter) in enumerate(zip(axs[r], letters)):
            skill_panel(ax, d, f"{prefix}{i}", letter, HET_TITLES[i] if r == 0 else None,
                        xlabel=r == 1, legends=False)
        axs[r, 0].set_ylabel(TITLES[g] + "\nPredictable component")   # row label: the signal g
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=6,
               **dict(LEGEND, fontsize=7.5))
    style.save(fig, NAME_HET)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="redo the arrays even if the npz exists")
    args = ap.parse_args()
    for name, comp, draw in ((NAME, compute, plot), (NAME_RM, compute_rm, plot_rm),
                             (NAME_HET, compute_het, plot_het)):
        npz = style.data_path(name + ".npz")
        if args.recompute or not os.path.exists(npz):
            np.savez(npz, **comp(args.seed))
        elif name == NAME_HET and "ex0_I_s" not in np.load(npz):         # add the g = exp row
            np.savez(npz, **compute_het(args.seed, have=dict(np.load(npz))))
        draw(dict(np.load(npz)))


if __name__ == "__main__":
    main()
