"""The helper functions from `notebooks/hindcast_data.ipynb`, as a module.

    import sys; sys.path.insert(0, "../scripts")
    import smyle_handles as S, smyle_metrics as M

    c = S.get("Nov", lead="2-4")            # Nov init, lead 2-4 -> DJF
    rho_o = M.pearson_coeff(c.s, c.G)       # (lat, lon)
    rho_m = M.ensemble_SNR(c.F)
    RPC_rho = rho_o / rho_m

    I_o = M.calc_MI_sG(c.s, c.G, k=4)       # nats, ~15 s
    I_m = M.calc_MI_sF(c.F, k=4)            # nats, ~20 x that
    RPC_lam = M.lam_of(I_o) / M.lam_of(I_m)

Shapes are the notebook's and the handles' alike: F is (N, J, lat, lon), s and G are
(J, lat, lon), every return is (lat, lon).

NATS, not bits
--------------
`infomeasure.mutual_information` returns nats unless told otherwise, and these functions
do not pass `base`. `lam_of` uses `exp(-2I)`, which is the matching nats formula, so the
set is self-consistent -- but it is NOT interchangeable with `infomeasure_rpc.mi_bits`
and `infomeasure_rpc.lam`, which work in bits with `2^(-2I)`. Mixing a bits MI into
`lam_of` understates lambda; mixing a nats MI into the bits `lam` overstates it. Pick one
convention per figure. (`approach="metric"` is an alias for `approach="ksg"`, verified
equal in 0.6.2.)

Also unlike `infomeasure_rpc.mi_bits`, these do no copula/rank transform and leave
`noise_level` and `minkowski_p` at their defaults, so they will not reproduce the decadal
numbers in the git history to the digit even after a nats/bits conversion.

The one deviation from the notebook
-----------------------------------
`pearson_coeff` here averages its numerator over the SAMPLE axis only (`axis=0`). The
notebook's version calls a bare `np.mean(...)`, which also averages over lat/lon and
leaves a single scalar global covariance divided by a per-cell std -- a field that
correlates ~0 with the real correlation and can never go negative. That one added
`axis=0` is the whole change; `__main__` below checks the result against `np.corrcoef`.
"""
import numpy as np
from joblib import Parallel, delayed

import infomeasure


def pearson_coeff(s, G):
    """Pearson correlation between ensemble mean and obs, per grid cell -> (lat, lon)."""
    num = np.mean((s - np.mean(s, axis=0)) * (G - np.mean(G, axis=0)), axis=0)
    den = np.std(s, axis=0) * np.std(G, axis=0)
    return num / den


def ensemble_SNR(F):
    """sd(ensemble mean) / <sd(member)>_n -- the predictable signal fraction, rho_m.

    Note this is the mean of the members' standard deviations, as in the notebook, not
    the root of their mean variance; the two differ by a Jensen gap of a percent or so.
    """
    s = np.mean(F, axis=0)
    noise = np.mean(np.std(F, axis=1), axis=0)
    return np.std(s, axis=0) / noise


def ensemble_SNR_loo(F):
    """rho_m as < corr(s_-n, f_n) >_n, the perfect-model correlation. (lat, lon).

    Each member is verified against the mean of the other N-1, exactly as rho_o verifies
    the ensemble mean against observations -- so it needs no variance decomposition.

    It does NOT remove the finite-ensemble bias, it reverses its sign. With true
    predictable fraction p, E[corr(s_-n, f_n)] = p^2 / sqrt(p^2 + (1-p^2)/(N-1)) < p,
    whereas ensemble_SNR estimates sqrt(p^2 + (1-p^2)/N) > p. Low and high brackets of
    the same quantity; see ensemble_SNR_debiased for a direct estimate of p.
    """
    N = F.shape[0]
    tot = F.sum(axis=0)
    acc = 0.0
    for n in range(N):
        loo = (tot - F[n]) / (N - 1)
        a, b = loo - loo.mean(0), F[n] - F[n].mean(0)
        acc = acc + (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return acc / N


def ensemble_SNR_debiased(F):
    """rho_m with the finite-ensemble inflation removed: sig^2 = (N var(s) - tot^2)/(N-1).

    var(ensemble mean) = sig^2 + noise^2/N, so ensemble_SNR's numerator is inflated and
    RPC comes out too small. This subtracts that term and estimates p directly.
    """
    N = F.shape[0]
    tot2 = F.var(axis=1).mean(axis=0)
    sig2 = (N * F.mean(axis=0).var(axis=0) - tot2) / (N - 1)
    return np.sqrt(np.maximum(sig2, 0.0) / tot2)


import numpy as np
from scipy.stats import rankdata, norm

def copula_transform(x, scores="uniform", tie_tol=0.05):
    """Marginal rank transform to pseudo-observations.

    x      : (n,) or (n, d) array, rows are samples, columns are variables.
    scores : "uniform" -> U = R / (n + 1),  in (0, 1)
             "normal"  -> Z = Phi^{-1}(R / (n + 1))
    """
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"expected (n,) or (n, d), got shape {x.shape}")

    n, d = x.shape
    if n < 2:
        raise ValueError(f"need n >= 2 samples, got n = {n}")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or inf; drop or impute before ranking")

    r = np.empty_like(x)
    for j in range(d):
        n_unique = np.unique(x[:, j]).size
        if n_unique < (1.0 - tie_tol) * n:
            raise ValueError(
                f"column {j}: only {n_unique}/{n} distinct values. Marginal is not "
                "continuous, so averaged ranks are not valid pseudo-observations. "
                "Use a mixed discrete-continuous estimator instead."
            )
        r[:, j] = rankdata(x[:, j], method="average")

    u = r / (n + 1.0)
    if scores == "uniform":
        out = u
    elif scores == "normal":
        out = norm.ppf(u)
    else:
        raise ValueError(f"scores must be 'uniform' or 'normal', got {scores!r}")

    return out[:, 0] if squeeze else out

                      
def calc_MI_sG(s, G, k=4, n_jobs=-1, use_copula=False):
    """I(s ; G) in NATS per grid cell -> (lat, lon).

    The notebook looped serially; this fans the same per-cell calls out over `n_jobs`,
    which changes nothing but the wall clock. Pass n_jobs=1 for the serial path.
    """
    T, m, n = s.shape
    s2 = np.ascontiguousarray(s.reshape(T, -1))
    G2 = np.ascontiguousarray(G.reshape(T, -1))

    def calc_point(p):
        if use_copula: 
            s_tmp = copula_transform(s2[:, p])
            g_tmp = copula_transform(G2[:, p])
        else:
            s_tmp = s2[:, p]
            g_tmp = G2[:, p]
        return infomeasure.mutual_information(s_tmp, g_tmp,
                                              approach="metric", k=k)

    out = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(calc_point)(p) for p in range(m * n))
    return np.asarray(out, dtype=float).reshape(m, n)


def calc_MI_sF(F, k=4, use_copula=False, n_jobs=-1):
    """< I(s_-n ; f_n) >_n in NATS per grid cell -> (lat, lon).

    F: shape (N, T, m, n). `s_-n` is the exact mean of the other N-1 members, so no
    member's own information leaks into its target.
    """
    N, T, m, n = F.shape
    F2 = np.ascontiguousarray(F.reshape(N, T, -1))
    Fsum = F2.sum(axis=0)                     # (T, space)

    def calc_point(p):
        fp = F2[:, :, p]                      # (N, T)
        total = Fsum[:, p]                    # (T,)
        vals = np.empty(N)
        for member in range(N):
            x = fp[member]
            loo_s = (total - x) / (N - 1)     # exact mean of all OTHER N-1 members
            if use_copula: 
                x = copula_transform(x)
                loo_s = copula_transform(loo_s)
            vals[member] = infomeasure.mutual_information(loo_s, x,
                                                          approach="metric", k=k)
        return vals.mean()

    out = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(calc_point)(p) for p in range(m * n))
    return np.asarray(out, dtype=float).reshape(m, n)

def calc_MI_sG_LOOavg(F, G, k=4, use_copula=False, n_jobs=-1):
    """< I(s_-n ; f_n) >_n in NATS per grid cell -> (lat, lon).

    F: shape (N, T, m, n). `s_-n` is the exact mean of the other N-1 members, so no
    member's own information leaks into its target.
    """
    N, T, m, n = F.shape
    F2 = np.ascontiguousarray(F.reshape(N, T, -1))
    G2 = np.ascontiguousarray(G.reshape(T, -1))
    Fsum = F2.sum(axis=0)                     # (T, space)

    def calc_point(p):
        gp = G2[:, p] # (T,)
        fp = F2[:, :, p]                      # (N, T)
        total = Fsum[:, p]                    # (T,)
        vals = np.empty(N)
        for member in range(N):
            x = fp[member]
            loo_s = (total - x) / (N - 1)     # exact mean of all OTHER N-1 members
            if use_copula: 
                gp = copula_transform(gp)
                loo_s = copula_transform(loo_s)
            vals[member] = infomeasure.mutual_information(loo_s, gp,
                                                          approach="metric", k=k)
        return vals.mean()

    out = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(calc_point)(p) for p in range(m * n))
    return np.asarray(out, dtype=float).reshape(m, n)
    
def lam_of(I):
    """sqrt(1 - exp(-2I)), the information analogue of a correlation. I in NATS.

    The 1e-6 is the notebook's, guarding the sqrt against tiny negative radicands. It
    also means lam_of(0) is 1e-3 rather than 0, so a cell with no information reads as
    a small positive lambda rather than exactly zero.
    """
    return np.sqrt(1 - np.exp(-2 * np.maximum(I, 0)) + 1e-6)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, m, n = 60, 6, 8

    # pearson_coeff must equal np.corrcoef cell by cell, including where it is negative
    sign = np.where(np.arange(m)[:, None] < m // 2, 1.0, -1.0) * np.ones((m, n))
    G = rng.standard_normal((T, m, n))
    s = sign * G + 0.6 * rng.standard_normal((T, m, n))
    got = pearson_coeff(s, G)
    ref = np.array([[np.corrcoef(s[:, i, j], G[:, i, j])[0, 1]
                     for j in range(n)] for i in range(m)])
    print("pearson_coeff vs np.corrcoef : max|diff| = %.2e" % np.abs(got - ref).max())
    print("                             : range %+.3f .. %+.3f, %.0f%% negative"
          % (got.min(), got.max(), 100 * (got < 0).mean()))

    # rho_m is bounded by 1 and rises with ensemble agreement
    F = (rng.standard_normal((1, T, m, n)) * 0.8
         + rng.standard_normal((12, T, m, n)) * 0.6)
    print("ensemble_SNR                 : range %.3f .. %.3f"
          % (ensemble_SNR(F).min(), ensemble_SNR(F).max()))

    # lam_of against its analytic Gaussian value, I = -0.5 ln(1 - r^2) nats
    for r in (0.3, 0.6, 0.9):
        I = -0.5 * np.log(1 - r ** 2)
        print("lam_of(I(r=%.1f)) = %.4f   vs |r| = %.1f" % (r, lam_of(I), r))
