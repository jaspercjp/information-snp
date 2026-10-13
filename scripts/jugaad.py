"""The thesis jugaad lambda and the Pearson rho RPC, per grid cell, parallel over rows.

Per cell, with N members and T times: f_jugaad = f.flatten() (NaN -> 0, rows (member, time)),
o_jugaad = tile(o, N), s_jugaad = tile(s, N) or s_-n flattened; I_o = I(o_jugaad; f_jugaad) (or
s_-n_jugaad with o_vs="loo"), I_m = I(s_jugaad; f_jugaad), np.histogram2d plug-in in bits;
lam = sqrt(1 - 2^(-2 I)), RPC = lam_o / lam_m. Bins: a rule per axis, scaled by alt, ceil(B * alt).
`maps(F, o)` with the defaults is bitwise `InfoTheoryMetrics.info_RPC` (tests/test_jugaad.py).
"""
import os
from multiprocessing import get_context

import numpy as np
from scipy.special import gammaln
from scipy.stats import iqr

KNUTH_MAX = 200                 # cap on pooled axes; T-value axes are capped at T
THESIS_ALTS = (0.5, 1.0, 1.5)


def fd_bins(x):
    """Freedman-Diaconis as the thesis: truncated, uncapped. None where IQR = 0 (undefined)."""
    q = iqr(x)
    return int(np.ptp(x) / (2 * q / np.power(len(x), 1 / 3))) if q > 0 else None


_CACHE = {}                                                         # edges, gammaln tables


def hist_counts_all(x, maxM):
    """`np.histogram(x, bins=M)[0]` for all M = 1..maxM (row M-1, zero-padded): numpy's bins are
    e_k <= x < e_{k+1} on edges k*((hi-lo)/M)+lo, so one searchsorted on sorted x gives every M."""
    xs = np.sort(np.asarray(x, dtype=float))
    lo, hi = float(xs[0]), float(xs[-1])
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5                                 # numpy's degenerate range
    if ("edges", maxM) not in _CACHE:
        M = np.repeat(np.arange(1, maxM + 1), np.arange(0, maxM))
        k = np.concatenate([np.arange(1, m) for m in range(1, maxM + 1)])
        _CACHE["edges", maxM] = (M, k.astype(float), M.astype(float))
    Mi, kf, Mf = _CACHE["edges", maxM]
    cum = np.full((maxM, maxM + 1), xs.size, dtype=np.int64)       # #(x < e_k); e_M -> n
    cum[:, 0] = 0
    cum[Mi - 1, kf.astype(np.intp)] = np.searchsorted(xs, kf * ((hi - lo) / Mf) + lo)
    return np.diff(cum, axis=1)


def knuth_logpost(x, maxM):
    """Knuth (2006) log posterior of M = 1..maxM equal-width bins, relative to M = 1."""
    n, M = len(x), np.arange(1, maxM + 1)
    if ("lgam", n) not in _CACHE:                                  # counts are integers:
        _CACHE["lgam", n] = gammaln(np.arange(n + 1) + 0.5)        # a table lookup
    occ = _CACHE["lgam", n][hist_counts_all(x, maxM)].sum(1) - (maxM - M) * gammaln(0.5)
    return n * np.log(M) + gammaln(M / 2) - gammaln(n + M / 2) - M * gammaln(0.5) + occ


def knuth_bins(x, maxM):
    """Knuth's optimal M. `InfoTheoryMetrics.optbins` returns this + 1 (off by one)."""
    return int(np.argmax(knuth_logpost(x, maxM))) + 1


def mi_bits(x, y, bins):
    """`InfoTheoryMetrics.I(x, y, nbins_alt=bins)[0]`: float32 p, log2, same operation order."""
    xy, _, _ = np.histogram2d(x, y, bins=bins)
    p = xy.astype(np.float32) / np.sum(xy)
    px, py = p.sum(axis=1), p.sum(axis=0)
    p, px, py = p[p > 0], px[px > 0], py[py > 0]
    H_x, H_y, H_xy = (-np.sum(q * np.log2(q)) for q in (px, py, p))
    return H_x + H_y - H_xy


def lam_bits(I):
    """Granger-Lin lambda from bits; I clipped at 0 (float32 round-off gives -1e-8)."""
    return np.sqrt(1 - np.power(2.0, -2 * np.maximum(np.asarray(I), 0.0)))


def cell(f, o, s, p=None, rule="fd", alts=THESIS_ALTS, tiled=True, loo_size="pooled",
         maxM=KNUTH_MAX, B=None, bins=None):
    """One cell: f, p (N, T); o (T,); s (T,) or (N, T) s_-n. -> (I_o, I_m, (B_f, B_o, B_s, B_p)).

    tiled: o and a 1-D s are sized on their tiled series (thesis) or their T values (cap T).
    loo_size: (N, T) axes are sized on all N*T rows ("pooled") or their T centres (cap T).
    bins: fixed (B_f, B_o, B_s, B_p) before alt, skipping the rule."""
    f = np.nan_to_num(np.asarray(f, dtype=float))
    N, T = f.shape
    f_j, o_j = f.ravel(), np.tile(o, N)
    s_j = np.tile(s, N) if np.ndim(s) == 1 else np.asarray(s, dtype=float).ravel()
    p_j = None if p is None else np.nan_to_num(np.asarray(p, dtype=float)).ravel()

    def nb(x, cap):
        return fd_bins(x) if rule == "fd" else knuth_bins(x, cap) if rule == "knuth" else int(B)

    def axis(a, a_j):
        if np.ndim(a) == 1:
            return nb(a_j, maxM) if tiled else nb(np.asarray(a, float), min(maxM, T))
        return nb(a_j, maxM) if loo_size == "pooled" else nb(np.nanmean(a, 0), min(maxM, T))

    if bins is None:
        Bf = nb(f_j, maxM)
        bins = (Bf, axis(o, o_j), axis(s, s_j), Bf if p is None else axis(p, p_j))
    Bf, Bo, Bs, Bp = (None if b is None else int(b) for b in bins)
    if None in (Bf, Bo, Bs, Bp):                                     # FD undefined here
        return np.full(len(alts), np.nan), np.full(len(alts), np.nan), (0, 0, 0, 0)
    up = lambda b, a: int(np.ceil(b * a))                           # noqa: E731
    x2, B2 = (f_j, Bf) if p is None else (p_j, Bp)
    I_o = np.array([mi_bits(o_j, x2, (up(Bo, a), up(B2, a))) for a in alts])
    I_m = np.array([mi_bits(s_j, f_j, (up(Bs, a), up(Bf, a))) for a in alts])
    return I_o, I_m, (Bf, Bo, Bs, Bp)


def loo_means(F):
    """Leave-one-out member means s_-n, NaN-skipping. (N, T, ...) in and out."""
    ok = ~np.isnan(F)
    out = np.nan_to_num(np.array(F, dtype=float), nan=0.0)
    tot, cnt = out.sum(axis=0, keepdims=True), ok.sum(axis=0, keepdims=True)
    out -= tot
    np.negative(out, out=out)                                       # tot - x, exactly
    with np.errstate(invalid="ignore", divide="ignore"):
        out /= cnt - ok
    return out


_G = {}


def _row(j):
    F, o, S, P, pre, kw = (_G[k] for k in ("F", "o", "S", "P", "bins", "kw"))
    na, nx = len(kw["alts"]), F.shape[-1]
    Io, Im, bins = np.full((na, nx), np.nan), np.full((na, nx), np.nan), np.zeros((4, nx), int)
    for i in range(nx):
        if np.all(np.isnan(o[:, j, i])):
            continue
        Io[:, i], Im[:, i], bins[:, i] = cell(
            F[:, :, j, i], o[:, j, i], S[..., j, i], None if P is None else P[:, :, j, i],
            bins=None if pre is None else pre[:, j, i], **kw)
    return Io, Im, bins


def maps(F, o, s="full", o_vs="f", bins=None, n_jobs=None, alts=THESIS_ALTS, **kw):
    """Per-cell jugaad lambda. F (N, T, ny, nx), o (T, ny, nx).

    s: "full" (NaN-skipping member mean), "loo" (s_-n) or an explicit (T, ny, nx) mean.
    o_vs: the obs paired with each member "f" (thesis) or with s_-n "loo". `kw` -> `cell`.
    Returns I_o, I_m (bits), lam_o, lam_m, lam (n_alt, ny, nx) and bins (4, ny, nx);
    lam is NaN where lam_m = 0 (a rule gave s one bin).
    """
    F, o = np.asarray(F, dtype=float), np.asarray(o, dtype=float)
    S = (np.nanmean(F, axis=0) if s == "full" else loo_means(F)) if isinstance(s, str) else s
    _G.update(F=F, o=o, S=np.asarray(S, dtype=float), P=loo_means(F) if o_vs == "loo" else None,
              bins=bins, kw=dict(kw, alts=tuple(alts)))
    n_jobs = n_jobs or int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    if n_jobs > 1:
        with get_context("fork").Pool(n_jobs) as pool:
            rows = pool.map(_row, range(F.shape[2]))
    else:
        rows = [_row(j) for j in range(F.shape[2])]
    _G.clear()
    I_o, I_m, bins = (np.stack([r[k] for r in rows], axis=1) for k in range(3))
    lam_o, lam_m = lam_bits(I_o), lam_bits(I_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(lam_m > 0, lam_o / lam_m, np.nan)
    return dict(I_o=I_o, I_m=I_m, lam_o=lam_o, lam_m=lam_m, lam=lam, bins=bins,
                alts=np.array(alts))


def corr_t(a, b):
    """Correlation along axis -3 over the times where both are finite."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(-3)
    am = np.where(ok, a, 0.0).sum(-3) / n
    bm = np.where(ok, b, 0.0).sum(-3) / n
    da = np.where(ok, a - am[..., None, :, :], 0)
    db = np.where(ok, b - bm[..., None, :, :], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (da * db).sum(-3) / np.sqrt((da ** 2).sum(-3) * (db ** 2).sum(-3))


def rho_measures(F, o, s):
    """rho_o = corr(s, o); rho_m = <corr(s_-n, f_n)>_n; and the ensemble-mean sd(s)/<sd(f_n)>."""
    L = loo_means(F)
    rho_m = np.nanmean(np.stack([corr_t(L[n], F[n]) for n in range(F.shape[0])]), 0)
    return corr_t(s, o), rho_m, np.nanstd(s, axis=0) / np.nanmean(np.nanstd(F, axis=1), axis=0)


def pooled_rho(F, Y):
    """|corr| per cell between pooled f (NaN -> 0) and Y of the same (N, T, ...) layout."""
    X = np.where(np.isnan(F), 0.0, F).reshape(-1, *F.shape[2:])
    Y = Y.reshape(-1, *F.shape[2:])
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs((Xc * Yc).mean(0) / (Xc.std(0) * Yc.std(0)))
