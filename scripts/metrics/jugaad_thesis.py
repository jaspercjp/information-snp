"""The thesis jugaad lambda, reproduced exactly and generalised to other binning rules.

    import jugaad_thesis as JT

    r = JT.jugaad_maps(F, o)                         # thesis: FD bins, alt 0.5/1/1.5, full mean s
    r = JT.jugaad_maps(F, o, rule="knuth", maxM=400) # Knuth (2006) OptBins, corrected
    r = JT.jugaad_maps(F, o, rule="fixed", B=8)      # the same B on every axis
    r = JT.jugaad_maps(F, o, s="loo")                # leave-one-out mean in lam_m
    r["lam_o"], r["lam_m"], r["lam"]                 # each (n_alt, ny, nx)

Per cell, with N members and T times (the layout of `InfoTheoryMetrics.info_RPC`):

    f_jugaad = f.flatten()          (N*T,)  rows ordered (member, time); NaN -> 0
    o_jugaad = tile(o, N)           (N*T,)  row (n, t) holds o[t]
    s_jugaad = tile(s, N)           (N*T,)  s = full ensemble mean (NaN-skipping)
    I_o = I(o_jugaad ; f_jugaad)    bits, np.histogram2d plug-in
    I_m = I(s_jugaad ; f_jugaad)
    lam_o = sqrt(1 - 2^(-2 I_o)),  lam_m = sqrt(1 - 2^(-2 I_m)),  lam = lam_o / lam_m

Bin counts per axis come from the rule applied to that axis's own pooled/tiled series,
then are scaled by `alt` and rounded up: `(ceil(B_o * alt), ceil(B_f * alt))`.

`rule="fd"` with the defaults matches `info_RPC` + `calc_RPC.ipynb` bit for bit
(`scripts/tests/test_jugaad_thesis.py`), including its float32 probabilities and its
`int()`-truncated, uncapped Freedman-Diaconis rule.

Two things to keep in view when reading the output
--------------------------------------------------
The tiled series are T distinct values repeated N times, so M = N*T is not an honest
sample count; both FD (h ~ M^(-1/3)) and Knuth (its likelihood counts every copy) choose
more bins than T values support, and the plug-in bias (Bx-1)(By-1)/(2 M_eff ln 2) bits
pushes lam_o up with bin count. And s contains f_n, so I(f; s) has a self-inclusion
floor (rho analogue >= sqrt(1/N)); `s="loo"` swaps in s_-n to show its size.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

from multiprocessing import get_context

import numpy as np
from scipy.special import gammaln
from scipy.stats import iqr

RULES = ("fd", "knuth", "fixed")
THESIS_ALTS = (0.5, 1.0, 1.5)


# --------------------------------------------------------------------------------------
# bin-count rules, each on one 1-D series
# --------------------------------------------------------------------------------------

def fd_bins(x):
    """Freedman-Diaconis exactly as `InfoTheoryMetrics.freedman_diaconis_bins`: truncated, no cap."""
    return int(np.ptp(x) / (2 * iqr(x) / np.power(len(x), 1 / 3)))


def knuth_logpost(x, maxM):
    """Knuth (2006) log posterior of M equal-width bins, for M = 1..maxM. Index i <-> M = i+1.

    Relative to M = 1, where it is exactly 0, so any M with logp > 0 beats one bin.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    M = np.arange(1, maxM + 1)
    c = hist_counts_all(x, maxM)                                    # (maxM, maxM), 0-padded
    occ = gammaln(c + 0.5).sum(axis=1) - (maxM - M) * gammaln(0.5)  # drop the padding
    return (n * np.log(M) + gammaln(M / 2) - gammaln(n + M / 2)
            - M * gammaln(0.5) + occ)


def hist_counts_all(x, maxM):
    """`np.histogram(x, bins=M)[0]` for every M = 1..maxM at once, row M-1 zero-padded.

    Replicates numpy's uniform-bin assignment exactly (index from the scaled offset, then
    the same one-step corrections against `linspace` edges), so Knuth sees the counts the
    loop over np.histogram would.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5                                 # numpy's degenerate range
    M = np.arange(1, maxM + 1)[:, None]
    X = np.broadcast_to(x, (maxM, len(x)))
    idx = (((X - lo) / (hi - lo)) * M).astype(np.intp)             # numpy's operation order
    idx[idx == M] -= 1
    step = (hi - lo) / M                                            # linspace: k*step + lo,
    edge = lambda k: np.where(k == M, hi, k * step + lo)           # noqa: E731  last = hi
    idx[X < edge(idx)] -= 1
    idx[(X >= edge(idx + 1)) & (idx != M - 1)] += 1
    flat = (np.arange(maxM)[:, None] * maxM + idx).ravel()
    return np.bincount(flat, minlength=maxM * maxM).reshape(maxM, maxM)


def knuth_bins(x, maxM):
    """Knuth's optimal M in 1..maxM. `InfoTheoryMetrics.optbins` returns this + 1 (see tests)."""
    return int(np.argmax(knuth_logpost(x, maxM))) + 1


# --------------------------------------------------------------------------------------
# plug-in MI, the thesis arithmetic
# --------------------------------------------------------------------------------------

def mi_bits(x, y, bins):
    """`InfoTheoryMetrics.I(x, y, nbins_alt=bins)[0]`: histogram2d plug-in, float32 p, log2."""
    xy, _, _ = np.histogram2d(x, y, bins=bins)
    p = xy.astype(np.float32) / np.sum(xy)
    px = p.sum(axis=1)
    py = p.sum(axis=0)
    p, px, py = p[p > 0], px[px > 0], py[py > 0]
    H_x = -np.sum(px * np.log2(px))                  # same operation order as the thesis,
    H_y = -np.sum(py * np.log2(py))                  # so equality is bitwise
    H_xy = -np.sum(p * np.log2(p))
    return H_x + H_y - H_xy


def lam_bits(I):
    """Granger-Lin lambda from MI in bits, as `calc_RPC.ipynb` cell 13.

    I is clipped at 0 first: float32 round-off can leave an independent pair at -1e-8,
    which the thesis formula turns into NaN. Nothing changes where I >= 0.
    """
    return np.sqrt(1 - np.power(2.0, -2 * np.maximum(np.asarray(I), 0.0)))


# --------------------------------------------------------------------------------------
# one cell, then the map
# --------------------------------------------------------------------------------------

def _rule_bins(x, rule, maxM, B):
    if rule == "fd":
        # IQR = 0 (more than half the rows identical, e.g. zero-filled NaN members in a
        # resample) makes the thesis rule int(inf) and crash; flag the cell instead.
        return fd_bins(x) if iqr(x) > 0 else None
    if rule == "knuth":
        return knuth_bins(x, maxM)
    return int(B)


def cell(f, o, s, rule="fd", alts=THESIS_ALTS, maxM=400, B=None, tiled=True, bins=None, p=None):
    """One grid cell. f `(N, T)`, o `(T,)`, s `(T,)` or `(N, T)` (the latter is s_-n, pooled).

    tiled : the rule sizes the o and s axes from their tiled N*T series (thesis) or, if
        False, from the T distinct values -- Knuth on a tiled series runs to `maxM`, because
        its likelihood rewards isolating each repeated value (optbins_characterise.py).
        Untiled Knuth is capped at T bins. A pooled s_-n has N*T distinct values and is
        always sized as is.
    p : `(N, T)` series paired with the tiled obs in I_o, pooled row (n, t) like f; default
        f itself (the thesis I(f; o)). p = s_-n gives I(s_-n; o), the ensemble-mean analogue.
        Its bin count B_p comes from the same rule on the pooled p (N*T distinct values).
    bins : precomputed (B_f, B_o, B_s[, B_p]), skipping the rule. Every rule here depends only on
        the multiset of values, so a permutation null can reuse the unpermuted bins exactly.
    Returns `(I_o, I_m, bins)`: I_o, I_m `(n_alt,)` in bits; bins = (B_f, B_o, B_s, B_p) before
    `alt` (B_p = B_f when p is f).
    """
    f = np.array(f, dtype=float)
    f[np.isnan(f)] = 0.0
    N, T = f.shape
    f_jug = f.flatten()
    o_jug = np.tile(o, N)
    s_jug = np.tile(s, N) if np.ndim(s) == 1 else np.asarray(s, dtype=float).flatten()
    if p is not None:
        p = np.array(p, dtype=float)
        p[np.isnan(p)] = 0.0
        p_jug = p.flatten()
    if bins is not None:
        Bf, Bo, Bs = (int(b) for b in bins[:3])
        Bp = int(bins[3]) if len(bins) > 3 else Bf
    elif tiled:
        Bf = _rule_bins(f_jug, rule, maxM, B)
        Bo = _rule_bins(o_jug, rule, maxM, B)
        Bs = _rule_bins(s_jug, rule, maxM, B)
    else:
        Bf = _rule_bins(f_jug, rule, maxM, B)
        Bo = _rule_bins(np.asarray(o, float), rule, min(maxM, T), B)
        Bs = (_rule_bins(np.asarray(s, float), rule, min(maxM, T), B) if np.ndim(s) == 1
              else _rule_bins(s_jug, rule, maxM, B))
    if bins is None:
        Bp = Bf if p is None else _rule_bins(p_jug, rule, maxM, B)
    I_o = np.empty(len(alts))
    I_m = np.empty(len(alts))
    if None in (Bf, Bo, Bs, Bp):                                 # FD undefined here
        return np.full(len(alts), np.nan), np.full(len(alts), np.nan), (0, 0, 0, 0)
    for k, a in enumerate(alts):
        bf = int(np.ceil(Bf * a))
        if p is None:
            I_o[k] = mi_bits(o_jug, f_jug, (int(np.ceil(Bo * a)), bf))
        else:
            I_o[k] = mi_bits(o_jug, p_jug, (int(np.ceil(Bo * a)), int(np.ceil(Bp * a))))
        I_m[k] = mi_bits(s_jug, f_jug, (int(np.ceil(Bs * a)), bf))
    return I_o, I_m, (Bf, Bo, Bs, Bp)


def loo_means(F):
    """Leave-one-out ensemble means, NaN-skipping like `s = f.mean("n")`. `(N, T, ...)` in and out."""
    ok = ~np.isnan(F)
    out = np.nan_to_num(np.array(F, dtype=float), nan=0.0)    # the one full-size array
    tot = out.sum(axis=0, keepdims=True)
    cnt = ok.sum(axis=0, keepdims=True)
    out -= tot                                                  # in place: -(tot - x), then
    np.negative(out, out=out)                                   # negate (exact), as tot - x
    with np.errstate(invalid="ignore", divide="ignore"):
        out /= cnt - ok
    return out


_G = {}


def _row(j):
    F, o, S, kw, pre, P = _G["F"], _G["o"], _G["S"], _G["kw"], _G["bins"], _G["P"]
    nx = F.shape[-1]
    na = len(kw["alts"])
    Io, Im, bins = np.full((na, nx), np.nan), np.full((na, nx), np.nan), np.zeros((4, nx), int)
    for i in range(nx):
        s = S[..., j, i]
        if np.all(np.isnan(o[:, j, i])):
            continue
        b = None if pre is None else pre[:, j, i]
        pp = None if P is None else P[:, :, j, i]
        Io[:, i], Im[:, i], bins[:, i] = cell(F[:, :, j, i], o[:, j, i], s, bins=b, p=pp, **kw)
    return Io, Im, bins


def jugaad_maps(F, o, s="full", rule="fd", alts=THESIS_ALTS, maxM=400, B=None, tiled=True,
                bins=None, o_vs="f", n_jobs=None):
    """Per-cell jugaad lambda over a map. F `(N, T, ny, nx)`, o `(T, ny, nx)`.

    s : "full" (thesis; NaN-skipping member mean), "loo" (s_-n), or an explicit `(T, ny, nx)`.
    tiled, bins : see `cell`; tiled=True is the thesis. `bins` is `(3 or 4, ny, nx)`.
    o_vs : what the tiled obs is paired with in I_o -- "f" (thesis: each member, I(f; o))
        or "loo" (the leave-one-out mean of the other members, I(s_-n; o)). I_m is the same
        either way.
    Cells whose obs are all NaN are skipped (NaN out). Returns a dict of `(n_alt, ny, nx)`
    arrays I_o, I_m (bits), lam_o, lam_m, lam, plus `bins` `(4, ny, nx)` = (B_f, B_o, B_s, B_p)
    before `alt`, and for rule="knuth" the boolean `cap_hit` `(3, ny, nx)`.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    F = np.asarray(F, dtype=float)
    o = np.asarray(o, dtype=float)
    if isinstance(s, str):
        S = np.nanmean(F, axis=0) if s == "full" else loo_means(F)
    else:
        S = np.asarray(s, dtype=float)
    if o_vs not in ("f", "loo"):
        raise ValueError(f"o_vs must be 'f' or 'loo', got {o_vs!r}")
    P = loo_means(F) if o_vs == "loo" else None
    ny = F.shape[2]
    _G.update(F=F, o=o, S=S, bins=bins, P=P, kw=dict(rule=rule, alts=tuple(alts), maxM=maxM, B=B, tiled=tiled))
    n_jobs = n_jobs or int(_os.environ.get("SLURM_CPUS_ON_NODE", 1))
    if n_jobs > 1:
        with get_context("fork").Pool(n_jobs) as pool:
            rows = pool.map(_row, range(ny))
    else:
        rows = [_row(j) for j in range(ny)]
    _G.clear()
    I_o = np.stack([r[0] for r in rows], axis=1)
    I_m = np.stack([r[1] for r in rows], axis=1)
    bins = np.stack([r[2] for r in rows], axis=1)
    lam_o, lam_m = lam_bits(I_o), lam_bits(I_m)
    with np.errstate(divide="ignore", invalid="ignore"):             # lam_m = 0 when a rule
        lam = np.where(lam_m > 0, lam_o / lam_m, np.nan)              # picks 1 bin for s
    out = dict(I_o=I_o, I_m=I_m, lam_o=lam_o, lam_m=lam_m, lam=lam, bins=bins,
               rule=rule, alts=np.array(alts), N=F.shape[0], T=F.shape[1])
    if rule == "knuth":
        cap_os = maxM if tiled else min(maxM, F.shape[1])
        cap_s = cap_os if S.ndim == 3 else maxM                  # pooled s_-n: N*T values
        out["cap_hit"] = bins >= np.array([maxM, cap_os, cap_s, maxM])[:, None, None]
    out["tiled"] = tiled
    out["o_vs"] = o_vs
    return out
