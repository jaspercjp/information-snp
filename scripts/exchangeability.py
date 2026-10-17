"""Member exchangeability from mutual information: pooled vs per-member, bias valid by construction.

Independent of jugaad.py: nothing here tiles the observations, re-bins a subset, or turns a
missing value into a sample. Arrays (any units; every MI and entropy in bits, log2):

    x  (R, T, *space)   members by time [by grid point], finite
    g  (T, *space)      observations, T samples only (never tiled)

Per grid point, with X[r] = mean of the other R - 1 members (leave-one-out) and Y = x:
    I_cond   = <MI(X[r]; Y[r])>_r            each from the T pairs of member r
    I_pool_T = <MI(T pairs)>_subsamples      n_sub subsamples of one random member per time step
    Delta    = I_pool_T - I_cond             either sign; 0 in expectation under exchangeability
    I_pool   = MI(all R T pairs)             not size-matched, so I_pool - I_cond is biased
    I_MXY    = I(M; X, Y) = H(X, Y) - H(X, Y | M),  M the member label (all members have T pairs)

Binning: equiprobable edges computed ONCE per grid point from pooled X (all R T values) and from
pooled Y, B = max(3, round(T^(1/3))) bins per axis (sensitivity: `bin_choices`). Every estimate,
observed or permuted, member or pooled, uses those edges: the data are reduced to integer codes
once and only counted afterwards.

Null: the (X, Y) pairs are built once; member labels are then permuted independently within each
time step (or each block of `block` consecutive steps, one permutation per block), carrying each
pair with its label, and the whole pipeline is rerun on every permutation. The same permutation
tables serve every grid point (spatial structure kept). For each statistic: observed, null mean and
SD, corrected = observed - null mean, and p: two-sided about the null mean for Delta (and
I_pool - I_cond), upper tail for I_MXY. (For I_cond and I_pool_T alone, observed - null mean is the
departure from exchangeability, not the plug-in MI bias: the null keeps the X-Y dependence.)

estimator="copula": Gaussian-copula MI, normal scores from ranks over the POOLED R T values per
grid point (computed once, like the edges), MI = -1/2 log2(1 - r^2), and I_MXY from Gaussian
entropies of the pooled and per-member covariances of (X, Y).

`obs_mi`: I_o = <MI(g; X[r])>_r, each from the T pairs (g_t, X[r, t]) with g's edges from its own
T values and X's pooled edges; null: g permuted in time (blocks of `block`), p upper tail.
"""
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

STATS = ("I_cond", "I_pool_T", "Delta", "I_pool", "I_pool_minus_cond", "I_MXY")
TWO_SIDED = ("Delta", "I_pool_minus_cond")
UPPER = ("I_MXY",)
LOG2_2PIE = np.log2(2 * np.pi * np.e)


def n_bins(T):
    """B = max(3, round(T^(1/3))), rounding halves up."""
    return max(3, int(np.floor(T ** (1 / 3) + 0.5)))


def bin_choices(T):
    """(B/2, B, 2B) for the sensitivity runs; B/2 rounded half up, at least 2."""
    B = n_bins(T)
    return max(2, int(np.floor(B / 2 + 0.5))), B, 2 * B


def _flat(x):
    """(R, T, *space) -> (R, T, P) float, space shape; raises on non-finite values."""
    x = np.asarray(x, dtype=float)
    if x.ndim < 2:
        raise ValueError("x must be (R, T, *space)")
    if not np.all(np.isfinite(x)):
        raise ValueError("x has non-finite values: drop or fill them explicitly first")
    R, T = x.shape[:2]
    if R < 3:
        raise ValueError("need R >= 3 members")
    return x.reshape(R, T, -1), x.shape[2:]


def loo_mean(x):
    """X[r] = mean of x over the other R - 1 members. (R, T, ...) in and out, same units."""
    x = np.asarray(x, dtype=float)
    return (x.sum(axis=0, keepdims=True) - x) / (x.shape[0] - 1)


def pooled_edges(v, B):
    """Interior equiprobable edges (B - 1, P) from all R T values of v (R, T, P) per point."""
    R, T, P = v.shape
    q = np.arange(1, B) / B
    return np.quantile(v.reshape(R * T, P), q, axis=0)


def codes(v, edges):
    """Bin index in [0, B) of every value: v (..., P), edges (B - 1, P) -> int16 (..., P)."""
    c = np.zeros(v.shape, np.int16)
    for e in edges:                                                  # B - 1 comparisons
        c += v >= e
    return c


def entropy_counts(c, axis=-1):
    """Plug-in entropy in bits of the count histograms along `axis`."""
    c = np.asarray(c, dtype=float)
    n = c.sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = c / n
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=axis)
    return h


def mi_counts(c):
    """Plug-in MI in bits of joint count tables c (..., B, B)."""
    c = np.asarray(c, dtype=float)
    return (entropy_counts(c.sum(-1)) + entropy_counts(c.sum(-2))
            - entropy_counts(c.reshape(*c.shape[:-2], -1)))


def _count(j, K):
    """Histograms of integer codes j (..., n) in [0, K) along the last axis -> (..., K)."""
    lead = j.shape[:-1]
    g = np.arange(int(np.prod(lead)), dtype=np.int64).reshape(lead + (1,)) * K
    return np.bincount((g + j).ravel(), minlength=g.size * K).reshape(lead + (K,))


def _stats_bins(j, B, picks):
    """Statistics from joint codes j = cx B + cy, (Q, R, T) -> {name: (Q,)}."""
    Q, R, T = j.shape
    K = B * B
    cm = _count(j, K)                                                # (Q, R, K) member tables
    I_cond = mi_counts(cm.reshape(Q, R, B, B)).mean(1)
    cp = cm.sum(1)                                                   # (Q, K) all R T pairs
    I_pool = mi_counts(cp.reshape(Q, B, B))
    I_MXY = entropy_counts(cp) - entropy_counts(cm).mean(1)
    sub = j[:, picks, np.arange(T)]                                  # (Q, n_sub, T)
    I_pool_T = mi_counts(_count(sub, K).reshape(Q, -1, B, B)).mean(1)
    return dict(I_cond=I_cond, I_pool_T=I_pool_T, Delta=I_pool_T - I_cond, I_pool=I_pool,
                I_pool_minus_cond=I_pool - I_cond, I_MXY=I_MXY)


def _gauss_mi(a, b):
    """-1/2 log2(1 - r^2) of the correlation along the last axis."""
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    r = (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return -0.5 * np.log2(1 - np.minimum(r * r, 1 - 1e-12))


def _gauss_h(a, b):
    """Gaussian entropy in bits of (a, b) along the last axis (ddof 0 covariance)."""
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    det = (a * a).mean(-1) * (b * b).mean(-1) - (a * b).mean(-1) ** 2
    return LOG2_2PIE + 0.5 * np.log2(np.maximum(det, 1e-300))


def _stats_copula(zx, zy, picks):
    """Copula statistics from pooled normal scores zx, zy (Q, R, T) -> {name: (Q,)}."""
    Q, R, T = zx.shape
    I_cond = _gauss_mi(zx, zy).mean(1)
    px, py = zx.reshape(Q, -1), zy.reshape(Q, -1)
    I_pool = _gauss_mi(px, py)
    I_MXY = _gauss_h(px, py) - _gauss_h(zx, zy).mean(1)
    t = np.arange(T)
    I_pool_T = _gauss_mi(zx[:, picks, t], zy[:, picks, t]).mean(1)
    return dict(I_cond=I_cond, I_pool_T=I_pool_T, Delta=I_pool_T - I_cond, I_pool=I_pool,
                I_pool_minus_cond=I_pool - I_cond, I_MXY=I_MXY)


def normal_scores(v):
    """Normal scores from average ranks over all R T values per point: (R, T, P) -> same."""
    R, T, P = v.shape
    r = rankdata(v.reshape(R * T, P), axis=0)
    return ndtri((r - 0.5) / (R * T)).reshape(R, T, P)


def perm_tables(rng, n, R, T, block=1):
    """n member-label permutations per time step, one per block of `block` steps: (n, T, R)."""
    nb = -(-T // block)
    base = np.broadcast_to(np.arange(R, dtype=np.int16), (n, nb, R))
    p = rng.permuted(base, axis=-1)
    return np.repeat(p, block, axis=1)[:, :T]


def _permute(a, tab):
    """a (P, R, T) with labels permuted by tab (k, T, R): out[k, p, r, t] = a[p, tab[k, t, r], t]."""
    T = a.shape[2]
    return np.moveaxis(a[:, tab.transpose(0, 2, 1), np.arange(T)], 0, 1)


def _prepare(x, B, estimator):
    """Pairs once: integer joint codes (bins) or pooled normal scores (copula), (P, R, T)."""
    xf, space = _flat(x)
    X, Y = loo_mean(xf), xf
    if estimator == "bins":
        cx = codes(X, pooled_edges(X, B))
        cy = codes(Y, pooled_edges(Y, B))
        return (np.moveaxis(cx.astype(np.int32) * B + cy, -1, 0),), space
    if estimator == "copula":
        return tuple(np.moveaxis(normal_scores(v), -1, 0) for v in (X, Y)), space
    raise ValueError("estimator must be 'bins' or 'copula'")


def exchangeability_test(x, B=None, n_sub=200, n_perm=1000, block=1, estimator="bins", seed=0,
                         max_elems=2 ** 24, return_null=False):
    """Pooled vs per-member MI and I(M; X, Y) with a member-label permutation null.

    x: (R, T, *space) members by time [by grid point], finite. B: bins per axis (default
    n_bins(T)); n_sub: size-matched pooled subsamples; n_perm: permutations (1000 for an index,
    200 for gridded fields); block: time-block length of the permutation; seed: one
    np.random.default_rng stream (subsample table first, then all permutation tables), so
    results do not depend on max_elems (the chunk size, in array elements).
    Returns {stat: {"obs", "null_mean", "null_sd", "corrected"[, "p"]} each (*space,) in bits}
    for stat in STATS, plus "meta" (and "null": {stat: (n_perm, *space)} if return_null).
    """
    R, T = np.shape(x)[:2]
    B = n_bins(T) if B is None else int(B)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, R, size=(n_sub, T))
    tabs = perm_tables(rng, n_perm, R, T, block)
    arrays, space = _prepare(x, B, estimator)
    P = arrays[0].shape[0]
    run = (lambda a: _stats_bins(a[0], B, picks)) if estimator == "bins" else \
        (lambda a: _stats_copula(a[0], a[1], picks))
    obs = run(arrays)                                                # {stat: (P,)}
    null = {k: np.empty((n_perm, P)) for k in STATS}
    per = P * max(R, n_sub) * T
    step = max(1, int(max_elems // per))
    for i in range(0, n_perm, step):
        tab = tabs[i:i + step]
        k = len(tab)
        out = run(tuple(_permute(a, tab).reshape(k * P, R, T) for a in arrays))
        for s in STATS:
            null[s][i:i + k] = out[s].reshape(k, P)
    res = {}
    for s in STATS:
        m, sd = null[s].mean(0), null[s].std(0, ddof=1)
        e = dict(obs=obs[s], null_mean=m, null_sd=sd, corrected=obs[s] - m)
        if s in TWO_SIDED:
            e["p"] = (1 + (np.abs(null[s] - m) >= np.abs(obs[s] - m)).sum(0)) / (n_perm + 1)
        elif s in UPPER:
            e["p"] = (1 + (null[s] >= obs[s]).sum(0)) / (n_perm + 1)
        res[s] = {k: np.reshape(v, space) for k, v in e.items()}
    res["meta"] = dict(R=R, T=T, B=B, n_sub=n_sub, n_perm=n_perm, block=block,
                       estimator=estimator, seed=seed, units="bits")
    if return_null:
        res["null"] = {s: null[s].reshape((n_perm,) + space) for s in STATS}
    return res


def obs_mi(g, x, B=None, n_perm=1000, block=1, seed=0):
    """I_o = <MI(g; X[r])>_r from T pairs per member, never tiling g, with a time-permutation null.

    g: (T, *space) observations; x: (R, T, *space) members. Edges: g's from its own T values,
    X's from pooled X (as exchangeability_test), B per axis (default n_bins(T)). Null: g
    permuted in time, in blocks of `block` steps (block order shuffled), n_perm times, seeded.
    Returns {"obs", "null_mean", "null_sd", "corrected", "p" (upper tail)} each (*space,), bits.
    """
    xf, space = _flat(x)
    R, T, P = xf.shape
    gf = np.asarray(g, dtype=float).reshape(T, -1)
    if gf.shape[1] != P or not np.all(np.isfinite(gf)):
        raise ValueError("g must be finite, (T, *space) matching x")
    B = n_bins(T) if B is None else int(B)
    X = loo_mean(xf)
    cx = np.moveaxis(codes(X, pooled_edges(X, B)), -1, 0).astype(np.int32)   # (P, R, T)
    cg = codes(gf, pooled_edges(gf[None], B)).T.astype(np.int32)             # (P, T)
    rng = np.random.default_rng(seed)
    blocks = np.array_split(np.arange(T), -(-T // block))
    orders = [np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
              for _ in range(n_perm)]

    def stat(cgt):                                                   # cgt (k, P, T)
        j = cgt[:, :, None, :] * B + cx[None]                         # (k, P, R, T)
        k = j.shape[0]
        return mi_counts(_count(j, B * B).reshape(k, P, R, B, B)).mean(-1)

    o = stat(cg[None])[0]
    null = np.concatenate([stat(cg[:, np.stack(orders[i:i + 50])].transpose(1, 0, 2))
                           for i in range(0, n_perm, 50)])
    m = null.mean(0)
    e = dict(obs=o, null_mean=m, null_sd=null.std(0, ddof=1), corrected=o - m,
             p=(1 + (null >= o).sum(0)) / (n_perm + 1))
    return {k: np.reshape(v, space) for k, v in e.items()}
