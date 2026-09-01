"""Vectorised KSG mutual information -- the MI twin of the notebook's Pearson functions.

    import sys; sys.path.insert(0, "../scripts")
    import mi_vectorized as V, smyle_metrics as M

    I_o = V.mi_member_vs_obs(c.F, c.G)   # (N, lat, lon)     <- pearson_coeff_pairwise
    I_m = V.mi_member_vs_member(c.F)     # (N, N, lat, lon)  <- pearson_coeff_F_pairwise
    lam_o, lam_m = M.lam_of(I_o), M.lam_of(I_m)

Same contract as the Pearson versions in `notebooks/dcpp-decadal-analysis-2.ipynb`: in
`(N, T, lat, lon)`, out one statistic per (pair, cell), entry `[i, j]` the number between
members i and j. NATS, KSG-1 at k=4, copula-transformed -- the convention of
`smyle_metrics.calc_MI_sG(..., use_copula=True)`, so `smyle_metrics.lam_of` applies
directly and the numbers are comparable to the notebook's `MI_FG_pairwise`.

What is vectorised, and what is not
-----------------------------------
`mi_pairwise.MI_F_pairwise` calls `infomeasure.mutual_information` once per (pair, cell)
-- 13.5M Python-level calls on the DCPP grand ensemble, each building two scipy KDTrees
over 48 points. This module never calls the estimator. It reimplements KSG-1 as array
arithmetic over a leading batch axis, so each numpy call settles a whole cache-sized
block of `(B, T, T)` problems at once. The remaining loop is over those blocks, not over
pairs: ~10^5 trips for the grand ensemble at the default `max_bytes`, each doing a handful
of whole-array operations, against 13.5M estimator calls that each built two KDTrees.

The trick that makes it work is that T is small. A KDTree is the right structure when n
is large; at T = 48-120 the brute-force `(T, T)` distance matrix is both smaller and
branch-free, and it batches. Cost is O(B T^2) arithmetic with no Python in the loop
instead of O(B T log T) with a Python call and two tree builds per problem.

Measured, Sherlock serc node, per problem on one core (same on real SLP and on synthetic
gaussians -- the data values do not change the cost):

                        T = 48        T = 120
    mi_pairwise         500 us        1000 us
    this module          15.6 us        90 us
    speedup               32x            11x

The T^2 term is why the advantage shrinks on the monthly cubes: the KDTree's T log T
scales better than brute force, and by T = 120 it has clawed back two thirds of the gap.
It is still 11x ahead, and would stop being so somewhere past T ~ 500.

Whole-cube wall clock, 16 threads on that node:

    DCPP grand ensemble  N=101, T=48,  37x72   13.45M pairs x cells    41.7 s   11 core-min
    decadal monthly      N=46,  T=120, 37x72    2.76M pairs x cells    53.2 s   14 core-min

against 121 core-min for `mi_pairwise` on the first of those -- 11x less core time. Thread
scaling saturates near 6x on 16 cores because the kernel is memory-bandwidth-bound rather
than compute-bound, so past ~8 threads you are buying very little; spend the cores on
several start dates at once instead.

Exactness
---------
`mi_batch` reproduces `infomeasure.mutual_information(x, y, approach="metric", k=k,
noise_level=0)` to floating-point noise, including on tie-heavy copula lattices where the
answer is decided by tie-breaking. `test_mi_vectorized.py` checks that against the
installed 0.6.2. Three details are load-bearing:

* **The k-th joint distance includes the point itself.** infomeasure queries the KDTree
  with `k+1` neighbours, so `eps_i` is the (k+1)-th smallest of the distance row that
  contains the self-distance 0. Here that is `partition(dz, k)[..., k]`, same thing.
* **Marginal counts are strict.** infomeasure passes `r = nextafter(eps, -inf)` to
  `query_ball_point`, which counts `dist <= r`. For finite floats that is exactly
  `dist < eps`, which is what the comparison below does.
* **The self-subtraction is conditional.** infomeasure subtracts `(eps > 0)`, not 1. If
  eps is exactly 0 (a duplicated joint point) the negative radius makes the ball count 0
  and nothing is subtracted. Dropping the condition would put a -1 there.

Both cube kinds
---------------
Nothing here knows what the sample axis means, so the same call works on

    dcpp_handles         F (N, T, lat, lon)   T = 48-60 start dates, lead-window means
    dcpp_decadal_handles F (N, T, lat, lon)   T = 120 months within ONE hindcast
    any uninitialised    F (N, T, lat, lon)   same shape; the member-vs-member statistic
                                              needs no observations at all

and `*space` is arbitrary: `(N, T)` for a single index series, `(N, T, lat, lon)` for a
field, `(N, T, lev, lat, lon)` if it ever comes to that. Cost scales as T^2, so the
120-month decadal cube is 6.25x the per-problem work of the 48-start cube; chunking keeps
peak memory flat regardless, but budget the wall clock accordingly.

Cells that are NaN anywhere in their series (land masks, obs gaps -- GPCP before 1979)
come out NaN rather than propagating garbage through the rank transform.

The dither is not cosmetic
--------------------------
Ranks land on a lattice of spacing 1/(T+1), so after the copula transform max-norm
distances are massively degenerate -- at T=48 only 46 of the 1128 pairwise distances are
distinct. KSG decides its answer by comparing distances, so with exact ties the result is
decided by how they break, and that is the dither's job.

* `noise_level=0` with `copula=True` is a different estimator, not a faster one. It sits
  ~0.05 nats from the dithered answer, comparable to the signal in a well-correlated pair.
* Even at 1e-10 the dither realisation is worth ~0.03 nats peak-to-peak at T=48. That is
  a floor on this estimator, inherited from the notebook's convention. `seed` makes it
  reproducible; vary it to see the spread.
* With `copula=False` the data is continuous, every distance is distinct, and the dither
  is an exact no-op.

The dither is applied ONCE per (member, cell) series, not once per pair, exactly as in
`mi_pairwise`: member i brings the same jittered series to all of its pairings. Dithering
per pair instead would make I(f_i; f_j) depend on which pair it came from.

Reading a single number
-----------------------
Do not. On the DCPP grand ensemble the off-diagonal MI runs -0.29 .. 0.54 nats with mean
0.0022, and 52% of entries are negative: the typical inter-member signal is an order of
magnitude below the dither floor and a single (pair, cell) estimate is mostly noise, with
per-entry correlation 0.22 against |Pearson|. The conditional mean is what survives, and
it tracks |rho| cleanly. Aggregate over cells, pairs, or a region before interpreting.
`mi_pairwise`'s docstring has the binned table.

The diagonal
------------
I(f_i; f_i) is infinite, not 1. `pearson_coeff_F_pairwise` puts 1.0 there; the honest
analogue is `np.inf`, which is what is returned, and which `lam_of` maps to 1.0 -- so the
notebook's `rho_o / rho_m_pairwise[i, i]` pattern behaves the same after `lam_of`. Raw MI
diagonals will overflow anything that averages them.
"""
import numpy as np
from scipy.special import digamma
from scipy.stats import rankdata

__all__ = [
    "mi_batch",
    "mi_member_vs_obs",
    "mi_member_vs_member",
    "mi_ensmean_vs_obs",
    "mi_loomean_vs_member",
    "mi_loomean_vs_obs",
    "copula_rank",
    "dither",
    "have_jax",
]

# Working set per chunk, PER THREAD. 8 MB is not a memory limit, it is a cache target:
# the kernel makes ~6 passes over its (B, T, T) blocks, so throughput is set by whether
# those blocks stay in last-level cache between passes. Measured on a Sherlock serc node,
# 200k problems at T=48, single thread:
#
#     max_bytes    512 KB   2 MB    8 MB   32 MB   64 MB
#     k problems/s   52.0   62.7    66.5    35.7    31.8
#
# so the 512 MB that "one big vectorised call" suggests is 2.6x SLOWER than 8 MB, and
# below ~1 MB the blocks get too small to amortise the per-chunk Python. The same optimum
# holds at T=120. Raise it only if you have measured on your node.
DEFAULT_MAX_BYTES = 8 << 20

# JAX wants the opposite of a cache target: few, large kernel launches. Reusing the 8 MB
# above would give 20 problems per launch at T=120 -- half a million launches for one
# grand ensemble, measuring dispatch overhead rather than the kernel. Resolved per backend
# by `_resolve_max_bytes`, so `backend="jax"` gets this unless you say otherwise.
JAX_MAX_BYTES = 1 << 31


def _resolve_max_bytes(max_bytes, backend):
    """None -> the right default for this backend. An explicit value always wins."""
    if max_bytes is not None:
        return max_bytes
    return DEFAULT_MAX_BYTES if backend == "numpy" else JAX_MAX_BYTES


# --------------------------------------------------------------------------------------
# preprocessing: the copula transform and the dither, vectorised over everything but time
# --------------------------------------------------------------------------------------

def copula_rank(X, axis=-1):
    """Average-rank pseudo-observations R / (T + 1) along `axis`. Shape preserved.

    The vectorised equal of `smyle_metrics.copula_transform(..., scores="uniform")`, minus
    its per-column continuity check -- that check calls `np.unique` once per series, which
    at 13.5M series costs more than the estimator does.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[axis]
    return rankdata(X, method="average", axis=axis) / (T + 1.0)


def dither(X, noise_level=1e-10, seed=0):
    """Add iid N(0, noise_level) to break exact ties. See the module docstring."""
    if not noise_level:
        return X
    rng = np.random.default_rng(seed)
    return X + rng.normal(0.0, noise_level, X.shape)


def _prepare(X, copula=True, noise_level=1e-10, seed=0):
    """Copula-transform then dither a `(..., T)` stack."""
    X = np.asarray(X, dtype=float)
    if copula:
        X = copula_rank(X, axis=-1)
    return dither(X, noise_level, seed)


# --------------------------------------------------------------------------------------
# the kernel: KSG-1 on a batch of (T,)-by-(T,) problems, no Python in the inner loop
# --------------------------------------------------------------------------------------

def _digamma_table(n):
    """digamma(0..n-1) as a lookup table. Counts are integers, so this replaces ~10^9
    transcendental evaluations with a gather. digamma(0) is -inf and is never indexed."""
    with np.errstate(divide="ignore"):
        return digamma(np.arange(n, dtype=float))


def _mi_kernel_numpy(X, Y, k, dig):
    """KSG-1 MI in nats for a batch of problems. X, Y: (B, T) float64 -> (B,).

    Peak extra memory is three (B, T, T) float64 arrays. Every step is a whole-array
    operation; there is no loop over the B axis.
    """
    T = X.shape[1]
    dx = np.abs(X[:, :, None] - X[:, None, :])          # (B, T, T)
    dy = np.abs(Y[:, :, None] - Y[:, None, :])
    dz = np.maximum(dx, dy)

    # eps_i = k-th nearest joint neighbour EXCLUDING self = (k+1)-th smallest INCLUDING
    # the self-distance 0, which is what infomeasure's query(k=k+1) returns. partition in
    # place, so no fourth (B, T, T) array.
    dz.partition(k, axis=-1)
    eps = dz[..., k].copy()                             # (B, T)
    del dz

    # Strict `< eps` matches nextafter(eps, -inf) with a <= ball. Self sits inside that
    # ball iff eps > 0, and infomeasure removes it on exactly that condition.
    e = eps[:, :, None]
    self_in = (eps > 0).astype(np.intp)
    nx = np.count_nonzero(dx < e, axis=-1) - self_in    # (B, T)
    ny = np.count_nonzero(dy < e, axis=-1) - self_in

    local = dig[nx + 1] + dig[ny + 1]
    return digamma(k) + digamma(T) - local.mean(axis=-1)


def _chunk_size(T, max_bytes, itemsize=8):
    """How many problems fit a `max_bytes` working set -- three (B, T, T) arrays + slack."""
    per = 3.5 * T * T * itemsize
    return max(1, int(max_bytes // per))


# --------------------------------------------------------------------------------------
# the JAX kernel -- same arithmetic, jitted, chunks padded to one static shape
# --------------------------------------------------------------------------------------

def have_jax():
    """True if JAX is importable. JAX is optional; numpy is the default backend."""
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


def _jax_kernel(k, jdt):
    """Build (and let JAX cache) a jitted KSG-1 kernel for one k and dtype."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import digamma as jdigamma

    @jax.jit
    def kernel(X, Y):
        T = X.shape[-1]
        dx = jnp.abs(X[:, :, None] - X[:, None, :])
        dy = jnp.abs(Y[:, :, None] - Y[:, None, :])
        dz = jnp.maximum(dx, dy)
        # top_k on the negated row gives the k+1 smallest; the last of those is eps.
        eps = -jax.lax.top_k(-dz, k + 1)[0][..., k]
        e = eps[:, :, None]
        inside = (eps > 0).astype(jnp.int32)
        nx = jnp.sum(dx < e, axis=-1).astype(jnp.int32) - inside
        ny = jnp.sum(dy < e, axis=-1).astype(jnp.int32) - inside
        local = jdigamma((nx + 1).astype(jdt)) + jdigamma((ny + 1).astype(jdt))
        return (jdigamma(jnp.asarray(k, jdt)) + jdigamma(jnp.asarray(T, jdt))
                - jnp.mean(local, axis=-1))

    return kernel


def _run_jax(X, Y, k, chunk, dtype):
    """Drive the jitted kernel over fixed-size, padded chunks. X, Y: (B, T) -> (B,)."""
    import jax
    import jax.numpy as jnp

    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    jdt = jnp.float64 if dtype == "float64" else jnp.float32

    kernel = _jax_kernel(k, jdt)
    B, T = X.shape
    out = np.empty(B, dtype=float)
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        xb, yb = X[lo:hi], Y[lo:hi]
        pad = chunk - (hi - lo)
        if pad:                     # keep a single compiled shape; the padding is dropped
            xb = np.concatenate([xb, np.zeros((pad, T))])
            yb = np.concatenate([yb, np.zeros((pad, T))])
        res = np.asarray(kernel(jnp.asarray(xb, jdt), jnp.asarray(yb, jdt)))
        out[lo:hi] = res[: hi - lo]
    return out


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------

def mi_batch(X, Y, k=4, copula=True, noise_level=1e-10, seed=0, backend="numpy",
             max_bytes=None, dtype="float64", n_jobs=1, prepared=False):
    """I(x ; y) in NATS for a whole stack of problems at once.

    Parameters
    ----------
    X, Y : array-like, `(..., T)`
        The two variables, sample axis LAST. Leading shapes must match or broadcast; the
        result carries that leading shape. `(T,)` in gives a float out.
    k : int
        KSG's number of neighbours. 4 is the notebook's and Kraskov et al.'s default.
    copula : bool
        Rank-transform each series first. True matches
        `smyle_metrics.calc_MI_sG(..., use_copula=True)` and the notebook's decadal runs.
    noise_level, seed : float, int
        Tie-breaking dither, applied after the copula transform. Read the module docstring
        before setting `noise_level=0`; it is not a free speedup.
    backend : {"numpy", "jax"}
        Leave this at "numpy". MEASURED on a serc node, s1961 grand ensemble at T=120:
        numpy 18.1 s against jax float64 1017.2 s on the same 735k-problem subset --
        JAX-on-CPU is ~56x SLOWER, projecting 4.3 h against numpy's 4.6 min for the full
        cube. XLA gains nothing here: the kernel is top_k plus boolean reductions over
        600 MB intermediates, the numpy path replaces digamma with an integer lookup
        table, and its chunking is tuned to stay in cache. Untested on GPU, which is the
        only place "jax" might pay.
    max_bytes : int
        Working-set target per chunk. Peak RSS is about `n_jobs * max_bytes` plus the
        inputs -- each thread holds its own `(B, T, T)` block.
    dtype : {"float64", "float32"}
        JAX only. float32 roughly triples GPU throughput and moves the answer by ~1e-6
        nats on continuous data; on copula ranks it can flip near-ties, so check it
        against float64 on a subset before trusting it.
    n_jobs : int
        numpy only. Threads over chunks; the ufuncs release the GIL, so this scales.
    prepared : bool
        Skip the copula transform and the dither -- X and Y are already pseudo-
        observations. The cube wrappers use this so a member is transformed once rather
        than once per pair.

    Returns
    -------
    ndarray with X's leading shape, or a float for 1-D input. Problems containing a
    non-finite sample come back NaN.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape[-1] != Y.shape[-1]:
        raise ValueError(f"sample axes differ: {X.shape[-1]} vs {Y.shape[-1]}")
    scalar = X.ndim == 1 and Y.ndim == 1
    lead = np.broadcast_shapes(X.shape[:-1], Y.shape[:-1])
    T = X.shape[-1]
    if T < k + 1:
        raise ValueError(f"need T >= k + 1 = {k + 1} samples, got {T}")

    X = np.ascontiguousarray(np.broadcast_to(X, lead + (T,)).reshape(-1, T))
    Y = np.ascontiguousarray(np.broadcast_to(Y, lead + (T,)).reshape(-1, T))

    good = np.isfinite(X).all(-1) & np.isfinite(Y).all(-1)
    if not prepared:
        X = _prepare(X, copula, noise_level, seed)
        Y = _prepare(Y, copula, noise_level, seed + 1)

    out = np.full(X.shape[0], np.nan)
    idx = np.flatnonzero(good)
    if idx.size == good.size:
        out = _dispatch(X, Y, k, backend, max_bytes, dtype, n_jobs)
    elif idx.size:
        out[idx] = _dispatch(np.ascontiguousarray(X[idx]), np.ascontiguousarray(Y[idx]),
                             k, backend, max_bytes, dtype, n_jobs)
    return float(out[0]) if scalar else out.reshape(lead)


def _dispatch(X, Y, k, backend, max_bytes, dtype, n_jobs):
    """Chunked evaluation of the kernel over a contiguous `(B, T)` pair."""
    T = X.shape[-1]
    itemsize = 4 if (backend == "jax" and dtype == "float32") else 8
    chunk = _chunk_size(T, _resolve_max_bytes(max_bytes, backend), itemsize)

    if backend == "jax":
        return _run_jax(X, Y, k, chunk, dtype)
    if backend != "numpy":
        raise ValueError(f"backend must be 'numpy' or 'jax', got {backend!r}")

    dig = _digamma_table(T + 2)
    bounds = list(range(0, X.shape[0], chunk))
    if n_jobs == 1 or len(bounds) == 1:
        return np.concatenate([_mi_kernel_numpy(X[lo:lo + chunk], Y[lo:lo + chunk], k, dig)
                               for lo in bounds])
    from joblib import Parallel, delayed
    parts = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_mi_kernel_numpy)(X[lo:lo + chunk], Y[lo:lo + chunk], k, dig)
        for lo in bounds)
    return np.concatenate(parts)


# --------------------------------------------------------------------------------------
# cube wrappers -- the MI analogues of the notebook's Pearson functions
# --------------------------------------------------------------------------------------

def _prep_cube(F, copula, noise_level, seed):
    """`(N, T, *space)` -> pseudo-observation `(N, C, T)`, plus `space, N, T, C`.

    The transform runs in the cube's OWN `(N, T, C)` layout and only then moves the sample
    axis last. That ordering is not cosmetic: it makes the dither draw the same numbers in
    the same places as `mi_pairwise.MI_F_pairwise`, so for a given seed this module
    reproduces that function bit for bit rather than merely in distribution.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim < 2:
        raise ValueError(f"expected (N, T, *space), got shape {F.shape}")
    N, T = F.shape[:2]
    space = F.shape[2:]
    C = int(np.prod(space)) if space else 1
    X = F.reshape(N, T, C)
    if copula:
        X = rankdata(X, method="average", axis=1) / (T + 1.0)
    if noise_level:
        X = X + np.random.default_rng(seed).normal(0.0, noise_level, X.shape)
    return np.ascontiguousarray(X.transpose(0, 2, 1)), space, N, T, C


def _prep_field(o, copula, noise_level, seed):
    """`(T, *space)` -> pseudo-observation `(C, T)`. The single-field twin of `_prep_cube`."""
    o = np.asarray(o, dtype=float)
    T = o.shape[0]
    C = int(np.prod(o.shape[1:])) if o.ndim > 1 else 1
    X = o.reshape(T, C)
    if copula:
        X = rankdata(X, method="average", axis=0) / (T + 1.0)
    if noise_level:
        X = X + np.random.default_rng(seed).normal(0.0, noise_level, X.shape)
    return np.ascontiguousarray(X.T)


def mi_member_vs_obs(F, o, k=4, copula=True, noise_level=1e-10, seed=0, **kw):
    """I(f_n ; o) per member and cell -> `(N, *space)`. NATS.

    The mutual-information analogue of `pearson_coeff_pairwise` -- same shapes, same
    (member, cell) layout, but a KSG estimate rather than a correlation.

    F: `(N, T, *space)`. o: `(T, *space)`. The notebook tiles obs to `(N, T, *space)`
    before calling its Pearson version; that is unnecessary here but accepted.
    """
    R, space, N, T, C = _prep_cube(F, copula, noise_level, seed)
    o = np.asarray(o, dtype=float)
    if o.ndim == np.ndim(F):                       # already tiled to (N, T, *space)
        o = o[0]
    O = _prep_field(o, copula, noise_level, seed + 1)          # (C, T)
    out = mi_batch(R.reshape(-1, T), np.broadcast_to(O, (N, C, T)).reshape(-1, T),
                   k=k, prepared=True, **kw)
    return out.reshape((N,) + space)


def mi_member_vs_member(F, k=4, copula=True, noise_level=1e-10, seed=0, verbose=False,
                        **kw):
    """I(f_i ; f_j) for every member pair and cell -> `(N, N, *space)`. NATS.

    The MI analogue of `pearson_coeff_F_pairwise`. KSG's max-norm neighbourhoods are
    symmetric in their two arguments, so only the N(N-1)/2 upper triangle is estimated and
    then mirrored -- exactly, not approximately. The diagonal is `+inf`; see the module
    docstring.

    Works on either cube kind: `dcpp_handles` (T = start dates) and
    `dcpp_decadal_handles` (T = months within one hindcast) differ only in what T means,
    and an uninitialised ensemble needs no observations for this statistic at all.
    """
    R, space, N, T, C = _prep_cube(F, copula, noise_level, seed)
    ii, jj = np.triu_indices(N, k=1)
    P = ii.size
    total = P * C

    n_jobs = kw.pop("n_jobs", 1)
    backend = kw.get("backend", "numpy")
    max_bytes = _resolve_max_bytes(kw.pop("max_bytes", None), backend)
    if backend != "numpy":
        # Same chunked walk as below, but serial and with JAX-sized blocks -- the jitted
        # kernel is already parallel inside. Gathering all P*C problems up front instead
        # would materialise two (P*C, T) host arrays: 21 GB for one grand ensemble.
        itemsize = 4 if kw.get("dtype") == "float32" else 8
        chunk = _chunk_size(T, max_bytes, itemsize)
        parts = []
        for lo in range(0, total, chunk):
            hi = min(lo + chunk, total)
            p, c = np.divmod(np.arange(lo, hi), C)
            parts.append(mi_batch(R[ii[p], c], R[jj[p], c], k=k, prepared=True,
                                  max_bytes=max_bytes, **kw))
            if verbose:
                print(f"  {hi}/{total} ({100 * hi / total:.1f}%)", flush=True)
        return _mirror(np.concatenate(parts), ii, jj, N, C, space)

    # One flat problem index p*C + c, walked in cache-sized chunks. Nothing iterates over
    # pairs in Python: a chunk is ~130 problems at T=48 and the kernel settles all of them
    # in a handful of whole-array calls, so the loop trades 13.5M estimator invocations for
    # ~10^5 trips of vectorised work.
    #
    # The gather is done INSIDE the worker and there is exactly one thread pool for the
    # whole run. Going through `mi_batch` per chunk instead would build a fresh joblib
    # Parallel (and its 16 threads) thousands of times, which cost more than the kernel.
    chunk = _chunk_size(T, max_bytes)
    dig = _digamma_table(T + 2)
    bounds = list(range(0, total, chunk))

    def work(lo):
        hi = min(lo + chunk, total)
        p, c = np.divmod(np.arange(lo, hi), C)
        X = np.ascontiguousarray(R[ii[p], c])
        Y = np.ascontiguousarray(R[jj[p], c])
        good = np.isfinite(X).all(-1) & np.isfinite(Y).all(-1)
        if good.all():
            return _mi_kernel_numpy(X, Y, k, dig)
        part = np.full(hi - lo, np.nan)
        idx = np.flatnonzero(good)
        if idx.size:
            part[idx] = _mi_kernel_numpy(np.ascontiguousarray(X[idx]),
                                         np.ascontiguousarray(Y[idx]), k, dig)
        return part

    if n_jobs == 1:
        parts = []
        for n, lo in enumerate(bounds):
            parts.append(work(lo))
            if verbose and n % 200 == 0:
                print(f"  {lo}/{total} ({100 * lo / total:.1f}%)", flush=True)
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(work)(lo) for lo in bounds)
    upper = np.concatenate(parts)
    return _mirror(upper, ii, jj, N, C, space)


def _mirror(upper, ii, jj, N, C, space):
    """Upper-triangle values -> a full symmetric `(N, N, *space)` with an infinite diagonal."""
    out = np.empty((N, N, C))
    upper = upper.reshape(ii.size, C)
    out[ii, jj] = upper
    out[jj, ii] = upper
    out[np.diag_indices(N)] = np.inf
    return out.reshape((N, N) + space)


def mi_ensmean_vs_obs(s, o, k=4, copula=True, noise_level=1e-10, seed=0, **kw):
    """I(s ; o) per cell -> `(*space,)`. Vectorised `smyle_metrics.calc_MI_sG`.

    s, o: `(T, *space)`.
    """
    space = np.shape(s)[1:]
    S = _prep_field(s, copula, noise_level, seed)
    O = _prep_field(o, copula, noise_level, seed + 1)
    return mi_batch(S, O, k=k, prepared=True, **kw).reshape(space)


def mi_loomean_vs_member(F, k=4, copula=True, noise_level=1e-10, seed=0, **kw):
    """< I(s_-n ; f_n) >_n per cell -> `(*space,)`. Vectorised `smyle_metrics.calc_MI_sF`.

    `s_-n` is the exact mean of the other N-1 members, so no member's own information
    leaks into its target.
    """
    F = np.asarray(F, dtype=float)
    N = F.shape[0]
    loo = (F.sum(axis=0, keepdims=True) - F) / (N - 1)          # raw, as in calc_MI_sF
    R, space, _, T, C = _prep_cube(F, copula, noise_level, seed)
    L, _, _, _, _ = _prep_cube(loo, copula, noise_level, seed + 1)
    out = mi_batch(L.reshape(-1, T), R.reshape(-1, T), k=k, prepared=True, **kw)
    return out.reshape(N, C).mean(axis=0).reshape(space)


def mi_loomean_vs_obs(F, o, k=4, copula=True, noise_level=1e-10, seed=0, **kw):
    """< I(s_-n ; o) >_n per cell -> `(*space,)`. Vectorised `calc_MI_sG_LOOavg`.

    The leave-one-out ensemble mean verified against observations, averaged over which
    member was held out -- the notebook's `lam_o_LOO`.
    """
    F = np.asarray(F, dtype=float)
    N = F.shape[0]
    loo = (F.sum(axis=0, keepdims=True) - F) / (N - 1)
    L, space, _, T, C = _prep_cube(loo, copula, noise_level, seed)
    O = _prep_field(o, copula, noise_level, seed + 1)            # (C, T)
    out = mi_batch(L.reshape(-1, T), np.broadcast_to(O, (N, C, T)).reshape(-1, T),
                   k=k, prepared=True, **kw)
    return out.reshape(N, C).mean(axis=0).reshape(space)
