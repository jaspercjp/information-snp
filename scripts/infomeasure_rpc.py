"""RPC information metrics computed with the `infomeasure` package.

Replaces the hand-written estimators used by `calc_RPC_KSG.ipynb` with
`infomeasure` >= 0.6, and computes the lambda metrics under three MI estimators:

    binning  -> approach="discrete"  (free parameter: nbins)
    KSG      -> approach="ksg"       (free parameter: k)
    ordinal  -> approach="ordinal"   (free parameter: embedding_dim)

Conventions pinned here, because every one of them silently changes the numbers:

* ``base=2`` on every call. infomeasure returns nats by default; the whole
  project works in bits and ``lambda = sqrt(1 - 2^-2I)`` requires bits.
* ``noise_level=0`` for KSG. infomeasure dithers by 1e-10 by default; the
  hand-written estimator did not, and NaNs are filled with 0.0 here, so
  dithering changes how the resulting exact ties are broken.
* ``minkowski_p=inf`` (max norm) for KSG, matching the hand-written MI.
* A **manual rank/copula transform** before KSG. infomeasure has no built-in
  one (``normalize=True`` is a z-score, not a rank transform) and the
  hand-written estimator rank-transformed every variable to uniform marginals.
* ``k=5`` rather than infomeasure's default of 4, which is this project's default.

With those four set, ``mi_bits`` reproduces the committed hand-rolled KSG number
at lat 32 / lon 21 to six decimals: I(F;g) = 0.186764 bits at k=5.

The ensemble-entry convention
-----------------------------
``infomeasure``'s discrete and ordinal MI estimators accept only 1-D variables
(``shape (n_samples,)``); passing an ``(n_samples, n_dims)`` ensemble raises.
Only KSG takes the joint 49-dimensional form used by `calc_RPC_KSG.ipynb`.
A three-way comparison therefore has to reduce the ensemble to one series, and
must use the *same* reduction for all three or it compares estimators and
ensemble-entry conventions at once. The reduction used here is the **ensemble
mean**, which is also the closest analogue of the published rho:

    I_o = I(s ; g)                    s = ensemble mean, g = observations
    I_m = < I(s_-i ; f_i) >_i         leave-one-out perfect-model analogue

``I_m`` deliberately uses the leave-one-out mean ``s_-i`` rather than the full
ensemble mean ``s``: ``s`` contains ``f_i``, so ``I(s; f_i)`` is inflated by
self-information. The joint 49-D KSG form is available via
``ensemble_entry="joint"`` for context, but it is not comparable to the other two.
"""

from __future__ import annotations

import os
import warnings
from multiprocessing import Pool

import numpy as np
import xarray as xr
from scipy.stats import rankdata

import infomeasure as im

# --------------------------------------------------------------------------- #
# estimator configuration
# --------------------------------------------------------------------------- #

#: name -> (infomeasure approach, free-parameter name, values to sweep, default index)
#:
#: The sweeps span each estimator's own bias/resolution trade-off, so the
#: leading axis of every returned field means "this estimator's free parameter"
#: rather than one shared axis. Defaults (index 1) are the project's k=5 and
#: nbins=5, and embedding_dim=3 for ordinal -- see ``ORDINAL_NOTE``.
ESTIMATORS = {
    "discrete": ("discrete", "nbins", (3, 5, 10), 1),
    "ksg": ("ksg", "k", (3, 5, 10), 1),
    "ordinal": ("ordinal", "embedding_dim", (2, 3, 4), 1),
}

ORDINAL_NOTE = """\
embedding_dim is bounded hard by T=120. Ordinal with embedding_dim=d builds d!
patterns from T-d+1 windows, so d=4 gives 24 patterns and d=5 gives 120 patterns
for 116 windows -- fully saturated. Measured permutation nulls at lat 32/lon 21
bear this out (40 permutations, bits):

    d   patterns  windows        I     null   bias
    1          1      120   0.0000   0.0000   zero by construction
    2          2      119   0.0728   0.0052   nearly blind
    3          6      118   0.5157   0.1789   35%
    4         24      117   2.0993   1.8346   87%
    5        120      116   4.1871   4.3330   103% -- excess goes negative

d=3 is the only defensible default, but note it still carries 35% bias, so the
ordinal column is only reported with its null subtracted.\
"""

NANFILL = "zero"      # "zero" matches the published code
LOO_MEMBERS = 12      # held-out members averaged for the model side
N_PERM = 20           # permutations for the per-estimator bias floor
SEED = 0


# --------------------------------------------------------------------------- #
# preprocessing -- calc_RPC.ipynb cell 3, verbatim apart from the time guard
# --------------------------------------------------------------------------- #

def load_data(data_dir="../data"):
    """Load and preprocess the ensemble, ensemble mean and observations.

    Identical to `calc_RPC.ipynb` cell 3 (verified bit-identical, max|diff| = 0)
    with one guard: the observation file on disk now carries a plain
    ``datetime64`` time axis rather than a ``cftime`` one, so
    ``obs.indexes["time"].to_datetimeindex()`` raises ``AttributeError``. The
    call is a no-op on a ``DatetimeIndex``, so it is applied only when needed.
    """
    f = xr.open_dataset(
        os.path.join(data_dir, "ensembles/large_psl_decadal_ensemble_no_extrapolate.nc")
    ).drop_dims(["bnds"])
    obs = xr.open_dataset(os.path.join(data_dir, "obs/hadslp2/hadslp2_monthly_1850_2004.nc"))
    obs = obs.where(obs["time.year"] >= 1962, drop=True)
    obs = obs.where(obs["time.year"] < 1972, drop=True)
    obs = obs.sortby("lon")

    with warnings.catch_warnings(action="ignore"):
        idx = obs.indexes["time"]
        obs["time"] = idx.to_datetimeindex() if hasattr(idx, "to_datetimeindex") else idx
    obs = obs.interp_like(f, kwargs={"fill_value": "extrapolate"})

    obs_lin_trend = xr.polyval(
        obs["time"], obs.psl.polyfit(dim="time", deg=1).polyfit_coefficients
    )
    obs["psl"] = obs["psl"] - obs_lin_trend

    f_lin_trend = xr.polyval(
        f["time"], f.psl.polyfit(dim="time", skipna=True, deg=1).polyfit_coefficients
    )
    f["psl"] = f["psl"] - f_lin_trend

    s = f.mean("n")
    return f, s, obs


def var_RPC(ensemble, signal, observation):
    """Published variance-based RPC. Never touches an MI estimator, so it is the
    cross-check that preprocessing is right (72.1% area-weighted)."""
    rho_o = xr.corr(signal.psl, observation.psl, dim="time")
    rho_o = xr.where(rho_o > 0, rho_o, 0)
    var_s = signal.psl.var(dim="time", skipna=True)
    var_total = ensemble.psl.var(dim="time", skipna=True).mean("n")
    rho_m = np.sqrt(var_s / var_total)
    return rho_o, rho_m, rho_o / rho_m


# --------------------------------------------------------------------------- #
# the infomeasure adapter
# --------------------------------------------------------------------------- #

def copula(x):
    """Rank transform to uniform marginals, as the hand-written KSG did."""
    x = np.asarray(x, dtype=float)
    return (rankdata(x) - 0.5) / len(x)


def rank_bins(x, nbins):
    """Equiprobable rank bins -> integer labels for the discrete estimator.

    infomeasure's ``DiscreteMIEstimator`` has no ``bins`` parameter -- it takes
    already-discrete data -- so the binning is done here, the same way
    `calc_RPC_KSG.ipynb`'s ``discrete_ensemble_info`` did it.
    """
    return np.minimum((copula(x) * nbins).astype(int), nbins - 1)


def mi_bits(x, y, estimator, param=None):
    """I(x; y) in bits via infomeasure, with this project's conventions pinned.

    Parameters
    ----------
    x : array, shape (T,) or (T, d)
        ``(T, d)`` is accepted only for ``estimator="ksg"``.
    y : array, shape (T,)
    estimator : {"discrete", "ksg", "ordinal"}
    param : int, optional
        The estimator's free parameter (nbins / k / embedding_dim). Defaults to
        that estimator's default from ``ESTIMATORS``.
    """
    approach, _, values, default_idx = ESTIMATORS[estimator]
    if param is None:
        param = values[default_idx]

    if estimator == "ksg":
        x = (np.column_stack([copula(x[:, j]) for j in range(x.shape[1])])
             if np.ndim(x) == 2 else copula(x))
        return float(im.mutual_information(
            x, copula(y), approach=approach, base=2, k=param,
            noise_level=0, minkowski_p=np.inf, normalize=False))

    if np.ndim(x) == 2:
        raise ValueError(
            "the %r estimator takes 1-D variables only; reduce the ensemble first"
            % estimator)

    if estimator == "discrete":
        return float(im.mutual_information(
            rank_bins(x, param), rank_bins(y, param), approach=approach, base=2))

    return float(im.mutual_information(
        x, y, approach=approach, base=2, embedding_dim=param))


def lam(I_bits):
    """lambda = sqrt(1 - 2^-2I), clipped at I = 0.

    All three estimators can return small negative values from sampling error;
    the clip is what `calc_RPC_KSG.ipynb` did too.
    """
    return np.sqrt(1.0 - 2.0 ** (-2.0 * np.maximum(np.asarray(I_bits, float), 0.0)))


# --------------------------------------------------------------------------- #
# gamma = I / H: matched estimator pairs
# --------------------------------------------------------------------------- #

#: level -> spec. One ``level`` string selects the MI approach, the entropy
#: approach and the transform each one sees, so a gamma cannot be assembled from
#: a numerator and a denominator of different kinds. That is the failure mode
#: that broke three earlier versions of this metric: a discrete plug-in I over a
#: differential kNN H, and a 1-D H normalising one side while a 49-D H
#: normalised the other.
#:
#: ``kind`` records what the denominator is.
#:
#: * ``"shannon"`` -- H >= 0 and bounded by log2(n_states), so gamma is a
#:   genuine fraction of the target's uncertainty and lies in [0, 1].
#: * ``"differential"`` -- H is unbounded below and shifts by log2(scale) under
#:   a change of units, so gamma means nothing until the scale is pinned. Every
#:   differential rung therefore standardises, which fixes the denominator at
#:   ``0.5*log2(2*pi*e) - J`` for negentropy J >= 0: bounded above by 2.047 bits,
#:   a shape measure rather than a spread measure.
#:
#: ``mi_transform`` and ``h_transform`` differ on the kNN rung on purpose. MI is
#: invariant to monotone marginal transforms, so the copula step there is an
#: estimator-quality choice (worth 4x, see commit f8f1420) that does not change
#: the population quantity. Differential entropy is *not* invariant, so its
#: transform defines which quantity is being estimated -- and on copula data the
#: answer is h(U[0,1]) = 0, leaving the estimator to report pure bias
#: (+0.43 bits at T=120). The denominator must therefore see standardised
#: values, never ranks.
GAMMA_LADDER = {
    "gaussian": dict(mi=None, h=None, param="", values=(0,), default_idx=0,
                     mi_transform="standardize", h_transform="standardize",
                     kind="differential"),
    "discrete": dict(mi="discrete", h="discrete", param="nbins",
                     values=(3, 5, 10), default_idx=1,
                     mi_transform="width_bins", h_transform="width_bins",
                     kind="shannon"),
    "miller_madow": dict(mi="miller_madow", h="miller_madow", param="nbins",
                         values=(3, 5, 10), default_idx=1,
                         mi_transform="width_bins", h_transform="width_bins",
                         kind="shannon"),
    "ordinal": dict(mi="ordinal", h="ordinal", param="embedding_dim",
                    values=(2, 3), default_idx=1,
                    mi_transform="none", h_transform="none", kind="shannon"),
    "kernel": dict(mi="kernel", h="kernel", param="bandwidth",
                   values=(0.3, 0.5, 1.0), default_idx=1,
                   mi_transform="standardize", h_transform="standardize",
                   kind="differential"),
    "knn": dict(mi="ksg", h="metric", param="k", values=(3, 5, 10), default_idx=1,
                mi_transform="copula", h_transform="standardize",
                kind="differential"),
}

#: H of a standardised Gaussian, in bits. The ceiling of every differential rung
#: and the exact denominator of the analytic one.
H_GAUSS = 0.5 * np.log2(2.0 * np.pi * np.e)

#: gamma is a ratio, so a denominator near zero manufactures arbitrarily large
#: values out of estimator noise. Below this many bits the point is dropped.
H_FLOOR = 0.10

KERNEL_NAME = "gaussian"      # infomeasure's kernel estimators require it by name


def width_bins(x, nbins):
    """Equal-width bins over the observed range -> integer labels.

    ``rank_bins`` cannot be used for gamma. Equiprobable bins put exactly T/nbins
    samples in every bin by construction, so H is identically log2(nbins) at
    every grid point (verified exact) and the denominator carries no information
    at all -- gamma would be I rescaled by a constant. Equal-width bins let H
    respond to the shape of the distribution, which is the whole point of
    normalising by it.
    """
    x = np.asarray(x, dtype=float)
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    return np.clip(np.digitize(x, edges[1:-1]), 0, nbins - 1)


def standardize(x):
    """Zero mean, unit variance. Pins the scale of a differential entropy."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _apply(x, transform, param):
    if transform == "copula":
        return copula(x)
    if transform == "standardize":
        return standardize(x)
    if transform == "width_bins":
        return width_bins(x, param)
    if transform == "none":
        return np.asarray(x, dtype=float)
    raise ValueError("transform %r not recognized" % transform)


def _approach_kwargs(spec, param):
    """The free parameter under the name infomeasure expects for this level."""
    if spec["mi"] == "kernel":
        return dict(kernel=KERNEL_NAME, bandwidth=param)
    if spec["param"] in ("k", "embedding_dim"):
        return {spec["param"]: param}
    return {}


def entropy_bits(x, level, param=None):
    """H(x) in bits from the entropy estimator matched to ``level``.

    Mirrors ``mi_bits``: ``base=2`` always, the level's own ``h_transform``
    applied here rather than by the caller, and the level's free parameter
    passed under whichever keyword infomeasure wants for that family.

    The ``minkowski_p`` that ``mi_bits`` pins for KSG has no analogue to set
    here: every Minkowski norm coincides in one dimension, and these entropies
    are all of 1-D series, so ``metric`` returns the same value for p = 2 and
    p = inf (verified). ``calc_RPC_KSG.ipynb``'s hand-written H used p = 2; the
    distinction is moot at this dimension.
    """
    spec = GAMMA_LADDER[level]
    if param is None:
        param = spec["values"][spec["default_idx"]]
    z = _apply(x, spec["h_transform"], param)
    if spec["h"] is None:                       # analytic Gaussian rung
        return float(H_GAUSS)
    return float(im.entropy(z, approach=spec["h"], base=2,
                            **_approach_kwargs(spec, param)))


def mi_bits_matched(x, y, level, param=None):
    """I(x; y) in bits from the MI estimator matched to ``level``.

    Separate from ``mi_bits`` because that function is pinned to the three
    lambda estimators and their transforms; this one follows ``GAMMA_LADDER``.
    """
    spec = GAMMA_LADDER[level]
    if param is None:
        param = spec["values"][spec["default_idx"]]
    a = _apply(x, spec["mi_transform"], param)
    b = _apply(y, spec["mi_transform"], param)
    if spec["mi"] is None:                      # analytic Gaussian rung
        r = float(np.corrcoef(a, b)[0, 1])
        r = min(abs(r), 1.0 - 1e-12)
        return float(-0.5 * np.log2(1.0 - r * r))
    if spec["mi"] == "ksg":
        return float(im.mutual_information(a, b, approach="ksg", base=2, k=param,
                                           noise_level=0, minkowski_p=np.inf,
                                           normalize=False))
    return float(im.mutual_information(a, b, approach=spec["mi"], base=2,
                                       **_approach_kwargs(spec, param)))


def gamma_pair(x, y, level, param=None, n_perm=0, seed=SEED):
    """I(x; y), H(y) and gamma = I/H from one matched estimator family.

    ``y`` is the prediction target and the variable whose entropy normalises, so
    gamma reads as the fraction of the target's uncertainty that ``x`` accounts
    for. Both sides of a gamma ratio must use the same convention: it is the
    mismatch, not the choice, that produced gamma = 121.

    ``n_perm`` > 0 adds ``I_null``, the mean I over time-permuted targets, and
    ``gamma_excess`` built from ``I - I_null``. There is deliberately no entropy
    null: every estimator here except ``ordinal`` reads only the sample
    distribution of ``y``, so permuting time leaves H exactly unchanged and a
    permutation test says nothing about its bias. Use the bias-corrected rung
    (``miller_madow``) to address that instead.
    """
    spec = GAMMA_LADDER[level]
    if param is None:
        param = spec["values"][spec["default_idx"]]

    I = mi_bits_matched(x, y, level, param)
    H = entropy_bits(y, level, param)
    out = dict(I=I, H=H, gamma=I / H if H > H_FLOOR else np.nan, kind=spec["kind"])

    if n_perm:
        rng = np.random.default_rng(seed)
        I_null = float(np.mean([mi_bits_matched(x, rng.permutation(y), level, param)
                                for _ in range(n_perm)]))
        out["I_null"] = I_null
        out["gamma_excess"] = ((I - I_null) / H) if H > H_FLOOR else np.nan
    return out


# --------------------------------------------------------------------------- #
# one grid point
# --------------------------------------------------------------------------- #

def _prepare(f_t, g_t, nanfill=NANFILL):
    """Apply the NaN policy and return (f, g) ready for the estimators."""
    if nanfill == "zero":
        return np.nan_to_num(f_t, nan=0.0), np.nan_to_num(g_t, nan=0.0)
    if nanfill == "drop":
        keep = ~(np.isnan(f_t).any(axis=0) | np.isnan(g_t))
        return f_t[:, keep], g_t[keep]
    raise ValueError("nanfill %r not recognized (use 'zero' or 'drop')" % nanfill)


def _ens_mean(f_raw, members=None):
    """Ensemble mean over ``members``, skipping NaNs, then zero-filled.

    The order matters. ``s = f.mean("n")`` in `calc_RPC.ipynb` is xarray's
    *skipna* mean, so the published signal is the mean over the members that
    actually have data at each time step. Averaging the zero-filled ensemble
    instead shrinks the mean toward zero wherever members are missing -- and
    they are missing in bulk: members 10..19 have all of calendar 1971 as NaN.
    At lat 32 / lon 21 the two differ by 0.024 bits in I_o, so this is taken in
    the published order: nanmean first, zero-fill only what is still NaN.
    """
    sub = f_raw if members is None else f_raw[members]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN time steps
        m = np.nanmean(sub, axis=0)
    return np.nan_to_num(m, nan=0.0)


def point_metrics(f_t, g_t, estimator, param=None, held_out=None,
                  n_perm=N_PERM, nanfill=NANFILL, seed=SEED,
                  ensemble_entry="mean"):
    """Observation- and model-side MI, lambdas and permutation nulls at one point.

    Parameters
    ----------
    f_t : array, shape (N_members, T)
    g_t : array, shape (T,)
    ensemble_entry : {"mean", "joint"}
        ``"mean"`` reduces the ensemble to its mean and is the only choice all
        three estimators support. ``"joint"`` uses the 49-D form and is KSG-only;
        it is context, not a comparable number.

    Returns
    -------
    dict with keys ``I_o``, ``I_m``, ``lam_o``, ``lam_m``, ``lam``,
    ``null_o``, ``null_m``.
    """
    N = f_t.shape[0]
    if held_out is None:
        held_out = np.unique(np.linspace(0, N - 1, LOO_MEMBERS).round().astype(int))
    rng = np.random.default_rng(seed)

    # f_raw keeps the NaNs so the ensemble mean can skip them, as `f.mean("n")`
    # does; F is the zero-filled ensemble the estimators actually see.
    f_raw = f_t if nanfill == "zero" else None
    F, g = _prepare(f_t, g_t, nanfill)

    if ensemble_entry == "joint":
        if estimator != "ksg":
            raise ValueError("ensemble_entry='joint' is KSG-only")
        x_o = F.T
        model_pairs = [(np.delete(F, i, axis=0).T, F[i]) for i in held_out]
    elif ensemble_entry == "mean":
        if f_raw is None:                      # nanfill="drop": no NaNs left
            x_o = F.mean(axis=0)
            model_pairs = [(np.delete(F, i, axis=0).mean(axis=0), F[i])
                           for i in held_out]
        else:
            keep = [np.delete(np.arange(N), i) for i in held_out]
            x_o = _ens_mean(f_raw)
            model_pairs = [(_ens_mean(f_raw, m), F[i])
                           for m, i in zip(keep, held_out)]
    else:
        raise ValueError("ensemble_entry %r not recognized" % ensemble_entry)

    I_o = mi_bits(x_o, g, estimator, param)
    I_m = float(np.mean([mi_bits(a, b, estimator, param) for a, b in model_pairs]))

    # Permutation null: the same estimator on a time-shuffled target. This is
    # the estimator's own bias floor, which at N=49 / T=120 is not negligible
    # and differs several-fold between the three estimators -- so raw lambda
    # differences between estimators partly measure bias, not skill.
    null_o = null_m = np.nan
    if n_perm:
        null_o = float(np.mean([mi_bits(x_o, rng.permutation(g), estimator, param)
                                for _ in range(n_perm)]))
        a0, b0 = model_pairs[len(model_pairs) // 2]
        null_m = float(np.mean([mi_bits(a0, rng.permutation(b0), estimator, param)
                                for _ in range(n_perm)]))

    lo, lm = lam(I_o), lam(I_m)
    return dict(I_o=I_o, I_m=I_m, lam_o=float(lo), lam_m=float(lm),
                lam=float(lo / lm) if lm > 0 else np.nan,
                null_o=null_o, null_m=null_m)


# --------------------------------------------------------------------------- #
# the full grid
# --------------------------------------------------------------------------- #

def _do_row(task):
    """One latitude row, for every estimator and every value of its parameter.

    The row slices travel in the task payload rather than through a ``Pool``
    initializer. The full ensemble is 385 MB, and on macOS ``Pool`` spawns
    rather than forks, so ``initargs`` would pickle all 385 MB once per worker
    and the pool never finishes starting. A row is 6 MB.
    """
    lat, F, G, kw = task                      # F: (N, T, n_lon)  G: (T, n_lon)
    n_lon = G.shape[1]
    n_perm, nanfill, seed = kw["n_perm"], kw["nanfill"], kw["seed"]
    entry = kw["ensemble_entry"]

    out = {}
    for name, (_, _, values, default_idx) in ESTIMATORS.items():
        out[name] = dict(
            I_o=np.full((len(values), n_lon), np.nan),
            I_m=np.full((len(values), n_lon), np.nan),
            null_o=np.full(n_lon, np.nan),
            null_m=np.full(n_lon, np.nan),
        )

    N = F.shape[0]
    held_out = np.unique(np.linspace(0, N - 1, LOO_MEMBERS).round().astype(int))

    for lon in range(n_lon):
        f_t = F[:, :, lon]
        g_t = G[:, lon]
        for name, (_, _, values, default_idx) in ESTIMATORS.items():
            for j, v in enumerate(values):
                # nulls only at the default parameter -- they cost n_perm calls each
                np_here = n_perm if j == default_idx else 0
                r = point_metrics(f_t, g_t, name, param=v, held_out=held_out,
                                  n_perm=np_here, nanfill=nanfill,
                                  seed=seed + lat * 1000 + lon,
                                  ensemble_entry=entry)
                out[name]["I_o"][j, lon] = r["I_o"]
                out[name]["I_m"][j, lon] = r["I_m"]
                if np_here:
                    out[name]["null_o"][lon] = r["null_o"]
                    out[name]["null_m"][lon] = r["null_m"]
    return lat, out


def grid_metrics(ensemble, observation, n_perm=N_PERM, nanfill=NANFILL,
                 seed=SEED, ensemble_entry="mean", processes=None, progress=True):
    """Run all three estimators over the whole lat/lon grid.

    Returns
    -------
    dict
        ``result[estimator]["I_o"]`` has shape ``(n_params, n_lat, n_lon)``;
        ``result[estimator]["null_o"]`` has shape ``(n_lat, n_lon)``. The
        leading axis is that estimator's *own* free parameter, and
        ``result["params"][estimator]`` records its values -- so nothing has to
        remember which shared axis meant what.
    """
    F = ensemble.psl.to_numpy()
    G = observation.psl.to_numpy()
    n_lat, n_lon = G.shape[1], G.shape[2]

    result = {"params": {name: dict(name=p, values=list(v), default=v[d])
                         for name, (_, p, v, d) in ESTIMATORS.items()},
              "ensemble_entry": ensemble_entry}
    for name, (_, _, values, _) in ESTIMATORS.items():
        result[name] = dict(
            I_o=np.full((len(values), n_lat, n_lon), np.nan),
            I_m=np.full((len(values), n_lat, n_lon), np.nan),
            null_o=np.full((n_lat, n_lon), np.nan),
            null_m=np.full((n_lat, n_lon), np.nan),
        )

    kwargs = dict(n_perm=n_perm, nanfill=nanfill, seed=seed,
                  ensemble_entry=ensemble_entry)
    tasks = [(lat, F[:, :, lat, :], G[:, lat, :], kwargs) for lat in range(n_lat)]

    if processes == 1:
        it = map(_do_row, tasks)
        pool = None
    else:
        pool = Pool(processes=processes)
        it = pool.imap_unordered(_do_row, tasks)
    try:
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(it, total=n_lat, desc="infomeasure RPC (lat rows)")
            except ImportError:
                pass
        for lat, out in it:
            for name in ESTIMATORS:
                result[name]["I_o"][:, lat, :] = out[name]["I_o"]
                result[name]["I_m"][:, lat, :] = out[name]["I_m"]
                result[name]["null_o"][lat, :] = out[name]["null_o"]
                result[name]["null_m"][lat, :] = out[name]["null_m"]
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return result


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #

def save_cache(path, result):
    flat = {}
    for name in ESTIMATORS:
        for field, arr in result[name].items():
            flat["%s__%s" % (name, field)] = arr
    for name, spec in result["params"].items():
        flat["params__%s" % name] = np.array(spec["values"])
    flat["ensemble_entry"] = np.array(result["ensemble_entry"])
    np.savez_compressed(path, **flat)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    result = {"params": {}, "ensemble_entry": str(z["ensemble_entry"])}
    for name, (_, p, v, d) in ESTIMATORS.items():
        result[name] = {field: z["%s__%s" % (name, field)]
                        for field in ("I_o", "I_m", "null_o", "null_m")}
        vals = list(z["params__%s" % name])
        result["params"][name] = dict(name=p, values=vals, default=v[d])
    return result


def lambda_fields(result, estimator, floor=False):
    """lambda_o, lambda_m, lambda for one estimator, at its default parameter.

    Parameters
    ----------
    floor : bool
        If True, subtract each side's own permutation null before forming
        lambda. Without this the three estimators are not comparable, because
        their bias floors differ several-fold. Points whose model side fails to
        clear its own null give lambda = NaN rather than a divide-by-zero.
    """
    d = ESTIMATORS[estimator][3]
    I_o = result[estimator]["I_o"][d]
    I_m = result[estimator]["I_m"][d]
    if floor:
        I_o = I_o - result[estimator]["null_o"]
        I_m = I_m - result[estimator]["null_m"]
    lo, lm = lam(I_o), lam(I_m)
    return lo, lm, np.where(lm > 0, lo / np.where(lm > 0, lm, 1.0), np.nan)


# --------------------------------------------------------------------------- #
# gamma over the grid
# --------------------------------------------------------------------------- #

GAMMA_TREATMENTS = ("as-is", "deseasonalised")

#: fields returned per level per treatment, all shape (n_lat, n_lon)
GAMMA_FIELDS = ("I_o", "H_o", "gamma_o", "I_o_null", "gamma_o_excess",
                "I_m", "H_m", "gamma_m", "I_m_null", "gamma_m_excess")


def deseason(a, period=12):
    """Subtract the month-of-year climatology along the leading (time) axis.

    The preprocessing in ``load_data`` removes a linear trend and no seasonal
    cycle, and these are monthly fields, so gamma on the raw series is largely
    measuring agreement of the annual cycle. Both treatments are always computed.
    """
    a = np.asarray(a, dtype=float)
    out = a.copy()
    for phase in range(period):
        sel = slice(phase, None, period)
        out[sel] = a[sel] - a[sel].mean(axis=0, keepdims=True)
    return out


def _loo_means(members):
    """(n_members, T) -> (n_members, T) of the nan-aware mean over the others."""
    valid = ~np.isnan(members)
    total = np.nansum(members, axis=0)
    count = valid.sum(axis=0)
    num = total - np.where(valid, np.nan_to_num(members, nan=0.0), 0.0)
    den = count - valid
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return np.nan_to_num(out, nan=0.0)


def gamma_point(members, target, level, param=None, held_out=None,
                n_perm=N_PERM, seed=SEED, post=None):
    """Observation- and model-side gamma at one grid point.

    ``members`` is ``(n_members, T)`` with NaNs intact so the leave-one-out means
    can skip them; ``target`` is the ``(T,)`` observation series, zero-filled.

    The model side forms gamma per held-out member and averages, matching the
    lambda convention -- averaging I and H first and dividing once is a different
    number. The observation side normalises by H of the observations and the
    model side by H of the held-out member, so each gamma is the fraction of
    *its own* prediction target's uncertainty that is accounted for. Using one
    side's denominator for both is what produced gamma = 121.

    ``post`` is applied to each 1-D series *after* the leave-one-out and ensemble
    means are formed -- ``deseason`` for the anomaly treatment. The order is not
    cosmetic: members 10..19 are NaN for all of calendar 1971, so subtracting a
    month-of-year climatology from the raw ensemble first would return NaN for
    every January..December of those members and delete them from the mean. The
    means are nan-aware, so they must come first and the anomalies second.
    """
    filled = np.nan_to_num(members, nan=0.0)
    loo = _loo_means(members)
    n_members = members.shape[0]
    if held_out is None:
        held_out = np.unique(np.linspace(0, n_members - 1, LOO_MEMBERS)
                             .round().astype(int))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        signal = np.nan_to_num(np.nanmean(members, axis=0), nan=0.0)

    if post is not None:
        signal, target = post(signal), post(target)
        filled = np.stack([post(filled[i]) for i in range(n_members)])
        loo = np.stack([post(loo[i]) for i in range(n_members)])

    obs_side = gamma_pair(signal, target, level, param, n_perm=n_perm, seed=seed)

    # The model-side null is measured on one member and reused for the rest, as
    # `point_metrics` does: it costs n_perm estimator calls per member otherwise,
    # and the floor is a property of the estimator at this T and dimension rather
    # than of the member. The middle held-out member stands in for all of them.
    rows = [gamma_pair(loo[i], filled[i], level, param,
                       n_perm=n_perm if i == held_out[len(held_out) // 2] else 0,
                       seed=seed + 1)
            for i in held_out]
    nulls = [r["I_null"] for r in rows if "I_null" in r]
    I_m_null = nulls[0] if nulls else np.nan

    return dict(
        I_o=obs_side["I"], H_o=obs_side["H"], gamma_o=obs_side["gamma"],
        I_o_null=obs_side.get("I_null", np.nan),
        gamma_o_excess=obs_side.get("gamma_excess", np.nan),
        I_m=float(np.mean([r["I"] for r in rows])),
        H_m=float(np.mean([r["H"] for r in rows])),
        gamma_m=float(np.nanmean([r["gamma"] for r in rows])),
        I_m_null=I_m_null,
        gamma_m_excess=float(np.nanmean(
            [(r["I"] - I_m_null) / r["H"] if r["H"] > H_FLOOR else np.nan
             for r in rows])) if nulls else np.nan,
    )


def _gamma_row(task):
    """One latitude row, every level and treatment. Row slices travel in the
    payload for the same reason ``_do_row`` does it: the full ensemble is 385 MB
    and macOS spawns rather than forks."""
    lat, F_row, G_row, kw = task          # F_row: (N, T, n_lon)  G_row: (T, n_lon)
    n_lon = G_row.shape[1]
    levels, treatments = kw["levels"], kw["treatments"]

    post = {"as-is": None, "deseasonalised": deseason}

    out = {(level, treatment): {field: np.full(n_lon, np.nan)
                                for field in GAMMA_FIELDS}
           for level in levels for treatment in treatments}

    held_out = np.unique(np.linspace(0, F_row.shape[0] - 1, LOO_MEMBERS)
                         .round().astype(int))
    for lon in range(n_lon):
        for treatment in treatments:
            for level in levels:
                res = gamma_point(F_row[:, :, lon], G_row[:, lon], level,
                                  param=kw["params"].get(level),
                                  held_out=held_out, n_perm=kw["n_perm"],
                                  seed=kw["seed"] + lat * 1000 + lon,
                                  post=post[treatment])
                for field in GAMMA_FIELDS:
                    out[(level, treatment)][field][lon] = res[field]
    return lat, out


def gamma_grid(ensemble, observation, levels=None, treatments=GAMMA_TREATMENTS,
               params=None, n_perm=N_PERM, seed=SEED, processes=None, progress=True):
    """Run the whole gamma ladder over the lat/lon grid, both treatments.

    Returns
    -------
    dict
        ``result[level][treatment][field]`` -> ``(n_lat, n_lon)``, with ``field``
        drawn from ``GAMMA_FIELDS``. ``result["params"][level]`` records which
        free parameter value was used.
    """
    levels = list(GAMMA_LADDER) if levels is None else list(levels)
    params = dict(params or {})
    for level in levels:
        spec = GAMMA_LADDER[level]
        params.setdefault(level, spec["values"][spec["default_idx"]])

    F = ensemble.psl.to_numpy()
    G = np.nan_to_num(observation.psl.to_numpy(), nan=0.0)
    n_lat, n_lon = G.shape[1], G.shape[2]

    result = {"params": params, "levels": levels, "treatments": list(treatments)}
    for level in levels:
        result[level] = {t: {field: np.full((n_lat, n_lon), np.nan)
                             for field in GAMMA_FIELDS} for t in treatments}

    kwargs = dict(levels=levels, treatments=list(treatments), params=params,
                  n_perm=n_perm, seed=seed)
    tasks = [(lat, F[:, :, lat, :], G[:, lat, :], kwargs) for lat in range(n_lat)]

    if processes == 1:
        it, pool = map(_gamma_row, tasks), None
    else:
        pool = Pool(processes=processes)
        it = pool.imap_unordered(_gamma_row, tasks)
    try:
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(it, total=n_lat, desc="gamma ladder (lat rows)")
            except ImportError:
                pass
        for lat, out in it:
            for (level, treatment), fields in out.items():
                for field, arr in fields.items():
                    result[level][treatment][field][lat, :] = arr
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return result


def save_gamma_cache(path, result):
    flat = {"__levels": np.array(result["levels"]),
            "__treatments": np.array(result["treatments"]),
            "__params": np.array([result["params"][l] for l in result["levels"]],
                                 dtype=float)}
    for level in result["levels"]:
        for treatment in result["treatments"]:
            for field, arr in result[level][treatment].items():
                flat["%s|%s|%s" % (level, treatment, field)] = arr
    np.savez_compressed(path, **flat)


def load_gamma_cache(path):
    z = np.load(path, allow_pickle=False)
    levels = [str(x) for x in z["__levels"]]
    treatments = [str(x) for x in z["__treatments"]]
    result = {"levels": levels, "treatments": treatments,
              "params": dict(zip(levels, z["__params"]))}
    for level in levels:
        result[level] = {t: {field: z["%s|%s|%s" % (level, t, field)]
                             for field in GAMMA_FIELDS} for t in treatments}
    return result


def area_weighted_fraction(field, lats, threshold=1.0):
    """Fraction of the globe where ``field > threshold``, cos(lat) area-weighted.

    The published percentages are area-weighted: rho gives 72.1% weighted vs
    66.4% unweighted, and 72.1% is the published figure.
    """
    w = np.cos(np.deg2rad(np.asarray(lats)))[:, None] * np.ones(field.shape[1])
    good = np.isfinite(field)
    return float(np.sum(w[good] * (field[good] > threshold)) / np.sum(w[good]))
