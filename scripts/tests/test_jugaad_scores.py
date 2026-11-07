"""Checks of jugaad.cell(..., scores=True): lambda from Knuth bins on normal scores.

    python scripts/tests/test_jugaad_scores.py      (or pytest)
"""
import os
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jugaad as J                                                   # noqa: E402


def ens(N=12, T=40, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(T)
    F = 0.6 * s + rng.standard_normal((N, T))
    o = 0.5 * s + rng.standard_normal(T)
    return F, o


def test_normal_scores_are_rank_quantiles_and_keep_nan():
    v = np.array([3.0, -1.0, 10.0, 3.0, np.nan, 0.5])
    z = J.normal_scores(v)
    ok = np.isfinite(v)
    assert np.allclose(z[ok], ndtri((rankdata(v[ok]) - 0.5) / ok.sum()))
    assert np.isnan(z[4]) and z[0] == z[3]                          # ties share a score
    assert J.normal_scores(np.ones((2, 3))).shape == (2, 3)


def test_scores_make_lambda_o_invariant_to_a_monotonic_obs_distortion():
    F, o = ens()
    S = J.loo_means(F)
    kw = dict(rule="knuth", tiled=False, loo_size="T", alts=(1.0,))
    base = J.cell(F, o, S, p=S, scores=True, **kw)
    bent = J.cell(F, np.expm1(3 * o), S, p=S, scores=True, **kw)
    assert np.allclose(base[0], bent[0]) and np.allclose(base[1], bent[1])
    raw = J.cell(F, o, S, p=S, **kw)[0], J.cell(F, np.expm1(3 * o), S, p=S, **kw)[0]
    assert not np.allclose(*raw)                                     # the raw estimator moves


def test_default_is_unchanged():
    F, o = ens(seed=3)
    S = J.loo_means(F)
    kw = dict(rule="knuth", tiled=False, loo_size="T")
    a, b = J.cell(F, o, S, p=S, **kw), J.cell(F, o, S, p=S, scores=False, **kw)
    assert all(np.array_equal(x, y) for x, y in zip(a[:2], b[:2])) and a[2] == b[2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
