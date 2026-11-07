"""Checks of monotonic_demo.py: the exact rho curve and the rank estimator's invariance.

    python scripts/tests/test_monotonic_demo.py      (or pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import monotonic_demo as M                                           # noqa: E402


def test_g_is_identity_at_zero_and_increasing():
    u = np.linspace(-3, 3, 101)
    assert np.array_equal(M.g(u, 0.0), u)
    for a in (0.5, 1.0, 2.0):
        assert np.all(np.diff(M.g(u, a)) > 0)
        assert np.allclose(M.g(u, 1e-8), u, atol=1e-6)


def test_rho_exact_matches_simulation():
    rng = np.random.default_rng(0)
    n, r = 1_000_000, 0.8
    x = rng.standard_normal(n)
    u = r * x + np.sqrt(1 - r * r) * rng.standard_normal(n)
    for a in (0.0, 0.5, 1.0):
        assert abs(np.corrcoef(x, M.g(u, a))[0, 1] - M.rho_exact(a, r)) < 3e-3


def test_rank_lambda_is_the_same_at_every_a():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(51)
    u = 0.8 * x + 0.6 * rng.standard_normal(51)
    vals = [M.lam_rank(x, M.g(u, a), 4) for a in M.A]
    assert np.allclose(vals, vals[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
