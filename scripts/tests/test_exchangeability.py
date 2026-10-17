"""Checks of exchangeability.py: construction rules, vectorisation and seeding.

    python scripts/tests/test_exchangeability.py      (or pytest)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import exchangeability as E                                          # noqa: E402


def ens(R=8, T=40, P=3, seed=0, beta=0.8):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((T, P))
    return beta * s + rng.standard_normal((R, T, P))


def test_bins_rule():
    assert [E.n_bins(T) for T in (8, 27, 33, 51, 120, 1000)] == [3, 3, 3, 4, 5, 10]
    assert E.bin_choices(51) == (2, 4, 8) and E.bin_choices(120) == (3, 5, 10)


def test_loo_excludes_member():
    x = ens()
    X = E.loo_mean(x)
    for r in range(x.shape[0]):
        assert np.allclose(X[r], np.delete(x, r, 0).mean(0))


def test_edges_equiprobable_and_fixed():
    x = ens(R=10, T=60, P=2)
    X = E.loo_mean(x)
    e = E.pooled_edges(X, 4)
    c = E.codes(X, e)
    for p in range(2):                                               # pooled counts equal
        assert np.all(np.bincount(c[..., p].ravel(), minlength=4) == 150)
    # a label permutation moves codes, never re-bins: the pooled histogram is unchanged
    tab = E.perm_tables(np.random.default_rng(1), 3, 10, 60)
    j = np.moveaxis(c, -1, 0)
    pj = E._permute(j, tab)                                          # (3, P, R, T)
    for k in range(3):
        for t in range(60):
            assert np.array_equal(np.sort(pj[k, :, :, t], axis=1), np.sort(j[:, :, t], axis=1))


def test_permutation_keeps_pairs_and_blocks():
    R, T = 6, 20
    tab = E.perm_tables(np.random.default_rng(2), 4, R, T, block=5)
    assert tab.shape == (4, T, R)
    for k in range(4):
        for t in range(T):
            assert sorted(tab[k, t]) == list(range(R))
        for b in range(0, T, 5):                                     # one permutation per block
            assert np.all(tab[k, b:b + 5] == tab[k, b])
    a = np.arange(R * T).reshape(1, R, T)                            # pair ids
    pa = E._permute(a, tab)[:, 0]
    assert np.all(pa % T == np.arange(T))                            # a pair stays at its time


def test_mi_counts():
    B = 4
    eye = np.eye(B)[None] * 10                                       # perfect dependence
    assert np.isclose(E.mi_counts(eye)[0], 2.0)                      # log2 4 bits
    assert np.isclose(E.mi_counts(np.ones((1, B, B)))[0], 0.0)
    rng = np.random.default_rng(3)
    c = rng.integers(0, 5, (B, B))
    p = c / c.sum()
    px, py = p.sum(1), p.sum(0)
    ref = sum(p[i, k] * np.log2(p[i, k] / (px[i] * py[k]))
              for i in range(B) for k in range(B) if p[i, k] > 0)
    assert np.isclose(E.mi_counts(c[None])[0], ref)


def test_cond_and_pool_by_hand():
    x = ens(R=5, T=30, P=1, seed=4)
    B = 3
    res = E.exchangeability_test(x, B=B, n_sub=20, n_perm=5, seed=0)
    X, Y = E.loo_mean(x[..., 0]), x[..., 0]
    eX = np.quantile(X.ravel(), [1 / 3, 2 / 3])
    eY = np.quantile(Y.ravel(), [1 / 3, 2 / 3])
    cx, cy = np.searchsorted(eX, X, "right"), np.searchsorted(eY, Y, "right")

    def mi(a, b):
        c = np.zeros((B, B))
        np.add.at(c, (a, b), 1)
        return E.mi_counts(c[None])[0]
    assert np.isclose(res["I_cond"]["obs"][0], np.mean([mi(cx[r], cy[r]) for r in range(5)]))
    assert np.isclose(res["I_pool"]["obs"][0], mi(cx.ravel(), cy.ravel()))
    picks = np.random.default_rng(0).integers(0, 5, (20, 30))       # the first draw of the stream
    t = np.arange(30)
    assert np.isclose(res["I_pool_T"]["obs"][0],
                      np.mean([mi(cx[picks[k], t], cy[picks[k], t]) for k in range(20)]))
    H = lambda a: E.entropy_counts(np.bincount(a, minlength=B * B)[None])[0]   # noqa: E731
    j = cx * B + cy
    assert np.isclose(res["I_MXY"]["obs"][0], H(j.ravel()) - np.mean([H(j[r]) for r in range(5)]))


def test_vectorised_equals_loop_and_seeded():
    x = ens(R=7, T=33, P=4, seed=5)
    for est in ("bins", "copula"):
        full = E.exchangeability_test(x, n_sub=30, n_perm=40, seed=7, estimator=est)
        small = E.exchangeability_test(x, n_sub=30, n_perm=40, seed=7, estimator=est,
                                       max_elems=1)                  # one permutation a chunk
        for p in range(4):
            one = E.exchangeability_test(x[..., p], n_sub=30, n_perm=40, seed=7, estimator=est)
            for s in E.STATS:
                for k in one[s]:
                    assert np.allclose(full[s][k][p], one[s][k]), (est, s, k)
                    assert np.allclose(small[s][k][p], one[s][k]), (est, s, k)


def test_copula_ranks_pooled():
    x = ens(R=6, T=25, P=1, seed=6)
    xf = x.reshape(6, 25, 1)
    z = E.normal_scores(xf)
    assert np.allclose(np.sort(z.ravel()), np.sort(E.ndtri((np.arange(150) + 0.5) / 150)))
    # per-member ranking would give each member the same set of scores; pooled does not
    assert not np.allclose(np.sort(z[0, :, 0]), np.sort(z[1, :, 0]))


def test_obs_never_tiled():
    x = ens(R=6, T=40, P=2, seed=8)
    g = np.random.default_rng(9).standard_normal((40, 2))
    res = E.obs_mi(g, x, B=3, n_perm=10, seed=0)
    X = E.loo_mean(x)
    for p in range(2):
        eg = np.quantile(g[:, p], [1 / 3, 2 / 3])                   # g's own T values
        eX = np.quantile(X[..., p].ravel(), [1 / 3, 2 / 3])
        cg = np.searchsorted(eg, g[:, p], "right")
        vals = []
        for r in range(6):
            c = np.zeros((3, 3))
            np.add.at(c, (cg, np.searchsorted(eX, X[r, :, p], "right")), 1)
            assert c.sum() == 40                                     # T pairs, not R T
            vals.append(E.mi_counts(c[None])[0])
        assert np.isclose(res["obs"][p], np.mean(vals))


def test_rejects_nonfinite():
    x = ens()
    x[0, 0, 0] = np.nan
    try:
        E.exchangeability_test(x, n_perm=2, n_sub=2)
    except ValueError:
        return
    raise AssertionError("NaN accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
