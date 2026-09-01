"""Checks `mi_vectorized` against the thing it replaces: infomeasure 0.6.2 itself.

    python test_mi_vectorized.py            # correctness, ~1 min
    python test_mi_vectorized.py --bench    # + a timing ladder vs mi_pairwise
    python test_mi_vectorized.py --jax      # + the JAX backend against numpy

Every "reference" number here comes from `infomeasure.mutual_information(..., approach=
"metric")`, called the same way `smyle_metrics` and `mi_pairwise` call it. The claim being
tested is equality to floating-point noise, not agreement in distribution -- so the
reference is always given data that has ALREADY been copula-transformed and dithered, and
`noise_level=0`, which removes infomeasure's internal RNG from the comparison.

The tie test is the one that matters. On copula ranks the max-norm distances are massively
degenerate, so KSG's answer is decided entirely by how `<` and `<=` are applied at the
neighbourhood boundary. A reimplementation that got the strictness or the conditional
self-subtraction wrong would still pass on continuous data and fail here.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SCRIPTS = os.environ.get(
    "SNP_SCRIPTS",
    "/oak/stanford/groups/cyaolai/JasperChen/Research/SNP/information-snp/scripts")
for p in (HERE, REPO_SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import mi_vectorized as V                                            # noqa: E402

try:                       # absent from the JAX environments; --jax alone still runs
    import infomeasure
except ImportError:
    infomeasure = None

TOL = 1e-11
FAILURES = []


def check(name, got, want, tol=TOL):
    got, want = np.asarray(got, float), np.asarray(want, float)
    finite = np.isfinite(got) & np.isfinite(want)
    same_nonfinite = np.array_equal(np.isfinite(got), np.isfinite(want))
    err = np.abs(got[finite] - want[finite]).max() if finite.any() else 0.0
    ok = same_nonfinite and err <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<58s} max|diff| = {err:.3e}")
    if not ok:
        FAILURES.append(name)
    return ok


def ref_mi(x, y, k):
    """infomeasure's answer on already-prepared data, with its own dither switched off."""
    return infomeasure.mutual_information(x, y, approach="metric", k=k, noise_level=0.0)


# --------------------------------------------------------------------------------------

def test_continuous(rng):
    print("\ncontinuous data, no ties -- the easy case")
    for T in (12, 25, 48, 120):
        for k in (1, 3, 4, 7):
            x = rng.standard_normal(T)
            y = 0.7 * x + rng.standard_normal(T)
            got = V.mi_batch(x, y, k=k, copula=False, noise_level=0)
            check(f"T={T:<4d} k={k}", got, ref_mi(x, y, k))


def test_copula_ties(rng):
    print("\ncopula ranks, massively tied distances -- the case that discriminates")
    for T in (13, 48, 60):
        for k in (1, 4, 8):
            x = rng.standard_normal(T)
            y = 0.5 * x + rng.standard_normal(T)
            xr = V.copula_rank(x)
            yr = V.copula_rank(y)
            n_dist = np.unique(np.abs(xr[:, None] - xr[None, :])).size
            got = V.mi_batch(xr, yr, k=k, copula=False, noise_level=0)
            check(f"T={T:<4d} k={k}  ({n_dist} distinct marginal distances)",
                  got, ref_mi(xr, yr, k))


def test_duplicates(rng):
    print("\nduplicated joint points -- forces eps == 0, the conditional self-subtraction")
    for T, rep in ((24, 3), (40, 5)):
        base = rng.standard_normal(T // rep + 1)
        x = np.repeat(base, rep)[:T]
        y = np.repeat(base * 0.5, rep)[:T]         # exact duplicates in the JOINT space
        got = V.mi_batch(x, y, k=4, copula=False, noise_level=0)
        check(f"T={T} with {rep}-fold repeats", got, ref_mi(x, y, 4))


def test_batching(rng):
    print("\nbatching, broadcasting, chunking, NaN")
    T, B = 40, 37
    X = rng.standard_normal((B, T))
    Y = 0.6 * X + rng.standard_normal((B, T))
    want = np.array([ref_mi(X[b], Y[b], 4) for b in range(B)])
    check("(B, T) batch vs a loop over infomeasure",
          V.mi_batch(X, Y, k=4, copula=False, noise_level=0), want)

    # tiny max_bytes forces many chunks; the answer must not move
    check("same, forced into ~20 chunks",
          V.mi_batch(X, Y, k=4, copula=False, noise_level=0, max_bytes=2 * T * T * 8 * 4),
          want)
    check("same, n_jobs=4 over chunks",
          V.mi_batch(X, Y, k=4, copula=False, noise_level=0,
                     max_bytes=2 * T * T * 8 * 4, n_jobs=4), want)

    # a 3-D leading shape must round-trip
    X3 = X[:6].reshape(2, 3, T)
    Y3 = Y[:6].reshape(2, 3, T)
    check("(2, 3, T) leading shape preserved",
          V.mi_batch(X3, Y3, k=4, copula=False, noise_level=0),
          want[:6].reshape(2, 3))

    # broadcasting one y against many x
    check("broadcast (B, T) against (T,)",
          V.mi_batch(X, Y[0], k=4, copula=False, noise_level=0),
          np.array([ref_mi(X[b], Y[0], 4) for b in range(B)]))

    # NaN in one series -> NaN out, neighbours unaffected
    Xn = X.copy()
    Xn[5, 3] = np.nan
    got = V.mi_batch(Xn, Y, k=4, copula=False, noise_level=0)
    ok = np.isnan(got[5]) and np.allclose(np.delete(got, 5), np.delete(want, 5), atol=TOL)
    print(f"  [{'ok ' if ok else 'FAIL'}] NaN series isolated, others untouched")
    if not ok:
        FAILURES.append("NaN isolation")


def test_symmetry(rng):
    print("\nsymmetry and the Gaussian limit")
    T = 200
    X = rng.standard_normal((50, T))
    Y = 0.8 * X + rng.standard_normal((50, T))
    a = V.mi_batch(X, Y, k=4, copula=False, noise_level=0)
    b = V.mi_batch(Y, X, k=4, copula=False, noise_level=0)
    check("I(x;y) == I(y;x) exactly", a, b, tol=0.0)

    # I = -0.5 ln(1 - r^2) nats for a bivariate normal; KSG at n=4000 gets ~2 decimals
    for r in (0.3, 0.6, 0.9):
        n = 4000
        x = rng.standard_normal(n)
        y = r * x + np.sqrt(1 - r * r) * rng.standard_normal(n)
        got = V.mi_batch(x, y, k=4, copula=False, noise_level=0)
        want = -0.5 * np.log(1 - r * r)
        ok = abs(got - want) < 0.02
        print(f"  [{'ok ' if ok else 'FAIL'}] Gaussian r={r}: got {got:.4f}, "
              f"analytic {want:.4f}")
        if not ok:
            FAILURES.append(f"gaussian r={r}")


def test_wrappers(rng):
    print("\ncube wrappers vs infomeasure, on a small (N, T, lat, lon) cube")
    N, T, LA, LO = 6, 30, 3, 4
    sig = rng.standard_normal((1, T, LA, LO))
    F = sig + 1.2 * rng.standard_normal((N, T, LA, LO))
    o = (sig[0] + 1.0 * rng.standard_normal((T, LA, LO)))

    # -- mi_member_vs_member: rebuild the reference the way mi_pairwise.MI_F_pairwise does
    R, _, _, _, C = V._prep_cube(F, True, 1e-10, 0)                 # (N, C, T)
    ii, jj = np.triu_indices(N, 1)
    want = np.empty((N, N, C))
    for a, b in zip(ii, jj):
        for c in range(C):
            want[a, b, c] = want[b, a, c] = ref_mi(R[a, c], R[b, c], 4)
    want[np.diag_indices(N)] = np.inf
    got = V.mi_member_vs_member(F, k=4)
    check("mi_member_vs_member  (N, N, lat, lon)", got, want.reshape(N, N, LA, LO))

    # the same numbers must come out of the existing joblib implementation
    try:
        import mi_pairwise
        ref = mi_pairwise.MI_F_pairwise(F, k=4, use_copula=True, noise_level=1e-10,
                                        seed=0, n_jobs=1)
        check("mi_member_vs_member == mi_pairwise.MI_F_pairwise (same seed)", got, ref)
    except ImportError:
        print("  [skip] mi_pairwise not importable")

    # -- mi_member_vs_obs
    Ro = V._prep_field(o, True, 1e-10, 1)                            # (C, T)
    want = np.array([[ref_mi(R[n, c], Ro[c], 4) for c in range(C)] for n in range(N)])
    check("mi_member_vs_obs       (N, lat, lon)",
          V.mi_member_vs_obs(F, o, k=4), want.reshape(N, LA, LO))
    check("mi_member_vs_obs accepts the notebook's tiled obs",
          V.mi_member_vs_obs(F, np.broadcast_to(o, (N,) + o.shape), k=4),
          want.reshape(N, LA, LO))

    # -- mi_ensmean_vs_obs, the vectorised calc_MI_sG
    s = F.mean(axis=0)
    Rs = V._prep_field(s, True, 1e-10, 0)
    Ro = V._prep_field(o, True, 1e-10, 1)
    want = np.array([ref_mi(Rs[c], Ro[c], 4) for c in range(C)])
    check("mi_ensmean_vs_obs       (lat, lon)",
          V.mi_ensmean_vs_obs(s, o, k=4), want.reshape(LA, LO))

    # -- mi_loomean_vs_member, the vectorised calc_MI_sF
    loo = (F.sum(0, keepdims=True) - F) / (N - 1)
    L, _, _, _, _ = V._prep_cube(loo, True, 1e-10, 1)
    want = np.mean([[ref_mi(L[n, c], R[n, c], 4) for c in range(C)] for n in range(N)],
                   axis=0)
    check("mi_loomean_vs_member       (lat, lon)",
          V.mi_loomean_vs_member(F, k=4), want.reshape(LA, LO))

    # -- mi_loomean_vs_obs, the vectorised calc_MI_sG_LOOavg
    L, _, _, _, _ = V._prep_cube(loo, True, 1e-10, 0)
    Ro = V._prep_field(o, True, 1e-10, 1)
    want = np.mean([[ref_mi(L[n, c], Ro[c], 4) for c in range(C)] for n in range(N)],
                   axis=0)
    check("mi_loomean_vs_obs           (lat, lon)",
          V.mi_loomean_vs_obs(F, o, k=4), want.reshape(LA, LO))


def test_shapes(rng):
    print("\nshape-agnosticism: index series, field, and a 120-month decadal cube")
    N, T = 5, 40
    F1 = rng.standard_normal((N, T))                       # a bare index, no space axes
    out = V.mi_member_vs_member(F1, k=4)
    ok = out.shape == (N, N) and np.isinf(out[0, 0]) and np.allclose(out, out.T)
    print(f"  [{'ok ' if ok else 'FAIL'}] (N, T) index series -> (N, N)")
    if not ok:
        FAILURES.append("index-series shape")

    F3 = rng.standard_normal((4, 24, 2, 3, 2))             # (N, T, lev, lat, lon)
    out = V.mi_member_vs_member(F3, k=4)
    ok = out.shape == (4, 4, 2, 3, 2)
    print(f"  [{'ok ' if ok else 'FAIL'}] (N, T, lev, lat, lon) -> (N, N, lev, lat, lon)")
    if not ok:
        FAILURES.append("5-d cube shape")

    # the decadal cube kind: T = 120 months rather than 48 start dates
    Fd = rng.standard_normal((4, 120, 3, 3))
    out = V.mi_member_vs_member(Fd, k=4)
    ok = out.shape == (4, 4, 3, 3) and np.isfinite(out[0, 1]).all()
    print(f"  [{'ok ' if ok else 'FAIL'}] T=120 monthly decadal cube -> (N, N, lat, lon)")
    if not ok:
        FAILURES.append("decadal cube shape")

    # a land-masked cell must not poison its neighbours
    Fm = rng.standard_normal((4, 30, 2, 2))
    Fm[:, :, 0, 0] = np.nan
    out = V.mi_member_vs_member(Fm, k=4)
    ok = np.isnan(out[0, 1, 0, 0]) and np.isfinite(out[0, 1, 1, 1])
    print(f"  [{'ok ' if ok else 'FAIL'}] NaN cell masked, rest of the field finite")
    if not ok:
        FAILURES.append("masked cell")


def test_jax(rng):
    print("\nJAX backend against numpy")
    if not V.have_jax():
        print("  [skip] jax not importable in this environment")
        return
    T, B = 48, 500
    X = rng.standard_normal((B, T))
    Y = 0.6 * X + rng.standard_normal((B, T))
    want = V.mi_batch(X, Y, k=4, copula=False, noise_level=0)
    check("jax float64 vs numpy",
          V.mi_batch(X, Y, k=4, copula=False, noise_level=0, backend="jax"), want,
          tol=1e-9)
    got32 = V.mi_batch(X, Y, k=4, copula=False, noise_level=0, backend="jax",
                       dtype="float32")
    print(f"  [info] jax float32 vs float64: max|diff| = "
          f"{np.abs(got32 - want).max():.3e} nats")


def bench(rng):
    print("\ntiming: this module vs mi_pairwise on the same cube")
    try:
        import mi_pairwise
    except ImportError:
        mi_pairwise = None

    for N, T, LA, LO in ((10, 48, 6, 8), (20, 48, 8, 12), (12, 120, 6, 8)):
        F = rng.standard_normal((N, T, LA, LO))
        n_prob = N * (N - 1) // 2 * LA * LO
        t0 = time.perf_counter()
        V.mi_member_vs_member(F, k=4)
        t_vec = time.perf_counter() - t0

        line = (f"  N={N:<3d} T={T:<4d} grid={LA}x{LO:<3d} "
                f"{n_prob:>8d} problems | vectorised {t_vec:7.2f} s")
        if mi_pairwise is not None:
            t0 = time.perf_counter()
            mi_pairwise.MI_F_pairwise(F, k=4, n_jobs=1)
            t_ref = time.perf_counter() - t0
            line += f" | mi_pairwise 1 core {t_ref:8.2f} s | speedup {t_ref / t_vec:6.1f}x"
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--jax", action="store_true")
    args = ap.parse_args()

    print(f"infomeasure {getattr(infomeasure, '__version__', 'ABSENT')}"
          f"  numpy {np.__version__}")
    rng = np.random.default_rng(12345)
    if infomeasure is not None:
        test_continuous(rng)
        test_copula_ties(rng)
        test_duplicates(rng)
        test_batching(rng)
        test_symmetry(rng)
        test_wrappers(rng)
    else:
        print("\n[skip] infomeasure absent -- reference comparisons cannot run here")
    test_shapes(rng)
    if args.jax:
        test_jax(rng)
    if args.bench:
        bench(rng)

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES
                  else f"{len(FAILURES)} FAILURES: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
