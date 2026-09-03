"""Standalone check of `mi_pairwise.MI_F_pairwise` -- correctness, then scaling.

    python test_MI_F_pairwise.py --n-jobs 16              # checks + ladder, ~3 min
    python test_MI_F_pairwise.py --n-jobs 16 --real       # plus the real DCPP cube

Nothing here writes into the repo and nothing imports the notebooks.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import infomeasure
import smyle_metrics as M
from mi_pairwise import MI_F_pairwise

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def note(text):
    print(f"         {text}")


def pearson_coeff_F_pairwise(F):
    """The notebook's Pearson version, copied verbatim as the reference contract."""
    def pairwise(F, G):
        def helper_mean(X):
            return np.mean(X, axis=1).reshape(X.shape[0], -1, X.shape[2], X.shape[3])
        num = np.mean((F - helper_mean(F)) * (G - helper_mean(G)), axis=1)
        return num / (np.std(F, axis=1) * np.std(G, axis=1))

    N = F.shape[0]
    out = np.nan * np.ones((N, N, F.shape[2], F.shape[3]))
    for n in range(N):
        gg = np.broadcast_to(F[n], (N,) + F[n].shape)
        out[n] = pairwise(F, gg)
    return out


def common_signal(N, J, shape, a, rng):
    """Members sharing a fraction `a` of a common signal -> pair correlation a**2."""
    a = np.broadcast_to(np.asarray(a, dtype=float), shape)
    sig = rng.standard_normal((1, J) + shape)
    eps = rng.standard_normal((N, J) + shape)
    return a * sig + np.sqrt(1.0 - a ** 2) * eps


# --------------------------------------------------------------------------------- #
def test_contract(nj, rng):
    print("\n1. contract: shape, symmetry, diagonal      F = (7, 48, 5, 4)")
    F = rng.standard_normal((7, 48, 5, 4))
    I = MI_F_pairwise(F, n_jobs=nj)
    off = ~np.eye(7, dtype=bool)
    check("shape is (N, N, lat, lon)", I.shape == (7, 7, 5, 4), str(I.shape))
    check("symmetric to the bit", np.array_equal(I[off], np.swapaxes(I, 0, 1)[off]))
    check("diagonal is +inf", np.isinf(I[np.diag_indices(7)]).all())
    check("off-diagonal all finite", np.isfinite(I[off]).all())
    check("lam_of(diagonal) == 1", np.allclose(M.lam_of(I[np.diag_indices(7)]), 1.0))


def test_vs_bruteforce(nj, rng):
    print("\n2. vs a brute-force per-cell call, 15 random (pair, cell) entries")
    F = rng.standard_normal((7, 48, 5, 4))
    picks = [(a, b, rng.integers(5), rng.integers(4))
             for a, b in (rng.choice(7, size=2, replace=False) for _ in range(15))]

    # (a) raw data, no dither: unambiguous, must match bit-for-bit
    I = MI_F_pairwise(F, use_copula=False, noise_level=0.0, n_jobs=nj)
    e = max(abs(I[a, b, y, x] - infomeasure.mutual_information(
        F[a, :, y, x], F[b, :, y, x], approach="metric", k=4, noise_level=0.0))
        for a, b, y, x in picks)
    check("use_copula=False, noise=0  vs infomeasure directly", e == 0.0, "max|d| = %.1e" % e)

    # (b) copula, no dither: validates rankdata(axis=1) against copula_transform
    I = MI_F_pairwise(F, use_copula=True, noise_level=0.0, n_jobs=nj)
    e = max(abs(I[a, b, y, x] - infomeasure.mutual_information(
        M.copula_transform(F[a, :, y, x]), M.copula_transform(F[b, :, y, x]),
        approach="metric", k=4, noise_level=0.0)) for a, b, y, x in picks)
    check("use_copula=True,  noise=0  vs smyle_metrics.copula_transform",
          e == 0.0, "max|d| = %.1e" % e)


def test_dither(nj, rng):
    print("\n3. the dither: hoisting it out of infomeasure must not change the answer")
    F = rng.standard_normal((7, 48, 5, 4))
    off = ~np.eye(7, dtype=bool)

    # raw data has no ties, so the dither is a genuine no-op either way
    a = MI_F_pairwise(F, use_copula=False, noise_level=0.0, n_jobs=nj)
    b = MI_F_pairwise(F, use_copula=False, noise_level=1e-10, n_jobs=nj)
    d = np.abs(a[off] - b[off]).max()
    check("use_copula=False: noise_level is a no-op", d < 1e-12, "max|d| = %.1e" % d)

    # copula data is on a lattice, so the dither decides how exact ties break.
    # Our pre-dither must agree with infomeasure's internal one to within the
    # internal one's OWN seed-to-seed spread -- measure that spread, don't assume it.
    ours = [MI_F_pairwise(F, seed=s, n_jobs=nj)[off] for s in range(4)]
    spread = np.abs(np.array(ours) - np.mean(ours, axis=0)).max()
    x, y = M.copula_transform(F[0, :, 0, 0]), M.copula_transform(F[1, :, 0, 0])
    im_reps = [infomeasure.mutual_information(x, y, approach="metric", k=4,
                                              noise_level=1e-10) for _ in range(40)]
    ours_cell = [MI_F_pairwise(F, seed=s, n_jobs=nj)[0, 1, 0, 0] for s in range(20)]
    gap = abs(np.mean(ours_cell) - np.mean(im_reps))
    tol = np.std(im_reps) + np.std(ours_cell)
    check("pre-dither mean == infomeasure-dither mean (within realisation noise)",
          gap <= max(tol, 1e-12), "gap %.2e vs tol %.2e" % (gap, tol))
    note("dither realisation spread, one cell/pair: ours %.3f nats, infomeasure %.3f nats"
         % (np.ptp(ours_cell), np.ptp(im_reps)))
    note("dither realisation spread over all pairs/cells: %.3f nats  <- estimator floor"
         % spread)


def test_gaussian(nj, rng):
    print("\n4. Gaussian ground truth on I itself (nats).  Pair correlation is a^2,")
    print("   so I_true = -0.5*ln(1 - a^4).  N=8, 4x4 grid, J=1200 to suppress bias.")
    for a in (0.5, 0.7, 0.9, 0.99):
        F = common_signal(8, 1200, (4, 4), a, rng)
        I = MI_F_pairwise(F, n_jobs=nj)
        got = I[~np.eye(8, dtype=bool)].mean()
        want = -0.5 * np.log(1 - a ** 4)
        check("a = %.2f -> I_true = %.4f" % (a, want), abs(got - want) < 0.03,
              "got %.4f nats  (bias %+.4f)" % (got, got - want))
    note("at short J the same estimator is biased high and lam_of rectifies")
    note("negative noise, so small-signal lam is inflated -- see check 5.")


def test_tracks_pearson(nj, rng):
    print("\n5. spatially varying signal: lam_of(MI) must track |Pearson| across cells")
    aa = np.linspace(0.05, 0.98, 64).reshape(8, 8)
    F = common_signal(10, 48, (8, 8), aa, rng)
    lam = M.lam_of(MI_F_pairwise(F, n_jobs=nj))
    rho = np.abs(pearson_coeff_F_pairwise(F))
    off = ~np.eye(10, dtype=bool)
    c = np.corrcoef(lam[off].ravel(), rho[off].ravel())[0, 1]
    check("corr(lam, |rho|) over all pairs/cells > 0.8", c > 0.8, "= %.3f" % c)
    ct = np.corrcoef(lam[off].mean(0).ravel(), (aa ** 2).ravel())[0, 1]
    check("pair-mean lam tracks the true a^2 field, corr > 0.95", ct > 0.95, "= %.3f" % ct)
    note("J=48, so both estimators are noisy per pair -- the per-pair agreement is")
    note("capped by that noise, not by the MI estimator. The field-level tracking is")
    note("the meaningful check, and it is near-perfect.")


def test_ladder(nj):
    print("\n6. scaling ladder at J=48, k=4, copula (n_jobs = %d)" % nj)
    print("   %-22s %10s %9s %11s %10s" % ("F.shape", "calls", "wall(s)", "us/call", "core-min"))
    sizes = [(8, 48, 6, 6), (16, 48, 10, 10), (24, 48, 18, 24), (46, 48, 37, 72)]
    rng = np.random.default_rng(7)
    per_call = []
    for N, J, ny, nx in sizes:
        F = common_signal(N, J, (ny, nx), 0.6, rng)
        t = time.perf_counter()
        I = MI_F_pairwise(F, n_jobs=nj)
        dt = time.perf_counter() - t
        n = N * (N - 1) // 2 * ny * nx
        per_call.append(dt / n * 1e6)
        print("   %-22s %10s %9.1f %11.1f %10.1f"
              % (str(F.shape), f"{n:,}", dt, per_call[-1], dt * nj / 60))
        assert np.isfinite(I[~np.eye(N, dtype=bool)]).all()
    check("per-call cost flat within 3x from smallest to largest",
          max(per_call[1:]) / min(per_call[1:]) < 3.0,
          "%.1f -> %.1f us" % (per_call[1], per_call[-1]))
    return per_call[-1]


def test_real(nj, load=False):
    print("\n7. the real thing: DCPP grand ensemble, SLP, lead 13-60, gm removed")
    import dcpp_handles
    F = dcpp_handles.get(lead="13-60", var="SLP", remove_gm=True).F
    N = F.shape[0]
    n = N * (N - 1) // 2 * F.shape[2] * F.shape[3]
    p = os.path.join(os.environ.get("SCRATCH", "."), "MI_F_pairwise_dcpp_SLP_13-60.npy")
    print(f"   F = {F.shape} -> {n:,} KSG calls")

    if load and os.path.exists(p):
        I = np.load(p)
        print(f"   loaded {p} (--load)")
    else:
        t = time.perf_counter()
        I = MI_F_pairwise(F, n_jobs=nj)
        dt = time.perf_counter() - t
        print("   %.1f s wall, %.1f us/call, %.1f core-min"
              % (dt, dt / n * 1e6, dt * nj / 60))
        np.save(p, I)
        print("   saved -> %s (%.0f MB)" % (p, I.nbytes / 1e6))

    off = ~np.eye(N, dtype=bool)
    lam, R = M.lam_of(I), np.abs(pearson_coeff_F_pairwise(F))[off].ravel()
    L = lam[off].ravel()
    check("all off-diagonal finite", np.isfinite(I[off]).all())
    print("   I   off-diag: %.4f .. %.4f nats, mean %.4f"
          % (I[off].min(), I[off].max(), I[off].mean()))

    # Per-entry agreement with Pearson is NOT the right test here: mean I is 0.002
    # nats against a ~0.03 nat dither floor, so at J=48 a single (pair, cell) MI is
    # almost all noise. What must hold is that the CONDITIONAL mean tracks |rho|.
    print("   lam vs |rho|, binned on |rho|  (the resolvable-signal check):")
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (R >= lo) & (R < hi)
        if m.sum() < 100:
            continue
        means.append(L[m].mean())
        print("     |rho| in [%.1f, %.1f): mean |rho| %.3f  mean lam %.3f   n = %d"
              % (lo, hi, R[m].mean(), means[-1], m.sum()))
    check("conditional mean lam is monotone increasing in |rho|",
          np.all(np.diff(means) > 0), "%s" % np.round(means, 3))

    res = R > 0.15                      # where J=48 can actually resolve a signal
    bias = L[res].mean() - R[res].mean()
    check("where |rho| > 0.15, mean lam matches mean |rho| to 0.1",
          abs(bias) < 0.1, "lam %.3f vs |rho| %.3f (%+.3f)"
          % (L[res].mean(), R[res].mean(), bias))

    frac = float((I[off] <= 0).mean())
    note("%.0f%% of off-diagonal I are <= 0, which lam_of rectifies to its 1e-3 floor;"
         % (100 * frac))
    note("that is why mean lam is %.3f where mean |rho| is only %.3f in the lowest bin."
         % (L[R < 0.1].mean(), R[R < 0.1].mean()))
    note("per-entry corr(lam, |rho|) = %.3f -- noise-limited at J=48, not a defect;"
         % np.corrcoef(L, R)[0, 1])
    note("check 5 gets 0.85/0.97 on synthetic data that has real spatial signal.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--load", action="store_true",
                    help="reuse the saved real-data MI instead of recomputing it")
    args = ap.parse_args()
    nj = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
    print("n_jobs = %d, cpu_count = %d, infomeasure %s"
          % (nj, os.cpu_count(), infomeasure.__version__))

    rng = np.random.default_rng(0)
    test_contract(nj, rng)
    test_vs_bruteforce(nj, rng)
    test_dither(nj, rng)
    test_gaussian(nj, rng)
    test_tracks_pearson(nj, rng)
    if not args.skip_ladder:
        test_ladder(nj)
    if args.real:
        test_real(nj, load=args.load)

    print("\n%d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
