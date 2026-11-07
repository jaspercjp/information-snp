"""lambda RPC from KSG mutual information on normal scores, with the Eade bootstrap, by row chunks.

    python scripts/ksg_lambda.py run VAR CASE [--rows a:b] [--boot 200]   -> $SNP_OUT/ksg_chunks/...
    python scripts/ksg_lambda.py merge VAR CASE                          -> $SNP_OUT/boot_ksg_VAR_CASE.npz
    python scripts/ksg_lambda.py compare                                 -> figures/slides/lambda-ksg-vs-knuth-*,
                                                                            figures/data/lambda-ksg-comparison.json

Per grid cell, F (N, T) members, o (T,) observations, s the member mean, s_-n the mean of the others:
    lambda_o = lambda(I(o; s))        KSG on the T pairs (o_t, s_t): o never repeated per member
    lambda_m = <lambda(I(f_n; s_-n))>_n  KSG on each member's T pairs (f_nt, s_-n,t), averaged over a
                                      seeded random 16 of the members (all if fewer): the analogue
                                      of rho_m = <corr(f_n, s_-n)>_n (jugaad.rho_measures). Pooling
                                      the N T pairs is wrong for KSG: s_-n,t is nearly the same for
                                      every member at a time t, so the pooled sample is T tight
                                      clusters and KSG overestimates I (lambda_m 0.64 vs rho_m 0.29
                                      on SLP lead 2-4); per member, lambda_o and lambda_m are both
                                      from T pairs, so their small-sample biases match.
    RPC = lambda_o / lambda_m,  lambda = sqrt(1 - exp(-2 I)), I in nats, clipped at 0
Each variable is replaced by its normal scores Phi^-1((rank - 1/2)/n) over its own sample (o and s
over T; f and s_-n over N T) before KSG, so lambda depends only on ranks; then a 1e-9 jitter
breaks the exact ties that bootstrap resampling creates. KSG: Kraskov et al. (2004) estimator 1,
k = 4, max-norm. Bootstrap: boot.draws (time resampled with replacement, whole years for monthly
cases, N - 3 members without), the same draws for every cell; B = 200 draws by default.
Choice and evidence: figures/supp-info/estimator-toys (scripts/estimator_toys.py).
"""
import argparse
import glob
import json
import os
from multiprocessing import get_context

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

import boot
import jugaad as J
from data import OUT, load

K, M_MEMBERS, JITTER = 4, 16, 1e-9
FLOOR = 0.02
CHUNKS = os.path.join(OUT, "ksg_chunks")


def ksg_mi(x, y, k=K):
    """Kraskov-Stoegbauer-Grassberger estimator 1 (nats), max-norm; x, y (n,) distinct points."""
    n = x.size
    xy = np.column_stack([x, y])
    eps = cKDTree(xy).query(xy, k=k + 1, p=np.inf)[0][:, -1]
    r = np.nextafter(eps, 0)                                         # strictly inside eps
    nx = cKDTree(x[:, None]).query_ball_point(x[:, None], r, p=np.inf, return_length=True) - 1
    ny = cKDTree(y[:, None]).query_ball_point(y[:, None], r, p=np.inf, return_length=True) - 1
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def lam(I):
    return float(np.sqrt(1 - np.exp(-2 * max(I, 0.0))))


def ksg_lambda(x, y, rng):
    """lambda from KSG on normal scores of x and y, ties broken by a tiny jitter."""
    xs = J.normal_scores(x) + JITTER * rng.standard_normal(x.size)
    ys = J.normal_scores(y) + JITTER * rng.standard_normal(y.size)
    return lam(ksg_mi(xs, ys))


def cell(f, o, rng, m=M_MEMBERS):
    """(lambda_o, lambda_m) for one cell: f (N, T), o (T,)."""
    N, T = f.shape
    s = f.mean(0)
    s_loo = (f.sum(0, keepdims=True) - f) / (N - 1)
    lo = ksg_lambda(o, s, rng)
    lm = float(np.mean([ksg_lambda(f[n], s_loo[n], rng)
                        for n in rng.choice(N, min(m, N), replace=False)]))
    return lo, lm


_G = {}


def _row(j):
    F, o, draws, seed = _G["F"], _G["o"], _G["draws"], _G["seed"]
    nx = F.shape[-1]
    nb = len(draws)
    po, pm = np.full(nx, np.nan), np.full(nx, np.nan)
    bo, bm = np.full((nb, nx), np.nan, np.float32), np.full((nb, nx), np.nan, np.float32)
    for i in range(nx):
        f, g = F[:, :, j, i], o[:, j, i]
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(g))):
            continue
        rng = np.random.default_rng([seed, j, i])
        po[i], pm[i] = cell(f, g, rng)
        for b, (mi, ti) in enumerate(draws):
            bo[b, i], bm[b, i] = cell(f[mi[:, 0]][:, ti], g[ti], rng)
    return po, pm, bo, bm


def chunk_path(var, case, a, b):
    return os.path.join(CHUNKS, f"{var}_{case}_rows{a:03d}-{b:03d}.npz")


def run(var, case, rows, nboot, seed):
    F, o, _, lats, lons = load(var, case)
    N, T = F.shape[:2]
    a, b = rows if rows else (0, len(lats))
    draws = list(boot.draws(var, case, N, T, nboot, seed))           # the same for every cell
    _G.update(F=F, o=o, draws=draws, seed=seed)
    procs = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    with get_context("fork").Pool(procs) as pool:
        out = pool.map(_row, range(a, b), chunksize=1)
    os.makedirs(CHUNKS, exist_ok=True)
    np.savez_compressed(chunk_path(var, case, a, b), rows=np.arange(a, b),
                        lam_o=np.stack([r[0] for r in out]), lam_m=np.stack([r[1] for r in out]),
                        B_lam_o=np.stack([r[2] for r in out], axis=1),
                        B_lam_m=np.stack([r[3] for r in out], axis=1), N=N, T=T)
    print(f"-> {chunk_path(var, case, a, b)}", flush=True)


def merge(var, case):
    F, o, _, lats, lons = load(var, case)
    parts = sorted(glob.glob(os.path.join(CHUNKS, f"{var}_{case}_rows*.npz")))
    z = [np.load(p) for p in parts]
    rows = np.concatenate([q["rows"] for q in z])
    if not np.array_equal(np.sort(rows), np.arange(len(lats))):
        raise ValueError(f"{var} {case}: chunks cover rows {rows.min()}-{rows.max()}, "
                         f"{rows.size} of {len(lats)}")
    order = np.argsort(rows)
    cat = lambda k, ax: np.concatenate([q[k] for q in z], axis=ax).take(order, axis=ax)  # noqa
    lam_o, lam_m = cat("lam_o", 0), cat("lam_m", 0)
    B_o, B_m = cat("B_lam_o", 1), cat("B_lam_m", 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rpc = np.where(lam_m > FLOOR, lam_o / lam_m, np.nan)
        B = np.where(B_m > FLOOR, B_o / B_m, np.nan).astype(np.float32)
    path = os.path.join(OUT, f"boot_ksg_{var}_{case}.npz")
    np.savez_compressed(path, lats=lats, lons=lons, N=int(z[0]["N"]), T=int(z[0]["T"]),
                        lam_o=lam_o, lam_m=lam_m, rpc=rpc, B_lam=B, B_lam_o=B_o, B_lam_m=B_m)
    print(f"-> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("run", "merge", "compare"))
    ap.add_argument("var", nargs="?")
    ap.add_argument("case", nargs="?")
    ap.add_argument("--rows", default=None, help="a:b, latitude rows of this chunk")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2014)
    a = ap.parse_args()
    if a.what == "run":
        rows = tuple(int(v) for v in a.rows.split(":")) if a.rows else None
        run(a.var, a.case, rows, a.boot, a.seed)
    elif a.what == "merge":
        merge(a.var, a.case)
    else:
        import plots
        plots.ksg_comparison()


if __name__ == "__main__":
    main()
