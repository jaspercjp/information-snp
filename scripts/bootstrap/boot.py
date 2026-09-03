"""One-file member+window bootstrap of lam_o, lam_m, rho_o, rho_m. Run, then combine.

    python boot.py run     --task-id $SLURM_ARRAY_TASK_ID --n-tasks 40 --B 500
    python boot.py combine --B 500

Replaces boot_v3.py + boot_kernels.py + boot_worker.py + boot_combine.py. Everything the
four-file version did that is NOT here was measured to be worthless or worse:

  * the numpy KSG kernel, its chunking (`max_bytes`), backend dispatch and joblib
    threading -- all of it exists to make a six-pass (B,T,T) kernel tolerable. The fused
    kernel below is 2.77x faster and holds a whole problem in registers, so there is
    nothing to chunk or block.  MEASURED: chunk tuning in EITHER direction is slower.
  * the `pruned` and `fixed` kernels: 2.53x and 1.79x, and neither is exact.
  * float64 + noise_level=1e-10: 1.78x slower for a 0.01%-of-spread difference.
  * `n_loo` member subsampling: changes the estimand of the max-over-members form.

Kept because it earns its keep: the draws are regenerated in full by every task from
`--boot-seed` and each task computes only its slice, so results are independent of how the
work is chunked; and `combine` refuses to summarise a partial set.

Contract: `eps_i` is the (k+1)-th smallest max-norm distance INCLUDING the self-distance 0,
counts are strict `<` with self removed exactly when `eps_i > 0`. `NOISE=1e-4` (not 1e-10)
because float32 resolves ~6e-8 near 0.5, so a 1e-10 dither rounds away and reproduces the
`noise_level=0` pathology; 1e-4 is 1e3x the resolution and 1/137 of a rank step.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
from numba import jit, prange
from scipy.special import digamma
from scipy.stats import rankdata

STAGE = os.environ.get("SNP_STAGE", os.path.join(os.environ["SCRATCH"], "snp_boot_stage"))
REPO = os.environ.get("SNP_REPO", "/oak/stanford/groups/cyaolai/JasperChen/Research/SNP/"
                                  "information-snp")
# smyle_metrics comes from $SNP_REPO, inserted ahead of whatever snp_path put on the
# path, because `pearson_coeff_loo_pairwise` currently exists ONLY in the main checkout's
# WORKING TREE -- it is not in HEAD, and HEAD's copy of this module is a different, older
# 259-line file. A run therefore depends on uncommitted code, so the resolved path is
# printed: a bootstrap that silently picked up the other smyle_metrics would produce rho
# numbers that no committed revision can reproduce.
sys.path.insert(0, os.path.join(REPO, "scripts"))
from smyle_metrics import ensemble_SNR_loo, pearson_coeff_loo_pairwise  # noqa: E402
import smyle_metrics as _sm  # noqa: E402

NOISE = 1e-4


def lam_of(I):
    """sqrt(1 - exp(-2I)); the 1e-6 is the notebook's, so lam_of(0) is 1e-3 not 0."""
    return np.sqrt(1 - np.exp(-2 * np.maximum(I, 0)) + 1e-6)


@jit(nopython=True, cache=True, parallel=True, fastmath=False, nogil=True)
def mi(X, Y, k, dig, c0):
    """KSG-1 in nats over a batch of prepared problems. X, Y: (B, T) float32 -> (B,).

    `fastmath=False` is deliberate: every comparison here is a tie decision.
    """
    B, T = X.shape
    out = np.empty(B, dtype=np.float64)
    for b in prange(B):
        x, y = X[b], Y[b]
        best = np.empty(k + 1, dtype=np.float32)
        acc = 0.0
        for i in range(T):
            xi, yi = x[i], y[i]
            for q in range(k + 1):
                best[q] = np.float32(np.inf)
            for j in range(T):                          # eps: keep the k+1 smallest
                a, c = abs(xi - x[j]), abs(yi - y[j])
                d = a if a > c else c
                if d < best[k]:
                    p = k
                    while p > 0 and best[p - 1] > d:
                        best[p] = best[p - 1]
                        p -= 1
                    best[p] = d
            eps = best[k]
            s = 1 if eps > 0 else 0
            nx = ny = 0
            for j in range(T):                          # marginal counts, strict <
                if abs(xi - x[j]) < eps:
                    nx += 1
                if abs(yi - y[j]) < eps:
                    ny += 1
            acc += dig[nx - s + 1] + dig[ny - s + 1]
        out[b] = c0 - acc / T
    return out


def prep(V, seed):
    """Copula rank + dither. `(n, T, C)` -> `(n, C, T)` float32, sample axis last."""
    T = V.shape[1]
    R = rankdata(V, method="average", axis=1) / (T + 1.0)
    R += np.random.default_rng(seed).standard_normal(R.shape) * NOISE
    return np.ascontiguousarray(R.transpose(0, 2, 1), dtype=np.float32)


def run(a):
    F, o = np.load(f"{STAGE}/F.npy"), np.load(f"{STAGE}/o.npy")
    N, T = F.shape[:2]
    C, M, W = int(np.prod(F.shape[2:])), a.n_members, a.n_months
    Fc, oc = F.reshape(N, T, C), o.reshape(T, C)
    A = Fc.sum(0)                                       # gather-free subset totals
    rng = np.random.default_rng(a.boot_seed)
    draws = [(rng.choice(N, M, replace=False), int(rng.integers(0, T - W + 1)))
             for _ in range(a.B)]                       # the FULL list, in every task
    idx = np.array_split(np.arange(a.B), a.n_tasks)[a.task_id]
    with np.errstate(divide="ignore"):
        dig = digamma(np.arange(W + 2, dtype=float))
    c0 = float(digamma(a.k) + digamma(W))
    Io, Im = np.empty((idx.size, C)), np.empty((idx.size, M, C))
    Ro, Rm = np.empty((idx.size, C)), np.empty((idx.size, C))

    print(f"smyle_metrics <- {_sm.__file__}", flush=True)
    for n, b in enumerate(idx):
        m, s = draws[b]
        w = slice(s, s + W)
        raw = Fc[m, w]                                                      # (M, W, C)
        excl = np.setdiff1d(np.arange(N), m)
        loo = (A[w] - Fc[excl, w].sum(0) - raw) / (M - 1)
        L, Rr = prep(loo, a.seed), prep(raw, a.seed + 2)
        O = prep(oc[w][None], a.seed + 1)[0]                                # (C, W)
        Io[n] = mi(L.reshape(-1, W),
                   np.ascontiguousarray(np.broadcast_to(O, (M, C, W)).reshape(-1, W)),
                   a.k, dig, c0).reshape(M, C).mean(0)
        Im[n] = mi(L.reshape(-1, W), Rr.reshape(-1, W), a.k, dig, c0).reshape(M, C)
        F4 = raw.reshape(M, W, C, 1)                    # Pearson: raw, no copula/dither
        Ro[n] = pearson_coeff_loo_pairwise(F4, oc[w].reshape(W, C, 1)).reshape(M, C).mean(0)
        Rm[n] = ensemble_SNR_loo(F4).reshape(M, C).mean(0)
        print(f"[{a.task_id}] {n + 1}/{idx.size}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(f"{a.out_dir}/part_{a.task_id:04d}.npz", rep_index=idx,
                        I_o=Io.reshape((-1,) + F.shape[2:]),
                        I_m=Im.reshape((-1, M) + F.shape[2:]).astype(np.float32),
                        rho_o=Ro.reshape((-1,) + F.shape[2:]),
                        rho_m=Rm.reshape((-1,) + F.shape[2:]),
                        n_members=M, n_months=W, k=a.k, boot_seed=a.boot_seed)


def combine(a):
    d = [np.load(f) for f in sorted(glob.glob(f"{a.parts}/part_*.npz"))]
    if not d:
        raise SystemExit(f"no parts in {a.parts}")
    for key in ("n_members", "n_months", "k", "boot_seed"):
        if len({int(x[key]) for x in d}) > 1:
            raise SystemExit(f"parts disagree on {key} -- runs are mixed in {a.parts}")
    order = np.argsort(np.concatenate([x["rep_index"] for x in d]))
    # One check for both failure modes: a missing replicate and a duplicated one.
    if not np.array_equal(np.concatenate([x["rep_index"] for x in d])[order],
                          np.arange(a.B)):
        raise SystemExit(f"parts do not form exactly replicates 0..{a.B - 1}; "
                         f"rerun the missing array elements or delete stale parts")
    get = lambda key: np.concatenate([x[key] for x in d], axis=0)[order]  # noqa: E731
    lam_o, lam_m = lam_of(get("I_o")), lam_of(get("I_m"))
    rho_o, rho_m = get("rho_o"), get("rho_m")
    lam_m_mean = lam_m.mean(1)
    # rho_m is a correlation and straddles zero, so the ratio is masked, not clamped:
    # clamping the denominator once gave a domain mean of -3.2e9. lam_m is floored at
    # 1e-3 by lam_of, so its ratio needs no mask.
    out = dict(I_o=get("I_o"), lam_o=lam_o, lam_m=lam_m, rho_o=rho_o, rho_m=rho_m,
               lam_m_meanmem=lam_m_mean, lam_m_maxmem=np.nanmax(lam_m, axis=1),
               ratio=lam_o / lam_m_mean, rho_ratio=rho_o / np.where(rho_m > 0, rho_m, np.nan),
               rho_ratio_valid_frac=np.mean(rho_m > 0),
               n_members=int(d[0]["n_members"]), n_months=int(d[0]["n_months"]),
               k=int(d[0]["k"]), boot_seed=int(d[0]["boot_seed"]))
    for name in ("lam_o", "lam_m_meanmem", "rho_o", "rho_m"):
        out[f"{name}_sd"] = out[name].std(0, ddof=1)
        out[f"{name}_ci"] = np.percentile(out[name], [2.5, 97.5], axis=0)
    np.savez_compressed(a.out, **out)
    print(f"{len(d)} parts, B={a.B}, n_members={out['n_members']}, "
          f"n_months={out['n_months']}, k={out['k']}, "
          f"rho_ratio valid on {out['rho_ratio_valid_frac']:.1%}")
    for name in ("lam_o", "lam_m_meanmem", "ratio", "rho_o", "rho_m", "rho_ratio"):
        x = np.nanmean(out[name], axis=tuple(range(1, lam_o.ndim)))       # domain mean
        print(f"  {name:<14} mean {np.nanmean(x):+.4f}  sd {np.nanstd(x, ddof=1):.4f}  "
              f"95% [{np.nanpercentile(x, 2.5):+.4f}, {np.nanpercentile(x, 97.5):+.4f}]")
    print(f"-> {a.out}")


if __name__ == "__main__":
    # Config comes from the environment, not argparse: this only ever runs under Slurm,
    # which already exports it, and SLURM_ARRAY_TASK_ID/COUNT arrive that way for free.
    env = lambda k, d: int(os.environ.get(k, d))                          # noqa: E731
    a = SimpleNamespace(
        task_id=env("SLURM_ARRAY_TASK_ID", 0), n_tasks=env("SLURM_ARRAY_TASK_COUNT", 1),
        B=env("B", 500), n_members=env("NMEMBER", 82), n_months=env("NMONTH", 72),
        k=env("K", 4), seed=env("SEED", 0), boot_seed=env("BOOTSEED", 0),
        parts=os.environ.get("PARTS", f"{STAGE}/parts"),
        out=os.environ.get("OUT", f"{REPO}/data/lam_rho_boot_SLP_s1961.npz"))
    a.out_dir = a.parts
    (run if sys.argv[1] == "run" else combine)(a)
