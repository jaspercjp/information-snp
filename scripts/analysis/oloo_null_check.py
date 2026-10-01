"""Can lambda_o = lambda(I(s_-n_jugaad; o_jugaad)) see skill? Synthetic check, ~1 min.

    python scripts/analysis/oloo_null_check.py

Members F = S + noise share a signal S (sd 0.5). Two obs series: one independent of
everything ("null"), one with corr(S, o) = 0.5 ("skill"; Gaussian target lambda = 0.5).
Both numerators, FD bins (thesis) and a common fixed B.

Why the s_-n numerator fails with FD: s_-n[t] differs from s[t] only by ~f_n/N, so the
pooled s_-n is T tight clusters of N rows, and the tiled obs is T exact repeats. FD sizes
both axes on N*T rows (20-45 bins), about one bin per distinct time, and the plug-in MI
then approaches the entropy of the binned series whatever the dependence.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import numpy as np

import jugaad_thesis as JT                                           # noqa: E402


def main(N=62, T=51, seed=2):
    rng = np.random.default_rng(seed)
    S = 0.5 * rng.standard_normal((T, 8, 8))
    F = S + rng.standard_normal((N, T, 8, 8))
    o_null = rng.standard_normal((T, 8, 8))
    o_skill = S + np.sqrt(0.75) * rng.standard_normal((T, 8, 8))     # corr(S, o) = 0.5
    rows = [("FD (thesis)", dict(rule="fd"))] + \
        [(f"fixed B={b}", dict(rule="fixed", B=b)) for b in (2, 3, 4, 6, 8, 12)]
    print(f"N={N} T={T}; mean lambda_o over 64 cells")
    print(f"{'bins':<14} {'I(f;o) null':>12} {'skill':>7} | {'I(s_-n;o) null':>15} {'skill':>7}")
    for lab, kw in rows:
        v = [np.nanmean(JT.jugaad_maps(F, o, alts=(1.0,), o_vs=ov, n_jobs=1, **kw)["lam_o"])
             for ov in ("f", "loo") for o in (o_null, o_skill)]
        print(f"{lab:<14} {v[0]:>12.3f} {v[1]:>7.3f} | {v[2]:>15.3f} {v[3]:>7.3f}")


if __name__ == "__main__":
    main()
