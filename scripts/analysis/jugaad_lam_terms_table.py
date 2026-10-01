"""How large the jugaad lambda terms are, per case: lambda_o under both numerators, lambda_m.

    python scripts/analysis/jugaad_lam_terms_table.py
    -> $SCRATCH/snp_jugaad_fd_optbins/figs/jugaad_lam_terms.{md,json}

    lambda_o (f)     = lambda(I(f_jugaad;    o_jugaad))   thesis      <case>.npz fd_tiled
    lambda_o (s_-n)  = lambda(I(s_-n_jugaad; o_jugaad))               oloo_<case>.npz
    lambda_m         = lambda(I(f_jugaad;    s_jugaad))   full mean s (same in both files)

FD bins, alt = 1; cos-lat area-weighted mean, the 10-90th percentile over cells, and the
area-mean at alt = 0.5 and 1.5 (the thesis binning band). Both ratios are given too.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import json
import os

import numpy as np

from jugaad_fd_optbins_figs import CASES, FIG, SHORT, wfrac, wmean   # noqa: E402
from jugaad_fd_optbins_run import OUT                                 # noqa: E402


def summ(W, a):
    """(mean at alt=1, p10, p90, mean at alt 0.5, mean at alt 1.5) for an (n_alt, ny, nx) array."""
    x = a[1]
    p10, p90 = np.nanpercentile(x, [10, 90])
    return [wmean(W, x), float(p10), float(p90), wmean(W, a[0]), wmean(W, a[2])]


def main():
    rows, out = [], {}
    for var, lead in CASES:
        pf, po = (os.path.join(OUT, f"{var}_lead{lead}.npz"),
                  os.path.join(OUT, f"oloo_{var}_lead{lead}.npz"))
        if not (os.path.exists(pf) and os.path.exists(po)):
            print(f"skip {var} {lead}: missing file")
            continue
        z, q = np.load(pf), np.load(po)
        W = np.cos(np.deg2rad(z["lats"]))[:, None] * np.ones(z["lons"].size)
        lo_f, lm = z["fd_tiled__lam_o"], z["fd_tiled__lam_m"]
        lo_s = q["lam_o"]
        assert np.allclose(q["lam_m"], lm, equal_nan=True)            # same lambda_m
        with np.errstate(invalid="ignore", divide="ignore"):
            r_f, r_s = lo_f[1] / lm[1], lo_s[1] / lm[1]
        cid = f"{var} {SHORT[lead]}"
        e = {"N": int(z["N"]), "T": int(z["T"]),
             "lam_o_f": summ(W, lo_f), "lam_o_sloo": summ(W, lo_s), "lam_m": summ(W, lm),
             "ratio_f": wmean(W, r_f), "ratio_f_gt1": wfrac(W, r_f > 1, np.isfinite(r_f)),
             "ratio_sloo": wmean(W, r_s), "ratio_sloo_gt1": wfrac(W, r_s > 1, np.isfinite(r_s)),
             "bins_median_f_o_s_sloo": [int(np.median(b)) for b in q["bins"]]}
        out[cid] = e
        rows.append((cid, e))
    f3 = lambda v: f"{v[0]:.3f} ({v[1]:.2f}–{v[2]:.2f}) [{v[3]:.2f}, {v[4]:.2f}]"   # noqa: E731
    md = ["Area-weighted mean at 1.0× FD bins, (10th–90th percentile over cells), "
          "[area mean at 0.5×, 1.5× bins].", "",
          "| case | N | T | λ_o = λ(I(f; o)) | λ_o = λ(I(s₋ₙ; o)) | λ_m = λ(I(f; s)) | "
          "λ_o/λ_m with f (% >1) | with s₋ₙ (% >1) | FD bins f/o/s/s₋ₙ |", "|" + "---|" * 9]
    for cid, e in rows:
        md.append(f"| {cid} | {e['N']} | {e['T']} | {f3(e['lam_o_f'])} | {f3(e['lam_o_sloo'])} | "
                  f"{f3(e['lam_m'])} | {e['ratio_f']:.3f} ({100 * e['ratio_f_gt1']:.0f}) | "
                  f"{e['ratio_sloo']:.3f} ({100 * e['ratio_sloo_gt1']:.0f}) | "
                  f"{'/'.join(map(str, e['bins_median_f_o_s_sloo']))} |")
    os.makedirs(os.path.join(FIG, "json"), exist_ok=True)
    with open(os.path.join(FIG, "json", "jugaad_lam_terms.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(FIG, "jugaad_lam_terms.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
