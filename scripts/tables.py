"""The two binning tables, from the point_<case>.npz files of run.py.

    python scripts/tables.py    -> figures/data/jugaad_fd_optbins_table.{md,json}, jugaad_lam_terms.{md,json}

Area means and shares are cos-lat weighted. dI = max - min over the thesis band alt 0.5/1/1.5.
"""
import json

import numpy as np

from data import CASES, out
from style import data_path

SHORT = {"cmip5": "CMIP5", "s1961": "s1961", "s1961moy": "s1961 moy", "2-4": "lead 2-4", "s1978": "s1978"}
RULES = (("fd", "FD (thesis)"), ("knuth", "Knuth"))
FLOOR = 0.02        # lambda_m at or below: ratio undefined (float32 round-off, raw ratio ~1e5)


def wmean(W, a, m=None):
    ok = np.isfinite(a) if m is None else (np.isfinite(a) & m)
    return float((W * np.where(ok, a, 0)).sum() / (W * ok).sum()) if ok.any() else np.nan


def wfrac(W, m, valid):
    return float((W * (m & valid)).sum() / (W * valid).sum())


def ratio(lam_o, lam_m):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lam_m > FLOOR, lam_o / lam_m, np.nan)


def gauss_I(rho):
    return -0.5 * np.log2(1 - np.minimum(rho, 0.999999) ** 2)


def band(z, key):
    """I, lambda (n_alt, ny, nx), their alt=1 slices (`<k>1`) and band widths (`d<k>`)."""
    k1 = int(np.argmin(np.abs(z[f"{key}__alts"] - 1.0)))
    d = {k: z[f"{key}__{k}"] for k in ("I_o", "I_m", "lam_o", "lam_m")}
    d["lam"] = ratio(d["lam_o"], d["lam_m"])
    d.update({f"{k}1": d[k][k1] for k in list(d)})
    d.update({f"d{k}": np.nanmax(d[k], 0) - np.nanmin(d[k], 0) for k in ("I_o", "I_m", "lam")})
    d["bins"], d["alts"] = z[f"{key}__bins"][:3], z[f"{key}__alts"]    # f, o, s axes
    return d


def load_case(var, case):
    z = np.load(out("point", var, case))
    W = np.cos(np.deg2rad(z["lats"]))[:, None] * np.ones(z["lons"].size)
    return dict(z=z, var=var, case=case, lats=z["lats"], lons=z["lons"], W=W, N=int(z["N"]),
                T=int(z["T"]), id=f"{var} {SHORT[case]}", rules={k: band(z, k) for k, _ in RULES})


def rule_row(d, key):
    z, W, r = d["z"], d["W"], d["rules"][key]
    valid, lo, hi = np.isfinite(r["lam1"]), np.nanmin(r["lam"], 0), np.nanmax(r["lam"], 0)
    e = {}
    for q in ("I_o", "I_m"):
        ok = np.isfinite(r[f"{q}1"])
        with np.errstate(divide="ignore", invalid="ignore"):
            e.update({q: wmean(W, r[f"{q}1"]), f"d{q}": wmean(W, r[f"d{q}"]),
                      f"med_rel_d{q}": float(np.nanmedian(r[f"d{q}"] / r[f"{q}1"])),
                      f"area_d{q}_gt_{q}": wfrac(W, r[f"d{q}"] > r[f"{q}1"], ok),
                      f"alt_{q}": [wmean(W, x) for x in r[q]]})
    e.update(lam_o=wmean(W, r["lam_o1"]), lam_m=wmean(W, r["lam_m1"]), lam=wmean(W, r["lam1"]),
             dlam=wmean(W, r["dlam"]), area_lam_gt1=wfrac(W, r["lam1"] > 1, valid),
             area_band_gt1=wfrac(W, lo > 1, valid), area_band_lt1=wfrac(W, hi < 1, valid),
             area_band_contains1=wfrac(W, (lo <= 1) & (hi >= 1), valid),
             area_lam_undefined=1 - wfrac(W, valid, np.isfinite(r["lam_o1"])),
             bins_median=[int(np.median(x)) for x in r["bins"]],
             bins_range=[[int(x.min()), int(x.max())] for x in r["bins"]],
             bins_eq1_frac=[float((x == 1).mean()) for x in r["bins"]])
    if key == "knuth":                                  # caps: 200 pooled, T on T values
        cap = np.array([200, min(200, d["T"]), min(200, d["T"])])[:, None, None]
        e["cap_hit_frac"] = [float(x.mean()) for x in r["bins"] >= cap]
    ll = ratio(z[f"{key}_loo__lam_o"], z[f"{key}_loo__lam_m"])[0]
    e["loo"] = dict(I_m=wmean(W, z[f"{key}_loo__I_m"][0]), lam_m=wmean(W, z[f"{key}_loo__lam_m"][0]),
                    lam=wmean(W, ll), area_lam_gt1=wfrac(W, ll > 1, np.isfinite(ll)),
                    bins_s_median=int(np.median(z[f"{key}_loo__bins"][2])))
    return e


MD_COLS = [("I_o", ".3f", 1), ("dI_o", ".3f", 1), ("med_rel_dI_o", ".2f", 1), ("area_dI_o_gt_I_o", ".0f", 100),
           ("I_m", ".3f", 1), ("dI_m", ".3f", 1), ("med_rel_dI_m", ".2f", 1), ("lam", ".3f", 1),
           ("dlam", ".3f", 1), ("area_lam_gt1", ".1f", 100), ("area_band_gt1", ".1f", 100),
           ("area_band_lt1", ".1f", 100), ("area_band_contains1", ".1f", 100),
           ("area_lam_undefined", ".1f", 100)]


def binning_table(ds):
    out_, md = {}, ["Binning uncertainty vs estimate. I in bits at alt=1; dI = max-min over alt "
                    "0.5/1/1.5; medians and shares over cells (cos-lat area-weighted shares).", "",
                    "| case | rule | I_o | dI_o | med dI_o/I_o | % dI_o>I_o | I_m | dI_m | med dI_m/I_m "
                    "| λ | dλ | % λ>1 | % band >1 | % band <1 | % band ∋1 | % λ undef. | bins f/o/s "
                    "| % 1 bin f/o/s | cap f/o/s % |", "|" + "---|" * 19]
    for d in ds:
        z, W, o = d["z"], d["W"], {"N": d["N"], "T": d["T"], "rules": {}}
        for key, lab in RULES:
            e = o["rules"][lab] = rule_row(d, key)
            pct = lambda v: "/".join(f"{100 * x:.0f}" for x in v)                     # noqa: E731
            md.append(f"| {d['id']} | {lab} | " + " | ".join(format(e[k] * m, f) for k, f, m in MD_COLS)
                      + f" | {'/'.join(map(str, e['bins_median']))} | {pct(e['bins_eq1_frac'])} | "
                      f"{pct(e.get('cap_hit_frac', []))} |")
        fd, kn = d["rules"]["fd"], d["rules"]["knuth"]
        with np.errstate(divide="ignore", invalid="ignore"):
            o["fd_minus_knuth"] = {k: wmean(W, fd[f"{k}1"] - kn[f"{k}1"]) for k in ("I_o", "I_m", "lam")}
            o["fd_minus_knuth"]["med_rel_I_o"] = float(np.nanmedian((fd["I_o1"] - kn["I_o1"]) / fd["I_o1"]))
        B, fx = z["fixed_B"], lambda k: [wmean(W, z[f"fixed_B{b}__{k}"][0]) for b in z["fixed_B"]]  # noqa: E731
        o["fixed_B"] = {"B": B.tolist(), "I_o": fx("I_o"), "I_m": fx("I_m"),
                        "lam": [wmean(W, ratio(z[f"fixed_B{b}__lam_o"], z[f"fixed_B{b}__lam_m"])[0]) for b in B]}
        ro, rm = z["rho_o"], z["rho_m"]
        o["pooled_pearson"] = dict(rho_o=wmean(W, ro), rho_m=wmean(W, rm), rho_m_loo=wmean(W, z["rho_m_loo"]),
                                   ratio=wmean(W, ro / rm), gauss_I_o=wmean(W, gauss_I(ro)),
                                   gauss_I_m=wmean(W, gauss_I(rm)), area_ratio_gt1=wfrac(W, ro > rm, np.isfinite(ro)))
        out_[d["id"]] = o
    md += ["", "| case | FD−Knuth I_o | FD−Knuth I_m | FD−Knuth λ | I_o at B=2 → 64 | "
           "Gaussian-equiv. I_o, I_m | pooled ρ_o/ρ_m | % ρ_o/ρ_m>1 |", "|" + "---|" * 8]
    for cid, v in out_.items():
        f, p, fx = v["fd_minus_knuth"], v["pooled_pearson"], v["fixed_B"]
        md.append(f"| {cid} | {f['I_o']:+.3f} | {f['I_m']:+.3f} | {f['lam']:+.3f} | {fx['I_o'][0]:.3f} → "
                  f"{fx['I_o'][-1]:.3f} | {p['gauss_I_o']:.3f}, {p['gauss_I_m']:.3f} | {p['ratio']:.3f} | "
                  f"{100 * p['area_ratio_gt1']:.1f} |")
    write("jugaad_fd_optbins_table", out_, md)


def lam_terms_table(ds):
    """lambda_o with I(f; o) and with I(s_-n; o), and lambda_m, FD bins."""
    def summ(W, a):
        return [wmean(W, a[1]), *map(float, np.nanpercentile(a[1], [10, 90])),
                wmean(W, a[0]), wmean(W, a[2])]
    out_, md = {}, ["Area-weighted mean at 1.0× FD bins, (10th–90th percentile over cells), "
                    "[area mean at 0.5×, 1.5× bins].", "",
                    "| case | N | T | λ_o = λ(I(f; o)) | λ_o = λ(I(s₋ₙ; o)) | λ_m = λ(I(f; s)) | "
                    "λ_o/λ_m with f (% >1) | with s₋ₙ (% >1) | FD bins f/o/s/s₋ₙ |", "|" + "---|" * 9]
    f3 = lambda v: f"{v[0]:.3f} ({v[1]:.2f}–{v[2]:.2f}) [{v[3]:.2f}, {v[4]:.2f}]"   # noqa: E731
    for d in ds:
        z, W = d["z"], d["W"]
        lo_f, lm, lo_s = z["fd__lam_o"], z["fd__lam_m"], z["oloo_fd__lam_o"]
        with np.errstate(invalid="ignore", divide="ignore"):
            r_f, r_s = lo_f[1] / lm[1], lo_s[1] / lm[1]
        e = out_[d["id"]] = {"N": d["N"], "T": d["T"], "lam_o_f": summ(W, lo_f),
                             "lam_o_sloo": summ(W, lo_s), "lam_m": summ(W, lm),
                             "ratio_f": wmean(W, r_f), "ratio_f_gt1": wfrac(W, r_f > 1, np.isfinite(r_f)),
                             "ratio_sloo": wmean(W, r_s), "ratio_sloo_gt1": wfrac(W, r_s > 1, np.isfinite(r_s)),
                             "bins_median_f_o_s_sloo": [int(np.median(b)) for b in z["oloo_fd__bins"]]}
        md.append(f"| {d['id']} | {d['N']} | {d['T']} | {f3(e['lam_o_f'])} | {f3(e['lam_o_sloo'])} | "
                  f"{f3(e['lam_m'])} | {e['ratio_f']:.3f} ({100 * e['ratio_f_gt1']:.0f}) | "
                  f"{e['ratio_sloo']:.3f} ({100 * e['ratio_sloo_gt1']:.0f}) | "
                  f"{'/'.join(map(str, e['bins_median_f_o_s_sloo']))} |")
    write("jugaad_lam_terms", out_, md)


def write(name, res, md):
    with open(data_path(name + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(data_path(name + ".md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ds = [load_case(*c) for c in CASES]
    binning_table(ds)
    lam_terms_table(ds)
