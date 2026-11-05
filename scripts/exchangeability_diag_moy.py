"""Why CanESM5 SLP s1961moy rejects (exact block null): memory longer than the block, or members?

    python scripts/exchangeability_diag_moy.py {block24,drop1,first5,last5}   (~6 min each)
        -> one JSON line: field significance of I_MXY and Delta for that variant
    python scripts/exchangeability_diag_moy.py members
        -> per-member term H(pool) - H(member) of I(M; X, Y) at the rejecting cells, zonal shares
           (needs grid --case SLP:s1961moy --model CanESM5 --block 12 --redetrend first)

Variants of the reference run (block 12, all 120 months): blocks of 24 months; months 13-120
(drops the first year); months 1-60; months 61-120. Each detrends its own window inside the test.
Summary: figures/data/exchangeability-grid-SLP-s1961moy-CanESM5-diagnostics.json.
"""
import json
import os
import sys

import numpy as np

import exchangeability as E
import exchangeability_check as C
from data import OUT, load

VAR, CS, MODEL = "SLP", "s1961moy", "CanESM5"
WINDOWS = {"block24": (slice(None), 24), "drop1": (slice(12, None), 12),
           "first5": (slice(None, 60), 12), "last5": (slice(60, None), 12)}


def variant(which):
    F, o, s, lats, lons = load(VAR, CS)
    x = C.model_members(VAR, CS, MODEL, F)
    raw = C.raw_members(VAR, CS, MODEL, x)
    W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
    win, L = WINDOWS[which]
    res = E.exchangeability_test(raw[:, win], n_perm=200, block=L, seed=3, return_null=True,
                                 detrend=True)
    out = {"diag": which, "T": int(raw[:, win].shape[1]), "block": L}
    for st in ("I_MXY", "Delta"):
        f = C.field_test(res, st, W)
        out[st] = {k: round(v, 4) for k, v in f.items() if k != "null_area"}
    print(json.dumps(out))


def members():
    F, o, s, lats, lons = load(VAR, CS)
    x = C.model_members(VAR, CS, MODEL, F)
    lab = json.load(open(os.path.join(OUT, "member_models.json")))["SLP s1961"]
    names = np.array(lab["members"])[np.array(lab["model_of"]) == MODEL]
    z = np.load(os.path.join(OUT, f"exchangeability_grid_{VAR}_{CS}_{MODEL}_block12_redetrend.npz"))
    rej = (z["I_MXY_p"] <= 0.05).ravel()
    (a,), _ = E._prepare(x, E.n_bins(x.shape[1]), "bins")          # (P, R, T) joint codes
    cm = E._count(a, E.n_bins(x.shape[1]) ** 2)
    term = E.entropy_counts(cm.sum(1))[:, None] - E.entropy_counts(cm)   # (P, R) bits
    W = np.cos(np.deg2rad(lats))[:, None] * np.ones(lons.size)
    w = W.ravel()
    for name, m in (("rejecting", rej), ("others", ~rej)):
        t = (w[m, None] * term[m]).sum(0) / w[m].sum()
        print(name, " ".join(f"{n.split(':')[1]}:{v:.3f}" for n, v in
                             sorted(zip(names, t), key=lambda q: -q[1])))
    r2 = rej.reshape(lats.size, lons.size)
    for la0, la1 in ((-90, -30), (-30, 30), (30, 90)):
        band = (lats >= la0) & (lats < la1)
        print(f"lat {la0}..{la1}: share with p <= 0.05 {(W[band] * r2[band]).sum() / W[band].sum():.3f}")


if __name__ == "__main__":
    members() if sys.argv[1] == "members" else variant(sys.argv[1])
