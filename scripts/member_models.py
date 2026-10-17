"""Which model each member of the six production cases comes from: $SNP_OUT/member_models.json.

    python scripts/member_models.py        (~5 min, ~10 GB: rebuilds each case from the raw data)

make_cases.py writes F without member labels. This rebuilds every case through the same loaders
(dcpp_decadal_handles + thesis_detrend for sYYYY, dcpp_handles for 2-4), keeps model_of and the
member names, and checks the rebuilt F against data/cases bit for bit ("matches_case_file").
"""
import gc
import json
import os
import sys

W = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [W, os.path.join(W, "data-acquisition")]
import numpy as np                                                  # noqa: E402
import make_cases as MC                                             # noqa: E402
from data import DATA, OUT, PRODUCTION                              # noqa: E402

out = {}
for var, case in [c for cs in PRODUCTION.values() for c in cs]:
    if case.startswith("s"):
        import dcpp_decadal_handles as H
        c = H.get(int(case[1:5]), var=var, verbose=False, anom="none")
        F = MC.thesis_detrend(c.F)
    else:
        import dcpp_handles as H
        c = H.get(lead=case, var=var, remove_gm=var.upper() == "SLP")
        F = np.asarray(c.F, float)
    model_of, members = [str(m) for m in c.model_of], [str(m) for m in c.members]
    del c
    Fc = np.load(os.path.join(DATA, "cases", f"{var}_{case}.npz"), mmap_mode="r")["F"]
    same = Fc.shape == F.shape and bool(np.array_equal(np.asarray(Fc), F, equal_nan=True))
    del F, Fc; gc.collect()
    u, n = np.unique(model_of, return_counts=True)
    print(var, case, "identical to case file:", same, dict(zip(u.tolist(), n.tolist())), flush=True)
    out[f"{var} {case}"] = {"model_of": model_of, "members": members, "matches_case_file": same}
with open(os.path.join(OUT, "member_models.json"), "w") as fh:
    json.dump(out, fh, indent=1)
