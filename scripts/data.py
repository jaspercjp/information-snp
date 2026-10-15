"""Paths and cases. Inputs: data-acquisition/make_cases.py's preprocessed per-case arrays,
members F (N, T, ny, nx), obs o and ensemble mean s (T, ny, nx)."""
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("SNP_DATA", os.path.join(REPO, "data"))
OUT = os.environ.get("SNP_OUT", os.path.join(os.environ.get("SCRATCH", DATA), "snp_jugaad"))

# PRECT has no s1961 (GPCP starts in 1979); the production decadal row uses s1978.
CASES = [("SLP", "cmip5"), ("SLP", "s1961"), ("TREFHT", "s1961"), ("SLP", "s1961moy"),
         ("TREFHT", "s1961moy"), ("SLP", "2-4"), ("PRECT", "2-4"), ("TREFHT", "2-4")]
PRODUCTION = {"lead2-4": [("SLP", "2-4"), ("PRECT", "2-4"), ("TREFHT", "2-4")],
              "decadal": [("SLP", "s1961"), ("PRECT", "s1978"), ("TREFHT", "s1961")]}


def load(var, case):                                            # -> F, o, s, lats, lons
    z = np.load(os.path.join(DATA, "cases", f"{var}_{case}.npz"))
    return z["F"], z["o"], z["s"], z["lats"], z["lons"]


def out(kind, var, case):
    return os.path.join(OUT, f"{kind}_{var}_{case}.npz")
