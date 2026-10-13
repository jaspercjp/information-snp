"""The pooled CMIP6 DCPP-A grand ensemble at one lead window: samples are START DATES.

    import dcpp_handles as G
    c = G.get(lead="2-4", var="SLP", remove_gm=True)    # DJF, the November models
    c.F (N, J, lat, lon)   c.G (J, lat, lon)   c.lats, c.lons, c.models, c.YEAR

Lead 2-4 is DJF for the November-initialised models, which are the five pooled here
(`MODELS`, pinned; CanESM5 starts in January and NorCPM1 in October, so neither has a
DJF at lead 2-4):

    HadGEM3-GC31-MM (DePreSys4) 10    EC-Earth3 16    MIROC6 10
    MPI-ESM1-2-LR 16                  MRI-ESM2-0 10   -> N = 62

Detrend per model, THEN pool. Each model is loaded through its own `Cube`, so each is
detrended within its own member set (Eade et al. 2014 correct each model separately),
and only then are the member axes concatenated. Pooling first would leave each model's
mean offset from a blended climatology in the anomalies as spurious signal.

Common start dates. `get` intersects the init years across the models (requiring
observations wherever they exist) and detrends each on that common set, so `c.J` is the
intersection. Observations: one series for the pool, from the first November model.

SLP needs `remove_gm=True`: EC-Earth3's `psl` carries a spatially uniform mass offset that
varies by start date (sd 28.8 Pa); see `_cube_io.remove_global_mean`. Leave it off for
TREFHT, whose global mean is real predictable signal.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cube_io import Cube, MONTH_ABBR, canon                       # noqa: E402
from paths import DCPP_CUBES, DEPRESYS_CUBES                       # noqa: E402

# The models the lead 2-4 pool uses, in pooling order (DePreSys4 first, then the
# data/dcpp_cubes/ tree in sorted order). Pinned: the old code discovered them from disk.
MODELS = ["HadGEM3-GC31-MM", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR", "MRI-ESM2-0"]
CUBE_DIRS = {m: DEPRESYS_CUBES if m == "HadGEM3-GC31-MM" else os.path.join(DCPP_CUBES, m)
             for m in MODELS}
REFERENCE_INIT = 11             # the convention pooled windows are defined against


def _path(model, var, lead):
    return os.path.join(CUBE_DIRS[model], f"{canon(var)}_lead{lead}.npz")


def _meta(model, var, lead):
    """(year, month, obs_ok) straight from the npz header, without loading cubes."""
    z = np.load(_path(model, var, lead), mmap_mode="r")
    return (np.asarray(z["year"]), np.asarray(z["month"]), np.asarray(z["obs_ok"]))


class Pooled:
    """The grand ensemble as one Cube-shaped object. Built by `get`."""

    def __init__(self, var, lead, models=None, require_obs=None, detrend_data=True,
                 smooth=True, remove_gm=False):
        var = canon(var)
        models = list(models or MODELS)
        missing = [m for m in models if not os.path.exists(_path(m, var, lead))]
        if missing:
            raise FileNotFoundError(f"no {var} lead {lead} cube for {missing}; "
                                    f"run build_dcpp_cube.py / build_depresys4_cube.py")

        # ---- intersect start years, requiring observations wherever they exist
        sets, inits = {}, {}
        for m in models:
            y, mo, ok = _meta(m, var, lead)
            inits[m] = int(mo[0])
            keep = np.ones(len(y), bool)
            want_obs = ok.any() if require_obs is None else require_obs
            if want_obs:
                keep &= ok
            sets[m] = set(y[keep].tolist())
        common = sorted(set.intersection(*sets.values()))
        if not common:
            raise ValueError(f"no start years common to {models} for {var} lead {lead}")
        self.dropped = {m: len(sets[m]) - len(common) for m in models}

        # ---- load each model separately, so each detrends on its own member set
        parts, names, owner = [], [], []
        ref_obs = first = None
        for m in models:
            c = Cube(var, lead, MONTH_ABBR[inits[m]], cubes=CUBE_DIRS[m], years=common,
                     require_obs=require_obs, detrend_data=detrend_data,
                     smooth=smooth, remove_gm=remove_gm)
            if first is None:
                first = c
            elif not (np.allclose(c.lats, first.lats) and np.allclose(c.lons, first.lons)
                      and np.array_equal(c.YEAR, first.YEAR)):
                raise ValueError(f"{m} is not on {models[0]}'s grid / start years")
            parts.append(c.F)
            names += [f"{m}:{s}" for s in np.asarray(c.members)]
            owner += [m] * c.F.shape[0]
            if inits[m] == REFERENCE_INIT and ref_obs is None:
                ref_obs = c.G
        # one observed series for the pool, on the November convention
        self.G = ref_obs if ref_obs is not None else first.G

        self.F = np.concatenate(parts, axis=0)
        self.members = np.array(names)
        self.model_of = np.array(owner)
        self.models = models
        self.var, self.lead = var, lead
        self.units, self.lats, self.lons = first.units, first.lats, first.lons
        self.YEAR = first.YEAR
        self.J, self.N = self.F.shape[1], self.F.shape[0]

    def __repr__(self):
        return (f"<Pooled DCPP {self.var} lead {self.lead} ({len(self.models)} models) "
                f"[{self.units}]: N={self.N} J={self.J}>")


def get(lead="2-4", var="SLP", models=None, **kw):
    """The pooled grand ensemble for one (variable, window)."""
    return Pooled(var, lead, models=models, **kw)
