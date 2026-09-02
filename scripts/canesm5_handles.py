"""Handles onto the CanESM5 DCPP decadal hindcasts. Same interface as `smyle_handles`.

    import sys; sys.path.insert(0, "../scripts")
    import canesm5_handles as C, smyle_metrics as M

    C.inventory()                      # which (variable, lead) cubes exist
    c = C.get(lead="13-60")            # 4-year mean, lead years 2-5
    rho_m = M.ensemble_SNR_debiased(c.F)

Everything `smyle_handles` returns, this returns, with the same names and shapes:

    c.F  (N, J, lat, lon) members          c.YEAR, c.MONTH  (J,) init year, month
    c.G  (J, lat, lon)    observations     c.lats, c.lons, c.members, c.N, c.J
    c.s  (J, lat, lon)    ensemble mean    c.target, c.label, c.units
    c.loo_mean(n)         s_-n, the mean over the other N-1 members
    c.gm(field)           cos(lat)-weighted global mean of a (lat, lon) field

Arrives Eade-smoothed (11.25x12.5 deg; 15x15 for TREFHT) on both model and
observations, and detrended within the init month. See `_cube_io` and `_smooth`.

CanESM5 initialises in JANUARY -- read this before using it
-----------------------------------------------------------
Unlike every other system here, CanESM5's lead 1 is 1 January, and its `sYYYY` label
is a year behind: `s1960` output BEGINS 1961-01. Two consequences:

* **The season is "Jan", not "Nov".** `season` defaults to "Jan" accordingly.
* **There is no DJF window.** Lead 2-4 is February-March-April here, sharing only
  one month with the DJF that lead 2-4 means for a November system. So CanESM5
  cannot join a seasonal pool, and `build_dcpp_cube.py` refuses to build it
  (`--min-overlap`). Only `lead="13-60"` exists.

The 4-year window IS poolable: lead 13-60 covers Jan-Dec of forecast years 2-5
against Nov-Oct for DePreSys4, which is 46 of 48 months in common, so the two target
essentially the same period. `c.target` reports "yr2-5" for both.

Ensemble size differs by variable
---------------------------------
ESGF publishes CanESM5 `psl` for only 20 of its 40 members across all 60 starts (the
other 20 have 3 starts each), while `tas` and `pr` are complete at 40 x 60. So
`c.N` is 20 for SLP and 40 for TREFHT and PRECT. That is not a bug in the cube; it
is what is published. It matters because rho_m's finite-ensemble bias depends on N,
so SLP and TREFHT numbers from this model are not equally noisy.

If `inventory()` is empty, build a cube:

    python scripts/build_dcpp_cube.py --models CanESM5 --var SLP --leads 13-60
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cube_io import Cube, target_season                      # noqa: F401
from _cube_io import inventory as _inventory
from _cube_io import seasons_of as _seasons_of
from _cube_io import summary as _summary

MODEL = "CanESM5"
CUBES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "dcpp_cubes", MODEL)


def get(season="Jan", lead="13-60", var="SLP", **kw):
    """One handle. CanESM5 has a single init month, January."""
    return Cube(var, lead, season, cubes=CUBES, **kw)


def by_season(lead="13-60", var="SLP", **kw):
    """The seasonal handles -- just January for this system."""
    return [Cube(var, lead, s, cubes=CUBES, **kw)
            for s in _seasons_of(var, lead, CUBES)]


# These three default to the SMYLE cube directory in `_cube_io`, so they are rebound
# here. Without that, `inventory()` silently lists SMYLE's cubes instead of this
# model's -- a wrong answer with no error, which is worse than a traceback.
def inventory(cubes=CUBES):
    return _inventory(cubes)


def seasons_of(var, lead, cubes=CUBES):
    return _seasons_of(var, lead, cubes)


def summary(var, lead, cubes=CUBES):
    return _summary(var, lead, cubes)


if __name__ == "__main__":
    for var, lead in inventory():
        summary(var, lead)
