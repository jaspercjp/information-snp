"""Handles onto the EC-Earth3 DCPP decadal hindcasts. Same interface as `smyle_handles`.

    import sys; sys.path.insert(0, "../scripts")
    import ecearth3_handles as E, smyle_metrics as M

    E.inventory()                      # which (variable, lead) cubes exist
    c = E.get(lead="2-4")              # Nov init, lead 2-4 -> DJF
    c = E.get(lead="13-60")            # 4-year mean, lead years 2-5

Everything `smyle_handles` returns, this returns, with the same names and shapes:

    c.F  (N, J, lat, lon) members          c.YEAR, c.MONTH  (J,) init year, month
    c.G  (J, lat, lon)    observations     c.lats, c.lons, c.members, c.N, c.J
    c.s  (J, lat, lon)    ensemble mean    c.target, c.label, c.units
    c.loo_mean(n)         s_-n, the mean over the other N-1 members
    c.gm(field)           cos(lat)-weighted global mean of a (lat, lon) field

Arrives Eade-smoothed (11.25x12.5 deg; 15x15 for TREFHT) on both model and
observations, and detrended within the init month. See `_cube_io` and `_smooth`.

Initialised 1 November, annually, like DePreSys4
------------------------------------------------
Lead 1 is November of the start year, so lead 2-4 is DJF and lead 13-60 is the
4-year mean over lead years 2-5 -- the same windows, on the same calendar months, as
`depresys4_handles`. Verified against it: `s1968` lead 13-60 covers 1969-11..1973-10
in both systems. So EC-Earth3 pools with DePreSys4 directly, at both windows, unlike
CanESM5.

Two caveats that bear on rho_m
------------------------------
**The 16 members are not exchangeable.** They span three initialisation variants --
10 with `i1`, 5 with `i2`, 1 with `i4`. Those are different initialisation methods,
not perturbed starts from one method, so treating all 16 as one ensemble violates the
exchangeability that `sigma_sig / sigma_tot` assumes. Pass `i_index` at BUILD time
(`build_dcpp_cube.py --i-index 1`) to get a clean 10-member cube if that matters for
the result you are computing.

**J is 51, not 60.** No EC-Earth3 member is published across all 60 starts -- every
one is missing between 1 and 6. A cube must be rectangular, so `largest_block` kept
all 16 members and dropped the 9 starts that were not covered by every member. The
alternative, keeping all 60 starts, would have left zero members.

Chunking: EC-Earth3 publishes ~11 files per (start, member) where the other models
publish one, so cube builds open ~9,000 files per (variable, window). That is a build
cost only; the cube is the same shape.

If `inventory()` is empty, build a cube:

    python scripts/build_dcpp_cube.py --models EC-Earth3 --var SLP --leads 2-4,13-60
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cube_io import Cube, target_season                      # noqa: F401
from _cube_io import inventory as _inventory
from _cube_io import seasons_of as _seasons_of
from _cube_io import summary as _summary

MODEL = "EC-Earth3"
CUBES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "dcpp_cubes", MODEL)


def get(season="Nov", lead="2-4", var="SLP", **kw):
    """One handle. EC-Earth3 has a single init month, November."""
    return Cube(var, lead, season, cubes=CUBES, **kw)


def by_season(lead="2-4", var="SLP", **kw):
    """The seasonal handles -- just November for this system."""
    return [Cube(var, lead, s, cubes=CUBES, **kw)
            for s in _seasons_of(var, lead, CUBES)]


# Rebound to this model's cube directory; the `_cube_io` versions default to SMYLE's
# and would otherwise report the wrong system with no error.
def inventory(cubes=CUBES):
    return _inventory(cubes)


def seasons_of(var, lead, cubes=CUBES):
    return _seasons_of(var, lead, cubes)


def summary(var, lead, cubes=CUBES):
    return _summary(var, lead, cubes)


if __name__ == "__main__":
    for var, lead in inventory():
        summary(var, lead)
