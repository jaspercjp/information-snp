"""Handles onto the DePreSys4 decadal hindcasts. Same interface as `smyle_handles`.

    import sys; sys.path.insert(0, "../scripts")
    import depresys4_handles as D, smyle_metrics as M

    D.inventory()                     # which (variable, lead) cubes exist
    c = D.get(lead="2-4")             # Nov init, lead months 2-4 -> DJF
    c = D.get(lead="13-60")           # annual mean over lead YEARS 2-5 (Eade's window)

Everything `smyle_handles` returns, this returns, with the same names and shapes, so the
two systems can be compared line for line:

    c.F  (N, J, lat, lon) members          c.YEAR, c.MONTH  (J,) init year, month
    c.G  (J, lat, lon)    observations     c.lats, c.lons, c.members, c.N, c.J
    c.s  (J, lat, lon)    ensemble mean    c.target, c.label, c.units
    c.loo_mean(n)         s_-n, the mean over the other N-1 members
    c.gm(field)           cos(lat)-weighted global mean of a (lat, lon) field

Variables are SLP (Pa), TREFHT (K) and PRECT (mm/day) -- this project's names for
HadGEM3's psl, tas and pr, so one name means one field across both systems.

DePreSys4 initialises on 1 November ONCE A YEAR (s1960..s2018), so unlike SMYLE there is
a single season: `season` defaults to "Nov" and `by_season` returns one handle. Lead 1 is
November of the start year. Detrending is over the 59 start dates.

Long windows overlap heavily between consecutive starts -- lead 13-60 spans four years,
so neighbouring samples share three of them. The 59 samples are far from independent at
that range; treat effective J as much smaller when judging significance.

If `inventory()` is empty, build a cube:

    python scripts/build_depresys4_cube.py --var SLP --leads 2-4,13-60
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cube_io import Cube, target_season                      # noqa: F401,E402
from _cube_io import inventory as _inventory
from _cube_io import seasons_of as _seasons_of
from _cube_io import summary as _summary

CUBES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "depresys4_cubes")


def get(season="Nov", lead="2-4", var="SLP", **kw):
    """One handle. DePreSys4 has a single init month, so `season` is "Nov" or "all"."""
    return Cube(var, lead, season, cubes=CUBES, **kw)


def by_season(lead="2-4", var="SLP", **kw):
    """The seasonal handles -- just November for this system."""
    return [Cube(var, lead, s, cubes=CUBES, **kw)
            for s in _seasons_of(var, lead, CUBES)]


# Rebound to this model's cube directory. Previously these were re-exported straight
# from `_cube_io`, whose defaults point at `data/smyle_cubes` -- so a bare
# `inventory()` here listed SMYLE's cubes with no error and no warning, and
# `seasons_of("SLP", "13-60")` raised FileNotFoundError on a SMYLE path.
def inventory(cubes=CUBES):
    return _inventory(cubes)


def seasons_of(var, lead, cubes=CUBES):
    return _seasons_of(var, lead, cubes)


def summary(var, lead, cubes=CUBES):
    return _summary(var, lead, cubes)


if __name__ == "__main__":
    for var, lead in inventory():
        summary(var, lead)
