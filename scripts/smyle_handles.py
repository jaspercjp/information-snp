"""Handles onto the CESM2 SMYLE hindcasts, by season and lead window.

    import sys; sys.path.insert(0, "../scripts")
    import smyle_handles as S

    S.inventory()                      # which (variable, lead) cubes exist
    c = S.get("Nov", lead="2-4")       # Nov init, lead months 2-4 -> DJF target
    for c in S.by_season(lead="1-3"):  # Feb, May, Aug, Nov in turn

A season is an INITIALIZATION month (SMYLE starts in Feb/May/Aug/Nov); the target
season is that month shifted by the lead window, reported as `c.target`. Use
`S.get("all", ...)` to pool all four -- though that leaves their anomaly variances
unequal, so a pooled statistic is a variance-weighted blend. Each handle arrives
detrended within its init month, with obs-less samples dropped.

    c.F  (N, J, lat, lon) members          c.YEAR, c.MONTH  (J,) init year, month
    c.G  (J, lat, lon)    observations     c.lats, c.lons, c.members, c.N, c.J
    c.s  (J, lat, lon)    ensemble mean    c.target, c.label   e.g. "DJF"
    c.loo_mean(n)         s_-n, the mean over the other N-1 members
    c.gm(field)           cos(lat)-weighted global mean of a (lat, lon) field

Variables are SLP (CESM's PSL, Pa), TREFHT (K) and PRECT (mm/day; CESM stores
m/s, the cube rescales). `c.units` reports it. Only SLP has
observations wired up -- for the other two `c.G` is all-NaN and `c.has_obs` is False,
so rho_m / lambda_m work but anything with an _o does not.

`get` and `by_season` also take `require_obs=False` (keep samples whose window runs
past the obs record) and `detrend_data=False` (raw values, not anomalies).

If `inventory()` is empty, build a cube:

    python scripts/build_smyle_cube.py --var SLP --leads 1-3,2-4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cube_io import Cube, inventory, seasons_of, summary, target_season  # noqa: F401,E402


def get(season="all", lead="1-3", var="SLP", **kw):
    """One handle. `season` is an init month name (Feb/May/Aug/Nov) or 'all'."""
    return Cube(var, lead, season, **kw)


def by_season(lead="1-3", var="SLP", **kw):
    """The four seasonal handles, in init-month order."""
    return [Cube(var, lead, s, **kw) for s in seasons_of(var, lead)]


if __name__ == "__main__":
    for var, lead in inventory():
        summary(var, lead)
