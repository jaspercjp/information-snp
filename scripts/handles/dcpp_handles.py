"""One handle onto the pooled CMIP6 DCPP grand ensemble, across all models at once.

    import sys; sys.path.insert(0, "../scripts")
    import dcpp_handles as G, smyle_metrics as M

    G.inventory()                          # which models can be pooled, per window
    c = G.get(lead="13-60")                # 4-year mean, lead years 2-5, all models
    c = G.get(lead="2-4", var="TREFHT")    # DJF, the November models only

    rho_o = M.pearson_coeff(c.s, c.G)
    rho_m = M.ensemble_SNR_debiased(c.F)   # RPC = rho_o / rho_m

Presents exactly the interface of `smyle_handles` / `depresys4_handles`, so anything
written against a single system works unchanged on the grand ensemble:

    c.F  (N, J, lat, lon) members          c.YEAR, c.MONTH  (J,) init year, month
    c.G  (J, lat, lon)    observations     c.lats, c.lons, c.members, c.N, c.J
    c.s  (J, lat, lon)    ensemble mean    c.target, c.label, c.units
    c.loo_mean(n)         s_-n             c.gm(field)   area-weighted global mean

plus two things a pooled handle needs:

    c.model_of   (N,)     which model each member came from
    c.by_model   {name: slice}   member-axis slice per model, for like-for-like work

Why this exists: Eade et al. (2014)'s decadal RPC comes from a 70-member MULTI-MODEL
grand ensemble (DePreSys 37 + four CMIP5 models 29), not a single system. A 10-member
DePreSys4 is penalised far harder by the finite-ensemble bias in sigma_sig -- which
Eade does not correct -- so reproducing their result needs comparable ensemble size.

Detrend per model, THEN pool
----------------------------
Each model is loaded through its own `Cube`, so each is detrended within its own
member set, and only then are the member axes concatenated. That ordering is not
cosmetic. Pooling first and detrending once would remove a single blended
climatology, leaving every model's mean offset from that blend sitting in the
anomalies as spurious ensemble-mean variance -- inflating sigma_sig and depressing
RPC. Eade are explicit about correcting each model separately "such that each model
is treated in the same way, regardless of initialization method".

Common start dates
------------------
The models cover different starts (DePreSys4 1960-2018, EC-Earth3 51 of 1960-2019,
CanESM5 1960-2019), so `get` intersects the init YEARS across the models being pooled
and detrends each on that common set. `c.J` is therefore the intersection, not the
largest member. `c.dropped` reports what each model lost.

Aligning on the `sYYYY` label is the right join: it is DCPP's own convention, and for
lead 13-60 it puts CanESM5's Jan-Dec window and the November models' Nov-Oct window
46 of 48 months apart -- the same four years, offset by two months.

What pooling does to rho_m, stated plainly
------------------------------------------
For a multi-model ensemble, sigma_tot picks up inter-model spread as well as internal
variability, and the ensemble mean averages over structural error as well as noise.
That is a different quantity from a single model's predictable fraction -- defensibly
so, since model uncertainty is genuinely part of forecast uncertainty, and it is what
Eade computed. But it is not interchangeable with a single-system rho_m, and models
with more members dominate it. `max_members` subsamples each model to a common count
if you want an equally-weighted pool; `c.by_model` lets you recompute per model.

Observations
------------
Every model's cube carries observations on its own target months. Pooled, there can
be only one observed series, so the NOVEMBER convention is used (the reference these
windows are defined against) and the largest discrepancy against any other member
model's obs is reported as `c.obs_spread`. For lead 13-60 that is CanESM5's two-month
offset: max 84.5 Pa but RMS 13.3 Pa against an obs sd of 35.6, and the two obs series
correlate 0.976 per cell on area average (worst cell 0.836). It costs a little skill;
it is not why SLP looks the way it does.

SLP needs `remove_gm=True`
--------------------------
EC-Earth3's dcppA-hindcast `psl` carries a spatially uniform mass offset that varies
by start date with sd 28.8 Pa (range 101054-101225 Pa), against 1.1 Pa for DePreSys4,
5.8 Pa for CanESM5 and 6.3 Pa in HadSLP2r. Being uniform and shared by every member,
it survives ensemble averaging untouched and contaminates the ensemble mean of every
grid cell. It is the same size at every lead, but it competes with a 48-month mean's
much smaller real variance: it is 5% of the lead 2-4 ensemble-mean variance and 50%
of the lead 13-60 one. Pass `remove_gm=True` for any SLP work, which takes the
area-weighted global mean off model and observations alike. At lead 13-60 that moves
global rho_o from -0.006 to +0.101 and the tropics from +0.014 to +0.201. Leave it
off for TREFHT, whose global mean is real predictable signal.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cube_io import Cube, MONTH_ABBR, canon, target_season   # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data")

# model -> cube directory. DePreSys4 IS a DCPP model (MOHC dcppA-hindcast); its cubes
# predate the multi-model tree, so they live in their own directory. Everything else
# is discovered from data/dcpp_cubes/<MODEL>/, which is exactly where
# build_dcpp_cube.py writes -- so a newly built model joins the pool without an edit
# here. Ten more models in the archive span 1960-2014 and could be added; see
# `available()` for what is actually on disk.
CUBE_DIRS = {"HadGEM3-GC31-MM": os.path.join(_DATA, "depresys4_cubes")}
_TREE = os.path.join(_DATA, "dcpp_cubes")
if os.path.isdir(_TREE):
    for _m in sorted(os.listdir(_TREE)):
        if os.path.isdir(os.path.join(_TREE, _m)):
            CUBE_DIRS.setdefault(_m, os.path.join(_TREE, _m))
ALIAS = {"DePreSys4": "HadGEM3-GC31-MM"}
REFERENCE_INIT = 11             # the convention pooled windows are defined against


def _path(model, var, lead):
    return os.path.join(CUBE_DIRS[model], f"{canon(var)}_lead{lead}.npz")


def _meta(model, var, lead):
    """(year, month, obs_ok) straight from the npz header, without loading cubes."""
    z = np.load(_path(model, var, lead), mmap_mode="r")
    return (np.asarray(z["year"]), np.asarray(z["month"]), np.asarray(z["obs_ok"]))


def available(var="SLP", lead="13-60"):
    """The models that actually have a cube for this (variable, window)."""
    return [m for m in CUBE_DIRS if os.path.exists(_path(m, var, lead))]


def inventory():
    """Print which models can be pooled for each (variable, window) on disk."""
    found = {}
    for var in ("SLP", "TREFHT", "PRECT"):
        for lead in ("2-4", "13-60", "1-3"):
            got = available(var, lead)
            if got:
                found[(var, lead)] = got
    if not found:
        print("no poolable cubes; build them with scripts/build_dcpp_cube.py")
        return found
    print("poolable DCPP cubes:")
    for (var, lead), models in sorted(found.items()):
        bits = []
        tot = 0
        for m in models:
            y, mo, ok = _meta(m, var, lead)
            z = np.load(_path(m, var, lead), mmap_mode="r")
            n = z["cube"].shape[1]
            tot += n
            bits.append(f"{m}(N={n},J={len(y)},init={int(mo[0])})")
        print(f"  {var:7s} lead {lead:6s} -> {tot:3d} members: {', '.join(bits)}")
    return found


class Pooled:
    """The grand ensemble as one Cube-shaped object. Built by `get`."""

    def __init__(self, var, lead, models=None, require_obs=None, detrend_data=True,
                 smooth=True, max_members=None, years=None, remove_gm=False,
                 verbose=True):
        var = canon(var)
        models = [ALIAS.get(m, m) for m in (models or available(var, lead))]
        models = [m for m in models if os.path.exists(_path(m, var, lead))]
        if not models:
            raise ValueError(f"no cubes for {var} lead {lead}; "
                             f"try scripts/build_dcpp_cube.py")

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
        common = set.intersection(*sets.values()) if sets else set()
        if years is not None:
            common &= set(int(v) for v in years)
        if not common:
            raise ValueError(f"no start years common to {models} for {var} lead {lead}")
        common = sorted(common)
        self.dropped = {m: len(sets[m]) - len(common) for m in models}

        # ---- load each model separately, so each detrends on its own member set
        parts, names, owner, g_by_model = [], [], [], {}
        ref_obs, ref_model = None, None
        first = None
        for m in models:
            season = MONTH_ABBR[inits[m]]
            c = Cube(var, lead, season, cubes=CUBE_DIRS[m], years=common,
                     require_obs=require_obs, detrend_data=detrend_data,
                     smooth=smooth, remove_gm=remove_gm)
            if first is None:
                first = c
            else:
                if not (np.allclose(c.lats, first.lats)
                        and np.allclose(c.lons, first.lons)):
                    raise ValueError(f"{m} is on a different grid "
                                     f"({len(c.lats)}x{len(c.lons)}) than "
                                     f"{models[0]} ({len(first.lats)}x{len(first.lons)})"
                                     f"; rebuild onto the observations' grid")
                if not np.array_equal(c.YEAR, first.YEAR):
                    raise ValueError(f"{m} start years do not line up after the "
                                     f"intersection -- this is a bug, not data")
            F = c.F
            if max_members is not None and F.shape[0] > max_members:
                F = F[:max_members]
            parts.append(F)
            names += [f"{m}:{s}" for s in np.asarray(c.members)[:F.shape[0]]]
            owner += [m] * F.shape[0]
            g_by_model[m] = c.G
            if inits[m] == REFERENCE_INIT and ref_obs is None:
                ref_obs, ref_model = c.G, m

        # Observations: one series for the pool, on the November convention that the
        # windows are defined against. Kept from the pass above rather than reloading
        # -- each reload is a full cube off disk.
        if ref_obs is None:                       # no November model in the pool
            ref_model = models[0]
            ref_obs = g_by_model[ref_model]
        self.G = ref_obs
        self.obs_model = ref_model
        self.obs_spread = {
            m: (float(np.nanmax(np.abs(g - ref_obs)))
                if np.isfinite(g).any() and np.isfinite(ref_obs).any() else float("nan"))
            for m, g in g_by_model.items() if m != ref_model}

        self.F = np.concatenate(parts, axis=0)
        self.members = np.array(names)
        self.model_of = np.array(owner)
        self.models = models
        self.init_months = inits
        self.by_model = {}
        i = 0
        for m, F in zip(models, parts):
            self.by_model[m] = slice(i, i + F.shape[0])
            i += F.shape[0]

        self.var, self.lead, self.season = var, lead, "pooled"
        self.units, self.lats, self.lons = first.units, first.lats, first.lons
        self.YEAR = first.YEAR
        self.MONTH = np.full(len(self.YEAR), REFERENCE_INIT)
        self.J, self.N = self.F.shape[1], self.F.shape[0]
        self.detrended = detrend_data
        self.smoothed = first.smoothed
        self.smooth_box_deg = first.smooth_box_deg
        self.gm_removed = first.gm_removed
        self.has_obs = bool(np.isfinite(self.G).any())
        self.W = (np.cos(np.deg2rad(self.lats))[:, None]
                  * np.ones(len(self.lons)))
        self.target = target_season(REFERENCE_INIT, lead)
        if verbose:
            print(repr(self))
            for m in models:
                sl = self.by_model[m]
                print(f"    {m:18s} N={sl.stop - sl.start:3d}  init month "
                      f"{inits[m]:2d}  dropped {self.dropped[m]} start(s)"
                      + ("" if m not in self.obs_spread else
                         f"  [its obs differ from {self.obs_model}'s by <= "
                         f"{self.obs_spread[m]:.4g} {self.units}]"))

    @property
    def s(self):
        return self.F.mean(axis=0)

    def loo_mean(self, n):
        return (self.F.sum(axis=0) - self.F[n]) / (self.N - 1)

    def gm(self, a):
        m = np.isfinite(a)
        return float((a[m] * self.W[m]).sum() / self.W[m].sum())

    def model_mean(self, model):
        """Ensemble mean of one model within the pool, on the pooled start dates."""
        return self.F[self.by_model[model]].mean(axis=0)

    @property
    def label(self):
        return (f"DCPP grand ensemble ({len(self.models)} models) {self.var}, "
                f"lead {self.lead} -> {self.target}")

    def __repr__(self):
        box = (f", smooth {self.smooth_box_deg[0]:g}x{self.smooth_box_deg[1]:g}deg"
               if self.smoothed else ", UNSMOOTHED")
        return (f"<Pooled {self.label} [{self.units}]: N={self.N} J={self.J}"
                f"{'' if self.has_obs else ', no obs'}"
                f"{'' if self.detrended else ', RAW'}{box}"
                f"{', gm removed' if self.gm_removed else ''}>")


def get(lead="13-60", var="SLP", models=None, **kw):
    """The pooled grand ensemble for one (variable, window)."""
    return Pooled(var, lead, models=models, **kw)


if __name__ == "__main__":
    inventory()
    for lead in ("2-4", "13-60"):
        for var in ("SLP", "TREFHT", "PRECT"):
            try:
                print()
                get(lead=lead, var=var)
            except Exception as e:
                print(f"  {var} lead {lead}: {type(e).__name__}: {e}")
