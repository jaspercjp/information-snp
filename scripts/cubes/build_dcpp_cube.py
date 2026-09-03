"""Reduce the multi-model CMIP6 DCPP hindcasts to (init, member) cubes, one per model.

    python scripts/build_dcpp_cube.py --plan                     # what would be built
    python scripts/build_dcpp_cube.py --var SLP --leads 13-60
    python scripts/build_dcpp_cube.py --models MIROC6 MRI-ESM2-0 --leads 2-4,13-60

Writes the same schema as `build_smyle_cube.py` / `build_depresys4_cube.py`, so the
cubes load through `_cube_io.Cube` unchanged:

    data/dcpp_cubes/<MODEL>/<VAR>_lead<a>-<b>.npz

ONE CUBE PER MODEL, deliberately. Eade et al. (2014) bias-correct each model
separately before pooling ("such that each model is treated in the same way,
regardless of initialization method"), and `_cube_io.detrend` removes the mean and
trend within a cube. Pooling first would remove one blended climatology instead of
one per model and leave inter-model mean offsets masquerading as signal. So: build
per model, detrend per model via the handles, then concatenate on the member axis.

Three things this has to handle that the single-model builders did not
--------------------------------------------------------------------
**Initialisation month varies by model.** CanESM5's `s1960` output begins 1961-01;
DePreSys4, EC-Earth3, MIROC6 and MRI-ESM2-0 all begin YYYY-11. Lead 1 is the first
output month, so a lead window means different calendar months in different models.
The init month is read from the manifest, never assumed, and `--require-target`
refuses to build a window whose target months do not match the reference November
system. Consequence worth knowing: CanESM5 has no complete DJF at lead 2-4 at all
(its first is lead 12-14), so it cannot join a seasonal pool -- only the 4-year one,
where a Jan-Dec window shares 46 of 48 months with a Nov-Oct one.

**Chunking varies by model.** EC-Earth3 publishes 11 files per (start, member);
CanESM5, MIROC6 and MRI-ESM2-0 publish 1. Only chunks overlapping the requested
window are opened, and the month count is asserted, so a short window fails loudly
rather than averaging over whatever it found.

**Coverage is ragged.** CanESM5 `psl` is published for only 20 of its 40 members
across all starts; EC-Earth3 has no member complete over all 60 starts. A cube must
be rectangular, so `largest_block` trims to the biggest (starts x members)
rectangle and prints exactly what it dropped.

Averaging matches the observations, which is the point of reusing `obs_windows`
from `build_smyle_cube`: both sides take an unweighted mean of the same set of
monthly means, on the same grid, with the observations never resampled. Note the
model side is bilinearly interpolated onto that grid -- point-sampled, not
area-averaged -- so it retains sub-grid variance the observational analyses have
already smoothed away. That inflates sigma_tot and therefore depresses rho_m.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import argparse
import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import xarray as xr

from build_smyle_cube import OBS_SPEC, UNITS, obs_grid, obs_windows, target_months, to_grid

warnings.simplefilter("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
MANIFEST = os.path.join(ROOT, "data", "dcpp_files.json")
RAW = os.path.join(os.environ.get("SCRATCH", os.path.join(ROOT, "data")), "dcpp")
OUTDIR = os.path.join(ROOT, "data", "dcpp_cubes")

CMIP_NAME = {"SLP": "psl", "TREFHT": "tas", "PRECT": "pr"}
SCALE = {"PRECT": 86400.0}                 # kg m-2 s-1 -> mm/day
VARS = list(CMIP_NAME)
REFERENCE_INIT = 11                        # the November systems everything pools with

_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)
_SPAN = re.compile(r"_(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$")


# ------------------------------------------------------------------- manifest use
def load_manifest(path=MANIFEST):
    with open(path) as f:
        return json.load(f)


def lead1_index(man, model):
    """{start: absolute month index of lead 1}, from the earliest file of each start.

    Lead 1 is the first output month, and it must be read rather than reconstructed
    from the `sYYYY` label. The two disagree: CanESM5's `s1960` output BEGINS
    1961-01, so `s{Y}` initialises in January of Y+1, not of Y. Deriving lead 1 as
    `Y*12 + init_month - 1` therefore lands a whole year early for CanESM5 -- which
    would silently verify every window against the wrong observed months. DePreSys4,
    EC-Earth3, MIROC6 and MRI-ESM2-0 all begin Y-11, where the two agree.
    """
    earliest = {}
    for rows in man[model].values():
        for r in rows:
            if not r["first_month"]:
                continue
            k = r["start"]
            if k not in earliest or r["first_month"] < earliest[k]:
                earliest[k] = r["first_month"]
    out = {}
    for s, fm in earliest.items():
        y, m = int(fm[:4]), int(fm[5:])
        out[s] = y * 12 + m - 1
    return out


def init_month(man, model, l1=None):
    """The calendar month of lead 1, or None if the manifest disagrees across starts."""
    l1 = l1 if l1 is not None else lead1_index(man, model)
    months = {v % 12 + 1 for v in l1.values()}
    return months.pop() if len(months) == 1 else None


def offset_years(man, model, l1=None):
    """Lead 1's year minus the `sYYYY` label. 0 for a Nov system, 1 for CanESM5."""
    l1 = l1 if l1 is not None else lead1_index(man, model)
    offs = {v // 12 - int(s[1:]) for s, v in l1.items()}
    return offs.pop() if len(offs) == 1 else None


def largest_block(rows, min_starts=20):
    """Trim ragged (start, member) coverage to the biggest usable rectangle.

    A cube must be rectangular, but the two ways of getting there are not
    equivalent. Dropping *members* costs ensemble size, which is what rho_m depends
    on and the whole reason for building a grand ensemble; dropping *starts* costs
    sample size for the correlation. EC-Earth3 makes the difference stark: no member
    is complete over all 60 starts, so a member-first trim leaves ZERO members,
    whereas a start-first trim keeps all 16.

    So evaluate both directions plus the alternating fixed point, and take whichever
    maximises J x N subject to keeping at least `min_starts` samples. Returns
    (starts, members, note).
    """
    pairs = {(r["start"], r["member"]) for r in rows}
    all_s = sorted({s for s, _ in pairs})
    all_m = sorted({m for _, m in pairs})

    def drop_starts_first():
        m = list(all_m)
        s = [x for x in all_s if all((x, y) in pairs for y in m)]
        return s, m

    def drop_members_first():
        m = [y for y in all_m if all((x, y) in pairs for x in all_s)]
        return (list(all_s) if m else []), m

    def alternate():
        s, m = list(all_s), list(all_m)
        for _ in range(50):
            m2 = [y for y in m if all((x, y) in pairs for x in s)]
            s2 = [x for x in s if all((x, y) in pairs for y in m2)]
            if (s2, m2) == (s, m) or not m2 or not s2:
                return s2, m2
            s, m = s2, m2
        return s, m

    cands = {"starts-first": drop_starts_first(),
             "members-first": drop_members_first(),
             "alternating": alternate()}
    ok = {k: v for k, v in cands.items()
          if v[0] and v[1] and len(v[0]) >= min_starts}
    if not ok:
        ok = {k: v for k, v in cands.items() if v[0] and v[1]}
    if not ok:
        return [], [], "no rectangular block survives"
    best = max(ok, key=lambda k: len(ok[k][0]) * len(ok[k][1]))
    s, m = ok[best]
    note = (f"{best}: kept {len(s)}/{len(all_s)} starts x {len(m)}/{len(all_m)} "
            f"members = {len(s)*len(m)} cells"
            + ("" if len(ok) == 1 else
               "; alternatives " + ", ".join(
                   f"{k} {len(v[0])}x{len(v[1])}" for k, v in cands.items()
                   if k != best)))
    return s, m, note


# ----------------------------------------------------------------------- reading
def read_one(paths, want, cm, scale, lat, lon):
    """Unweighted mean over exactly the months in `want`, on the target grid.

    Opens only the chunks that overlap. Asserts the month count, so a window that
    is short (a truncated run, a missing chunk) raises instead of quietly averaging
    over fewer months than the observations will.
    """
    acc = None
    got = []
    native = None
    for p in paths:
        m = _SPAN.search(os.path.basename(p))
        if m:
            lo = int(m.group(1)) * 12 + int(m.group(2)) - 1
            hi = int(m.group(3)) * 12 + int(m.group(4)) - 1
            if hi < min(want) or lo > max(want):
                continue
        with xr.open_dataset(p, decode_times=_CODER) as d:
            if "time_bnds" in d:
                stamps = d.time_bnds.values[:, 0]
            else:                                # no bounds: time is mid-month
                stamps = d.time.values
            months = np.array([t.year * 12 + t.month - 1 for t in stamps])
            sel = np.where(np.isin(months, list(want)))[0]
            if sel.size == 0:
                continue
            a = d[cm].values[sel]
            if native is None:
                native = (d["lat"].values, d["lon"].values)
            part = a.sum(axis=0)
            acc = part if acc is None else acc + part
            got += months[sel].tolist()
    if sorted(got) != sorted(want):
        raise ValueError(f"found {len(got)} of {len(want)} months "
                         f"({sorted(set(want) - set(got))[:3]}... missing)")
    mean = acc / len(got) * scale
    da = xr.DataArray(mean, dims=("lat", "lon"),
                      coords={"lat": native[0], "lon": native[1]})
    return to_grid(da, lat, lon).values


def parse_leads(text, nmonth=122):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        a, b = int(a), int(b or a)
        if not 1 <= a <= b <= nmonth:
            raise SystemExit(f"lead window {part!r} outside 1-{nmonth}")
        out.append((a, b))
    return out


def target_label(im, lead):
    """The calendar months a window covers, for a model initialising in month `im`."""
    a, b = lead
    first = ((im - 1 + a - 1) % 12) + 1
    last = ((im - 1 + b - 1) % 12) + 1
    return f"{first:02d}..{last:02d}"


def window_months(l1, lead):
    """The absolute months a lead window covers, given lead 1's absolute index."""
    a, b = lead
    return [l1 + k - 1 for k in range(a, b + 1)]


def target_overlap(l1_abs, start_year, lead):
    """Fraction of target months shared with a November system on the same `sYYYY`.

    A label mismatch is not automatically fatal. CanESM5's lead 13-60 covers Jan-Dec
    of forecast years 2-5 where DePreSys4's covers Nov-Oct: different labels, but 46
    of 48 months in common, so the four-year means target essentially the same
    period. At lead 2-4 the same two-month offset is fatal instead -- FMA against
    DJF, nothing shared. This tells the two cases apart.
    """
    mine = set(window_months(l1_abs, lead))
    ref = set(window_months(start_year * 12 + REFERENCE_INIT - 1, lead))
    return len(mine & ref) / len(ref)


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="default: every model in the manifest")
    ap.add_argument("--var", default="SLP", choices=VARS)
    ap.add_argument("--leads", default="13-60",
                    help="lead-month windows relative to EACH model's own init month")
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--manifest-path", default=MANIFEST)
    ap.add_argument("--plan", action="store_true",
                    help="report init months, coverage and target alignment; build nothing")
    ap.add_argument("--min-overlap", type=float, default=0.9,
                    help="skip a window sharing less than this fraction of its target "
                         "months with a November init. 0.9 keeps CanESM5's 4-year "
                         "window (46/48 shared) and rejects its FMA-for-DJF (0/3)")
    ap.add_argument("--min-starts", type=int, default=20,
                    help="when trimming ragged coverage, do not fall below this many "
                         "start dates just to keep members")
    ap.add_argument("--drop-members", nargs="+", default=None,
                    help="variant_labels to exclude before trimming, as `MODEL:member` "
                         "or bare `member` to apply to every model. Qualify them: "
                         "r6i1p1f1 exists in MIROC6, MRI-ESM2-0 and MPI-ESM1-2-LR too, "
                         "so a bare label costs a member in each. "
                         "`largest_block` maximises J x N over ALL "
                         "starts, so a single member missing one start year costs that "
                         "year rather than the member -- and after the obs_ok trim "
                         "(HadSLP2r ends 2019-12, so lead 13-60 stops at s2014) that "
                         "trade is usually the wrong way round. Dropping the blocking "
                         "member by hand is the fix. NorCPM1 r6i1p1f1 lacks s1989: "
                         "keeping it costs start year 1989, dropping it costs 1 of 20 "
                         "members. EC-Earth3 r1i4p1f1 lacks s2005-s2010 and r3i1p1f1 "
                         "lacks s1976, together worth 7 start years.")
    ap.add_argument("--i-index", default=None,
                    help="keep only members with this i index, e.g. 1 -- EC-Earth3 mixes "
                         "i1/i2/i4, which are different initialisations and are not "
                         "exchangeable perturbations of one another")
    a = ap.parse_args()

    man = load_manifest(a.manifest_path)
    models = a.models or sorted(man)
    var, cm = a.var, CMIP_NAME[a.var]
    leads = parse_leads(a.leads)
    LAT, LON = obs_grid(var)
    ref = {lead: target_label(REFERENCE_INIT, lead) for lead in leads}

    print(f"{var} ({cm}) on the {len(LAT)}x{len(LON)} observation grid; "
          f"windows {', '.join(f'{x}-{y}' for x, y in leads)}")
    print(f"reference (init month {REFERENCE_INIT}): "
          + "  ".join(f"{x}-{y} -> months {ref[(x,y)]}" for x, y in leads))

    plans = []
    for model in models:
        if cm not in man.get(model, {}):
            print(f"\n{model}: no {cm} in the manifest; skipping")
            continue
        rows = man[model][cm]
        if a.i_index:
            rows = [r for r in rows if f"i{a.i_index}p" in r["member"]]
        if a.drop_members:
            drop = {m.split(":", 1)[1] for m in a.drop_members
                    if m.startswith(f"{model}:")}
            drop |= {m for m in a.drop_members if ":" not in m}
            before = {r["member"] for r in rows}
            rows = [r for r in rows if r["member"] not in drop]
            gone = sorted(before & drop)
            if gone:
                print(f"\n{model}: dropped {len(gone)} member(s) on request: {gone}")
        l1 = lead1_index(man, model)
        im = init_month(man, model, l1)
        off = offset_years(man, model, l1)
        starts, members, note = largest_block(rows, min_starts=a.min_starts)
        iidx = sorted({r["member"].split("i")[1].split("p")[0] for r in rows})
        print(f"\n{model}: lead 1 = month {im} of sYYYY{off:+d}   "
              f"members {len(members)} (i-index {iidx})   starts {len(starts)}")
        print(f"  coverage: {note}")
        if im is None or off is None:
            print("  lead 1 is inconsistent across starts; skipping")
            continue
        if not starts or not members:
            print("  no rectangular block; skipping")
            continue
        for lead in leads:
            lab = target_label(im, lead)
            probe = starts[len(starts) // 2]
            ov = target_overlap(l1[probe], int(probe[1:]), lead)
            same = (lab == ref[lead]) and off == 0
            print(f"  lead {lead[0]}-{lead[1]}: months {lab}"
                  + ("  == reference" if same else
                     f"  != reference {ref[lead]}, "
                     f"{ov*100:.0f}% of target months shared"))
            if not same and ov < a.min_overlap:
                print(f"    overlap below --min-overlap {a.min_overlap:.2f}; skipping")
                continue
            plans.append((model, l1, im, lead, starts, members))

    if a.plan:
        print(f"\n{len(plans)} (model, window) cube(s) would be built into {a.outdir}")
        return

    for model, l1, im, lead, starts, members in plans:
        files = defaultdict(list)
        for r in man[model][cm]:
            files[(r["start"], r["member"])].append(
                os.path.join(a.raw, model, r["start"], cm, r["title"]))
        J, N = len(starts), len(members)
        cube = np.full((J, N, len(LAT), len(LON)), np.nan, np.float32)
        years = np.array([int(s[1:]) for s in starts])
        print(f"\n{model} {var} lead {lead[0]}-{lead[1]}: {J} starts x {N} members",
              flush=True)
        for j, s in enumerate(starts):
            want = window_months(l1[s], lead)
            for i, mem in enumerate(members):
                cube[j, i] = read_one(sorted(files[(s, mem)]), want, cm,
                                      SCALE.get(var, 1.0), LAT, LON)
            if (j + 1) % 10 == 0 or j + 1 == J:
                print(f"  {j+1}/{J} starts", flush=True)
        assert np.isfinite(cube).all(), f"NaNs in the {model} {var} lead {lead} cube"

        # Observations on EXACTLY the model's own target months. obs_windows rebuilds
        # them as target_months(y, mm, lead), so pass lead 1's own (year, month)
        # rather than the sYYYY label -- otherwise CanESM5's obs would sit a year off
        # the model's. This is what keeps both sides an unweighted mean of the same
        # set of monthly means.
        inits = [(l1[s] // 12, l1[s] % 12 + 1) for s in starts]
        obs = obs_windows(inits, [lead], var, LAT, LON)
        if obs is not None:
            g, ok = obs[lead]
        else:
            g = np.full((J, len(LAT), len(LON)), np.nan, np.float32)
            ok = np.zeros(J, bool)

        out = os.path.join(a.outdir, model)
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"{var}_lead{lead[0]}-{lead[1]}.npz")
        np.savez_compressed(
            path, cube=cube, obs=g, obs_ok=ok, year=years,
            month=np.full(J, im), members=np.array(members),
            lat=LAT, lon=LON, lead=np.array(lead), var=np.array(var),
            units=np.array(UNITS.get(var, "")), source_id=np.array(model),
            init_month=np.array(im), target_months=np.array(target_label(im, lead)))
        print(f"wrote {path}")
        print(f"  cube {cube.shape}  mean {cube.mean():.4g} {UNITS.get(var,'')}   "
              f"obs {int(ok.sum())}/{J} usable")


if __name__ == "__main__":
    main()
