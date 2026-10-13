"""Reduce the multi-model CMIP6 DCPP-A hindcasts to (start, member) lead-window cubes.

    python build_dcpp_cube.py --plan                     # what would be built
    python build_dcpp_cube.py --var SLP                  # lead 2-4, the four models
    python build_dcpp_cube.py --var TREFHT --models MIROC6

Writes one cube per (model, variable, lead window), in the schema `_cube_io.Cube` reads:

    data/dcpp_cubes/<MODEL>/<VAR>_lead<a>-<b>.npz
        cube   (J, N, lat, lon)   float32, raw -- NOT detrended; on the obs grid
        obs    (J, lat, lon)      float32, the observations on the same windows
        obs_ok (J,)               bool, False where the window runs past the obs record
        year, month (J,)          start year (sYYYY label), init month of lead 1
        members, lat, lon, lead, var, units, source_id, init_month, target_months

Reads data/dcpp_files.json (fetch_dcpp.py --manifest) and the raw files under
$SCRATCH/dcpp/<MODEL>/<sYYYY>/<psl|tas|pr>/. DePreSys4 (HadGEM3-GC31-MM), the fifth model
of the lead 2-4 pool, has its own builder, build_depresys4_cube.py.

ONE CUBE PER MODEL, deliberately: each model is detrended within its own member set on
load (Eade et al. 2014 bias-correct each model separately), then pooled.

Lead 1 is the first output month, read from the manifest rather than reconstructed from
the `sYYYY` label. All four default models begin YYYY-11, so lead 2-4 is DJF. Only
models whose lead window matches the November reference are built.

Chunking varies by model (EC-Earth3 publishes 11 files per (start, member), the others
1): only chunks overlapping the window are opened, and the month count is asserted.
Coverage is ragged (EC-Earth3 has no member complete over all 60 starts), so
`largest_block` trims to the biggest (starts x members) rectangle and prints what it
dropped.

Averaging matches the observations (`obs_grid.obs_windows`): both sides take an
unweighted mean of the same set of monthly means, on the same grid, with the
observations never resampled. The model side is bilinearly interpolated onto that grid.
"""
import argparse
import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import xarray as xr

from obs_grid import UNITS, obs_grid, obs_windows, to_grid
from paths import DCPP_CUBES, DCPP_MANIFEST, DCPP_RAW

warnings.simplefilter("ignore")

MODELS = ["EC-Earth3", "MIROC6", "MPI-ESM1-2-LR", "MRI-ESM2-0"]
CMIP_NAME = {"SLP": "psl", "TREFHT": "tas", "PRECT": "pr"}
SCALE = {"PRECT": 86400.0}                 # kg m-2 s-1 -> mm/day
VARS = list(CMIP_NAME)
REFERENCE_INIT = 11                        # the November systems everything pools with
MIN_STARTS = 20

_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)
_SPAN = re.compile(r"_(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$")


# ------------------------------------------------------------------- manifest use
def load_manifest(path=DCPP_MANIFEST):
    with open(path) as f:
        return json.load(f)


def lead1_index(man, model):
    """{start: absolute month index of lead 1}, from the earliest file of each start.

    Read, not reconstructed from the `sYYYY` label: CanESM5's `s1960` output BEGINS
    1961-01, a year after its label.
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


def largest_block(rows, min_starts=MIN_STARTS):
    """Trim ragged (start, member) coverage to the biggest usable rectangle.

    Dropping members costs ensemble size, dropping starts costs sample size; EC-Earth3
    has no member complete over all 60 starts, so a member-first trim leaves ZERO
    members whereas a start-first trim keeps all 16. So evaluate both directions plus
    the alternating fixed point and take whichever maximises J x N subject to keeping at
    least `min_starts` samples. Returns (starts, members, note).
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

    Opens only the chunks that overlap. Asserts the month count, so a short window
    raises instead of quietly averaging over fewer months than the observations will.
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


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--var", default="SLP", choices=VARS)
    ap.add_argument("--leads", default="2-4",
                    help="lead-month windows relative to EACH model's own init month")
    ap.add_argument("--raw", default=DCPP_RAW)
    ap.add_argument("--outdir", default=DCPP_CUBES)
    ap.add_argument("--manifest-path", default=DCPP_MANIFEST)
    ap.add_argument("--plan", action="store_true",
                    help="report init months, coverage and target alignment; build nothing")
    a = ap.parse_args()

    man = load_manifest(a.manifest_path)
    var, cm = a.var, CMIP_NAME[a.var]
    leads = parse_leads(a.leads)
    LAT, LON = obs_grid(var)
    ref = {lead: target_label(REFERENCE_INIT, lead) for lead in leads}

    print(f"{var} ({cm}) on the {len(LAT)}x{len(LON)} observation grid; "
          f"windows {', '.join(f'{x}-{y}' for x, y in leads)}")

    plans = []
    for model in a.models:
        if cm not in man.get(model, {}):
            print(f"\n{model}: no {cm} in the manifest; skipping")
            continue
        rows = man[model][cm]
        l1 = lead1_index(man, model)
        im = init_month(man, model, l1)
        off = offset_years(man, model, l1)
        starts, members, note = largest_block(rows)
        print(f"\n{model}: lead 1 = month {im} of sYYYY{off:+d}   "
              f"members {len(members)}   starts {len(starts)}")
        print(f"  coverage: {note}")
        if im is None or off is None or not starts or not members:
            print("  inconsistent lead 1 or no rectangular block; skipping")
            continue
        for lead in leads:
            lab = target_label(im, lead)
            if lab != ref[lead] or off != 0:
                print(f"  lead {lead[0]}-{lead[1]}: months {lab} != reference "
                      f"{ref[lead]}; skipping")
                continue
            print(f"  lead {lead[0]}-{lead[1]}: months {lab}  == reference")
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

        # Observations on EXACTLY the model's own target months: pass lead 1's own
        # (year, month) rather than the sYYYY label.
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
