"""Fetch the observational fields that verify TREFHT and PRECT, into data/obs/.

    python scripts/fetch_obs.py             # both
    python scripts/fetch_obs.py --var PRECT

SLP has its own script, `fetch_hadslp2r.py`; this one covers the other two.

    TREFHT  <- NCEP/NCAR Reanalysis 1, 2 m air temperature, T62 Gaussian, 1948-
    PRECT   <- GPCP v2.3 satellite+gauge precipitation, 2.5 deg, 1979-

Both come from NOAA PSL over plain HTTPS, no credentials. Output is normalised so
`build_smyle_cube.py` can treat every observational file the same way: latitude
ascending, longitude 0-360, one 3-D variable named `obs`, units already matching the
corresponding model cube (K for TREFHT, mm/day for PRECT -- no conversion needed at
verification time).

Two things to be aware of, both recorded in the file attributes:

**NCEP R1 is a reanalysis, not an observational analysis.** It is 2 m air temperature
over land AND ocean, which is the exact TREFHT variable -- unlike Berkeley Earth or
HadCRUT, whose ocean component is SST. The price is that it is partly model output.

**GPCP starts in 1979**, so it verifies only 164 of the 248 SMYLE initializations
(41 per season against 62). Global observed monthly precipitation does not exist before
the satellite era; the alternative is a land-only gauge analysis. `obs_ok` in the cube
marks which initializations are usable, and `smyle_handles` drops the rest.

PSL asks to be acknowledged: "data provided by the NOAA PSL, Boulder, Colorado, USA,
from their website at https://psl.noaa.gov".
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import argparse
import os
import shutil
from urllib.request import urlopen

import numpy as np
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
OBSDIR = os.path.join(_HERE, "..", "data", "obs")

SPEC = {
    "TREFHT": dict(
        url="https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/"
            "surface_gauss/air.2m.mon.mean.nc",
        raw="ncep_air2m.mon.mean.nc",
        out="ncep_air2m_monthly.nc",
        var="air", units="K", scale=1.0, offset=0.0,
        source="NCEP/NCAR Reanalysis 1 monthly 2 m air temperature, NOAA PSL",
    ),
    "PRECT": dict(
        url="https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc",
        raw="gpcp_precip.mon.mean.nc",
        out="gpcp_precip_monthly.nc",
        var="precip", units="mm/day", scale=1.0, offset=0.0,
        source="GPCP v2.3 monthly satellite-gauge precipitation, NOAA PSL",
    ),
}


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return
    print(f"  downloading {url}")
    with urlopen(url, timeout=900) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(path + ".part", path)


def normalise(spec, raw_path, out_path):
    """One 3-D `obs` variable, latitude ascending, longitude 0-360, CF time."""
    # PSL files carry a time_bnds whose fill value overflows the CF decoder, and it is
    # not needed here -- drop it rather than fight it.
    d = xr.open_dataset(raw_path,
                        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
                        drop_variables=["time_bnds", "climatology_bounds"])
    a = d[spec["var"]].astype("float64") * spec["scale"] + spec["offset"]

    if a["lat"].values[1] < a["lat"].values[0]:            # NCEP is north-to-south
        a = a.reindex(lat=a["lat"][::-1])
    a = a.assign_coords(lon=(a["lon"] % 360)).sortby("lon")

    out = xr.Dataset({"obs": a.transpose("time", "lat", "lon")})
    out["obs"].attrs = {"units": spec["units"], "source": spec["source"]}
    out.attrs["source"] = spec["source"]
    out.attrs["note"] = "normalised by scripts/fetch_obs.py: lat ascending, lon 0-360"
    out.to_netcdf(out_path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="all", choices=["all"] + list(SPEC))
    a = ap.parse_args()
    os.makedirs(OBSDIR, exist_ok=True)

    for var in (list(SPEC) if a.var == "all" else [a.var]):
        spec = SPEC[var]
        raw = os.path.join(OBSDIR, spec["raw"])
        out = os.path.join(OBSDIR, spec["out"])
        print(f"{var}:")
        download(spec["url"], raw)
        d = normalise(spec, raw, out)
        t = d["obs"]["time"].values
        print(f"  {out}")
        print(f"    {d.sizes['time']} months, {t[0]} -> {t[-1]}")
        print(f"    grid {d.sizes['lat']}x{d.sizes['lon']}, "
              f"lat {float(d.lat[0]):+.2f}..{float(d.lat[-1]):+.2f}, "
              f"lon {float(d.lon[0]):.2f}..{float(d.lon[-1]):.2f}")
        print(f"    mean {float(d.obs.mean()):.3f} {spec['units']}, "
              f"NaN {float(np.isnan(d.obs.values).mean()):.4f}")
        d.close()


if __name__ == "__main__":
    main()
