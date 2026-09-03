"""Fetch and parse HadSLP2r monthly SLP (1850-2019) into data/obs/.

    python scripts/fetch_hadslp2r.py      # -> data/obs/hadslp2r_monthly_1850_2019.nc

This is the observational side of the SMYLE verification. The parse is unchanged from
the version on `origin/infomeasure-updates`: same 72x37 ASCII layout, same roll to
0-360 and latitude flip, same hPa*100 -> Pa conversion. Only the download uses
stdlib `urllib` instead of `requests`, so it runs in an env without that dependency.

HadSLP2r rather than HadSLP2 because HadSLP2 ends 2004-12 and SMYLE runs to 2019. Over
the 1850-2004 overlap the two differ by at most 1 Pa, i.e. last-digit rounding.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import gzip
import os
import shutil
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr

NLON, NLAT = 72, 37
SRC = "https://www.metoffice.gov.uk/hadobs/hadslp2/data/hadslp2r.asc.gz"
_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(_HERE, "..", "data", "obs", "hadslp2r.asc.gz")
OUT = os.path.join(_HERE, "..", "data", "obs", "hadslp2r_monthly_1850_2019.nc")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if not os.path.exists(RAW):
        print(f"downloading {SRC}")
        with urlopen(SRC, timeout=600) as r, open(RAW + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(RAW + ".part", RAW)

    with gzip.open(RAW, "rt") as fh:
        tokens = fh.read().split()
    vals, times, i = [], [], 0
    while i < len(tokens):
        year, month = int(tokens[i]), int(tokens[i + 1])
        block = np.array(tokens[i + 2: i + 2 + NLON * NLAT],
                         dtype=np.float64).reshape(NLAT, NLON)
        times.append(pd.Timestamp(year=year, month=month, day=15))
        vals.append(block * 0.01 * 100.0)          # hPa*100 -> hPa -> Pa
        i += 2 + NLON * NLAT

    obs = xr.Dataset({"psl": (("time", "lat", "lon"), np.stack(vals))},
                     coords={"time": times,
                             "lat": np.arange(90.0, -95.0, -5.0),
                             "lon": np.arange(-180.0, 180.0, 5.0)})
    obs = obs.reindex(lat=obs.lat[::-1])
    obs = obs.assign_coords(lon=(obs.lon % 360)).sortby("lon")
    obs.psl.attrs = {"units": "Pa", "long_name": "mean sea level pressure"}
    obs.attrs["source"] = "HadSLP2r bulk ASCII, Met Office Hadley Centre"
    obs.to_netcdf(OUT)
    print(f"{OUT}\n  {obs.sizes['time']} months, "
          f"{obs.time.values[0]} -> {obs.time.values[-1]}")
    print(f"  grid {obs.sizes['lat']}x{obs.sizes['lon']}, "
          f"mean {float(obs.psl.mean()):.1f} Pa")


if __name__ == "__main__":
    main()
