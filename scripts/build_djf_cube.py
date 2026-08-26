"""Build the DJF hindcast cube: one winter mean per (start date, member), on the N48 grid.

    python scripts/build_djf_cube.py            # -> data/djf_cube.npz

`notebooks/granger_lin_lambda_m.ipynb` runs on a single start date (decadal1961) with
T=120 monthly steps, so its sample axis is *time within one hindcast*. This cube gives the
other axis: J=45 start dates, one seasonal mean each, so a per-cell series is 45 long and
successive entries are independent forecasts rather than successive months.

Source is the raw monthly psl already fetched under `.scratch/eade_replication/fetch`
(1305 files, 9.5 GB, start dates decadal1960..decadal2005). The grid, the periodic
padding and the lead-time convention are taken from `.scratch/eade_replication/code/
build_cube.py`, which is where they were established and verified against Eade's
smoothing lengths; `pad_cyclic`/`to_n48` are that file's, unchanged. What is new here is
the seasonal window: that builder took the lead-years-2-5 *annual* mean (48 months), this
one takes DJF.

Lead-time convention, asserted per file: for start-date index j every model's output
begins Jan(1961+j), so lead year 1 is 1961+j. Two DJF windows come out of one pass:

    djf     Dec(1961+j), Jan(1962+j), Feb(1962+j)   -- the first complete winter, lead 12-14 months
    djf25   the four winters inside lead years 2-5, averaged (12 months)

`djf` is the seasonal target and what the notebook uses; `djf25` is the winter-only
analogue of the published lead-2-5 mean, stored so the choice costs one line downstream.

Observations are HadSLP2r over the same windows and the same grid. Members missing from
disk, or with an incomplete window, are left NaN -- the notebook drops them.
"""
import json
import os
import warnings
from multiprocessing import Pool

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")

SCRATCH = os.environ.get(
    "EADE_SCRATCH",
    "/Users/jasperchen/Academics/Research/SNP/information-snp/.scratch/eade_replication")
OUT = "data/djf_cube.npz"

MODELS = ["CanCM4", "GFDL-CM2p1", "MIROC5", "MPI-ESM-LR"]
NMEM = {"CanCM4": 10, "GFDL-CM2p1": 10, "MIROC5": 6, "MPI-ESM-LR": 3}
LABELS = [f"{k}_r{i}i1p1" for k in MODELS for i in range(1, NMEM[k] + 1)]

LAT48 = np.arange(-90.0, 90.0 + 1e-9, 2.5)          # 73
LON48 = np.arange(0.0, 360.0 - 1e-9, 3.75)          # 96


def pad_cyclic(da):
    """Pad longitude periodically and latitude to the poles, so interp never extrapolates."""
    lon = da["lon"].values
    lo = da.isel(lon=[-1]).assign_coords(lon=[lon[-1] - 360.0])
    hi = da.isel(lon=[0]).assign_coords(lon=[lon[0] + 360.0])
    da = xr.concat([lo, da, hi], dim="lon")
    lat = da["lat"].values
    if lat[0] > -90.0:
        s = da.isel(lat=[0]).mean("lon").expand_dims(lon=da["lon"]).transpose(*da.dims)
        da = xr.concat([s.assign_coords(lat=[-90.0]), da], dim="lat")
    if lat[-1] < 90.0:
        n = da.isel(lat=[-1]).mean("lon").expand_dims(lon=da["lon"]).transpose(*da.dims)
        da = xr.concat([da, n.assign_coords(lat=[90.0])], dim="lat")
    return da


def to_n48(da):
    return pad_cyclic(da).interp(lat=LAT48, lon=LON48)


def djf_masks(years, months, j):
    """(first-winter mask, lead-2-5-winters mask) for start-date index j.

    A DJF season is labelled by the January it ends in: winter w of start date j is
    Dec(1961+j+w-1), Jan(1961+j+w), Feb(1961+j+w). w=1 is the first complete winter;
    lead years 2-5 hold w=2..5.
    """
    y1 = 1961 + j

    def winter(w):
        dec = (years == y1 + w - 1) & (months == 12)
        jf = (years == y1 + w) & np.isin(months, (1, 2))
        return dec | jf

    return winter(1), winter(2) | winter(3) | winter(4) | winter(5)


def one_file(task):
    """DJF means from one member file. Returns (j, label, djf, djf25, report)."""
    j, model, member, path = task
    label = f"{model}_{member}"
    if not os.path.exists(path):
        return j, label, None, None, {"file": os.path.basename(path), "j": j,
                                      "ok": False, "note": "missing on disk"}
    d = xr.open_dataset(path, use_cftime=True).psl
    years = np.array([t.year for t in d["time"].values])
    months = np.array([t.month for t in d["time"].values])
    m1, m25 = djf_masks(years, months, j)
    rep = {"file": os.path.basename(path), "j": j,
           "file_first": f"{years[0]}-{months[0]:02d}",
           "file_last": f"{years[-1]}-{months[-1]:02d}",
           "expected_lead1_year": int(1961 + j),
           "n_djf": int(m1.sum()), "n_djf25": int(m25.sum()),
           "ok": bool(m1.sum() == 3 and m25.sum() == 12
                      and years[0] == 1961 + j and months[0] == 1)}
    out = []
    for mask, want in ((m1, 3), (m25, 12)):
        out.append(to_n48(d.isel(time=np.where(mask)[0]).mean("time")).values
                   if mask.sum() == want else None)
    return j, label, out[0], out[1], rep


def obs_djf(js):
    """HadSLP2r DJF means over the same windows and grid, shape (J, 73, 96) each."""
    o = xr.open_dataset(f"{SCRATCH}/data/hadslp2r_monthly_1850_2019.nc").psl
    # HadSLP2r carries a plain datetime64 axis, the model files a cftime one; the .dt
    # accessor is the only year/month route that works on both.
    years = o["time"].dt.year.to_numpy()
    months = o["time"].dt.month.to_numpy()
    first, lead25 = [], []
    for j in js:
        m1, m25 = djf_masks(years, months, j)
        assert m1.sum() == 3 and m25.sum() == 12, (j, m1.sum(), m25.sum())
        first.append(to_n48(o.isel(time=np.where(m1)[0]).mean("time")).values)
        lead25.append(to_n48(o.isel(time=np.where(m25)[0]).mean("time")).values)
    return np.stack(first).astype(np.float32), np.stack(lead25).astype(np.float32)


def main():
    man = json.load(open(f"{SCRATCH}/data/full_manifest.json"))
    js = sorted({m["j"] for m in man})
    tasks = [(m["j"], m["model"], m["member"],
              f"{SCRATCH}/fetch/models/{m['model']}/psl/{m['filename']}") for m in man]

    shape = (len(js), len(LABELS), LAT48.size, LON48.size)
    cube = np.full(shape, np.nan, np.float32)
    cube25 = np.full(shape, np.nan, np.float32)
    report = []

    with Pool(processes=max(1, (os.cpu_count() or 4) - 2)) as pool:
        for k, (j, label, a, b, rep) in enumerate(pool.imap_unordered(one_file, tasks, 8)):
            report.append(rep)
            if label in LABELS:
                jj, mi = js.index(j), LABELS.index(label)
                if a is not None:
                    cube[jj, mi] = a
                if b is not None:
                    cube25[jj, mi] = b
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(tasks)} files", flush=True)

    obs, obs25 = obs_djf(js)
    np.savez_compressed(OUT, cube=cube, cube25=cube25, obs=obs, obs25=obs25,
                        j=np.array(js), members=np.array(LABELS), lat=LAT48, lon=LON48)

    bad = [r for r in report if not r.get("ok")]
    full = np.isfinite(cube).all(axis=(2, 3))
    print(f"\ncube {cube.shape} -> {OUT}")
    print(f"alignment report: {len(report)} files, {len(bad)} failing")
    for r in bad[:10]:
        print("  FAIL", r)
    print(f"complete (start date, member) pairs: {full.sum()}/{full.size}")
    incomplete = [(js[i], int((~full[i]).sum())) for i in range(len(js)) if not full[i].all()]
    if incomplete:
        print("  start dates with missing members (j, count):", incomplete)


if __name__ == "__main__":
    main()
