# data-acquisition

From raw downloads to the per-case input arrays the analysis reads,
`data/cases/<VAR>_<case>.npz`, and nothing more.

All paths come from `paths.py`: the data root is `$SNP_DATA`, else `<repo>/data`. Raw
DCPP files go to `$SNP_DCPP_RAW`, else `$SCRATCH/dcpp`. Scripts import each other as flat
modules, so run them from this folder (`python make_cases.py`) or put it on `sys.path`.

## Order

```
# 1. fetch (network; resumable; nothing here needs credentials)
python fetch_cmip5_thesis.py                 # CMIP5 decadal1961 psl (49 files) + HadSLP2
python fetch_hadslp2r.py                     # HadSLP2r, the SLP obs for DCPP (also parses it)
python fetch_obs.py                          # NCEP R1 2 m T (TREFHT), GPCP (PRECT)
python fetch_depresys4.py --manifest && python fetch_depresys4.py
python fetch_dcpp.py --manifest --with-replicas && python fetch_dcpp.py

# 2. build
python build_cmip5_thesis_ensemble.py        # thesis ensemble + HadSLP2 netCDF
for v in SLP TREFHT PRECT; do
  python build_depresys4_cube.py --var $v    # lead 2-4 (the cubes on disk: --leads 2-4,13-60)
  python build_dcpp_cube.py --var $v         # lead 2-4, EC-Earth3 MIROC6 MPI-ESM1-2-LR MRI-ESM2-0
done
python build_dcpp_monthly_cache.py           # SLP/TREFHT s1961, PRECT s1978, six models

# 3. cases
python make_cases.py                         # all nine, or e.g.: python make_cases.py SLP s1961
```

Steps 2-3 need the obs first: the obs grids define the grid every model is put onto.
The monthly cache is optional (`make_cases.py` builds a missing cache file on the fly),
but building it is the slow part, so step 2 does it once up front.

## Cases

`make_cases.py` writes `data/cases/<VAR>_<case>.npz` with `np.savez` (uncompressed):
`F` (N, T, ny, nx), `o` (T, ny, nx), `s` (T, ny, nx), all float64, and `lats`, `lons`.
`s` is `np.nanmean(F, axis=0)`, except for cmip5 where it is calc_RPC's xarray member mean.

| case | VAR | what | loader |
|---|---|---|---|
| `cmip5` | SLP | thesis CMIP5 decadal1961, 49 members, T = 120 months (1962-1971), 64 x 128 | `cmip5_thesis.arrays` (calc_RPC cell 3) |
| `s1961` | SLP, TREFHT | DCPP-A s1961, six models, monthly Jan 1962 - Dec 1971, obs grid, Eade smoothing, then `thesis_detrend` (line incl. intercept per member and cell; seasonal cycle kept) | `dcpp_decadal_handles` |
| `s1961moy` | SLP, TREFHT | as s1961, month-of-year climatology removed (per model; obs separately) before `thesis_detrend` | `dcpp_decadal_handles` |
| `s1978` | PRECT | as s1961 for s1978 (Jan 1979 - Dec 1988; GPCP starts in 1979) | `dcpp_decadal_handles` |
| `2-4` | SLP, TREFHT, PRECT | DJF means (lead months 2-4), one per start date, five November models pooled (N = 62); smoothed, detrended per model over the common start dates; SLP also has the area-weighted global mean removed | `dcpp_handles` |

Models. Lead 2-4 (`dcpp_handles.MODELS`, pinned): HadGEM3-GC31-MM (DePreSys4) 10,
EC-Earth3 16, MIROC6 10, MPI-ESM1-2-LR 16, MRI-ESM2-0 10 members. Decadal
(`dcpp_decadal_handles.MODELS`, pinned): CanESM5, EC-Earth3, HadGEM3-GC31-MM, MIROC6,
MPI-ESM1-2-LR, NorCPM1 (every model whose runs reach 120 months; MRI-ESM2-0's are 62).

## Scripts: what each reads and writes (paths under `data/`)

| script | reads | writes |
|---|---|---|
| `paths.py` | - | (the path constants) |
| `fetch_cmip5_thesis.py` | `cmip5_psl_provenance.json` (here; URL, SHA256, size per file) | `models/<model>/psl/*.nc`, `obs/hadslp2/hadslp2.asc.gz` |
| `fetch_hadslp2r.py` | network | `obs/hadslp2r.asc.gz`, `obs/hadslp2r_monthly_1850_2019.nc` |
| `fetch_obs.py` | network | `obs/ncep_air2m.mon.mean.nc` -> `obs/ncep_air2m_monthly.nc`, `obs/gpcp_precip.mon.mean.nc` -> `obs/gpcp_precip_monthly.nc` |
| `fetch_depresys4.py` | ESGF search | `depresys4_files.json`, `depresys4/<sYYYY>/<var>/*.nc` |
| `fetch_dcpp.py` | ESGF search | `dcpp_files.json`, `$SCRATCH/dcpp/<model>/<sYYYY>/<var>/*.nc` |
| `build_cmip5_thesis_ensemble.py` | `models/`, `obs/hadslp2/hadslp2.asc.gz` | `ensembles/large_psl_decadal_ensemble_no_extrapolate.nc`, `obs/hadslp2/hadslp2_monthly_1850_2004.nc` |
| `build_depresys4_cube.py` | `depresys4/`, obs | `depresys4_cubes/<VAR>_lead2-4.npz` |
| `build_dcpp_cube.py` | `dcpp_files.json`, `$SCRATCH/dcpp/`, obs | `dcpp_cubes/<model>/<VAR>_lead2-4.npz` |
| `build_dcpp_monthly_cache.py` | both manifests, `$SCRATCH/dcpp/`, `depresys4/`, obs grid | `dcpp_monthly_cache/<model>/<VAR>_s<YYYY>_<YYYY+1>-01_120mo.npz` |
| `make_cases.py` | the ensemble + HadSLP2 nc, the cubes, the cache, obs | `cases/<VAR>_<case>.npz` |

Modules without a CLI: `obs_grid.py` (obs file per variable, obs grid, bilinear
regridding with cyclic/pole padding, obs window means), `_smooth.py` (Eade box mean),
`_cube_io.py` (one model's lead-window cube: global-mean removal, smoothing, per-init-month
detrend), `dcpp_handles.py`, `dcpp_decadal_handles.py`, `cmip5_thesis.py`.

## Data sources

| data | source |
|---|---|
| CMIP5 decadal1961 psl | CEDA ESGF replica, `https://esgf.ceda.ac.uk/thredds/fileServer/esg_dataroot/cmip5/output1/...` (one URL per file in `cmip5_psl_provenance.json`) |
| HadSLP2 1850-2004 | https://www.metoffice.gov.uk/hadobs/hadslp2/data/hadslp2.asc.gz |
| HadSLP2r 1850-2019 | https://www.metoffice.gov.uk/hadobs/hadslp2/data/hadslp2r.asc.gz |
| NCEP/NCAR R1 2 m air temperature | https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface_gauss/air.2m.mon.mean.nc |
| GPCP v2.3 precipitation | https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc |
| DePreSys4 (HadGEM3-GC31-MM dcppA-hindcast) | ESGF search at esgf-data.dkrz.de, files from CEDA |
| CMIP6 dcppA-hindcast, six models | ESGF-1.5 bridge https://esgf-node.ornl.gov/esgf-1-5-bridge, files from the publishing data nodes and replicas |

## Manual steps and caveats

- No download is manual. `cmip5_psl_provenance.json` is the list behind the CMIP5 fetch;
  how it was originally assembled (which ESGF query, why these 49 members) is not
  scripted. It is a copy of `data/models/PROVENANCE_psl.json`.
- The two ESGF manifests (`dcpp_files.json`, 30 MB; `depresys4_files.json`, 8 MB) are not
  in git. Re-querying ESGF can give a different file list, and the DCPP cube builder
  trims each model to its largest complete (start x member) block from the manifest.
  Keep the existing manifests rather than regenerating them, if you want the cubes to
  come out the same.
- The on-disk MIROC6 lead 2-4 cubes hold 61 starts (s1960-s2020), while the current
  manifest gives 62 (s2021 too). No other pooled model has s2021 (EC-Earth3 stops at
  s2017), so it falls outside the common start dates, and the pooled lead 2-4 cases do
  not depend on it.
- Raw DCPP files are 215 GB under `$SCRATCH/dcpp`, which is purged after 90 days without
  modification. The cubes and the monthly cache on Oak are all `make_cases.py` needs. To
  keep the raw files, move them to Oak (and set `$SNP_DCPP_RAW`).

## Verification

Checked against the code this folder was trimmed from (branch `jugaad-fd-optbins`):
every `cases/*.npz` equals the old `load(var, lead)` output bitwise (F, o, s, lats, lons),
for all nine cases. Rebuilt into a temporary data root and compared bitwise with the files
on disk: the HadSLP2r, NCEP, GPCP and HadSLP2 obs files, the CMIP5 thesis ensemble, and a
sample of the cubes (DePreSys4 SLP 2-4 for s1960-s1964, MRI-ESM2-0 SLP 2-4, the MIROC6 SLP
s1961 monthly cache).
