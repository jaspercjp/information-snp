# Replicating the SMYLE analysis on another machine

Three scripts, in order. Nothing needs credentials.

```bash
python scripts/fetch_hadslp2r.py     #  ~40 MB, seconds   -> data/obs/
python scripts/fetch_smyle.py        #  ~48 GB, ~70 min   -> data/smyle/
python scripts/build_smyle_cube.py   #  ~2 min            -> data/smyle_cube.npz
```

Then open `notebooks/smyle_data.ipynb`.

## What each step gets you

`fetch_hadslp2r.py` downloads the Met Office HadSLP2r ASCII archive and parses it to
netCDF, 1850-2019, 5 degree grid, Pa.

`fetch_smyle.py` pulls monthly PSL, TREFHT and PRECT for CESM2 SMYLE from NSF NCAR
GDEX: the main BSMYLE case, 1958-2019, quarterly initializations (Feb/May/Aug/Nov),
members 001-020. 14,880 files. It is resumable -- files land in `.part` and are renamed
on success, so a re-run skips whatever is already complete. `--verify` reopens every
file and checks it holds its variable with 24 time steps. `--dry-run` prints URLs.

Keep the concurrency at 4. GDEX rate-limits by connection count: at 8 workers the
time-to-first-byte went from 0.2 s to 35 s and throughput fell to ~1.5 MB/s. At 4 it
holds ~11 MB/s. The script backs off for 90 s if any response takes over 20 s.

`build_smyle_cube.py` reduces the raw files to one lead-1-3 mean per (initialization,
member) on the N48 grid, with HadSLP2r over the same windows, and writes
`data/smyle_cube.npz` (104 MB). Requires ~48 GB of the raw data on disk; you can delete
`data/smyle/` afterwards if you only want the lead-1-3 window.

## Things that will bite you

**CESM time stamping.** A monthly mean is stamped at the END of its averaging period,
so record 0 of a 1970-02 init reads `time = 1970-03-01` but IS the February mean. Use
`time_bnds[:, 0]`. Record 0 is lead month 1 = the init month. Reading `time` naively
shifts everything a month, which at leads 1-3 is a 33-100% error.

**Detrend within each init month.** The sample axis cycles Feb/May/Aug/Nov, so one line
over all 247 leaves the seasonal cycle in -- and model and obs share it, which fakes
skill. `rho_o` comes out 0.759 that way against 0.450 done properly.

**J = 247, not 248.** The 2019-11 init verifies into Jan 2020; HadSLP2r stops at
Dec 2019.

## What is not in git

`data/` is ignored throughout -- every file under it is rebuilt by the scripts above.
The 24-month runs mean leads 1-24 are already on disk after `fetch_smyle.py`; change
`NLEAD` at the top of `build_smyle_cube.py` and rebuild to use a different window.
