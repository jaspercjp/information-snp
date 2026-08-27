"""Fetch CESM2 SMYLE monthly PSL, TREFHT and PRECT from NSF NCAR GDEX.

    python scripts/fetch_smyle.py                 # download (resumable)
    python scripts/fetch_smyle.py --verify        # open every file, report bad ones
    python scripts/fetch_smyle.py --dry-run

Main BSMYLE only (1958-2019, quarterly inits 02/05/08/11), members 001-020, so the
ensemble size is uniform at 20 across all 248 start dates. The XT / XT-beta / v2 / v3
variants are a different model version and are deliberately excluded; the 682
BSMYLE member files beyond 020 are excluded for the same uniformity reason.

File span is always init month .. init+23 months, so URLs are constructed from the
case name without a per-case catalog lookup. Downloads land in a .part file and are
renamed on success, so any file present with its final name is complete -- that is
what makes a re-run cheap.

No credentials required. Layout on disk:

    <dest>/<VAR>/b.e21.BSMYLE.f09_g17.<YYYY>-<MM>.<NNN>.cam.h0.<VAR>.<span>.nc

NOTE on time: CESM stamps a monthly mean at the END of its averaging period, so
record 0 of a 1970-02 init carries time=1970-03-01 but is the FEBRUARY mean. Use
time_bnds[:, 0] to label months. Record 0 is lead month 1 = the init month.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

# data.gdex.ucar.edu is the direct object store: measured ~2x the throughput of the
# /thredds/fileServer path and a much lower time-to-first-byte.
BASE = ("https://data.gdex.ucar.edu/d651065/SMYLE/archive/"
        "{case}/atm/proc/tseries/month_1/{case}.cam.h0.{var}.{span}.nc")
DEST = "/Users/jasperchen/Academics/Research/SNP/information-snp/data/smyle"
VARS = ["PSL", "TREFHT", "PRECT"]
YEARS = range(1958, 2020)          # main BSMYLE
INITS = ["02", "05", "08", "11"]
MEMBERS = range(1, 21)
# Throughput saturates around 4 workers (7.2 / 9.1 / 10.4 MB/s at 2 / 4 / 6). Going to
# 8 tripped server-side rate limiting: time-to-first-byte went from 0.2 s to 35 s and
# the effective rate fell to ~1.5 MB/s. Stay low.
NWORKERS = 4
RETRIES = 4
MIN_BYTES = 1_000_000
SLOW_SECONDS = 20.0        # a response this slow means we are being throttled
COOLDOWN = 90.0            # so back off for this long before trying again

_last_slow = [0.0]


def span(year, mm):
    """init month .. init+23 months, as YYYYMM-YYYYMM."""
    y0, m0 = year, int(mm)
    tot = (y0 * 12 + (m0 - 1)) + 23
    return f"{y0}{m0:02d}-{tot // 12}{tot % 12 + 1:02d}"


def jobs(dest):
    out = []
    for y in YEARS:
        for mm in INITS:
            sp = span(y, mm)
            for n in MEMBERS:
                case = f"b.e21.BSMYLE.f09_g17.{y}-{mm}.{n:03d}"
                for v in VARS:
                    name = f"{case}.cam.h0.{v}.{sp}.nc"
                    out.append((BASE.format(case=case, var=v, span=sp),
                                os.path.join(dest, v, name)))
    return out


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) >= MIN_BYTES:
        return "skip", path, 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    for attempt in range(RETRIES):
        # if any worker recently saw a throttled response, wait it out
        wait = COOLDOWN - (time.time() - _last_slow[0])
        if wait > 0:
            time.sleep(wait)
        t0 = time.time()
        try:
            with urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            size = os.path.getsize(tmp)
            if size < MIN_BYTES:
                raise OSError(f"short read {size}")
            with open(tmp, "rb") as f:                     # netCDF magic
                if f.read(3) not in (b"CDF", b"\x89HD"):
                    raise OSError("not a netCDF file")
            if time.time() - t0 > SLOW_SECONDS:
                _last_slow[0] = time.time()
            os.replace(tmp, path)
            return "ok", path, size
        except (HTTPError, URLError, OSError) as e:
            if os.path.exists(tmp):
                os.unlink(tmp)
            if attempt == RETRIES - 1:
                return "fail", f"{path}  {type(e).__name__}: {e}", 0
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workers", type=int, default=NWORKERS)
    a = ap.parse_args()

    todo = jobs(a.dest)
    have = sum(1 for _, p in todo
               if os.path.exists(p) and os.path.getsize(p) >= MIN_BYTES)
    print(f"{len(todo)} files ({len(todo)//len(VARS)} cases x {len(VARS)} vars), "
          f"{have} already present, {len(todo)-have} to go", flush=True)

    if a.dry_run:
        for u, p in todo[:3]:
            print("  ", u)
        print("   ...")
        return

    if a.verify:
        import xarray as xr
        bad = []
        for i, (_, p) in enumerate(todo):
            if not os.path.exists(p):
                bad.append((p, "missing"))
                continue
            try:
                with xr.open_dataset(p) as d:
                    v = os.path.basename(p).split(".cam.h0.")[1].split(".")[0]
                    if v not in d or d.sizes.get("time") != 24:
                        bad.append((p, f"var={v in d} time={d.sizes.get('time')}"))
            except Exception as e:
                bad.append((p, f"{type(e).__name__}: {e}"))
            if (i + 1) % 2000 == 0:
                print(f"  verified {i+1}/{len(todo)}, {len(bad)} bad", flush=True)
        print(f"\n{len(todo)-len(bad)}/{len(todo)} good, {len(bad)} bad")
        for p, why in bad[:40]:
            print("  BAD", os.path.basename(p), why)
        return

    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    total = 0
    fails = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, u, p): p for u, p in todo}
        for i, fut in enumerate(as_completed(futs)):
            status, info, size = fut.result()
            if status == "ok":
                n_ok += 1
                total += size
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                fails.append(info)
            if (i + 1) % 250 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                gb = total / 2**30
                rate = gb / el * 3600 if el else 0
                print(f"  {i+1}/{len(todo)}  ok={n_ok} skip={n_skip} fail={n_fail}  "
                      f"{gb:.1f} GB  {gb/el*1024:.1f} MB/s  eta "
                      f"{(len(todo)-i-1)/max(n_ok,1)*el/3600:.1f} h", flush=True)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min: {n_ok} downloaded, {n_skip} skipped, "
          f"{n_fail} failed, {total/2**30:.1f} GB")
    if fails:
        fp = os.path.join(a.dest, "failures.txt")
        open(fp, "w").write("\n".join(fails) + "\n")
        print(f"failures written to {fp}")


if __name__ == "__main__":
    main()
