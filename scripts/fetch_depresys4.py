"""Fetch Met Office DePreSys4 decadal hindcasts (CMIP6 DCPP) from ESGF.

    python scripts/fetch_depresys4.py --manifest    # query ESGF -> data/depresys4_files.json
    python scripts/fetch_depresys4.py               # download (resumable)
    python scripts/fetch_depresys4.py --verify      # reopen every file
    python scripts/fetch_depresys4.py --dry-run

DePreSys4 is `CMIP6.DCPP.MOHC.HadGEM3-GC31-MM.dcppA-hindcast`: the CMIP6 successor to
the HadCM3-based DePreSys that Eade et al. (2014) used. 59 start dates (s1960..s2018, no
gaps), initialised 1 November each year and run 10 years forward, 10 members
(r1..r10 i1p1f2), on the native N216 grid (`gn`, 432x324).

Unlike SMYLE, the filenames are not predictable from the case name -- each
(start, member, variable) is split into a variable number of yearly chunks -- so the
URL list is resolved once from the ESGF search API and cached as a manifest. Rebuild it
with --manifest if ESGF republishes.

Layout on disk groups by start date first, matching data/smyle/:

    <dest>/<sYYYY>/<VAR>/<filename>.nc

Downloads land in a .part file and are renamed on success, so a re-run skips whatever is
already complete. No credentials.

Two things that will bite you downstream, both differing from SMYLE:

* **360-day calendar.** HadGEM3-GC31-MM uses 30-day months. Open with cftime and do not
  assume a standard calendar when labelling lead months.
* **November initialisation, annual.** Lead month 1 is November of the start year, so
  DJF is lead months 2-4 -- the same offset as SMYLE's Nov inits, but there is only one
  start per year, so the 59 samples are annual and their multi-year windows overlap.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
DEST = os.path.join(ROOT, "data", "depresys4")
MANIFEST = os.path.join(ROOT, "data", "depresys4_files.json")

INDEX = "esgf-data.dkrz.de"
SEARCH = dict(project="CMIP6", source_id="HadGEM3-GC31-MM",
              experiment_id="dcppA-hindcast", frequency="mon",
              distrib="true", latest="true", replica="false")
VARS = ["psl", "tas", "pr"]
PAGE = 5000

NWORKERS = 6          # CEDA is the sole data node; keep this modest
RETRIES = 4
MIN_BYTES = 50_000    # the short 2-month chunks are genuinely small
SLOW_SECONDS = 25.0
COOLDOWN = 60.0
_last_slow = [0.0]


def _search(**kw):
    kw.setdefault("format", "application/solr+json")
    url = f"https://{INDEX}/esg-search/search?" + urllib.parse.urlencode(kw)
    with urllib.request.urlopen(url, timeout=300) as r:
        return json.load(r)


def http_url(doc):
    """ESGF packs urls as 'href|mime|service'; we want the plain HTTP one."""
    for u in doc.get("url", []):
        href, _, service = u.rpartition("|")
        if service == "HTTPServer":
            return href.split("|")[0]
    return None


def build_manifest(path=MANIFEST):
    out, total = [], 0
    for var in VARS:
        got, offset = 0, 0
        while True:
            d = _search(type="File", variable_id=var, limit=str(PAGE),
                        offset=str(offset), **SEARCH)
            docs = d["response"]["docs"]
            n_found = d["response"]["numFound"]
            for doc in docs:
                url = http_url(doc)
                if not url:
                    continue
                name = doc.get("title") or url.rsplit("/", 1)[-1]
                # s1960-r10i1p1f2 -> start s1960
                m = re.search(r"_(s\d{4})-(r\d+i\d+p\d+f\d+)_", name)
                start = m.group(1) if m else "unknown"
                out.append({"var": var, "start": start,
                            "member": m.group(2) if m else "unknown",
                            "filename": name, "size": doc.get("size", 0),
                            "url": url})
                total += doc.get("size", 0)
            got += len(docs)
            offset += PAGE
            print(f"  {var}: {got}/{n_found}", flush=True)
            if got >= n_found or not docs:
                break
    with open(path, "w") as f:
        json.dump(out, f)
    starts = sorted({r["start"] for r in out})
    print(f"\nwrote {path}")
    print(f"  {len(out)} files, {total / 2**30:.1f} GB")
    print(f"  {len(starts)} start dates {starts[0]}..{starts[-1]}, "
          f"{len({r['member'] for r in out})} members, {len(VARS)} variables")
    return out


def load_manifest(path=MANIFEST):
    if not os.path.exists(path):
        sys.exit(f"no manifest at {path}; run with --manifest first")
    with open(path) as f:
        return json.load(f)


def target(dest, rec):
    return os.path.join(dest, rec["start"], rec["var"], rec["filename"])


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) >= MIN_BYTES:
        return "skip", path, 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    for attempt in range(RETRIES):
        wait = COOLDOWN - (time.time() - _last_slow[0])
        if wait > 0:
            time.sleep(wait)
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            size = os.path.getsize(tmp)
            if size < MIN_BYTES:
                raise OSError(f"short read {size}")
            with open(tmp, "rb") as f:
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
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--manifest", action="store_true",
                    help="rebuild the file list from ESGF and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workers", type=int, default=NWORKERS)
    a = ap.parse_args()

    if a.manifest:
        build_manifest()
        return

    recs = load_manifest()
    todo = [(r["url"], target(a.dest, r)) for r in recs]
    have = sum(1 for _, p in todo
               if os.path.exists(p) and os.path.getsize(p) >= MIN_BYTES)
    gb = sum(r.get("size", 0) for r in recs) / 2**30
    print(f"{len(todo)} files ({gb:.1f} GB), {have} already present, "
          f"{len(todo) - have} to go", flush=True)

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
                with xr.open_dataset(p, decode_times=False) as d:
                    v = os.path.basename(p).split("_")[0]
                    if v not in d:
                        bad.append((p, f"missing variable {v}"))
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
            if (i + 1) % 500 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                g = total / 2**30
                print(f"  {i+1}/{len(todo)}  ok={n_ok} skip={n_skip} fail={n_fail}  "
                      f"{g:.1f} GB  {g/el*1024:.1f} MB/s  eta "
                      f"{(len(todo)-i-1)/max(n_ok,1)*el/3600:.1f} h", flush=True)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min: {n_ok} downloaded, "
          f"{n_skip} skipped, {n_fail} failed, {total/2**30:.1f} GB")
    if fails:
        fp = os.path.join(a.dest, "failures.txt")
        os.makedirs(a.dest, exist_ok=True)
        open(fp, "w").write("\n".join(fails) + "\n")
        print(f"failures written to {fp}")


if __name__ == "__main__":
    main()
