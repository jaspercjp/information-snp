"""Fetch CMIP6 DCPP decadal hindcasts (`dcppA-hindcast`) for a set of models.

    python scripts/fetch_dcpp.py --manifest        # query ESGF -> data/dcpp_files.json
    python scripts/fetch_dcpp.py --plan            # sizes and init months, no download
    python scripts/fetch_dcpp.py                   # download (resumable)
    python scripts/fetch_dcpp.py --verify          # reopen every file
    python scripts/fetch_dcpp.py --models CanESM5 --vars psl

This is `fetch_depresys4.py` generalised past one model, for building a multi-model
grand ensemble like the 70-member one in Eade et al. (2014). Defaults to the four
models requested: CanESM5 (40 members), EC-Earth3 (16), MIROC6 (10), MRI-ESM2-0 (10)
-- 76 members against DePreSys4's 10.

Index node
----------
`fetch_depresys4.py` uses `esgf-data.dkrz.de`, which currently indexes only 6 of the
16 models publishing `dcppA-hindcast` -- none of the four here. So this uses ORNL's
ESGF-1.5 bridge, which is Globus Search behind a Solr-compatible facade. Three
differences from a classic esg-search endpoint, all handled below:

* `fields=` is rejected (`extra_forbidden`) -- take the whole document.
* Repeated `variable_id=a&variable_id=b` silently keeps only the LAST value, so each
  variable needs its own query rather than one combined one.
* It rate-limits: parallel queries return HTTP 429 from Globus Search. Queries are
  therefore serial with a delay, and retried with backoff.

Layout on disk mirrors `data/depresys4/` with a model level inserted, and keeps the
CMIP variable spelling so `build_depresys4_cube.py`'s `CMIP_NAME` lookup carries over
unchanged:

    <dest>/<source_id>/<sYYYY>/<psl|tas|pr>/<filename>.nc

Default dest is `$SCRATCH/dcpp`. Note $SCRATCH purges files unmodified for 90 days and
`touch` does NOT reset the timer -- copy to $OAK anything that must outlive a quarter.

Lead convention is NOT assumed here. The manifest records each file's first month as
parsed from its name, and `--plan` prints the earliest month per model, so the
initialisation month is read off the data rather than guessed. It matters: `lead 2-4`
is DJF only for a November start, and DCPP-A groups differ.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
MANIFEST = os.path.join(ROOT, "data", "dcpp_files.json")
DEST = os.path.join(os.environ.get("SCRATCH", os.path.join(ROOT, "data")), "dcpp")

INDEX = "esgf-node.ornl.gov"
BRIDGE = f"https://{INDEX}/esgf-1-5-bridge"

MODELS = ["CanESM5", "EC-Earth3", "MIROC6", "MRI-ESM2-0"]
VARS = ["psl", "tas", "pr"]

PAGE = 1000           # bridge tolerates this; larger risks a timeout
QUERY_GAP = 8.0       # seconds between queries -- below this, Globus Search 429s
QUERY_RETRIES = 5
OFFSET_CAP = 9000     # the bridge 422s past offset 10000; partition before that

NWORKERS = 8
RETRIES = 4
MIN_BYTES = 50_000
_CHUNK = 1 << 20

# Per-socket-operation timeout, NOT a total transfer budget: urlopen applies it to the
# connect and to each read, and reads are _CHUNK at a time, so 120 s is generous even
# on a 1 MB/s link. It was 900 s, which is what a dead host costs per file -- and DIAS
# (MIROC6, MRI-ESM2-0) does not answer at all, so 12 workers x 900 s burned an entire
# 24 h allocation without fetching a byte or reaching the hosts that were up.
SOCK_TIMEOUT = 120

# A host that refuses N connections in a row gets benched, so the next file falls
# straight through to a replica instead of rediscovering the timeout. Benched, NOT
# written off: the Globus data node refuses connections under 12-way concurrency but
# serves fine on its own, and it is the only host for 576 NorCPM1 files. So the bench
# expires and the host is probed again -- a permanent ban turns a busy host into a
# missing model.
DEAD_AFTER = 3
BENCH_SECONDS = 600.0
_health = {}                 # host -> [consecutive zero-byte failures, benched_until]
_health_lock = threading.Lock()


def _host_of(url):
    return urllib.parse.urlparse(url).netloc


def _host_dead(host):
    with _health_lock:
        st = _health.get(host)
        return bool(st and st[1] > time.time())


def _host_ok(host):
    with _health_lock:
        _health[host] = [0, 0.0]


def _host_failed(host):
    """Record a zero-byte failure. True once the host is benched."""
    with _health_lock:
        st = _health.setdefault(host, [0, 0.0])
        st[0] += 1
        if st[0] >= DEAD_AFTER and st[1] <= time.time():
            st[1] = time.time() + BENCH_SECONDS
            st[0] = 0
            print(f"  host {host} refused {DEAD_AFTER} connections in a row; "
                  f"benching it for {BENCH_SECONDS/60:.0f} min", flush=True)
            return True
        return st[1] > time.time()


# ---------------------------------------------------------------- ESGF querying
def _get(url, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_dcpp/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _search(_whole=False, **kw):
    """One bridge query, with backoff on 429/5xx.

    Returns the Solr `response` block, or the whole document when `_whole` (needed
    for `facet_counts`, which sits outside `response`).
    """
    kw.setdefault("format", "application/solr+json")
    url = BRIDGE + "?" + urllib.parse.urlencode(kw)
    delay = QUERY_GAP
    for attempt in range(QUERY_RETRIES):
        try:
            d = _get(url)
            if "response" not in d:            # bridge reports errors with HTTP 200
                raise RuntimeError(str(d)[:200])
            return d if _whole else d["response"]
        except (HTTPError, URLError, RuntimeError, TimeoutError) as e:
            if attempt == QUERY_RETRIES - 1:
                raise
            print(f"    query failed ({str(e)[:80]}); retrying in {delay:.0f}s",
                  flush=True)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


_FIRST_MONTH = re.compile(r"_(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$")


def _http_url(urls):
    """The HTTPServer entry out of ESGF's 'url|mime|service' triples."""
    for u in urls or []:
        parts = u.split("|")
        if len(parts) == 3 and parts[2].lower() == "httpserver":
            return parts[0]
    return None


_BASE = dict(project="CMIP6", experiment_id="dcppA-hindcast", frequency="mon",
             replica="false", latest="true")


def _absorb(rows, docs):
    """Fold File documents into `rows`, keyed on filename so copies collapse.

    Every distinct download host for a filename is kept, in `urls`, primaries first.
    One host being down then costs one connection attempt rather than the file: DIAS
    serves MIROC6 and MRI-ESM2-0 as the primary and does not answer, while ORNL and
    the Globus data node hold reachable replicas of both. `url` stays as the first
    entry so manifests written before this change still load.
    """
    for d in docs:
        title = d.get("title")
        url = _http_url(d.get("url"))
        if not title or not url or d.get("retracted"):
            continue
        is_replica = bool(d.get("replica"))
        prev = rows.get(title)
        if prev is not None:
            if url not in prev["urls"]:
                if is_replica:
                    prev["urls"].append(url)
                else:
                    prev["urls"].insert(0, url)      # a primary outranks a replica
                    prev["replica"] = False
                prev["url"] = prev["urls"][0]
            continue
        m = _FIRST_MONTH.search(title)
        sub = d.get("sub_experiment_id") or [""]
        vl = d.get("variant_label") or [""]
        rows[title] = dict(
            title=title, url=url, urls=[url], size=int(d.get("size") or 0),
            start=(sub[0] if isinstance(sub, list) else sub),
            member=(vl[0] if isinstance(vl, list) else vl),
            first_month=(f"{m.group(1)}-{m.group(2)}" if m else None),
            last_month=(f"{m.group(3)}-{m.group(4)}" if m else None),
            replica=is_replica,
            checksum=(d.get("checksum") or [None])[0],
        )


def _page(rows, label, replicas=False, **kw):
    """Page one query into `rows`. Raises if it would need to cross OFFSET_CAP.

    `replicas=True` drops the `replica=false` filter so `_absorb` can collect every
    host serving each file. Costs ~3x the records for a widely replicated model.
    """
    base = dict(_BASE)
    if replicas:
        base.pop("replica", None)
    offset = 0
    while True:
        r = _search(type="File", limit=PAGE, offset=offset, **base, **kw)
        docs, total = r.get("docs", []), r.get("numFound", 0)
        if total > OFFSET_CAP and offset == 0:
            raise _TooDeep(total)
        _absorb(rows, docs)
        offset += len(docs)
        print(f"  {label} {offset}/{total} file records", flush=True)
        if not docs or offset >= total:
            return
        time.sleep(QUERY_GAP)


class _TooDeep(Exception):
    """numFound exceeds the bridge's offset cap; the query must be partitioned."""

    def __init__(self, total):
        super().__init__(f"{total} records > offset cap {OFFSET_CAP}")
        self.total = total


def _variants(model, var):
    """The variant_labels present, from a facet query. Used to partition."""
    d = _search(source_id=model, variable_id=var, type="File", limit=0,
                facets="variant_label", _whole=True, **_BASE)
    ff = d.get("facet_counts", {}).get("facet_fields", {}).get("variant_label", [])
    return ff[0::2]


def build_manifest(models, variables, path=MANIFEST, merge=True, replicas=False):
    """Query ESGF into `path`. MERGES into an existing manifest by default.

    Merging is the safe default because `--models` is how you add a model, and a
    replacing write would silently drop every model already surveyed -- the manifest
    is 22 MB of query results that take ~20 min to regenerate. Only the (model,
    variable) pairs actually queried are overwritten. Pass `--replace-manifest` for a
    clean rebuild.
    """
    out = {}
    if merge and os.path.exists(path):
        with open(path) as f:
            out = json.load(f)
        print(f"merging into {path}: {len(out)} model(s) already present "
              f"({', '.join(sorted(out))})")
    for model in models:
        out.setdefault(model, {})
        for var in variables:
            rows = {}
            try:
                _page(rows, f"{model:14s} {var:4s}", replicas=replicas,
                      source_id=model, variable_id=var)
            except _TooDeep as e:
                # EC-Earth3 publishes ~20k primary File records per variable
                # (~6-monthly chunks x 16 members x 60 starts), past the bridge's
                # offset cap. Split by member -- ~1.3k each, comfortably under.
                print(f"  {model} {var}: {e}; partitioning by member", flush=True)
                time.sleep(QUERY_GAP)
                mems = _variants(model, var)
                print(f"  {model} {var}: {len(mems)} members", flush=True)
                for i, mem in enumerate(mems):
                    time.sleep(QUERY_GAP)
                    _page(rows, f"{model:14s} {var:4s} [{mem} {i+1}/{len(mems)}]",
                          replicas=replicas, source_id=model, variable_id=var,
                          variant_label=mem)
            out[model][var] = sorted(rows.values(), key=lambda r: r["title"])
            nalt = sum(1 for r in rows.values() if len(r["urls"]) > 1)
            print(f"  -> {model} {var}: {len(rows)} unique files "
                  f"({nalt} with a fallback host)", flush=True)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:            # checkpoint after every variable
                json.dump(out, f, indent=1)
            time.sleep(QUERY_GAP)
    print(f"\nwrote {path}")
    return out


# ---------------------------------------------------------------------- planning
def local_path(dest, model, var, row):
    return os.path.join(dest, model, row["start"], var, row["title"])


def plan(man, dest):
    """Per-model volume, file count, start range and -- the important one -- the
    earliest output month, which is the initialisation month."""
    print(f"\n{'model':14s} {'var':4s} {'files':>7s} {'GB':>8s} {'members':>7s} "
          f"{'starts':>6s}  first month(s) seen")
    grand_n = grand_b = 0
    for model, byvar in man.items():
        for var, rows in byvar.items():
            n = len(rows)
            b = sum(r["size"] for r in rows)
            mem = len({r["member"] for r in rows})
            st = sorted({r["start"] for r in rows})
            firsts = sorted({r["first_month"][5:] for r in rows
                             if r["first_month"]})
            grand_n += n
            grand_b += b
            print(f"{model:14s} {var:4s} {n:7d} {b/2**30:8.1f} {mem:7d} "
                  f"{len(st):6d}  months {','.join(firsts)}")
        # the earliest file of the earliest start = the init month
        allrows = [r for rows in byvar.values() for r in rows if r["first_month"]]
        if allrows:
            e = min(allrows, key=lambda r: (r["start"], r["first_month"]))
            print(f"{'':14s} -> earliest: {e['start']} begins {e['first_month']}"
                  f"  => init month {e['first_month'][5:]}"
                  f"  ({'Nov, same as DePreSys4' if e['first_month'][5:] == '11' else 'NOT November -- lead windows must shift'})")
    have = sum(1 for model, byvar in man.items() for var, rows in byvar.items()
               for r in rows if os.path.exists(local_path(DEST, model, var, r)))
    print(f"\nTOTAL {grand_n} files, {grand_b/2**30:.1f} GB "
          f"({grand_b/2**40:.2f} TB); {have} already on disk at {dest}")
    return grand_n, grand_b


# --------------------------------------------------------------------- download
def _one(row, target):
    """Fetch one file, trying every known host for it before giving up."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    part = target + ".part"
    urls = [u for u in (row.get("urls") or [row.get("url")]) if u]
    last = "no url in the manifest"
    for url in urls:
        host = _host_of(url)
        if _host_dead(host):
            last = f"{host} written off earlier this run"
            continue
        for attempt in range(RETRIES):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "fetch_dcpp/1.0"})
                with urllib.request.urlopen(req, timeout=SOCK_TIMEOUT) as r, \
                        open(part, "wb") as f:
                    while True:
                        chunk = r.read(_CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                n = os.path.getsize(part)
                if n < MIN_BYTES:
                    raise IOError(f"only {n} bytes")
                if row["size"] and abs(n - row["size"]) > 0:
                    raise IOError(f"size {n} != published {row['size']}")
                os.replace(part, target)
                _host_ok(host)
                return n
            except Exception as e:
                got = os.path.getsize(part) if os.path.exists(part) else 0
                if os.path.exists(part):
                    os.remove(part)
                last = f"{host}: {str(e)[:100]}"
                # nothing arrived at all -> the host, not the file, is the problem
                if got == 0 and _host_failed(host):
                    break                     # stop retrying a written-off host
                if attempt == RETRIES - 1:
                    break
                time.sleep(3 * (attempt + 1))
    return f"FAIL {row['title']}: {last}"


def download(man, dest, workers=NWORKERS, dry_run=False):
    todo = []
    for model, byvar in man.items():
        for var, rows in byvar.items():
            for r in rows:
                t = local_path(dest, model, var, r)
                if os.path.exists(t) and (not r["size"]
                                          or os.path.getsize(t) == r["size"]):
                    continue
                todo.append((r, t))
    nbytes = sum(r["size"] for r, _ in todo)
    print(f"{len(todo)} files to fetch, {nbytes/2**30:.1f} GB -> {dest}")
    if dry_run or not todo:
        return
    done = fails = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, r, t): r for r, t in todo}
        for fut in as_completed(futs):
            res = fut.result()
            if isinstance(res, str):
                fails += 1
                print("  " + res, flush=True)
            else:
                done += 1
            if (done + fails) % 200 == 0:
                el = time.time() - t0
                print(f"  {done+fails}/{len(todo)}  {done} ok  {fails} failed  "
                      f"{el/60:.1f} min elapsed", flush=True)
    print(f"done: {done} ok, {fails} failed, {(time.time()-t0)/60:.1f} min")


def verify(man, dest):
    import xarray as xr
    bad = []
    for model, byvar in man.items():
        for var, rows in byvar.items():
            n_ok = 0
            for r in rows:
                p = local_path(dest, model, var, r)
                if not os.path.exists(p):
                    bad.append((p, "missing"))
                    continue
                try:
                    with xr.open_dataset(p, decode_times=False) as d:
                        assert var in d.variables, "variable absent"
                    n_ok += 1
                except Exception as e:
                    bad.append((p, str(e)[:80]))
            print(f"  {model:14s} {var:4s} {n_ok}/{len(rows)} open cleanly",
                  flush=True)
    if bad:
        print(f"\n{len(bad)} problem file(s):")
        for p, why in bad[:40]:
            print(f"  {os.path.basename(p)}: {why}")
    else:
        print("\nall files present and openable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--vars", nargs="+", default=VARS)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--manifest-path", default=MANIFEST)
    ap.add_argument("--manifest", action="store_true", help="query ESGF and merge the "
                    "result into the manifest")
    ap.add_argument("--with-replicas", action="store_true",
                    help="with --manifest, record every host serving each file, not "
                         "just the primary. Needed when the primary is down -- DIAS "
                         "serves MIROC6 and MRI-ESM2-0 and does not answer.")
    ap.add_argument("--replace-manifest", action="store_true",
                    help="with --manifest, write ONLY the models queried, discarding "
                         "everything else already in the file")
    ap.add_argument("--plan", action="store_true", help="print sizes, do not download")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=NWORKERS)
    a = ap.parse_args()

    if a.manifest or not os.path.exists(a.manifest_path):
        print(f"querying {INDEX} for {len(a.models)} model(s) x {len(a.vars)} var(s)")
        man = build_manifest(a.models, a.vars, a.manifest_path,
                             merge=not a.replace_manifest,
                             replicas=a.with_replicas)
        man = {m: {v: rows for v, rows in bv.items() if v in a.vars}
               for m, bv in man.items() if m in a.models}
    else:
        with open(a.manifest_path) as f:
            man = json.load(f)
        man = {m: {v: rows for v, rows in bv.items() if v in a.vars}
               for m, bv in man.items() if m in a.models}

    plan(man, a.dest)
    if a.plan:
        return
    if a.verify:
        verify(man, a.dest)
        return
    download(man, a.dest, workers=a.workers, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
