"""Fetch the raw inputs of the thesis CMIP5 case: 49 decadal1961 psl files and HadSLP2.

    python fetch_cmip5_thesis.py            # download whatever is missing, verify all
    python fetch_cmip5_thesis.py --verify   # checksums of what is on disk, no download

CMIP5 psl: the files listed in cmip5_psl_provenance.json next to this script (a copy of
data/models/PROVENANCE_psl.json: file, url, SHA256, size, version), from the CEDA ESGF
replica (https://esgf.ceda.ac.uk/thredds/fileServer/esg_dataroot/cmip5/output1/...):
CanCM4 10, GFDL-CM2p1 10, HadCM3 20, MIROC5 6, MPI-ESM-LR 3 members of `decadal1961`.
Written to data/models/<model>/psl/<file>; each is checked against its SHA256 and size.

HadSLP2 (1850-2004), the thesis observations:
https://www.metoffice.gov.uk/hadobs/hadslp2/data/hadslp2.asc.gz -> data/obs/hadslp2/.
The Met Office publishes no checksum for it.

Both are then turned into the ensemble and obs files by build_cmip5_thesis_ensemble.py.
No credentials.
"""
import argparse
import hashlib
import json
import os
import shutil
from urllib.request import urlopen

from paths import HADSLP2, MODELS_CMIP5

PROVENANCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "cmip5_psl_provenance.json")
HADSLP2_URL = "https://www.metoffice.gov.uk/hadobs/hadslp2/data/hadslp2.asc.gz"
HADSLP2_RAW = os.path.join(HADSLP2, "hadslp2.asc.gz")


def target(rec):
    """psl_Amon_<model>_decadal1961_... -> data/models/<model>/psl/<file>."""
    return os.path.join(MODELS_CMIP5, rec["file"].split("_")[2], "psl", rec["file"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  downloading {url}", flush=True)
    with urlopen(url, timeout=600) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(path + ".part", path)


def check(rec, path):
    if not os.path.exists(path):
        return "missing"
    if os.path.getsize(path) != rec["size"]:
        return f"size {os.path.getsize(path)} != {rec['size']}"
    if rec["ctype"].upper() != "SHA256" or sha256(path) != rec["sum"]:
        return "checksum mismatch"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true", help="check only, download nothing")
    a = ap.parse_args()

    with open(PROVENANCE) as f:
        recs = json.load(f)
    bad = []
    for rec in recs:
        p = target(rec)
        if not a.verify and not os.path.exists(p):
            download(rec["url"], p)
        why = check(rec, p)
        if why:
            bad.append((rec["file"], why))
    print(f"CMIP5 psl: {len(recs) - len(bad)}/{len(recs)} files present and verified")
    for f, why in bad:
        print(f"  BAD {f}: {why}")

    if not a.verify and not os.path.exists(HADSLP2_RAW):
        download(HADSLP2_URL, HADSLP2_RAW)
    print(f"HadSLP2: {HADSLP2_RAW} " + ("present" if os.path.exists(HADSLP2_RAW)
                                         else "MISSING"))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
