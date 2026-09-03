"""Add `import snp_path` next to a notebook's existing sys.path bootstrap. Idempotent.

    python scripts/tools/patch_notebook_imports.py notebooks/*.ipynb

After the scripts/ reorg, `sys.path.append("../scripts")` alone is no longer enough: the
modules live in subfolders (fetch, cubes, handles, mi, metrics, bootstrap, analysis,
tests), and `scripts/snp_path.py` is what puts those subfolders on the path. One extra
line per notebook restores `import mi_vectorized` and friends.

Every tracked notebook was patched in the reorg commit, but notebooks get edited live in
Jupyter, so a copy that was open at the time will still be missing the line. Run this on
it whenever you like -- it does nothing if the line is already there, and it only ever
inserts, never rewrites existing source.
"""
import json
import sys

LINE = "import snp_path  # noqa: F401  # scripts/ subfolders -> sys.path\n"


def patch(path):
    with open(path) as fh:
        nb = json.load(fh)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if any("snp_path" in ln for ln in src):
            return "already patched"
        for i, ln in enumerate(src):
            if "sys.path" in ln and "scripts" in "".join(src[i:i + 3]):
                j = i
                while j < len(src) and "scripts" not in src[j]:
                    j += 1
                if j < len(src) and not src[j].endswith("\n"):
                    src[j] += "\n"
                src.insert(j + 1, LINE)
                with open(path, "w") as fh:
                    json.dump(nb, fh, indent=1)
                    fh.write("\n")
                return f"patched after line {j + 1} of a code cell"
    return "NO sys.path/scripts cell found -- add `import snp_path` by hand"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        print(f"{p}: {patch(p)}")
