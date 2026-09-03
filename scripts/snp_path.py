"""Put `scripts/` and every one of its subfolders on `sys.path`.

The modules under `scripts/` import each other as top-level names -- `import
mi_vectorized`, `from smyle_metrics import lam_of` -- which worked for free while they all
lived in one directory. They are now grouped into subfolders (fetch, cubes, handles, mi,
metrics, bootstrap, analysis, tests), and the dependencies cross those boundaries freely,
so the flat namespace has to be reconstructed at import time. This module is the one place
that happens.

Import it once, as early as possible, and every sibling becomes importable regardless of
which subfolder it lives in:

    import sys; sys.path.append("../scripts")     # notebooks already do this
    import snp_path                               # noqa: F401

Every module under `scripts/` carries a 3-line header that locates this file from its own
`__file__`, so running any of them directly (`python scripts/mi/mi_vectorized.py`) works
too. Nothing here is a package: no `__init__.py`, no dotted imports, no relative imports.
That is deliberate -- it keeps `import mi_vectorized` valid from a notebook, an sbatch
script, and a bare interpreter alike, which a package layout would not.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def add_all():
    """Prepend `scripts/` and its immediate subfolders to `sys.path`, once each."""
    for p in [ROOT] + [os.path.join(ROOT, d) for d in sorted(os.listdir(ROOT))
                       if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith(
                           ("_", "."))]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return sys.path


add_all()
