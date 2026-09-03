"""Read the common-grid cubes built by `build_common_grid_cubes.py`.

    import common_grid_handles as CG

    CG.inventory()                              # what has been built
    c = CG.get("dcpp", "TREFHT", "13-60")       # (N, T, 37, 72), SLP's grid
    c = CG.get("monthly", "PRECT", 1961)

    rho = smyle_metrics.pearson_coeff(c.s, c.G) # every variable is now cell-comparable

Presents the same surface as `dcpp_handles` / `dcpp_decadal_handles` -- `.F`, `.G`, `.s`,
`.lats`, `.lons`, `.members`, `.model_of`, `.N`, `.T`, `.label` -- so anything written
against those works unchanged. It is a reader only; the pre-processing all happened in
the builder, and `.meta` carries what was done.

Every variable here is on SLP's grid and smoothed with ONE box, which the native handles
deliberately do not do. Read the builder's docstring before drawing conclusions from it:
the observations have been coarsened, and TREFHT no longer uses Eade's SAT box.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
CACHE = os.path.join(ROOT, "data", "common_grid")


def tag_of(dataset, var, window):
    return (f"dcpp_{var}_lead{window}" if dataset == "dcpp"
            else f"monthly_{var}_s{window}")


class CommonGridCube:
    """A built cube. Arrays are float64 in memory even though they are float32 on disk."""

    def __init__(self, dataset, var, window):
        self.dataset, self.var, self.window = dataset, var, window
        self.tag = tag_of(dataset, var, window)
        npz = os.path.join(CACHE, self.tag + ".npz")
        if not os.path.exists(npz):
            raise FileNotFoundError(
                f"{npz} not built. Run:\n"
                f"  python build_common_grid_cubes.py --datasets {dataset} "
                f"--vars {var} "
                f"{'--leads' if dataset == 'dcpp' else '--starts'} {window}")
        with np.load(npz, allow_pickle=False) as z:
            self.F = z["F"].astype(float)
            self.G = z["o"].astype(float)
            self.lats, self.lons = z["lats"], z["lons"]
            self.members, self.model_of = z["members"], z["model_of"]
        meta_path = os.path.join(CACHE, self.tag + ".meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.N, self.T = self.F.shape[0], self.F.shape[1]
        self.units = self.meta.get("units")
        self.smooth_box_deg = tuple(self.meta.get("smoothing_box_deg", ()))

    @property
    def s(self):
        return self.F.mean(axis=0)

    def loo_mean(self, n):
        return (self.F.sum(axis=0) - self.F[n]) / (self.N - 1)

    @property
    def by_model(self):
        """{model: index array} -- the members are not contiguous per model after a
        rebuild, so this returns indices rather than the native handles' slices."""
        return {m: np.flatnonzero(self.model_of == m)
                for m in dict.fromkeys(self.model_of.tolist())}

    @property
    def label(self):
        box = self.smooth_box_deg
        return (f"{self.meta.get('label', self.tag)} "
                f"[common grid {len(self.lats)}x{len(self.lons)}, "
                f"smooth {box[0]:g}x{box[1]:g}deg]" if box else self.tag)

    def __repr__(self):
        return (f"<CommonGridCube {self.tag}: N={self.N} T={self.T} "
                f"grid {len(self.lats)}x{len(self.lons)}>")


def get(dataset, var, window):
    """Load one built cube. `window` is a lead string (dcpp) or a start year (monthly)."""
    return CommonGridCube(dataset, var, window)


def inventory():
    """Print what has been built."""
    if not os.path.isdir(CACHE):
        print(f"nothing built; {CACHE} does not exist")
        return
    files = sorted(f for f in os.listdir(CACHE) if f.endswith(".npz"))
    if not files:
        print(f"nothing built in {CACHE}")
        return
    print(f"{CACHE}")
    for f in files:
        tag = f[:-4]
        mp = os.path.join(CACHE, tag + ".meta.json")
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        box = m.get("smoothing_box_deg", [])
        print(f"  {tag:<34s} N={m.get('N', '?'):<4} T={m.get('T', '?'):<4} "
              f"{'x'.join(map(str, m.get('common_grid', [])))}  "
              f"box {'x'.join(f'{b:g}' for b in box)}deg  "
              f"{os.path.getsize(os.path.join(CACHE, f)) / 1e6:6.0f} MB")


if __name__ == "__main__":
    inventory()
