"""Shared cartopy map panels for the pairwise figures: coastlines, north at the top.

    import tmp_mapaxes as MAP

    fig, axs = plt.subplots(2, 3, figsize=..., subplot_kw=MAP.subplot_kw(),
                            layout="compressed")
    im = MAP.show(axs[0, 0], field, lats, lons, cmap="jet", vmin=0, vmax=0.7)

Two problems this fixes at once.

**Orientation.** The figures used to hand a `(lat, lon)` array straight to `imshow`
with no `origin=`. The observational grids run `lat[0] = -90` upward, and
matplotlib's default is `origin="upper"`, so every map came out with the South Pole
at the top. `show()` passes `origin="lower"` and a real `extent`, so north is up.

**No geography.** A metric map with no coastline is very hard to read -- "the
tropical convective bands" and "the Southern Ocean" were assertions the reader could
not check. `show()` draws Natural Earth coastlines over every panel.

Projection choice
-----------------
The data are on 0-360 longitude, so the projection is
`PlateCarree(central_longitude=180)` and the data transform is plain `PlateCarree()`.
That keeps the Pacific in the middle and 0 degrees E at the left edge, which is
exactly where the old index-space panels put it -- so these figures are still
horizontally aligned with the versions that preceded them, and only the vertical flip
and the coastlines have changed.

`extent` is built from CELL EDGES, not centres, via the same midpoint rule
`_smooth._edges` uses. On the 5 deg HadSLP2r grid the centres run 0..355, so a
centre-based extent would shift the field half a cell (2.5 deg) west of the
coastlines. Latitude edges are clipped to [-90, 90].
"""
import numpy as np
import cartopy.crs as ccrs

# 110m is the right resolution for panels a few inches wide; 50m turns into a smear
# of ink at this size and costs more to draw. Overridable per call.
COAST = "110m"
COAST_LW = 0.45
COAST_COLOR = "0.10"

# The panels in this project span `jet` (near-black at the low end), `magma` and
# `inferno` (near-black over most of their range) and `RdBu_r` (white in the middle).
# No single line colour is legible on all of them, so every coastline is drawn twice:
# a wider translucent white stroke first, then the dark line on top. That reads as a
# thin dark line on light fills and as a haloed line on dark ones.
COAST_HALO = "white"
COAST_HALO_LW = 1.5          # multiple of COAST_LW
COAST_HALO_ALPHA = 0.75

#: the projection every panel is drawn in
PROJ = ccrs.PlateCarree(central_longitude=180)
#: the CRS the data itself is in
DATA = ccrs.PlateCarree()

#: grey under masked/NaN cells, the project's existing convention
BED = "0.75"


def subplot_kw():
    """`subplot_kw=` for `plt.subplots` when every panel in the figure is a map."""
    return {"projection": PROJ}


def add_ax(fig, nrows, ncols, idx, geo=True):
    """One axes in an `nrows x ncols` grid, map or plain.

    For figures that mix histograms with maps: `plt.subplots(subplot_kw=...)` would
    force the projection onto every panel, so those figures build their axes one at
    a time instead. `idx` is 1-based, as `add_subplot` wants.
    """
    if geo:
        return fig.add_subplot(nrows, ncols, idx, projection=PROJ)
    return fig.add_subplot(nrows, ncols, idx)


def grid(var):
    """(lats, lons) of the observational grid a variable is analysed on.

    Reads it from the project rather than taking it as an argument, so a caller that
    only has an npz of per-cell metrics -- `tmp_pairwise_figs.py` -- does not have to
    load a cube just to draw a coastline. Requires `<SNP_REPO>/scripts` on sys.path.
    """
    from build_smyle_cube import obs_grid
    lats, lons = obs_grid(var)
    return np.asarray(lats, float), np.asarray(lons, float)


def _edges(x, lo=None, hi=None):
    """Cell edges from centres: midpoints inside, half-steps at the ends.

    The same rule as `scripts/cubes/_smooth.py::_edges`, kept local so this module
    has no dependency on the repo beyond `grid()`.
    """
    x = np.asarray(x, float)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    mid = 0.5 * (x[1:] + x[:-1])
    first = x[0] - (mid[0] - x[0])
    last = x[-1] + (x[-1] - mid[-1])
    e = np.concatenate([[first], mid, [last]])
    if lo is not None:
        e = np.clip(e, lo, hi)
    return e


def extent(lats, lons):
    """`extent=` for imshow: [west, east, south, north] on cell edges."""
    le = _edges(lons)
    ae = _edges(lats, -90.0, 90.0)
    return [float(le[0]), float(le[-1]), float(ae[0]), float(ae[-1])]


def coastlines(ax, coast=None, coast_lw=COAST_LW, coast_color=COAST_COLOR,
               halo=True):
    """Haloed Natural Earth coastlines. See COAST_HALO for why the halo is there."""
    res = coast or COAST
    if halo:
        ax.coastlines(res, linewidth=coast_lw * COAST_HALO_LW, color=COAST_HALO,
                      alpha=COAST_HALO_ALPHA)
    ax.coastlines(res, linewidth=coast_lw, color=coast_color)


def show(ax, field, lats, lons, coast=None, bed=BED, coast_lw=COAST_LW,
         coast_color=COAST_COLOR, halo=True, **imshow_kw):
    """`field` (nlat, nlon) on a GeoAxes, right way up, with coastlines drawn over it.

    `imshow_kw` is passed through, so cmap/vmin/vmax/interpolation behave as before.
    Returns the image, for `fig.colorbar`.
    """
    F = np.asarray(field, float)
    if F.shape != (len(lats), len(lons)):
        raise ValueError(f"field is {F.shape}, grid is "
                         f"{(len(lats), len(lons))} -- reshape before calling")
    if bed:
        ax.set_facecolor(bed)
    im = ax.imshow(F, origin="lower", extent=extent(lats, lons), transform=DATA,
                   **imshow_kw)
    coastlines(ax, coast, coast_lw, coast_color, halo)
    ax.set_global()
    # A freshly created GeoAxes has xaxis/yaxis INVISIBLE, and an invisible axis does
    # not draw its label -- so `ax.set_xlabel(...)` silently does nothing. Several
    # panels in this project carry their area-mean in the xlabel and their row name in
    # the ylabel, and those vanished on the first cartopy pass. Turning the axes back
    # on with an empty tick list restores the labels and still draws no ticks.
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def hatch(ax, mask, lats, lons, pattern="///", color="black"):
    """Hatch the True cells of `mask` on a GeoAxes, in lon/lat rather than indices."""
    m = np.asarray(mask, float)
    if m.sum() == 0:
        return
    cs = ax.contourf(np.asarray(lons, float), np.asarray(lats, float), m,
                     levels=[0.5, 1.5], colors="none", hatches=[pattern],
                     extend="neither", transform=DATA)
    cs.set_edgecolor(color)
    cs.set_linewidth(0.0)


def strip_ticks(axs):
    """Remove ticks/labels from map axes. GeoAxes ignores `set_xticks([])` cleanly,
    but its gridlines are off by default anyway, so this only has to hide the frame
    ticks that a plain Axes would show."""
    for ax in np.ravel(np.asarray(axs, dtype=object)):
        try:
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception:
            pass
