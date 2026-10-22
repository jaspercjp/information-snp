"""Figure style and output folders, shared by every figure script.

Text is Helvetica Neue when its font files are found (.otf/.ttf in $SNP_FONT_DIR, ~/.local/share/fonts
or ~/.fonts), else TeX Gyre Heros, the Helvetica clone that ships with TeX Live (`ml system texlive`
not needed, the files are read directly). Regular weight everywhere, bold for panel titles
(`ax.set_title`, and `panel_label`). `save` writes figures/<section>/{png,pdf}/<name>, section
main-text for the names in MAIN_TEXT, supp-info otherwise; `data_path` is figures/data/<name>.
"""
import glob
import os

import matplotlib
from matplotlib import font_manager as fm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGROOT = os.path.join(REPO, "figures")
MAIN_TEXT = {"rpc_rho_lambda_oloo_decadal", "gaussian-array-lambda-rho", "rho-lam-histogram",
             "flattened-histograms", "members-not-exchangeable"}
GYRE = "/share/software/user/open/texlive/2019/texmf-dist/fonts/opentype/public/tex-gyre"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # categorical slots 1-4, validated


def _register():
    """Add the font files to matplotlib; returns the family name."""
    dirs = [os.environ.get("SNP_FONT_DIR"), os.path.expanduser("~/.local/share/fonts"),
            os.path.expanduser("~/.fonts")]
    for d in filter(None, dirs):
        files = [f for ext in ("otf", "ttf") for f in
                 glob.glob(os.path.join(d, "**", f"HelveticaNeue*.{ext}"), recursive=True)
                 + glob.glob(os.path.join(d, "**", f"Helvetica Neue*.{ext}"), recursive=True)]
        if files:
            for f in files:
                fm.fontManager.addfont(f)
            return fm.FontProperties(fname=sorted(files, key=len)[0]).get_name()
    files = glob.glob(os.path.join(GYRE, "texgyreheros-*.otf"))
    for f in files:
        fm.fontManager.addfont(f)
    return "TeX Gyre Heros" if files else "DejaVu Sans"


FONT = _register()
matplotlib.rcParams.update({
    "font.family": FONT, "font.weight": "normal", "font.size": 9,
    "axes.titleweight": "bold", "axes.titlesize": 10, "figure.titleweight": "normal",
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.sf": FONT, "mathtext.cal": FONT, "mathtext.tt": FONT,
    "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold", "mathtext.bfit": f"{FONT}:italic:bold", "mathtext.default": "it",
    "axes.unicode_minus": True, "pdf.fonttype": 42, "svg.fonttype": "none",
    "text.color": INK, "axes.labelcolor": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": MUTED,
    "legend.framealpha": 1.0, "legend.fontsize": 8.5, "patch.linewidth": 0.8,
})


def section(name):
    return "main-text" if name in MAIN_TEXT else "supp-info"


def save(fig, name, dpi=300):
    """-> figures/<section>/png/<name>.png and pdf/<name>.pdf."""
    sec = os.path.join(FIGROOT, section(name))
    for ext in ("png", "pdf"):
        os.makedirs(os.path.join(sec, ext), exist_ok=True)
        fig.savefig(os.path.join(sec, ext, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")
    print(f"-> {section(name)}/{{png,pdf}}/{name}", flush=True)


def data_path(name):
    os.makedirs(os.path.join(FIGROOT, "data"), exist_ok=True)
    return os.path.join(FIGROOT, "data", name)


def tidy(ax, grid=True):
    """Hairline grid, no top/right spines."""
    if grid:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set(color=MUTED, lw=0.8)
    ax.tick_params(length=3, width=0.8)


def panel_label(ax, letter, title=None, inside=False, **kw):
    """Bold '(a) title' at the top left of the axes, as the panel title; inside=True puts the
    bold '(a)' in the axes' upper-left corner and centres the title."""
    if not inside:
        ax.set_title(f"({letter})" + (f" {title}" if title else ""), loc="left", **kw)
        return
    ax.text(0.025, 0.975, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", fontsize=matplotlib.rcParams["axes.titlesize"], zorder=20,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    if title:
        ax.set_title(title, loc="center", **kw)
