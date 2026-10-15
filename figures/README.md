# figures

| folder | holds |
|---|---|
| `main-text/{png,pdf}/` | the paper's main-text figures, each as png and pdf |
| `supp-info/{png,pdf}/` | everything else, each as png and pdf; `thesis-*` are the original thesis figures |
| `data/` | what the figures show: the json area summaries, the binning tables (md + json), and the npz curves the demo figures replot from |
| `notes/` | `jugaad-fd-optbins.md` (the binning study behind the maps), `BOOTSTRAP.md` (the stippling), `NOTES.txt` |

Which folder a figure goes to is `scripts/style.py:MAIN_TEXT`; every script saves through `style.save`.

## Main text

| figure | script | inputs |
|---|---|---|
| `rpc_rho_lambda_oloo_decadal` | `scripts/plots.py` (`production("decadal", oloo=True)`) | `$SNP_OUT` point + boot (fd, lmloo, olmloo) for SLP s1961, PRECT s1978, TREFHT s1961 |
| `gaussian-array-lambda-rho` | `scripts/gaussian_demo.py` | none (synthetic, seeded); curves in `data/gaussian-array-lambda-rho.npz` |
| `flattened-histograms` | `scripts/flattened_hist.py` | `$SNP_DATA/cases/` for the three decadal rows above |
| `rho-lam-histogram` | not yet here | |

## Style

Text is Helvetica Neue when its font files are present (`.otf`/`.ttf` in `$SNP_FONT_DIR`,
`~/.local/share/fonts` or `~/.fonts`), else TeX Gyre Heros, the free Helvetica clone in TeX Live:
regular weight, bold panel titles. Sherlock has no Helvetica Neue, so the figures here are in
TeX Gyre Heros; with the font files in place, rerunning the scripts switches every figure.

## Not yet in the new style

- `rpc_rho_lambda_*` (main-text `oloo_decadal` and the three supp-info ones) and `rho_vs_lambda_*`
  need the bootstrap outputs, which are not on disk; `bash scripts/run_all.sh` regenerates them
  (and every other pipeline figure). The `rho_vs_lambda_*` pdfs are the png wrapped in a pdf.
- `thesis-*` were saved by hand from the notebooks; their pdfs are the png wrapped in a pdf.
  `thesis-gaussian-array` is the exception (`scripts/thesis-old/gaussian_array_metrics.py`).
