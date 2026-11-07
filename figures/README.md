# figures

| folder | holds |
|---|---|
| `main-text/{png,pdf}/` | the paper's main-text figures, each as png and pdf |
| `supp-info/{png,pdf}/` | everything else, each as png and pdf; `thesis-*` are the original thesis figures |
| `data/` | what the figures show: the json area summaries, the binning tables (md + json), and the npz curves the demo figures replot from |
| `slides/{png,pdf}/` | canonical renders of talk figures (`style.save(..., folder="slides")`) |
| `Manuscript/`, `Presentation/` | the figure sets: byte-copies of canonical renders, chosen by each set's `manifest.json` (see Figure sets below) |
| `notes/` | `jugaad-fd-optbins.md` (the binning study behind the maps), `BOOTSTRAP.md` (the stippling), `NOTES.txt` |

Which folder a figure goes to is `scripts/style.py:MAIN_TEXT`; every script saves through `style.save`.

## Main text

| figure | script | inputs |
|---|---|---|
| `rpc_rho_lambda_oloo_decadal` | `scripts/plots.py` (`production("decadal", oloo=True)`) | `$SNP_OUT` point + boot (fd, lmloo, olmloo) for SLP s1961, PRECT s1978, TREFHT s1961 |
| `gaussian-array-lambda-rho` | `scripts/gaussian_demo.py` | none (synthetic, seeded); curves in `data/gaussian-array-lambda-rho.npz` |
| `flattened-histograms` | `scripts/flattened_hist.py` | `$SNP_DATA/cases/` for the three decadal rows above |
| `rho-lam-histogram` | not yet here | |
| `members-not-exchangeable` | `scripts/exchangeability_check.py figure-main` | `$SNP_OUT/exchangeability_grid_{SLP,PRECT,TREFHT}_2-4.npz` from `exchangeability_check.py grid --case VAR:2-4` (200 permutations) |

## Figure sets

`Manuscript/` (the paper) and `Presentation/` (talks) are curated trees of figures rendered
elsewhere under `figures/`. Nothing in them is rendered: every file is a byte-identical
`shutil.copy2` of a canonical render, and each set's `manifest.json` (tracked; the copies are
gitignored) is the record for it: **dest -> source -> sha256 -> the command that regenerates the
source**, plus `section`, `workflow` and a free-text `note`. The same design as
AntarcticIceMeltFracture and DiffusionParameterCalibration (their ADR 0018); code in
`scripts/figure_sets.py`, CLI `scripts/figure_set.py`, tests `scripts/tests/test_figure_sets.py`.

    PY=/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/SNP_env/bin/python
    $PY scripts/figure_set.py Presentation list
    $PY scripts/figure_set.py Presentation add --source slides/png/NAME.png \
        --dest ExchangeabilityMaps/png/NAME.png --command "python scripts/... " --note "..."
    $PY scripts/figure_set.py Presentation export      # re-copy every source, refresh hashes
    $PY scripts/figure_set.py Presentation refresh     # accept new bytes after a re-render
    $PY scripts/figure_set.py Presentation verify      # offline check; exit 1 on any problem

- **Placement** is the `dest` path: `<Set>/<section>/{png,pdf}/<name>.<ext>`. `section` is the
  first folder inside the set (`MainText`/`SuppInformation` for the manuscript, a talk section
  such as `ExchangeabilityMaps` for the presentation); `verify` checks the two agree.
- **One entry per file**: a png and its pdf are two entries.
- **Re-rendering**: `style.save` copies each file it writes into every set that lists it, so the
  copies follow the canonical render, but the recorded sha256 does not: run `refresh` afterwards
  (that records that someone accepted the new bytes), or `verify` fails. `style.save` writes pdfs
  without a creation date, so re-rendering an unchanged figure keeps its hash.
- **Removing**: drop the entry (`FigureSet.remove_entries`) and delete the copy; `verify` flags any
  file in a set folder that no entry lists.
- Run all of this on a compute node, not a Sherlock login node.

### Manuscript

| section | figure | source script |
|---|---|---|
| `SuppInformation` | `estimator-toys` (png, pdf): seven lambda estimators on four toy problems with a known lambda, n = 33, 51, 120 | `scripts/estimator_toys.py` |

### Presentation

| section | figure | source script |
|---|---|---|
| `ExchangeabilityMaps` | `exchangeability-all-models`, `exchangeability-canesm5` (png, pdf) | `scripts/exchangeability_check.py figure-slides` |
| `RhoLambdaMaps` | `rho-lambda-lead2-4`, `rho-lambda-moy`, `rho-lambda-canesm5` (CanESM5, DJF lead 12-14) (png, pdf) | `scripts/plots.py slides` (needs `boot.py` fd + olmloo for the six cases: `.claude/scripts/rho_lambda_boot.sbatch`) |
| `ToyModels` | `monotonic-lambda-rho` (png, pdf): a monotonic non-linearity lowers rho, not lambda | `scripts/monotonic_demo.py` |

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
