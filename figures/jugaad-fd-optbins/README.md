# Thesis jugaad λ: Freedman–Diaconis vs Knuth OptBins, and how much the binning moves it

Code: branch `jugaad-fd-optbins`. The main pieces:
- `scripts/metrics/jugaad_thesis.py`
- `scripts/analysis/jugaad_fd_optbins_{run,figs}.py`
- `scripts/analysis/cmip5_thesis_reproduce.py`
- `scripts/tests/test_jugaad_thesis.py`

Raw outputs are in `$SCRATCH/snp_jugaad_fd_optbins/`.

The formulation is the thesis one:
- `f_jugaad` = members flattened, NaN → 0.
- `o_jugaad` and `s_jugaad` = the obs and the full ensemble mean, each tiled N times.
- `I` = `np.histogram2d` plug-in MI in bits.
- λ = √(1 − 2^(−2I)), and λ = λ_o/λ_m.

This implementation is bitwise equal to `InfoTheoryMetrics.info_RPC`.

## Cases

| case | what | N | T |
|---|---|---|---|
| CMIP5 | thesis decadal1961 psl, rebuilt (see below), calc_RPC cell 3 preprocessing | 49 | 120 |
| s1961 | DCPP-A s1961, monthly Jan 1962 – Dec 1971, thesis preprocessing (raw + linear detrend) | 92 (SLP), 112 (TREFHT) | 120 |
| s1961 moy | the same, month-of-year climatology removed first | 92 / 112 | 120 |
| lead 2-4 | DCPP-A DJF lead 2-4 across start dates (`dcpp_handles`) | 62 | 51 / 33 / 52 |

PRECT has no s1961 case, because GPCP starts in 1979.

## Binning rules

- **FD (thesis):** `int(ptp/(2·IQR/n^{1/3}))`, applied to the tiled or pooled series of each axis.
- **Knuth:**
  - Knuth (2006) posterior, on the T distinct obs and s values (capped at T) and on the N·T pooled members (cap 200).
  - On a tiled series, Knuth runs straight to its cap: its likelihood rewards isolating each of the T repeated values. So the tiled version is only kept as `knuth_tiled80`, a degenerate reference.
  - `InfoTheoryMetrics.optbins` returns the Knuth optimum + 1 whenever that optimum is above 1. The loop stores M bins at index M, then adds 1.
- **Band:** the thesis uncertainty band scales each rule's bin counts by alt = 0.5, 1 and 1.5. ΔI is the max − min of I over the band.

## What the numbers say (`jugaad_fd_optbins_table.md`)

1. **FD's binning range is about 0.17–0.25 bits in every case, whatever the size of I.**
   - With month-of-year anomalies (s1961 moy) or across start dates (lead 2-4), ΔI_o exceeds I_o itself over 56–99% of the area. The median ΔI_o/I_o is 1.2–1.4.
   - Only with the seasonal cycle left in (s1961 raw, CMIP5) is I large enough for the band to be small beside it: median ΔI_o/I_o is 0.11 for TREFHT s1961 and 0.8–0.9 for SLP.
2. **The two rules disagree by more than Knuth's whole estimate.**
   - Mean I_o is 0.15–0.66 bits higher under FD than under Knuth, and mean λ is 0.23–0.41 higher.
   - With one common B on every axis, mean I_o grows between B = 2 and B = 64 by 2.8× (TREFHT s1961) up to 34× (PRECT lead 2-4). For s1961 moy it starts from 0.000, so the growth is unbounded.
3. **λ ≤ 1 almost everywhere, under either rule.**
   - FD's λ band lies entirely below 1 over 68–99.6% of the area, and above 1 over at most 14%.
   - FD compresses λ toward 1, because both I_o and I_m are inflated by 22–95 bins per axis.
4. **For Gaussian (linear) dependence, the full-mean jugaad ratio is ρ_o = corr(s, o), not the RPC.**
   - Pooled over (n, t), corr(f; tile(o)) = cov(s, o)/(σ_f σ_o) and corr(f; tile(s)) = σ_s/σ_f.
   - So the ratio is corr(s, o) ≤ 1. This checks out numerically: for lead 2-4 the difference is 1e-14. For CMIP5 it's 0.015, because f is zero-filled there while s skips NaNs.
   - The pooled-Pearson column (`% ρ_o/ρ_m>1`) is 0 in every case.
   - The variance-based RPC on the same CMIP5 data averages 1.02, with ρ > 1 over 71.6% of the area (published 72.1%).
5. **With s₋ₙ (leave-one-out), λ falls further and never exceeds 1 over more than 0.4% of the area.** Knuth on the pooled s₋ₙ picks 56–196 bins, for the same reason tiling breaks it: the N copies of s₋ₙ[t] form tight clusters.
6. **Knuth degenerates on the short untiled series.**
   - It chooses 1 bin for o or s in up to 33% of cells. There I = 0 exactly, λ_m ≤ 0.02, and λ is left undefined (gray on the maps; up to 27% of the area for PRECT lead 2-4).

## CMIP5 reproduction

- **Data:** the thesis data were rebuilt with `scripts/cubes/build_cmip5_thesis_ensemble.py`.
  - The 49 psl files come from the CEDA ESGF replica, all checksum-verified. Provenance is in `data/models/PROVENANCE_psl.json`.
  - The obs are HadSLP2 from the Met Office.
- **Obs:** stamping obs on day 16 reproduces calc_RPC's saved cell-8 obs moments exactly (variance 158673.163, skew 0.129, kurtosis 2.216).
- **Ensemble:** the ensemble moments are within 0.4%.
- **Binning band:** the saved cell-20 band widths match within 1–6%. For λ_o the saved value is 0.196 and the rebuild gives 0.185; for λ, 0.140 and 0.146.
- **Stale output:** cell 16's saved γ means (0.71, 0.77) are stale. Its code prints lines that aren't in the saved output.
- **Mutation trap:** `info_RPC` writes NaN → 0 into its input ensemble in place. Anything computed from `f` after it sees zero-filled members.

## Figures

- `jugaad_lam_maps_<case>`: FD λ, Knuth λ and pooled Pearson ρ_o/ρ_m. Cells are stippled where the alt 0.5–1.5 band of λ contains 1.
- `jugaad_binning_uncertainty`: ΔI against I per cell, with the 1:1 line.
- `jugaad_bins_sensitivity`: mean I and λ against a common bin count, with FD and Knuth shown with their alt bands.
