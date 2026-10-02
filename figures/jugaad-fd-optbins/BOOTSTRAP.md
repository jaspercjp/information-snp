# Bootstrap significance for the ρ and λ RPC maps

This file documents the bootstrap behind the stippling in `rho_vs_lambda_<case>.png`. It
follows Eade et al. (2014, GRL) supporting information.

- **Code:**
  - Draws: `scripts/analysis/jugaad_rho_lambda_boot.py`
  - Figures: `scripts/analysis/jugaad_rho_vs_lambda_figs.py`
  - Measures: `scripts/metrics/jugaad_thesis.py`
- **Raw draws:** `$SCRATCH/snp_jugaad_fd_optbins/`
  - `boot_<VAR>_lead<LEAD>.npz` holds ρ and the FD λ.
  - `boot_knuth_<VAR>_lead<LEAD>.npz` holds the Knuth λ.

Each statistic gets its own test. The ρ dots say whether the ρ RPC differs from 1. The λ
dots say whether the λ RPC differs from 1. The two columns are paired only in that they use
the same resampled data.

---

## 0. Input data (once, before any resampling)

| case | loader | members N | time steps T | what one time step is |
|---|---|---|---|---|
| s1961 | `dcpp_decadal_handles.get(1961, anom="none")` | SLP 92, TREFHT 112 | 120 | one month, Jan 1962 – Dec 1971 |
| lead 2-4 | `dcpp_handles.get(lead="2-4")` | 62 | SLP 51, PRECT 33, TREFHT 52 | one start date (DJF mean, lead 2-4) |

- **Arrays:** F holds the members as (N, T, lat, lon) and o the observations as (T, lat, lon).
- **Preprocessing:** done once on the full data and never repeated inside a draw.
  - **s1961:** the thesis preprocessing. A least-squares line (intercept and slope) is removed per member and grid cell, and from the obs. The seasonal cycle is kept.
  - **lead 2-4:** `dcpp_handles` detrends across start dates per model, then pools the models. For SLP it also removes the area-weighted global mean (`remove_gm=True`).

---

## 1. The draws (shared by ρ and λ)

1. **Seed.** `rng = np.random.default_rng([2014, sum(ord(c) for c in VAR + LEAD)])`. This
   gives one fixed stream per case.
2. **For each draw b = 1 … 1000, in this exact order of random-number calls:**
   1. **Time indices `ti`, with replacement** (`time_index`).
      - **lead 2-4:** T start dates drawn independently: `rng.integers(0, T, T)`.
      - **s1961 (monthly):** whole calendar years. Draw `T // 12 = 10` year indices with replacement, then take all 12 months of each: `ti = year*12 + [0..11]`. This keeps the seasonal cycle and the within-year autocorrelation intact.
   2. **Members `mi`, without replacement:** `rng.choice(N, N - 3, replace=False)`. That is 59 members at lead 2-4, 89 for SLP s1961 and 109 for TREFHT s1961.
      - Eade: "repeatedly resampling the same members reduces the number of independent data points in the ensemble mean."
   3. **Resampled data:** `Fb = F[mi, :][:, ti]` (written `F[mi[:, None], ti]`, one copy) and `ob = o[ti]`.
      - The obs and every member get the same `ti`.
      - Duplicated years appear as duplicated time steps.
3. **Same draws for both columns.** The ρ run and the λ-only Knuth run (`--rule knuth`) use the
   same seed and make the same random-number calls in the same order, so draw b is the same
   (`ti`, `mi`) in both. I checked this directly: a code change was confirmed not to alter any
   draw.

---

## 2. ρ RPC on one draw (`stats` → `rho_measures`)

Given Fb, with N' = N − 3 members and T time steps, and ob:

1. **Ensemble mean** of the drawn members: `s = nanmean(Fb, axis=0)`.
2. **ρ_o = corr(s, ob)** along time, per grid cell (`_corr_t`).
   - It uses the times where both series are finite, with the means and sums taken over those times.
   - Duplicated years count as many times as they were drawn.
3. **Leave-one-out means** `s₋ₙ`: the NaN-skipping mean of the other N' − 1 drawn members (`jugaad_thesis.loo_means`).
4. **ρ_m = ⟨corr(s₋ₙ, f_n)⟩ₙ**, the average over the N' drawn members of each member's correlation with the mean of the others. Same `_corr_t`, no variance decomposition.
5. **RPC_ρ = ρ_o / ρ_m.**

- **Matches the project definition:** on NaN-free data, `rho_measures` equals `smyle_metrics.pearson_coeff` and `ensemble_SNR_loo` to 1e-12. The figure script checks this every time it runs.
- **Ensemble-mean ρ_m:** sd(s)/⟨sd(f_n)⟩ is computed only on the full data, for the json. It is not bootstrapped.

---

## 3. λ RPC on one draw (`jugaad_thesis.jugaad_maps`)

Same Fb and ob. The ensemble mean `s = nanmean(Fb, axis=0)` comes from the drawn members.
Then, independently per grid cell:

1. **Jugaad layout** (rows ordered member, time):
   - `f_jug`: the drawn members flattened, N'·T rows. NaN → 0, as in the thesis.
   - `o_jug`: ob tiled N' times. Row (n, t) holds ob[t].
   - `s_jug`: s tiled N' times.
2. **Bin counts, re-chosen on every draw.**
   - **FD (thesis):** `int(ptp / (2·IQR / n^(1/3)))` on each axis's pooled or tiled series, with no cap. If IQR = 0, the cell is NaN for that draw.
   - **Knuth OptBins (current figures):** Knuth's equal-width posterior, maximised over M.
     - The o and s axes are sized on their T values (duplicated years included), with M capped at T.
     - The f axis is sized on the N'·T pooled rows, with M capped at 200.
   - Bins are taken at 1.0×. The 0.5× / 1.5× band is not bootstrapped (see §5).
3. **MI:** `I_o = I(o_jug; f_jug)` and `I_m = I(s_jug; f_jug)`.
   - Computed with `np.histogram2d` using bins (B_o, B_f) and (B_s, B_f).
   - Plug-in estimate with float32 probabilities, in bits: I = H(x) + H(y) − H(x, y). This is the thesis `InfoTheoryMetrics.I`.
4. **λ:** `λ = sqrt(1 − 2^(−2 max(I, 0)))` for each of I_o and I_m.
5. **RPC_λ = λ_o / λ_m.** It is set to NaN when λ_m = 0, which happens when the rule gives s one bin, so I_m = 0 exactly.

- **λ_o uses I(f; o)**, member against obs, as in the thesis. It is not I(s₋ₙ; o).
- **λ_m uses the full drawn-member mean s**, as in the thesis.

---

## 4. From 1000 draws to a dot

For each statistic separately, per grid cell:

1. **Percentiles:** `q05 = nanpercentile(draws, 5)` and `q95 = nanpercentile(draws, 95)`. NaN draws are dropped. A cell is NaN only if every draw is NaN.
2. **Not significant (black dot):** `q05 ≤ 1 ≤ q95`. This is a 90% two-sided test, as in Eade.
3. **Significantly above 1:** `q05 > 1`. **Significantly below 1:** `q95 < 1`.
4. **What is plotted:**
   - The mapped colour is the full-data point estimate, not the bootstrap mean or median.
   - The figure script asserts that the point estimate stored with the draws equals the stored map.

The draws are percentile intervals with no bias correction and no multiple-testing correction,
as in Eade.

---

## 5. Masks and red rings (full data, not bootstrapped)

- **Grey, both columns:** corr(s, o) ≤ 0 on the full data. This follows Eade's Fig. 1.
- **Grey, λ column only:** the full-data λ RPC is undefined.
  - That means λ_m ≤ 0.02, almost always because Knuth gave s one bin.
  - This 0.02 floor is applied **only to the full-data map**. Bootstrap draws are NaN only where λ_m = 0 exactly.
- **Red ring (λ column):** drawn around a black dot when, on the full data,
  |λ(1.5× bins) − λ(0.5× bins)| > λ(1.0× bins).
  - "k× bins" means every axis's bin count is replaced by `ceil(k·B)`.
  - The test is on the mapped λ RPC.
  - For Knuth, the band uses the same rule as the map (sized on T values for o and s, on N·T for f).
- **Dot size:** the dot area scales with grid-cell area (1.6 pt² on the TREFHT T62 grid).

---

## 6. Known caveats

- **Knuth can force λ_o = 0.** When it picks one obs bin, which happens for the near-sinusoidal
  seasonal cycles in TREFHT s1961 over 38% of the area, λ_o = 0 on the full data and in nearly
  every draw. Those cells then count as "significantly below 1" by construction. That covers
  30 of the 73 percentage points for TREFHT s1961, 7 points for SLP s1961, and at most 1.2 at
  lead 2-4.
- **Resampled years enter as duplicates.**
  - In ρ they are repeated time steps in the correlations.
  - In λ they are repeated values in the bin-sizing series and repeated rows in the joint histogram (N' copies per drawn year).
- **No re-detrending inside a draw.** The anomalies come from the full-data preprocessing in §0.

---

## 7. Reproduce

```bash
PY=/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/SNP_env/bin/python
# rho + FD lambda draws (one file per case)
$PY scripts/analysis/jugaad_rho_lambda_boot.py SLP 2-4 --boot 1000
# Knuth lambda draws, same (ti, mi) as above
$PY scripts/analysis/jugaad_rho_lambda_boot.py SLP 2-4 --boot 1000 --rule knuth
# figures (Knuth lambda column; s1961 and lead 2-4)
LAM_RULE=knuth $PY scripts/analysis/jugaad_rho_vs_lambda_figs.py
```

Cases run: SLP, PRECT, TREFHT for lead 2-4; SLP and TREFHT for s1961 (GPCP, the PRECT obs,
starts in 1979).
