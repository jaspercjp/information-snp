# Bootstrap significance for the ρ and λ RPC maps

This file documents the bootstrap behind the stippling in `rho_vs_lambda_<case>.png`. It
follows Eade et al. (2014, GRL) supporting information.

- **Code:**
  - Draws: `scripts/boot.py`
  - Figures: `scripts/plots.py`
  - Measures: `scripts/jugaad.py`
- **Raw draws:** `$SNP_OUT` (default `$SCRATCH/snp_jugaad/`)
  - `boot_fd_<VAR>_<case>.npz` holds ρ and the FD λ.
  - `boot_knuth_<VAR>_<case>.npz` holds the Knuth λ (bins re-chosen per draw) of `rho_vs_lambda_{2-4,s1961}`.
  - `boot_lmloo_<VAR>_<case>.npz` and `boot_olmloo_<VAR>_<case>.npz` hold the Knuth λ of the production figures.

Each statistic gets its own test. The ρ dots say whether the ρ RPC differs from 1. The λ
dots say whether the λ RPC differs from 1. The two columns are paired only in that they use
the same resampled data.

> **Current production figures** (`rpc_rho_lambda_lead2-4`, `rpc_rho_lambda_decadal`, from
> `scripts/plots.py`) differ from the description below in three ways:
>
> 1. **Bins are fixed.** Each λ draw keeps the cell's **full-data Knuth bin counts**
>    (`boot.py ... lmloo`, giving `boot_lmloo_*`). Re-choosing Knuth inside a draw is
>    wrong here: resampled years repeat exactly, and Knuth on the T obs and s values then
>    resolves the repeats (4 → 42–44 bins on SLP lead 2-4, with λ_o 0.26 → 0.48 inside the
>    draws). The draw would then be a different estimator from the one mapped. Binning
>    uncertainty is shown only by the red rings.
> 2. **λ_m uses leave-one-out means.** λ_m = λ(I(f_jugaad; s₋ₙ_jugaad)): each member's rows are
>    paired with the flattened mean of the other drawn members, not with tiled s. The s₋ₙ axis is
>    sized on its T cluster centres, mean_n s₋ₙ[n, t] = s[t].
> 3. **PRECT is decadal s1978** (GPCP starts in 1979). Its row is in the decadal figure.
>
>
> The `rpc_rho_lambda_oloo_*` figures add a third column (`boot.py ... olmloo`): the same, with
> λ_o = λ(I(s₋ₙ_jugaad; o_jugaad)), its s₋ₙ axis sized on the T cluster centres like λ_m's, and
> green rings for its binning band.
>
> The ρ draws are unchanged. Caveat: with fixed bins, cells where Knuth gives the obs one bin
> have λ_o = 0 in every draw and count as significantly below 1 by construction. For
> TREFHT s1961 that is 37.6 of its 97.8 percentage points.

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
   1. **Time indices `ti`, with replacement** (`boot.draws`).
      - **lead 2-4:** T start dates drawn independently: `rng.integers(0, T, T)`.
      - **s1961 (monthly):** whole calendar years. Draw `T // 12 = 10` year indices with replacement, then take all 12 months of each: `ti = year*12 + [0..11]`. This keeps the seasonal cycle and the within-year autocorrelation intact.
   2. **Members `mi`, without replacement:** `rng.choice(N, N - 3, replace=False)`. That is 59 members at lead 2-4, 89 for SLP s1961 and 109 for TREFHT s1961.
      - Eade: "repeatedly resampling the same members reduces the number of independent data points in the ensemble mean."
   3. **Resampled data:** `Fb = F[mi, :][:, ti]` (written `F[mi[:, None], ti]`, one copy) and `ob = o[ti]`.
      - The obs and every member get the same `ti`.
      - Duplicated years appear as duplicated time steps.
3. **Same draws for both columns.** Every `boot.py` mode (`fd`, `knuth`, `lmloo`, `olmloo`) uses the
   same seed and makes the same random-number calls in the same order, so draw b is the same
   (`ti`, `mi`) in all of them. The rewrite into `scripts/` was checked against the stored
   draws bitwise (point estimates and the first draws of every bootstrap file), on the CPU
   generation each file was made on: last-bit float64 differences between Sherlock's node
   generations (AMD Milan sh03 vs Siena/Genoa sh04) are expected.

---

## 2. ρ RPC on one draw (`boot.fd_stats` → `jugaad.rho_measures`)

Given Fb, with N' = N − 3 members and T time steps, and ob:

1. **Ensemble mean** of the drawn members: `s = nanmean(Fb, axis=0)`.
2. **ρ_o = corr(s, ob)** along time, per grid cell (`jugaad.corr_t`).
   - It uses the times where both series are finite, with the means and sums taken over those times.
   - Duplicated years count as many times as they were drawn.
3. **Leave-one-out means** `s₋ₙ`: the NaN-skipping mean of the other N' − 1 drawn members (`jugaad.loo_means`).
4. **ρ_m = ⟨corr(s₋ₙ, f_n)⟩ₙ**, the average over the N' drawn members of each member's correlation with the mean of the others. Same `corr_t`, no variance decomposition.
5. **RPC_ρ = ρ_o / ρ_m.**

- **Matches the project definition:** on NaN-free data, `rho_measures` equals the project's earlier `smyle_metrics.pearson_coeff` and `ensemble_SNR_loo` to 1e-12 (checked when this was written; `scripts/tests/test_jugaad.py` checks ρ_o against `np.corrcoef`).
- **Ensemble-mean ρ_m:** sd(s)/⟨sd(f_n)⟩ is computed only on the full data, for the json. It is not bootstrapped.

---

## 3. λ RPC on one draw (`jugaad.maps`)

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
   - The point estimate is stored with the draws, so the map and its dots come from one file.

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
$PY scripts/boot.py SLP 2-4 fd        # rho + FD lambda draws -> boot_fd_SLP_2-4.npz
$PY scripts/boot.py SLP 2-4 knuth     # Knuth lambda, bins re-chosen per draw (rho_vs_lambda), same draws
$PY scripts/boot.py SLP 2-4 lmloo     # Knuth lambda, lambda_m from s_-n, fixed bins, same draws
$PY scripts/boot.py SLP 2-4 olmloo    # the same with lambda_o from I(s_-n; o)
$PY scripts/plots.py                  # every figure; or bash scripts/run_all.sh for all of it
```

Cases: `fd` for the eight `data.CASES` and PRECT s1978; `knuth` for lead 2-4 and SLP, TREFHT
s1961 (the Knuth column of `rho_vs_lambda_{2-4,s1961}`); `lmloo` and `olmloo` for the six
production rows (SLP, PRECT, TREFHT at lead 2-4; SLP s1961, PRECT s1978, TREFHT s1961).
