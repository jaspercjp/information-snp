# Pairwise exploratory figures — SLP

The original set: the pairwise formulation of the information RPC, measured on DCPP-A
sea-level pressure. `figures/pairwise-exploratory-prect/` is this same set rebuilt for
precipitation, and its README carries the side-by-side comparison — read that one for
the PRECT-vs-SLP contrast rather than duplicating it here.

`png/` holds the figures, `json/` the numbers behind them — including the
`pairnull_*` / `pairdenom_*` run summaries the figures are built from. Both carry the
same stems.

Two layouts appear throughout:

| | SLP s1961 (decadal) | SLP lead 2-4 (seasonal) |
|---|---|---|
| members N | 92 | 62 |
| samples T | 120 months | 51 start dates |
| cells C | 2664 (37×72) | 2664 |
| bins B | 4 | 3 |
| pairs | 4186 | 1891 |

---

## 1. The pairwise formulation, and why it was adopted

`figA_pairwise_vs_loo_SLP_*`, `figB_mm_residual_SLP_*`.

Replacing the leave-one-out `mean_n I(s_-n; o)` with the pairwise `mean_n I(f_n; o)`
collapses the permutation null's spread. Leave-one-out means differ by one member in
N, so their average retains nearly the full single-term variance; the members
themselves are near-independent and theirs averages down. Measured null sd ratio
(LOO / pairwise): **7.5×** at s1961, **5.0×** at lead 2-4.

The decadal numerator, from `json/pairnull_SLP_s1961_B4_P500.json`, is a weak signal
either way — debiased I = +0.0032 at SNR 1.49 in the pairwise form. This is the point
the PRECT README returns to: decadal PRECT reaches SNR 5.64 on the same machinery, so
the decadal SLP case is the harder one independent of anything about non-Gaussianity.

`figB` is the argument that Miller–Madow alone is not enough. MM's residual is a fixed
offset that averaging does not remove, while the noise it must be small against does,
so the ratio grows with the number of terms averaged — residual / null sd is **0.67**
for the s1961 numerator and **5.11** for its denominator. Every number in this
directory subtracts the empirical permutation null mean explicitly.

## 2. RPC: λ against ρ

`figC_pairwise_rpc_SLP_{s1961,lead2-4}`. Three ρ summaries × {pairwise, LOO} ×
{numerator, denominator, ratio}. From `json/figC_pairwise_rpc_SLP_s1961.json`:

| form | ρ variant | RPC_ρ | RPC_λ |
|---|---|---|---|
| pairwise | signed mean | **0.610** | 0.842 |
| pairwise | mean \|ρ\| | 0.959 | 0.842 |
| LOO | signed mean | 0.710 | 0.765 |
| LOO | mean \|ρ\| | 0.860 | 0.765 |

and at lead 2-4, where everything sits near unity: RPC_λ = 0.982 (pairwise) against
RPC_ρ = 1.033 signed / 1.011 mean|ρ|.

The signed-mean ρ collapses at s1961 — ρ_o = 0.0094 against mean|ρ| = 0.0908, a factor
of 10, because members disagree in sign and the average cancels. λ is non-negative and
cannot cancel, so its like-for-like ρ partner is mean|ρ|, not the signed mean the RPC
conventionally uses.

**Colour limits.** The four metric panels use `jet` from 0 to **0.7**, set by
`--vmax`. They were 0–1.0 in the first version of this set; 0.7 is the current
project setting and both variables were regenerated with it, so the SLP and PRECT
figC panels remain directly comparable. The ratio panels are unchanged at `RdBu_r`
0–2.

## 3. The non-Gaussian excess — the result that did NOT survive on SLP

`figD_excess_{member,lam_minus_rho}_SLP_s1961_{ksg,mm}`. λ_o − ρ_o with the
estimator's own bias subtracted, measured against a Gaussian-copula surrogate with the
same per-pair rank correlation. For a jointly Gaussian pair λ = |ρ| exactly, so this
is zero under Gaussianity.

| layout | estimator | λ_o | ρ_o (rank) | raw λ−ρ | estimator bias | **excess** | area +ve |
|---|---|---|---|---|---|---|---|
| pairwise (`member`) | KSG | 0.1467 | 0.0883 | +0.0584 | 0.0553 | **+0.0031** | 44.7% |
| pairwise (`member`) | MM B=4 | 0.0922 | 0.0883 | +0.0039 | 0.0108 | **−0.0069** | 24.8% |
| LOO (`lam_minus_rho`) | KSG | 0.1607 | 0.1031 | +0.0576 | 0.0467 | **+0.0109** | 46.7% |
| LOO (`lam_minus_rho`) | MM B=4 | 0.0969 | 0.1031 | −0.0062 | 0.0034 | **−0.0096** | 38.5% |

**The excess flips sign between KSG and Miller–Madow**, and is ~10% of ρ_o either way.
On SLP it therefore cannot be claimed. That constraint is what the PRECT rebuild was
for, and on PRECT it does not bind — all four configurations there are positive and
40–65% of ρ_o.

Note the naming: this set's `lam_minus_rho` stem is the **LOO** layout, which the
PRECT set names `loo`. The files were left under their original names so that the
prior handoffs' paths still resolve.

## 4. ρ_o is one field measured three ways

`figE_rho_o_three_ways_SLP_s1961`. Leave-one-out and plain ensemble-mean are the same
field — pattern r = **0.9999986**, slope 1.0032 — and pairwise is that pattern scaled
by σ_s/σ_f: r = 0.939, slope 5.92. Area means 0.0094 (pairwise), 0.0567 (LOO), 0.0569
(ensemble mean). A control, not a result.

## 5. Bootstrap

`figF_pairwise_bootstrap_SLP_s1961_B4`, 1000 draws of M=61 of N=92 members, members
only, time axis never resampled: RPC (ratio of area means) **0.849**, the 95% CI holds
1 on 63.6% of area, 30.9% differs from 1 after BH-FDR q=0.10, on 84.6% usable area.
The RPC is **below** 1, so those significant cells are not the RPC > 1 cells the
project is trying to explain away.

## 6. Marginal distributions — what the MI pipeline is blind to

`figG_marginal_moments_SLP_{s1961,lead2-4}`. **New.** Everything above measures
non-Gaussianity of the *dependence structure*: the pipeline copula-transforms, so
per-cell marginals are mapped to ranks and their shape is discarded by construction.
This figure measures that discarded piece — the pooled histogram of model and observed
values over all cells, with skewness and excess kurtosis computed two ways.

Three rows, because the answer depends entirely on which preprocessing stage is meant:
`raw`, `anomaly` (month-of-year climatology removed) and `PIPELINE` (+ the Eade
11.25°×12.5° box mean). **Only the bottom row is what figA–figF operate on.**

SLP s1961, the pipeline row:

| | model | obs |
|---|---|---|
| skewness, per-cell then area-averaged | −0.002 | −0.074 |
| excess kurtosis, per-cell then area-averaged | **+0.737** | **+0.797** |
| skewness, pooled + per-cell standardised | +0.061 | +0.024 |
| excess kurtosis, pooled + per-cell standardised | +0.735 | +0.715 |
| skewness, pooled in native units | +0.240 | +0.278 |
| excess kurtosis, pooled in native units | **+3.921** | **+4.248** |

Three things to take from it.

**SLP marginals are nearly symmetric but mildly heavy-tailed, and the model gets this
right.** Skewness is within 0.08 of zero on both sides; excess kurtosis is ~+0.75 on
both. At lead 2-4 the same quantities are +0.021/+0.010 skew and +0.269/+0.314 excess
kurtosis — the lead-window mean is an average over three months, so it is closer to
Gaussian than the monthly field, exactly as the central limit theorem requires.

**The pooled-in-native-units kurtosis (+3.9) is mostly an artefact, and that is why
two summaries are reported.** Pooling raw values across cells mixes cells of very
different variance, and a scale mixture of Gaussians is leptokurtic even when every
component is exactly Gaussian. Standardising each cell before pooling removes it and
returns +0.735 — which agrees with the per-cell-then-averaged +0.737, as it should.
**Quote the standardised or the per-cell number, never the native-units pooled one.**

**Smoothing Gaussianises, so the stage matters.** For SLP the effect is small
(per-cell excess kurtosis +0.736 unsmoothed → +0.737 smoothed), but for PRECT the same
box mean cuts it from +10.9 to +3.8. Any statement of the form "this field is
non-Gaussian" has to name the stage.

Model/observed pattern correlations of the per-cell maps are +0.48 (skewness) and
+0.42 (excess kurtosis) at s1961 — the model reproduces the geography of SLP's
non-Gaussianity only moderately.

**Read the observed maps as fields, not cell by cell.** The model has N·T = 11040
samples per cell against the observations' T = 120, so se(skew) is 0.023 against 0.224
and se(excess kurtosis) 0.047 against 0.447. The area *mean* of the observed moments
is fine — it averages unbiased estimates — but individual observed cells are noisy.
Both standard errors are annotated on the panels. The model's own figure is a floor,
not a true standard error: members share a forced signal and months are autocorrelated,
so N·T independent samples is an overstatement.

Moments are the bias-corrected G1 / G2 (scipy's `bias=False`), implemented in numpy and
asserted at run time against scipy (max |diff| 4e-16), against the analytic values for
the normal, exponential and uniform distributions, and — since the correction is
invisible at large n — against the known −6/(n+1) bias of the uncorrected g2 at n=20.

---

## Files

| stem | what |
|---|---|
| `figA_pairwise_vs_loo_SLP_{s1961,lead2-4}` | pairwise vs LOO: debiased signal, null sd, z, numerator and denominator |
| `figB_mm_residual_SLP_{s1961,lead2-4}` | why Miller–Madow alone fails |
| `figC_pairwise_rpc_SLP_{s1961,lead2-4}` | 3 ρ variants × {pairwise, LOO} × {numerator, denominator, RPC} |
| `figD_excess_{member,lam_minus_rho}_SLP_s1961_{ksg,mm}` | non-Gaussian excess of the DEPENDENCE, both estimators, both layouts |
| `figE_rho_o_three_ways_SLP_s1961` | pairwise / LOO / ensemble-mean ρ_o |
| `figF_pairwise_bootstrap_SLP_s1961_B4` | member-subsample bootstrap of the pairwise λ RPC |
| `figG_marginal_moments_SLP_{s1961,lead2-4}` | MARGINAL distributions: pooled histograms, skewness, excess kurtosis |
| `pairnull_SLP_*` / `pairdenom_SLP_*` (json only) | the underlying run summaries the figures read |

figD, figE and figF are decadal-only; figA, figB, figC and figG exist at both layouts.

`figA`/`figB` were regenerated when this directory was reorganised, to pick up the
`_SLP_s1961` / `_SLP_lead2-4` tag and the accompanying `.json` that the PRECT set has
and the original untagged `figA_pairwise_vs_loo.png` / `figB_mm_residual.png` did not.
The untagged originals are preserved at
`$SCRATCH/snp_figs_backup_20260921/pairwise-exploratory/`.

## A caveat that applies to every map in this directory

The maps are drawn with `imshow` straight from a `(lat, lon)` array. The observational
grids run **lat[0] = −90 upward**, and matplotlib's default is `origin="upper"`, so
every map in figA–figF has the **South Pole at the top**. `figG` passes
`origin="lower"` and is the right way up. This is a display-only issue — no computed
number is affected, and the area weights are applied in data space — but the older
panels are vertically mirrored relative to figG and should be fixed before any of them
is published.

## Reproducing

Scripts live in `.claude/scripts/` (gitignored). Order matters: the numerator and
denominator runs write the `pairnull_*`/`pairdenom_*` npz into
`$SCRATCH/snp_pairwise_null/`, and figA, figB, figC and figF all read them. figG reads
none of them — it goes straight to the data handles.

```
python .claude/scripts/tmp_pairwise_null_s1961.py  --var SLP --start 1961 --perm 500
python .claude/scripts/tmp_pairwise_denom_s1961.py --var SLP --start 1961 --perm 200
python .claude/scripts/tmp_pairwise_figs.py        --var SLP --start 1961 --bins 4
python .claude/scripts/tmp_pairwise_rpc.py         --var SLP --start 1961 --vmax 0.7
python .claude/scripts/tmp_pairwise_bootstrap.py   --var SLP --start 1961 --draws 1000
python .claude/scripts/tmp_excess_lam_map.py       --var SLP --start 1961 --estimator ksg --layout member
python .claude/scripts/tmp_rho_o_three_ways.py     --var SLP --start 1961
python .claude/scripts/tmp_marginal_moments.py     --var SLP --start 1961
```

with `--dataset seasonal --lead 2-4 --bins 3` for the seasonal case. Every script needs
`SNP_REPO` pointing at a checkout that has `data/` and the untracked
`scripts/metrics/jugaad_metrics.py`; a fresh worktree has neither.

`tmp_marginal_moments.py` costs about 1 min for SLP and 3.5 min for PRECT, peaking at
6.1 GB on the PRECT grid — run it alone, the job's cgroup ceiling is 17.2 GB.

Percentages of area throughout are cos(lat)-weighted, including the colour limits of
figG's maps: an unweighted percentile there is set by the near-polar rows, which carry
almost no area but the most extreme moments.
