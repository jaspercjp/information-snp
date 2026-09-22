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

`figD_excess_{member,loo}_SLP_s1961_{ksg,mm}`. λ_o − ρ_o with the estimator's own bias
subtracted, measured against a Gaussian-copula surrogate with the same per-pair rank
correlation. For a jointly Gaussian pair λ = |ρ| exactly, so this is zero under
Gaussianity.

| layout | estimator | λ_o | ρ_o (rank) | raw λ−ρ | estimator bias | **excess** | area +ve |
|---|---|---|---|---|---|---|---|
| pairwise (`member`) | KSG | 0.1467 | 0.0883 | +0.0584 | 0.0555 | **+0.0029** | 44.9% |
| pairwise (`member`) | MM B=4 | 0.0922 | 0.0883 | +0.0039 | 0.0109 | **−0.0070** | 25.0% |
| LOO (`loo`) | KSG | 0.1606 | 0.1031 | +0.0575 | 0.0468 | **+0.0107** | 46.5% |
| LOO (`loo`) | MM B=4 | 0.0969 | 0.1031 | −0.0062 | 0.0033 | **−0.0095** | 38.8% |

**The excess flips sign between KSG and Miller–Madow**, and is ~10% of ρ_o either way.
On SLP it therefore cannot be claimed. That constraint is what the PRECT rebuild was
for, and on PRECT it does not bind — all four configurations there are positive and
40–65% of ρ_o.

Two bookkeeping notes on this table.

**The excess moves in the fourth decimal between runs, and that is expected.**
Everything on the data side — λ_o, both ρ_o, the raw difference — reproduces bit for
bit. Only `estimator_bias` shifts, by ~8e-5, because the Gaussian-copula surrogate
stream is re-seeded per member block and the block size is a memory choice rather than
a statistical one. The excess is that bias subtracted, so it inherits the wobble: the
KSG pairwise figure has read +0.0031 and +0.0029 on two runs. The sign, which is the
whole point, does not move.

**This set's figD files used to be named `figD_excess_lam_minus_rho_*`.** They are the
LOO layout, so they are now `figD_excess_loo_*`, matching the PRECT set. Prior
handoffs that cite `figD_excess_member_SLP_s1961_mm.json` are unaffected — that stem
is unchanged — and the old names remain in git history.

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

## 7. Does high RPC live where both model and obs are heavy-tailed? — **no**

`figH_rpc_vs_kurtosis_SLP_s1961`. The hypothesis: outliers have an outsized effect on
a Pearson ρ and almost none on a rank/information λ, so RPC_ρ should be elevated where
the model *and* observed marginals both have fat tails, and RPC_λ should not be.

**The answer is no, and λ is what settles it.**

### λ cannot respond to outliers, and that is a structural fact

λ is built from equiprobable bins of the copula ranks, so it is invariant under any
monotone change of a cell's marginal — including clipping the tails, provided the clip
sits outside the outermost bin edge. At B=4 that edge is the 25th/75th percentile and
the clip is at 2.5σ, far beyond it. The script asserts this rather than assuming it:
bin membership is **identical for 100.0000%** of (cell, sample) pairs over 200 cells.

That makes λ a clean control, and it collapses the hypothesis to a question about ρ.

### The spatial coincidence is weak, and λ shares it exactly

Taking the user's framing literally, as a contingency statement:

| | RPC_ρ | RPC_λ |
|---|---|---|
| area with RPC > 1 | 39.1% | 37.0% |
| P(both excess kurtosis > 0 \| RPC > 1) | 81.1% | 81.9% |
| P(both excess kurtosis > 0 \| RPC ≤ 1) | 72.9% | 72.9% |
| **enrichment ratio** | **1.11** | **1.12** |

So RPC > 1 regions *are* slightly more likely to be jointly heavy-tailed — but λ is
enriched by exactly the same factor, and λ cannot see outliers. Whatever produces the
enrichment is a property both metrics share, not Pearson leverage.

Note also that the both-positive mask covers **76% of area** here, so as a binary
classifier it is barely selective. A stricter mask (both > 0.45, one observed standard
error) cuts that to 42%, and the inside/outside contrast grows for *both* metrics —
ρ 1.003 vs 0.944, λ 0.908 vs 0.779. Again no differential.

### On the primary statistic, λ is the *more* kurtosis-sensitive metric

The slope dRPC/dK, in RPC units per unit of joint excess kurtosis
(K = min(model, obs)), block-bootstrapped over 3×3-cell tiles:

| | slope | 95% CI | tile-level | latitude-controlled |
|---|---|---|---|---|
| RPC_ρ | +0.042 | [+0.026, +0.061] | +0.050 | +0.024 |
| RPC_λ | **+0.121** | [+0.063, +0.184] | +0.163 | +0.022 |
| difference ρ − λ | **−0.076** | [−0.132, −0.025], p = 0.002 | | |

The difference is *negative* and significant: λ's RPC rises with joint kurtosis about
three times faster than ρ's. That is the opposite of the prediction. Once latitude is
regressed out the two are indistinguishable (+0.024 vs +0.022), which says most of the
raw difference was shared latitudinal structure rather than anything about tails.

**Do not use the correlation for this comparison.** An earlier version of this analysis
reported corr(RPC_ρ, K) = +0.234 against corr(RPC_λ, K) = +0.112 with p = 0.004, and
read it as support. It is not: RPC_λ's per-cell spatial sd is 0.70 against RPC_ρ's
0.10, because λ is debiased and null-subtracted and the ratio of two small debiased
quantities is unstable. Noise in a field attenuates its correlation with anything, so
the smoother field wins regardless of mechanism. The slope is not attenuated that way,
which is why it is primary.

### The proposed mechanism is real, and tiny

Clipping every cell's series at ±2.5σ (1.9% of model values, 1.9% of obs) and
recomputing ρ gives ΔRPC_ρ = RPC_ρ(raw) − RPC_ρ(clipped) — the part of RPC that
outliers are actually responsible for:

- area mean ΔRPC_ρ = **+0.0013**, against RPC_ρ = 0.969. Outliers account for about
  **0.13%** of its value.
- but ΔRPC_ρ *does* rise with joint kurtosis, slope **+0.0078** [+0.0050, +0.0104],
  p < 0.001, and is +0.0028 inside the heavy-tail region against −0.0038 outside.

So the direction the hypothesis predicts is there and is statistically solid. It
accounts for **18%** of the RPC_ρ–kurtosis slope and a tenth of a percent of RPC_ρ
itself. The mechanism exists; it is not what drives the association.

See the PRECT README's section 7 for the decisive cross-check: on the variable with
five times the excess kurtosis, the association disappears entirely.

### Caveats

- **Decadal only.** Per-cell kurtosis needs samples: T = 120 gives se(excess
  kurtosis) = 0.45 on the observed side. At lead 2-4 (T = 51) it is 0.69, and for
  PRECT seasonal (T = 33) 0.85 — too noisy for a per-cell mask, so figH is not
  produced for the seasonal layouts.
- **The observed mask is part coin-flip.** With se 0.45, a cell whose true excess
  kurtosis is 0 is classified positive half the time. This attenuates toward the
  null, so the weak positive findings above are, if anything, understated — which is
  why the stricter mask is reported too.
- **mean\|ρ\|, not the signed mean.** The signed per-cell ratio is defined on only 29%
  of area here, and that 29% is selected by where \|ρ_m\| is large — not a sample you
  can run a spatial test on. `rho_sgn` is in the json for reference.
- Degrees of freedom are tiles, not cells: 312 tiles against 2549 usable cells.

## 8. figH2 — the same test with the ensemble mean as the model, LOO only

`figH2_rpc_vs_kurtosis_ensmean_SLP_s1961`. figH took the model's kurtosis from all
N·T pooled member values. But in the leave-one-out scheme the series that actually
enters ρ_o is `s_-n`, the leave-one-out ensemble mean — an outlier can only leverage
ρ_o if it is an outlier *of that series*. So figH2 measures the ensemble mean's
kurtosis instead, drops the pairwise form, and is the better-posed version of the
test. (Kurtosis is taken from the full ensemble mean; `s_-n` differs by one member in
N, checked: area means +1.7929 vs +1.8008, pattern r 0.9968.)

### Averaging the members makes SLP *more* heavy-tailed, not less

| | area-mean excess kurtosis |
|---|---|
| pooled members (figH) | +0.737 |
| **ensemble mean (figH2)** | **+1.801** |
| observations | +0.797 |

Pattern correlation between the two model versions is only **+0.33**, so they are
substantially different fields and figH was measuring the wrong one for this scheme.

This runs against the naive central-limit expectation, and the reason is
informative: averaging suppresses the unforced part by ~1/√N while leaving the common
forced signal untouched, so the ensemble mean's shape is the *predictable*
component's shape. For SLP that component is markedly more kurtotic than the weather
it is buried in. (PRECT goes the other way — see its README §7.)

The price is precision: the model kurtosis now has T samples rather than N·T, so its
per-cell standard error rises from 0.05 to **0.45**, the same as obs. Both maps are
now noisy, the joint mask is more of a coin flip than in figH, and a null result here
is partly a power problem.

### The hypothesis still fails, and now for a sharper reason

| | slope dRPC/dK | 95% CI | latitude-controlled |
|---|---|---|---|
| RPC_ρ | +0.145 | [+0.033, +0.274] | +0.006 |
| RPC_λ | +0.148 | [+0.029, +0.271] | +0.105 |
| difference ρ − λ | −0.001 | [−0.140, +0.139], p = 0.99 | |

The two metrics are **indistinguishable**, and after controlling for latitude it is λ
that retains the relationship while ρ loses it. Enrichment: ρ 1.11, λ 1.18 — again λ
is the larger. Inside the joint heavy-tail region RPC_ρ = 1.038 against 0.834
outside, which looks impressive until you see RPC_λ do the same thing (0.956 vs
0.708).

**The mechanism is 16× bigger here than in figH, though.** Clipping every correlated
series — `s_-n`, `f_n` and `o` alike — at ±2.5σ moves the area-mean RPC_ρ by
**+0.0207** against figH's +0.0013, i.e. 2.1% of RPC rather than 0.13%, and ΔRPC_ρ
rises with joint kurtosis (slope +0.026, p = 0.054; +0.028 inside the heavy-tail
region against +0.003 outside). That makes sense: the ensemble mean has far less
independent noise than a single member, so one extreme month carries more leverage
over its correlation. Outlier leverage on ρ is therefore a real, measurable, ~2%
effect in the LOO scheme — it is simply not what produces the RPC–kurtosis
association, because λ shows the association just as strongly and cannot see
outliers at all.

### Conditioning on where obs is heavy-tailed *relative to* the model

The second conditioner, added on request:
**D = excess kurtosis(obs) − excess kurtosis(ensemble mean)**, a difference rather
than a ratio because excess kurtosis is signed and routinely negative here. Positive
D means the observations carry extremes the model's ensemble mean does not.

This is the sharper condition, and it predicts the *opposite* sign to `K`: under `K`
both series are heavy-tailed so a shared extreme month can co-occur and inflate ρ_o,
whereas under `D` the extremes are one-sided and should pull the series apart.

It produces the largest effect anywhere in figH/figH2:

| | RPC_ρ | RPC_λ |
|---|---|---|
| slope on D | **+0.084** [+0.032, +0.110] | +0.052 [+0.006, +0.097] |
| RPC inside D > 1 se | **1.349** | 1.152 |
| RPC outside | 0.904 | 0.835 |
| RPC inside D > 2 se | **1.635** | 1.318 |
| enrichment P(D>1se \| RPC>1) / P(D>1se \| RPC≤1) | **2.10** | 1.50 |

RPC crosses *above 1* in that region — 1.35, rising to 1.64 at the stricter
threshold — and the enrichment is 2.1×, twice anything the `K` conditioning gave.

**But the decomposition shows it is a denominator effect, and has nothing to do with
outliers.** Splitting RPC = ρ_o/ρ_m and normalising each term by its own area mean:

| term | relative slope on D |
|---|---|
| ρ_o (numerator) | **+0.004** — flat |
| ρ_m (denominator) | **−0.091** |
| λ_o | −0.014 |
| λ_m | −0.103 |

RPC_ρ's relative slope on D, +0.0955, is +0.004 from the numerator minus (−0.091)
from the denominator. The numerator — the only term that touches the observations —
does not move at all. λ_m behaves almost identically to ρ_m.

The interpretation is straightforward once seen. D is mostly *negative* here (obs is
heavier than the ensemble mean on only 34% of area), so "large D" mainly means *the
ensemble mean is unusually close to Gaussian*, not that the observations are extreme.
Where the ensemble mean has occasional large forced excursions, members agree with it
strongly and ρ_m is high, which depresses RPC. Where it is well-behaved, ρ_m is lower
and RPC rises. That is a statement about the model's own signal-to-noise structure,
not about observed outliers leveraging a Pearson correlation — and λ, which cannot
see outliers, reproduces it.

So the strong D signal is real and worth knowing, but it is not evidence for the
hypothesis; it is a reason to be careful reading any RPC map, since a cell's RPC can
be raised by its denominator going quiet.

## 9. figI — just look at it: ensemble mean vs observations where RPC > 1

`figI_ensmean_vs_obs_regions_SLP_s1961`. Twenty standard climate-index boxes are
screened on their area-mean RPC_ρ, and the ones that clear 1 get a scatter of the
model ensemble mean against the observations, pooled over the box's cells and months.
The selected boxes are outlined on the RPC_ρ map in panel 1. Purpose: see whether
there is any obvious non-linearity in the relationship.

**There is not.** Every cloud is a plain tilted ellipse. The sharpest check available
without fitting anything is the Pearson-versus-Spearman gap — Spearman is invariant
to any monotone distortion and Pearson is not, so a gap indicates curvature. The
largest gap over the five selected boxes is **−0.056** (Azores High) and every other
is **≤ 0.018**. At this resolution and sample size the ensemble-mean-to-observation
relationship is linear to within the noise.

| box | ρ_o | λ_o | RPC_ρ (sign-blind) | RPC_ρ (signed) | r | r_s |
|---|---|---|---|---|---|---|
| Amundsen Sea Low | +0.211 | 0.124 | 1.85 | +2.81 | +0.21 | +0.21 |
| Subpolar N Atlantic | +0.116 | 0.142 | 1.66 | +3.70 | +0.10 | +0.11 |
| Barents / Kara | +0.135 | 0.064 | 1.65 | +62.4 * | +0.15 | +0.13 |
| Azores High | +0.092 | 0.163 | 1.40 | +3.10 | +0.10 | +0.04 |
| W trop Pacific (Nino4) | +0.119 | 0.076 | 0.81 | +1.08 | +0.12 | +0.11 |

λ_o is `mean_n I(s_-n; o)` expressed as λ — the leave-one-out form, Miller–Madow
corrected *and* permutation-null-mean subtracted. It is non-negative by construction,
so unlike ρ_o it carries no sign ambiguity; comparing it against |ρ_o| is the
like-for-like reading. On SLP the two are within ~0.05 of each other everywhere here,
so λ is not seeing much that ρ misses.

\* signed denominator collapsed; see below.

### The eastern tropical Pacific does not have RPC > 1 here

Worth saying plainly, since it was the example region. Niño3 gives ρ_o = +0.113 but
|ρ_m| = 0.235, so **RPC_ρ = 0.48**; Niño3.4 is 0.67. On PRECT it is worse — 0.18 and
0.11 (PRECT README §9). The reason is the denominator, not the skill: the eastern
tropical Pacific is strongly forced, so the members agree closely with each other,
ρ_m is large, and the ratio is small however well or badly the model tracks the
observations. RPC > 1 in this dataset is a **high-latitude** phenomenon — Amundsen
Sea Low, subpolar North Atlantic, Barents/Kara, Azores High — where ρ_m is small
because the members barely agree.

### Two things to be careful about when reading these panels

**Sign-blindness is doing work.** RPC_ρ here is |corr(ens mean; o)| over
mean_n |corr(s_-n; f_n)|, because the conventional signed denominator cancels box by
box: Barents/Kara has a signed ρ_m of ~0.002 and a signed RPC of **+62**, and the
Icelandic Low comes out at −3.9. The sign-blind version is stable and is what the
panels quote; both are on every title and in the json. But sign-blindness means a
box can score well on an *anti*correlation — see the PRECT Amundsen Sea Low panel,
RPC_ρ = 2.57 with ρ_o = **−0.22**. Whether a member anticorrelated with the
observations should count as skill is a modelling judgement, and the RPC does not
make it for you.

**RPC > 1 does not imply the region is interesting.** Several boxes clear the
threshold with a numerator that is essentially zero — PRECT's Azores High has
ρ_o = −0.022 and RPC_ρ = 1.20; its Aleutian Low has ρ_o = −0.024 and RPC_ρ = 1.00.
A ratio of two small numbers exceeding 1 is not evidence of a paradox, and the
scatter panel makes that obvious in a way the map does not.


### The seasonal layout is a different world — and it is where the nonlinearity lives

`figI_ensmean_vs_obs_regions_SLP_lead2-4`. At lead 2-4 the sample axis is 51 start
dates rather than months within one hindcast, and the model actually has skill:
ρ_o reaches +0.80 (Maritime Continent) against a decadal maximum of +0.21. **Eleven of
the twenty boxes clear RPC > 1**, against five in the decadal case.

The Pearson–Spearman gaps grow too, though they stay modest on SLP: Sahel **−0.082**
(r +0.484, r_s +0.402), Trop S Atlantic −0.054, SPCZ −0.050. That is three to four
times the decadal gaps, in the same direction — Pearson running above Spearman, which
is what a few dominant co-occurring events produce. See the PRECT README §8 for the
same effect an order of magnitude larger.

Note also that the signed RPC becomes usable here: with real skill the signed
denominator no longer cancels, and signed and sign-blind RPC agree closely in most
boxes (Trop N Atlantic 1.209 vs 1.182, Trop S Atlantic 1.032 vs 1.032). The decadal
pathology was a symptom of having almost no signal, not a property of the metric.

### Deliberate choices

- **Standardised per cell before pooling.** A box holds 27–160 cells whose variances
  differ by an order of magnitude; pooling them raw gives a bow-tie that mimics
  curvature but is only heteroscedasticity. Raw-unit standard deviations are in the
  json.
- **Boxes screened after the fact, not chosen by eye.** The candidate list is the
  standard index regions, fixed before looking at any RPC; the full pass/fail table
  for all twenty is printed and stored. Five of twenty clear the threshold on SLP.
- **No p-values.** Cells within a box are spatially correlated (11.25°×12.5° box
  smoothing) and share one forced signal in time, so the effective sample size is far
  below the 3 000–20 000 points plotted. A tight cloud is not proportionally strong
  evidence.

---

## 10. figJ — is any of it distinguishable from zero?

`figJ_region_significance_SLP_lead2-4`. figI shows boxes with RPC > 1 whose scatter is
a near-circular blob: Barents/Kara has ρ_o = 0.178 and λ_o = 0.108, and clears RPC > 1
only because ρ_m = 0.135 and λ_m are smaller still. Is any of that real?

**For Barents/Kara, no.** 500 permutations and 500 bootstrap draws over the 51 start
dates:

| statistic | value | permutation null | p |
|---|---|---|---|
| ρ_o (signed) | +0.178 | mean +0.003 | **0.184** |
| mean\|ρ_o\| | 0.178 | mean **0.115** | **0.170** |
| λ_o | 0.108 | sd 0.062 | **0.314** |
| RPC_ρ | 1.32 | — | 95% CI **[0.19, 2.40]** |

### The null mean of mean|ρ_o| is 0.115, and that is the whole point

A bootstrap alone cannot answer this question, because neither statistic has zero
expectation under independence. At T = 51 an independent pair has
sd(ρ) = 1/√50 = 0.141, so **E|ρ| = 0.141·√(2/π) = 0.113** — and the permutation null
reproduces exactly that in all twenty boxes (0.109–0.118). Barents/Kara's 0.178 is
therefore only **1.6× the pure-chance floor**, and λ_o sits 1.7 null standard
deviations out. Neither clears 5%, and the RPC interval spans 0.19 to 2.40.

**Five of the twenty boxes fail both tests**: Barents/Kara, Icelandic Low,
Mediterranean, S Ocean (Indian), Subpolar N Atlantic. Two of those — Barents/Kara and
the Mediterranean — are among the eleven that figI selected as RPC > 1. Their RPC
exceeds 1 because it is a ratio of two quantities that are both noise, which is the
caution in §9 now with a p-value attached. **No SLP box has an RPC_ρ interval entirely
above 1.**

The tropical boxes are unambiguous by contrast: Maritime Continent ρ_o +0.801 /
λ_o 0.670, Niño3 +0.760 / 0.596, IOD east +0.738 / 0.623, all p < 0.002.

### Two traps this figure had to avoid

**A λ bootstrap CI excluding zero means nothing.** λ = √(1−exp(−2·max(I−null, 0))) is
non-negative by construction, so its bootstrap can never reach 0 — Barents/Kara's λ_o
CI is [0.10, 0.36], which looks conclusive and is not. The permutation p of 0.314 is
the test, and the figure plots λ_o against its null's 95th percentile rather than
against zero.

**Start dates are resampled, not cells.** A box's cells share the same 51 start dates
and are spatially correlated by the 11.25°×12.5° smoothing, so they are not
independent replicates of the model–observation relationship; resampling them would
shrink the interval without adding information. The permutation likewise applies ONE
shuffle of the observed order to every cell, so the observed field's spatial
covariance survives and only its alignment with the model is destroyed.

λ is formed per cell and then area-averaged, matching figI and the stored pipeline. An
earlier version averaged the MI first and mapped afterwards; λ is concave, so that
read 0.063 against figI's 0.107. The table above uses the per-cell convention and
reproduces figI at 0.108.

---

## Files

| stem | what |
|---|---|
| `figA_pairwise_vs_loo_SLP_{s1961,lead2-4}` | pairwise vs LOO: debiased signal, null sd, z, numerator and denominator |
| `figB_mm_residual_SLP_{s1961,lead2-4}` | why Miller–Madow alone fails |
| `figC_pairwise_rpc_SLP_{s1961,lead2-4}` | 3 ρ variants × {pairwise, LOO} × {numerator, denominator, RPC} |
| `figD_excess_{member,loo}_SLP_s1961_{ksg,mm}` | non-Gaussian excess of the DEPENDENCE, both estimators, both layouts |
| `figE_rho_o_three_ways_SLP_s1961` | pairwise / LOO / ensemble-mean ρ_o |
| `figF_pairwise_bootstrap_SLP_s1961_B4` | member-subsample bootstrap of the pairwise λ RPC |
| `figG_marginal_moments_SLP_{s1961,lead2-4}` | MARGINAL distributions: pooled histograms, skewness, excess kurtosis |
| `figH_rpc_vs_kurtosis_SLP_s1961` | does high RPC sit where both sides are heavy-tailed? includes the winsorizing mechanism test |
| `figJ_region_significance_SLP_lead2-4` | permutation + bootstrap: is rho_o / lambda_o in each box distinguishable from zero? |
| `figI_ensmean_vs_obs_regions_SLP_{s1961,lead2-4}` | ensemble mean vs obs scatter in named regions with RPC > 1; the plain look-at-it diagnostic |
| `figH2_rpc_vs_kurtosis_ensmean_SLP_s1961` | figH with the ENSEMBLE MEAN as the model side, LOO only; adds the D = obs-minus-model kurtosis conditioner and the numerator/denominator decomposition |
| `pairnull_SLP_*` / `pairdenom_SLP_*` (json only) | the underlying run summaries the figures read |

figD, figE and figF are decadal-only; figA, figB, figC and figG exist at both layouts.

`figA`/`figB` were regenerated when this directory was reorganised, to pick up the
`_SLP_s1961` / `_SLP_lead2-4` tag and the accompanying `.json` that the PRECT set has
and the original untagged `figA_pairwise_vs_loo.png` / `figB_mm_residual.png` did not.
The untagged originals are preserved at
`$SCRATCH/snp_figs_backup_20260921/pairwise-exploratory/`.

## Map conventions

Every map panel in this directory is drawn through `tmp_mapaxes.py`, which fixes two
things that used to be wrong or missing.

**Orientation.** The panels used to hand a `(lat, lon)` array straight to `imshow`
with no `origin=`. The observational grids run **lat[0] = −90 upward** and
matplotlib's default is `origin="upper"`, so every map in figA–figF came out with the
**South Pole at the top**. They are now all `origin="lower"` on a cartopy
`PlateCarree` axes: **Arctic at the top, Antarctic at the bottom.** This was display
only — no number in this README changed because of it — but the figures committed
before this point are vertically mirrored relative to the current ones.

**Coastlines.** Natural Earth coastlines (110 m) are drawn over every map. Each is
stroked twice, a translucent white line under a dark one, because these panels span
`jet`, `magma`, `inferno` and `RdBu_r` and no single line colour is legible on all
four. Without them, claims like "the tropical convective bands" or "the Southern
Ocean" were assertions a reader could not check.

Longitude runs 0–360 with the Pacific in the middle, which is where the old
index-space panels put it, so these figures remain horizontally aligned with their
predecessors. `extent` is built from cell **edges**, not centres — on the 5° HadSLP2r
grid whose centres run 0–355, a centre-based extent would offset the field half a
cell from the coastlines.

**One trap if you add a panel.** A freshly created cartopy `GeoAxes` has its x- and
y-axis set *invisible*, and an invisible axis does not draw its label — so
`ax.set_xlabel(...)` silently does nothing. Several panels here carry their area mean
in the xlabel and their row name in the ylabel, and all of those vanished on the first
pass. `tmp_mapaxes.show` now re-enables both axes with an empty tick list, which
restores the labels and still draws no ticks. If a label you set does not appear, that
is why.

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
python .claude/scripts/tmp_rpc_vs_kurtosis.py       --var SLP --start 1961
python .claude/scripts/tmp_rpc_vs_kurtosis_ensmean.py --var SLP --start 1961
python .claude/scripts/tmp_region_scatter.py         --var SLP --start 1961
```

with `--dataset seasonal --lead 2-4 --bins 3` for the seasonal case. Every script needs
`SNP_REPO` pointing at a checkout that has `data/` and the untracked
`scripts/metrics/jugaad_metrics.py`; a fresh worktree has neither.

`tmp_marginal_moments.py` costs about 1 min for SLP and 3.5 min for PRECT, peaking at
6.1 GB on the PRECT grid — run it alone, the job's cgroup ceiling is 17.2 GB.

**Every one of those scripts now imports `tmp_mapaxes.py` from the same directory**,
which owns the projection, the orientation, the cell-edge extent and the coastlines.
Change a map convention there, once, rather than in seven places. It needs `cartopy`
(0.25.0 in `VirtualEnv/SNP_env`) and the Natural Earth 110 m coastline shapefile,
already cached under `~/.local/share/cartopy` — so no download happens at plot time.
`tmp_mapaxes.grid(var)` reads the observational grid out of the project, which is why
`tmp_pairwise_figs.py` now needs `SNP_REPO` too: it reads only per-cell npz and had no
lats/lons of its own to place a coastline against.

Percentages of area throughout are cos(lat)-weighted, including the colour limits of
figG's maps: an unweighted percentile there is set by the near-polar rows, which carry
almost no area but the most extreme moments.
