# Pairwise exploratory figures — PRECT

The SLP set in `figures/pairwise-exploratory/` rebuilt for precipitation, on the
hypothesis that PRECT carries more non-Gaussian dependence than SLP. It does, and by
enough to change the conclusion.

`png/` holds the figures, `json/` the numbers behind them — including the
`pairnull_*` / `pairdenom_*` run summaries the figures are built from.

---

## What is different about PRECT, before any result

Three things forced a departure from the SLP protocol. None is a choice.

**No s1961.** The observed product is GPCP, which begins in 1979, so
`dcpp_decadal_handles` offers PRECT only from `s1978` (SLP: `s1960`–`s2009`). The
decadal analogue of SLP's `s1961` is therefore **`s1978`**, covering Jan 1979 – Dec 1988.
It is a different decade, so decadal PRECT and decadal SLP are not the same experiment
run on two variables — keep that in mind before reading across the two READMEs.

**A bigger grid.** PRECT's native grid is 72×144 = **10368 cells**, against SLP's
37×72 = 2664. Everything costs 3.9× more, which is why the denominator run below is
hours rather than minutes.

**Fewer seasonal samples, hence B=2.** At lead 2-4 PRECT has **33 start dates**, not
SLP's 51 — again the GPCP start date. The project's occupancy rule
(`B = floor(sqrt(T/5))`) then returns **B=2** rather than SLP's B=3. B=2 is a coarse
binning and it attenuates λ; the seasonal numbers below are conservative because of it.
A B=3 sensitivity was *not* run: at T=33 it gives T/B² = 3.7, below the rule's floor
of 5.

| | SLP s1961 | PRECT s1978 | SLP lead 2-4 | PRECT lead 2-4 |
|---|---|---|---|---|
| members N | 92 | 112 | 62 | 62 |
| samples T | 120 months | 120 months | 51 starts | 33 starts |
| cells C | 2664 | 10368 | 2664 | 10368 |
| bins B | 4 | 4 | 3 | 2 |
| pairs | 4186 | 6216 | 1891 | 1891 |

---

## 1. The headline: the non-Gaussian excess is real on PRECT

`figD_excess_{member,loo}_PRECT_s1978_{ksg,mm}` — λ_o − ρ_o with the estimator's own
bias subtracted, measured on a Gaussian-copula surrogate with the same per-pair rank
correlation.

For a jointly Gaussian pair λ = |ρ| **exactly**, so this difference is zero under
Gaussianity and everything λ can see beyond ρ lives in it. The SLP result was the
blocking problem: the excess was ~10% of ρ_o and **flipped sign between estimators**,
so it could not be claimed. On PRECT all four configurations agree, and they agree
*positive*:

| layout | estimator | λ_o | ρ_o (rank) | raw λ−ρ | estimator bias | **excess** | area +ve |
|---|---|---|---|---|---|---|---|
| pairwise | KSG | 0.1987 | 0.0857 | 0.1131 | 0.0572 | **+0.0559** | 56.4% |
| pairwise | MM B=4 | 0.1323 | 0.0857 | 0.0467 | 0.0125 | **+0.0342** | 43.2% |
| LOO | KSG | 0.1995 | 0.0950 | 0.1045 | 0.0518 | **+0.0527** | 53.2% |
| LOO | MM B=4 | 0.1339 | 0.0950 | 0.0389 | 0.0078 | **+0.0312** | 49.8% |

Against SLP s1961, same quantity, same code:

| | SLP s1961 | PRECT s1978 |
|---|---|---|
| excess, KSG pairwise | +0.0029 | **+0.0559** |
| excess, MM pairwise | **−0.0070** *(sign flip)* | **+0.0342** |
| excess as % of ρ_o | 3.3% / −7.9% | **65% / 40%** |
| share of raw gap that is estimator artefact, **KSG pairwise** | 95% | **51%** |

Two things changed, and both matter. The excess is an order of magnitude larger
relative to ρ_o, and **the estimator disagreement is gone** — KSG and Miller–Madow
differ in magnitude (KSG is the more biased estimator and still reads higher after
subtraction) but no longer in sign. Half of KSG's raw gap survives the surrogate
subtraction on PRECT; on SLP a twentieth of it does.

Two corrections to this table, both made when the figures were regenerated with
coastlines:

- The last row previously read **81%** for SLP against 51% for PRECT. 81% is SLP's
  **LOO** value; its pairwise value is 95%, and every other row in the table is
  pairwise. The row now compares like with like. PRECT is insensitive to the choice
  (pairwise 51%, LOO 50%); SLP is not (95% against 81%), which is itself worth
  knowing.
- The SLP excess figures moved in the fourth decimal (+0.0031 → +0.0029, −0.0069 →
  −0.0070). The SLP numbers stored previously predate the chunked surrogate loop, and
  the surrogate stream is re-seeded per block, so the bias term shifts by ~8e-5. The
  PRECT column reproduced **bit for bit** — it was chunked both times. Signs and
  conclusions are untouched.

**It is also spatially coherent.** The excess is not diffuse: it concentrates in the
tropical convective bands — ITCZ, SPCZ, the monsoon regions — which is where
precipitation is most obviously non-Gaussian, while the estimator-bias panel is a
spatially flat field. That the signal has the geography one would predict a priori,
and the artefact does not, is a stronger argument than the area mean alone.

The caution that carries over from SLP: the pipeline copula-transforms, so this is
non-Gaussianity of the *dependence structure*, not of the marginals. PRECT's marginals
are wildly non-Gaussian and none of that is what is being measured here.

---

## 2. RPC: λ and ρ disagree, and they disagree in opposite directions at the two leads

`figC_pairwise_rpc_PRECT_{lead2-4,s1978}` — three ρ summaries × {pairwise, LOO}, ratio
of area means (the robust summary; the per-cell ratio is shown only to display how
badly it behaves).

**Colour limits.** The four metric panels use `jet` from 0 to **0.7** (`--vmax`),
lowered from the 1.0 these were first drawn with. Both variables were regenerated
together, so these panels and the SLP ones remain directly comparable. The ratio
panels are unchanged at `RdBu_r` 0–2. No number in the tables below is affected — the
change is display only.

**Lead 2-4 (seasonal).**

| form | ρ variant | RPC_ρ | RPC_λ |
|---|---|---|---|
| pairwise | signed mean | 1.103 | **1.003** |
| pairwise | mean \|ρ\| | 1.048 | **1.003** |
| pairwise | rms ρ | 1.025 | **1.003** |
| LOO | signed mean | 1.100 | 0.878 |
| LOO | mean \|ρ\| | 1.084 | 0.878 |
| LOO | rms ρ | 0.979 | 0.878 |

In the pairwise form λ puts RPC at **1.003** — as close to unity as this machinery can
resolve — while every ρ summary sits above 1, the signed one by 10%. This is the
cleanest instance in the project so far of the information metric and the variance
metric giving materially different answers to the question the project is actually
about. On SLP lead 2-4 the same panel gave RPC_λ = 0.982 against RPC_ρ = 1.033 / 1.011
/ 0.993, i.e. λ was closer to 1 than the *signed* ρ but not than its sign-blind
partners; here it is closer than all three.

Read it with the B=2 caveat from above: coarse bins attenuate λ, which pushes λ_o and
λ_m down together, and the ratio is more robust to that than either term.

**s1978 (decadal).** Here the disagreement runs the other way, and ρ is the metric that
breaks down:

| form | ρ variant | RPC_ρ | RPC_λ |
|---|---|---|---|
| pairwise | signed mean | **0.353** | 0.792 |
| pairwise | mean \|ρ\| | 0.965 | 0.792 |
| pairwise | rms ρ | 0.960 | 0.792 |
| LOO | signed mean | **0.348** | 0.868 |
| LOO | mean \|ρ\| | 0.776 | 0.868 |
| LOO | rms ρ | 0.637 | 0.868 |

The signed-mean ρ collapses: |signed mean| / across-member sd is **0.116** in the
pairwise form, meaning the members disagree in sign and the average cancels to almost
nothing (ρ_o = 0.0043 against mean|ρ| = 0.0893, a factor of 21). This is the same
pathology SLP s1961 showed at 0.155, worse. λ is sign-blind and cannot cancel, so its
fair ρ partner is mean|ρ|, not the signed mean — and whether a member anticorrelated
with observations should count toward skill is a modelling judgement, not a statistical
one.

The masking tells the same story. Per-cell ratios are usable on only **16% of area**
for pairwise signed ρ, against **86%** for λ. That is a real robustness advantage for
λ, but it comes from sign-blindness, not from information.

**The bootstrap.** `figF_pairwise_bootstrap_PRECT_*`, 1000 draws of M of N members,
members only, time axis never resampled:

| | lead 2-4 (M=41 of 62) | s1978 (M=75 of 112) |
|---|---|---|
| RPC, ratio of area means | 1.019 | 0.797 |
| area where 95% CI holds 1 | 52.7% | 37.5% |
| area ≠ 1 after BH-FDR q=0.10 | 46.0% | 63.4% |
| usable area (λ_m > 0.02) | 62.9% | 74.5% |

Note the decadal RPC is **below** 1, so those significant cells are not the RPC > 1
cells the project is trying to explain away — the same caveat that applied to SLP
s1961's bootstrap (RPC 0.849 there).

---

## 3. The pairwise formulation reproduces on PRECT

`figA_pairwise_vs_loo_*`, `figB_mm_residual_*`. The two findings the SLP work rested on
both hold, at both leads.

**The null floor collapses.** Replacing `mean_n I(s_-n; o)` with `mean_n I(f_n; o)`
shrinks the permutation null's spread, because the leave-one-out means differ by one
member in N and so their average keeps nearly the full single-term variance, while the
members are near-independent and theirs averages down:

| | null sd ratio, LOO / pairwise |
|---|---|
| SLP s1961 | 7.5× |
| **PRECT s1978** | **6.5×** |
| SLP lead 2-4 | 5.0× |
| **PRECT lead 2-4** | **5.6×** |

and that converts directly into resolved area. PRECT s1978 numerator: pairwise
debiased I = **+0.0141 nats at SNR 5.64, 41.8% of area after FDR**, against LOO's
+0.0142 at SNR 0.88 and 9.3%. Nearly the same signal, a sixth of the floor.

For scale, SLP s1961 pairwise was +0.0032 at SNR 1.49 — the decadal PRECT case is a
much stronger one than the decadal SLP case, independent of anything about
non-Gaussianity.

**Miller–Madow alone is still not sufficient, and at s1978 it is far from it.** MM's
residual is a fixed offset that averaging does not remove, while the noise it must be
small against does — so the ratio grows with the number of terms averaged. Residual /
null sd:

| | numerator | denominator |
|---|---|---|
| PRECT lead 2-4 | 0.21 (62 members) | 1.42 (1891 pairs) |
| **PRECT s1978** | **0.65** (112 members) | **6.22** (6216 pairs) |
| SLP s1961, for comparison | 0.67 | 5.11 |

Anything above ~0.10 means the empirical null mean must be subtracted explicitly. At
s1978 the denominator's MM residual is six times its own null spread — using MM alone
there would not be a small error, it would be the dominant term. Every number in this
directory subtracts the permutation null mean explicitly.

---

## 4. ρ_o is one field measured three ways

`figE_rho_o_three_ways_PRECT_s1978`. As on SLP, the leave-one-out and plain
ensemble-mean definitions are the same field to four decimals — pattern r = **0.99999**,
slope 1.003 — and the pairwise definition is that same pattern scaled down by
σ_s/σ_f: r = 0.944, slope 6.67. Area means 0.0043 (pairwise), 0.0231 (LOO), 0.0231
(ensemble mean). Nothing here is new; it is the control that says the three
denominators are not measuring different things.

---

## 5. Marginal distributions — the part §1 explicitly excludes

`figG_marginal_moments_PRECT_{s1978,lead2-4}`. **New.** §1 ends with the caution that
the pipeline copula-transforms, so the excess it measures is non-Gaussianity of the
*dependence structure*, and that "PRECT's marginals are wildly non-Gaussian and none
of that is what is being measured here". This figure measures the excluded part, so
the claim can be checked instead of asserted — pooled histograms of model and observed
values over all cells, plus skewness and excess kurtosis.

Three rows: `raw`, `anomaly` (month-of-year climatology removed) and `PIPELINE`
(+ the Eade 11.25°×12.5° box mean). **Only the bottom row is what figA–figF operate
on.** Two summaries of each moment, because they answer different questions:
computed within each cell and then cos(lat)-area-averaged, or pooled over every value
at once.

PRECT s1978, per-cell then area-averaged, at each stage:

| stage | skew model | skew obs | exkurt model | exkurt obs |
|---|---|---|---|---|
| raw | +1.611 | +1.090 | +6.987 | +2.463 |
| anomaly | +1.460 | +0.774 | +10.943 | +3.706 |
| **PIPELINE** | **+0.711** | **+0.473** | **+3.832** | **+2.218** |

and at lead 2-4 the pipeline row is +0.511 / +0.449 skew, +1.829 / +1.079 excess
kurtosis — closer to Gaussian, as a three-month mean of 33 start dates should be.

**The marginals really are wildly non-Gaussian, and §1's caution was warranted.**
Against SLP s1961's pipeline row (skew −0.002 model / −0.074 obs, excess kurtosis
+0.737 / +0.797), PRECT is a different regime: skewness is two orders of magnitude
larger and excess kurtosis 5× larger on the model side. Whatever else separates the
two variables, their marginals are not remotely alike — and figD's copula transform
is what stops that difference from contaminating the excess.

**The model overstates its own non-Gaussianity.** Model skewness exceeds observed by
50% (+0.711 vs +0.473) and model excess kurtosis by 73% (+3.832 vs +2.218), in the
same direction at both leads and at every preprocessing stage. That is a marginal-side
model bias which none of figA–figF can see, and it is the kind of thing a referee
asking "is the model's precipitation distribution right?" will want addressed.

**Smoothing does much of the Gaussianising.** The Eade box mean averages ~20 cells, so
the central limit theorem applies: per-cell excess kurtosis falls from +10.9 to +3.8
on the model side between the `anomaly` and `PIPELINE` rows. Any claim that "PRECT is
non-Gaussian" must name the stage — unsmoothed it is far more so than the pipeline
sees.

**Do not quote the pooled-in-native-units numbers.** Pooling across cells of unequal
variance is leptokurtic even under exact Gaussianity, which is why the pipeline row's
native-units excess kurtosis reads +12.55 (model) / +16.08 (obs) against +3.19 / +1.91
after per-cell standardisation. Column 2 of the figure is the fair comparison; column
1 is shown so the artefact is visible rather than hidden.

Model/observed pattern correlation of the per-cell maps is +0.545 (skewness) and
+0.348 (excess kurtosis) at s1978: the model gets the geography partly right. The
maps show the expected structure — the ITCZ, SPCZ and monsoon bands are the most
skewed and heaviest-tailed regions, the same geography figD's excess picks out.

Sample sizes are very unequal and the figure annotates both: N·T = 13440 per cell for
the model against T = 120 for the observations, so se(skew) is 0.021 against 0.224 and
se(excess kurtosis) 0.042 against 0.447. The observed *maps* are noisy; their area
means are not. The model's standard error is a floor — members share a forced signal.

## 6. What this does and does not settle

It settles the estimator-sign objection **for PRECT**: the non-Gaussian excess is
positive under both KSG and Miller–Madow, in both layouts, is 40–65% of ρ_o rather than
~10%, and is concentrated where the physics says it should be. The prior handoff
treated "no significant difference" as a live outcome; on this variable it is not.

It does not settle these:

- **Decadal PRECT and decadal SLP are different decades** (s1978 vs s1961). The
  variable is the intended contrast but it is not the only difference between the two
  columns. Running SLP at s1978 would isolate it, and is cheap — SLP's grid is 3.9×
  smaller.
- **The lead 2-4 comparison runs at B=2 against SLP's B=3.** Coarser bins attenuate λ,
  so the seasonal PRECT λ numbers are conservative, but "conservative" is not "equal"
  and the two are not strictly like-for-like.
- **ρ is still Pearson-on-raw against λ-on-ranks** in the RPC panels. §2 of the prior
  handoff argued for putting ρ on normal scores so the comparison is like-for-like;
  that has not been done here, and part of the §2 gap above is that transform rather
  than information. The `rho_o_rank` column of the figD table is the like-for-like
  version and is what the excess is computed against.
- **Significance of the excess itself has not been tested.** The surrogate subtraction
  removes the estimator's bias but the maps carry no permutation p-value. The area mean
  is large relative to SLP's, not relative to a null.

---

## Files

`png/` and `json/` carry the same stems.

| stem | what |
|---|---|
| `figA_pairwise_vs_loo_PRECT_{s1978,lead2-4}` | pairwise vs LOO: debiased signal, null sd, z, numerator and denominator. Columns paired, sharing a colour scale |
| `figB_mm_residual_PRECT_{s1978,lead2-4}` | why MM alone fails; the residual is a spatially flat field |
| `figC_pairwise_rpc_PRECT_{s1978,lead2-4}` | 3 ρ variants × {pairwise, LOO} × {numerator, denominator, RPC} |
| `figD_excess_{member,loo}_PRECT_s1978_{ksg,mm}` | non-Gaussian excess maps, both estimators, both layouts |
| `figE_rho_o_three_ways_PRECT_s1978` | pairwise / LOO / ensemble-mean ρ_o |
| `figF_pairwise_bootstrap_PRECT_s1978_B4`, `..._lead2-4_B2` | member-subsample bootstrap of the pairwise λ RPC |
| `figG_marginal_moments_PRECT_{s1978,lead2-4}` | MARGINAL distributions: pooled histograms, skewness, excess kurtosis, model vs obs |
| `pairnull_PRECT_*` / `pairdenom_PRECT_*` (json only) | the underlying run summaries the figures read |

figD and figE are decadal-only, as in the SLP set — both scripts take the sample axis
to be months within one hindcast.

## Reproducing

Scripts are the SLP ones under `.claude/scripts/` (gitignored), driven by `--var PRECT`.
Three changes were needed and are described where they live:

- `tmp_pairwise_figs.py` (figA/figB) was hard-coded to SLP s1961; it now takes
  `--var/--start/--dataset/--lead/--bins` and writes a json beside each figure.
- `tmp_excess_lam_map.py` (figD) now runs in blocks of `--chunk` members. The flat
  `(N·C, T)` arrays the SLP version built are 1.1 GB each at the PRECT grid and several
  are live at once, which exceeds a 16 GB job. Blocking changes the peak, not the
  result: verified against the stored SLP s1961 MM/member numbers, which it reproduces
  exactly on the data side (λ_o 0.0922, ρ_rank 0.0883, raw +0.0039). Only the
  surrogate-averaged bias term moves, by 0.0001, because the surrogate stream is
  re-seeded per block.
- `tmp_mi_fast.py` is a new drop-in for `jugaad_metrics.mi_plugin(..., mm=True)` that
  replaces the three `np.log` calls with one `n·log n` table — the counts are integers,
  so this is algebra, not approximation. 1.4× on the denominator's inner loop. The
  denominator run asserts it against the real function on its own data before starting
  (`max |diff| = 1.24e-15`).

Order matters: the numerator and denominator runs write the `pairnull_*`/`pairdenom_*`
npz into `$SCRATCH/snp_pairwise_null/`, and figA, figB, figC and figF all read them.

```
python .claude/scripts/tmp_pairwise_null_s1961.py  --var PRECT --start 1978 --perm 500
python .claude/scripts/tmp_pairwise_denom_s1961.py --var PRECT --start 1978 --perm 200 --chunk-pairs 40
python .claude/scripts/tmp_pairwise_figs.py        --var PRECT --start 1978 --bins 4
python .claude/scripts/tmp_pairwise_rpc.py         --var PRECT --start 1978 --vmax 0.7
python .claude/scripts/tmp_pairwise_bootstrap.py   --var PRECT --start 1978 --draws 1000
python .claude/scripts/tmp_excess_lam_map.py       --var PRECT --start 1978 --estimator ksg --layout member --chunk 8
python .claude/scripts/tmp_rho_o_three_ways.py     --var PRECT --start 1978
python .claude/scripts/tmp_marginal_moments.py     --var PRECT --start 1978
```

with `--dataset seasonal --lead 2-4 --bins 2` for the seasonal case.

**Run these one at a time.** The job's cgroup ceiling is 17.2 GB and the decadal
denominator alone holds ~7.3 GB; it is single-threaded, so nothing is gained by
overlapping it with other work and a concurrent figD run will push the job over the
limit. The decadal denominator takes ~5 h at 6216 pairs × P=200 on this grid.
`tmp_marginal_moments.py` is the cheap one — ~3.5 min and 6.1 GB peak on the PRECT
grid, and it reads no `pairnull_*`/`pairdenom_*` npz, going straight to the handles.

All of these scripts import **`tmp_mapaxes.py`** from the same directory, which owns
the projection, the orientation, the cell-edge extent and the coastlines — change a
map convention there once instead of in seven scripts. It needs `cartopy` (0.25.0 in
`VirtualEnv/SNP_env`) and the Natural Earth 110 m shapefile, already cached under
`~/.local/share/cartopy`, so nothing is downloaded at plot time.

**Map conventions.** Every map panel here goes through `tmp_mapaxes.py`: cartopy
`PlateCarree`, `origin="lower"` so the **Arctic is at the top and the Antarctic at the
bottom**, and Natural Earth 110 m coastlines stroked white-under-dark so they stay
legible across `jet`, `magma`, `inferno` and `RdBu_r` alike. `extent` comes from cell
edges, not centres, so the field registers against the coastline rather than sitting
half a cell off.

Until this pass, figA–figF passed no `origin=` at all and — because the observational
grids run lat[0] = −90 upward against matplotlib's `origin="upper"` default —
rendered with the **South Pole at the top**. That was display only; no number in this
README changed. But figures committed before this point are vertically mirrored
relative to the current ones.

Percentages of area throughout are cos(lat)-weighted.
