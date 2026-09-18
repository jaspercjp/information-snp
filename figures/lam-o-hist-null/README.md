# Null floors for the non-pooled lambda metrics, equiprobable bins

Does the information-theoretic skill metric `lam_o` clear its own permutation null? Three
cases, one estimator (equiprobable-bin plug-in MI), one question. Built by
`.claude/scripts/tmp_lam_o_hist_permtest.py`; raw npz, logs and the script live in
`$SCRATCH/snp_lamo_hist_final` and `$SCRATCH/snp_lamo_seasonal`.

## The cases, and what separates them

| folder | data | sample axis | N | samples |
|---|---|---|---|---|
| `decadal-s1961/` | one DCPP-A hindcast, s1961 SLP monthly | 120 calendar months | 92 | 120 |
| `seasonal-lead2-4/` | DCPP-A grand ensemble, lead 2-4 (first DJF) | start dates | 62 | 51 |
| `decadal-lead13-60/` | DCPP-A grand ensemble, lead 13-60 (yr 2-5) | start dates | 101 | 48 |

The first case has the most samples and the least skill; the second has the fewest
samples and the most skill. That trade-off is the whole story.

## Headline: only the seasonal case clears the floor

`lam_o = lam[I(s;o)]`, area-weighted, i.i.d. null. "debiased I" is `I_data - I_null`
in nats, +/- the null's own spread, which IS the estimator uncertainty on it.

| case | B | \|rho\| | debiased I | SNR | lam(debiased) | area > 2sd | area p<=0.05 |
|---|---|---|---|---|---|---|---|
| decadal s1961 | 4 | 0.112 | +0.0039 +/- 0.0184 | 0.2 | 0.090 | 7.8% | 8.9% |
| **seasonal lead 2-4** | **3** | **0.402** | **+0.0751 +/- 0.0294** | **2.6** | **0.291** | **44.1%** | **44.1%** |
| seasonal lead 2-4 | 4 | 0.402 | +0.0863 +/- 0.0478 | 1.8 | 0.315 | 38.6% | 39.9% |
| decadal lead 13-60 | 3 | 0.224 | +0.0267 +/- 0.0315 | 0.8 | 0.176 | 20.8% | 20.3% |

Against 5% of area expected at p<=0.05 by chance, the seasonal case returns 44% -- nine
times chance. The one-hindcast case returns 8.9%, i.e. chance. Lead 13-60 sits between
and is not resolved on average, though its best cells are.

## Why, in one formula

The plug-in MI of independent series satisfies `2*T*I_null ~ chi2_k` with `k=(B-1)^2`, so

    bias = k/(2T)        sd = sqrt(2k)/(2T)        detection needs I_signal > sqrt(2k)/T

Fewer samples RAISES the bar (rho > 0.33 at T=51 vs rho > 0.22 at T=120 for B=3), but
seasonal skill rises much further than the bar does. Measured null means match the
analytic bias to three decimals in every case, which is the calibration check.

## Reading the panels

- `floors/` -- DATA | FLOOR (null mean) | DATA-floor | z. The floor is the null MEAN,
  i.e. the estimator bias, so the excess is a bias-corrected level and NOT a
  significance statement: under the null, DATA exceeds the null mean half the time.
  The z panel and the p<=0.05 area carry the significance. `floor95` is in the JSONs.
- `debiased/` -- `I_data - I_null` mapped, its lambda, and its histogram against zero.
  This is the "true I with estimator uncertainty" view.
- `ratio/` -- the RPC analogue `lam_o/lam_m` against its own null. Worth noting it
  clears nothing anywhere: even in the seasonal case the ratio is 1.018 with z = 0.06,
  because lam_o and lam_m rise together.
- `draws/` -- data beside null draws on one colour scale. If they look alike, it is noise.
- `two-nulls/` -- i.i.d. months vs 12-month blocks (only meaningful for the monthly case).

Bin counts are sized by the real sample count: `T/B^2 >~ 5` gives B=4 at T=120 and B=3
at T=51. B=8 at T=120 and B=4 at T=51 are the lean-on-the-correction variants, kept for
the sensitivity.
