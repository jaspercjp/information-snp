# Is RPC = 1 more plausible under lambda than under rho?

The claim: measured by `lambda`, the RPC is not statistically distinguishable from 1;
measured by Pearson `rho`, that reading is far less tenable. Tested on the existing
member+window bootstrap bundles in `data/bundle/` -- 1000 draws each, with
`boot_lam_ratio` and `boot_rho_ratio` computed from the SAME resampled members and
window per draw, so every comparison is paired. Built by
`.claude/scripts/tmp_rpc_lambda_vs_rho.py`; no new estimation.

## Answer: supported per cell, in all 11 bundles

Fraction of the common mask whose 95% bootstrap CI contains 1, and the paired distance
from 1 in each metric's own bootstrap sd:

| bundle | mask | CI holds 1, lam | CI holds 1, rho | med \|z\| lam | med \|z\| rho | 1 closer under lam |
|---|---|---|---|---|---|---|
| dcpp SLP hindcast 13-60 | 99% | **75.6%** | 24.8% | **1.28** | 4.79 | **81%** |
| trefht hindcast 13-60 | 100% | **38.8%** | 11.7% | **3.14** | 10.62 | **81%** |
| prect hindcast 13-60 | 99% | **69.9%** | 21.9% | **1.55** | 4.55 | **80%** |
| trefht hindcast 2-4 | 100% | **72.4%** | 41.6% | **1.11** | 2.45 | **75%** |
| prect hindcast 2-4 | 97% | **84.2%** | 45.2% | **0.99** | 1.76 | **65%** |
| dcpp SLP decadal s1961 | 35% | **99.8%** | 74.4% | 0.66 | 0.14 | 24% |
| slp decadal s1990 | 31% | **100.0%** | 85.9% | 0.53 | 0.05 | 10% |
| trefht decadal s1961 | 77-80% | **97.1-97.4%** | 77.6-80.3% | 0.60 | 0.53 | 45% |
| trefht decadal s1990 | 62% | **98.8%** | 84.5% | 0.57 | 0.20 | 34% |
| prect decadal s1990 | 29% | **98.4%** | 81.3% | 0.61 | 0.08 | 18% |

lambda wins on "CI holds 1" in every bundle. The two blocks differ in WHY:

* **hindcast bundles** (sample axis = start dates, `rho_m > 0` nearly everywhere, so the
  rho ratio is well conditioned -- CI widths 0.1-1.4). Here rho decisively rejects 1:
  median `|z|` of 1.8-10.6 and 1 inside the CI on only 12-45% of area, against lambda's
  39-84%. Paired per cell, 1 is closer under lambda in 65-81% of area. **This is the
  claim, and this is where it is strongest.**
* **single-hindcast decadal bundles** (sample axis = months). Here `rho_m <= 0` over
  30-70% of area, so the rho ratio is undefined or explosive: CI widths 8-25 and domain
  values from -0.75 to +3.3. 1 sits inside those intervals often enough, but only
  because they are uninformative. lambda's CI width is a steady 1.8-2.6 and contains 1
  on 97-100% of area. The contrast here is CONDITIONING, not centrality.

## The structural advantage behind it

`rho_m` is a correlation and straddles zero, so `rho_o/rho_m` blows up as `rho_m -> 0`:
undefined on 30% of s1961 SLP cells, 65-71% at s1990. `lam_m` is information-derived and
cannot be negative, so the lambda ratio is defined on 100% of area in every bundle. The
headline numbers above therefore use the common mask where both exist, and the mask
column shows how much that costs.

## The caveat, stated plainly

On the DOMAIN MEAN -- one area-weighted number per draw, the best-conditioned form -- the
lambda ratio's CI is tight (typically +/-0.05) and excludes 1 in 4 of 11 bundles:

| bundle | lambda domain RPC | rho domain RPC |
|---|---|---|
| dcpp SLP hindcast 13-60 | +1.47 [1.35, 1.61] excludes 1 | -0.75 [-1.23, -0.43] excludes 1 |
| prect hindcast 13-60 | +1.54 [1.49, 1.59] excludes 1 | +0.91 [0.69, 1.27] holds 1 |
| trefht hindcast 13-60 | +1.24 [1.20, 1.28] excludes 1 | +0.83 [0.81, 0.85] excludes 1 |
| prect hindcast 2-4 | +1.13 [1.03, 1.24] excludes 1 | +1.11 [0.72, 2.05] holds 1 |
| trefht hindcast 2-4 | +1.04 [0.97, 1.12] holds 1 | +1.04 [0.95, 1.16] holds 1 |
| dcpp SLP decadal s1961 | +0.93 [0.81, 1.06] holds 1 | -0.12 [-13.6, +6.6] holds 1 |
| the four other decadal | 0.92-1.07, all hold 1 | 0.02-3.33, all hold 1, widths 3-27 |

So "not statistically distinguishable from 1" is a PER-CELL statement. Aggregated, the
lambda RPC is near 1 in value (0.92-1.54, median ~1.04) but precise enough that a 20-50%
departure is resolvable, and in the 13-60 windows it sits significantly above 1. Quoting
the per-cell result without this would overstate the case.

## Reading the panels

`maps/<tag>.png`, 3x3: column 1 lambda, column 2 rho, column 3 the distributions.
Row 1 RPC, row 2 `z = (RPC-1)/sd_boot`, row 3 the "CI contains 1" mask. The bottom-right
panel is the paired per-cell difference `|z_rho| - |z_lambda|`, positive where 1 is the
more plausible reading under lambda.
