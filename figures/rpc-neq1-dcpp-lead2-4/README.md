# Is the RPC significantly DIFFERENT from 1? DCPP, lead 2-4, all three variables

Two-sided test of `H0: RPC = 1` on every usable cell, against the member bootstrap in
`scripts/bootstrap/boot.py` / `metrics_bundle.py`:

    p = 2 * min( frac(draws <= 1), frac(draws >= 1) )

so a cell is resolved whether its RPC sits above OR below 1. The earlier one-sided
version could only reject upward and would have scored a cell at RPC = 0.3 as perfectly
consistent with 1, which is why this is the version to quote. BH-FDR at q = 0.10 over
all ~2600 usable cells, where ~130 clear at alpha = 0.05 by chance.

Built by `.claude/scripts/tmp_rpc_neq1_significance.py`. 1000 draws, 42 of 62 members,
`n_months` = the full record, i.e. MEMBERS ONLY, no time resampling. Plots use bicubic
interpolation.

## Result: lambda holds 1 over 72-85% of area, rho over 42-45%

| variable | metric | CI holds 1 | p<.05 | after FDR | of FDR: >1 | of FDR: <1 | median p | median \|z\| | median RPC |
|---|---|---|---|---|---|---|---|---|---|
| SLP | **lambda** | **82.4%** | 17.6% | **12.7%** | 10.7% | 2.1% | **0.44** | 1.27 | 1.07 |
| SLP | rho | 43.6% | 56.4% | 57.5% | 32.6% | 25.0% | 0.03 | 1.92 | 1.07 |
| TREFHT | **lambda** | **72.4%** | 27.6% | **22.8%** | 17.4% | 5.4% | **0.33** | 1.40 | 1.05 |
| TREFHT | rho | 41.6% | 58.4% | 59.4% | 36.6% | 22.8% | 0.01 | 2.46 | 1.03 |
| PRECT | **lambda** | **84.6%** | 15.4% | **10.2%** | 8.6% | 1.7% | **0.43** | 1.27 | 1.04 |
| PRECT | rho | 45.2% | 54.9% | 55.6% | 32.1% | 23.5% | 0.03 | 1.76 | 1.12 |

The point estimates do not distinguish the two metrics -- median RPC is 1.03-1.12 for
both, in every variable. The uncertainty does, and decisively:

* **lambda is consistent with 1 over the large majority of area** (72-85%), with a median
  two-sided p of 0.33-0.44, i.e. the typical cell is nowhere near resolvable.
* **rho resolves as different from 1 over the majority of area** (56-59% after FDR), with
  a median p of 0.01-0.03.
* The resolved area differs by a factor of **3-5** in every variable: 10-23% for lambda
  against 56-59% for rho.

A second point the two-sided form exposes, which the one-sided version hid: rho's
departures are split roughly evenly either side of 1 (32.6% above / 25.0% below for SLP,
36.6/22.8 TREFHT, 32.1/23.5 PRECT). A metric whose significant departures scatter
symmetrically on both sides looks like a noise signature rather than a coherent
signal-to-noise paradox. lambda's departures are lopsided toward >1 (10.7/2.1, 17.4/5.4,
8.6/1.7) but cover far too little area to carry a claim.

## Which error bar, and which way it biases

Members only: `n_members = 42` of 62, `n_months` = the whole record, so
`time_resampling: false`. The interval covers the finite-ENSEMBLE error -- how much RPC
moves had a different 42 members been run -- and not the finite-RECORD error from having
~50 start dates.

Adding time resampling widens intervals, which can only REDUCE the resolved area. So the
10-23% resolved for lambda is an UPPER bound and the 72-85% consistent with 1 is a LOWER
bound. Both biases run in favour of the conclusion, which is the right direction for it.

## Provenance of "dcpp"

All three are DCPP hindcasts at lead 2-4, but they arrive through two handles:

    SLP     dcpp_handles, lead="2-4"                        N=62, J=51
    TREFHT  common_grid_handles("dcpp", "TREFHT", "2-4")    N=62, J=52
    PRECT   common_grid_handles("dcpp", "PRECT",  "2-4")    N=62, J=33  (GPCP from 1979)

The common-grid route puts TREFHT and PRECT on the same 37x72 grid as SLP; going through
`dcpp_handles` instead would keep each variable on its own obs grid (NCEP T62 = 18048
cells, GPCP 2.5deg = 10368) and cost 4-7x the bootstrap time for the same question.

`maps/<tag>.png`: row 1 lambda, row 2 rho; columns are the RPC point estimate, the
two-sided p, and the holds-1 / p<.05 / FDR verdict. Grey is where rho_m <= 0 and the rho
ratio cannot be formed; lambda is defined everywhere.
