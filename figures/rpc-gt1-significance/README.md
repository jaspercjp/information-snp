# Where the lambda RPC exceeds 1, is the excess significant?

The cautious objection to "RPC = 1 is the plausible reading under lambda" is that some
cells DO come out above 1 on the point estimate. This tests exactly those cells, lead 2-4
(first DJF) only, against the member bootstrap already implemented in
`scripts/bootstrap/boot.py` / `metrics_bundle.py`:

    H0: RPC = 1     H1: RPC > 1     p = fraction of bootstrap draws with ratio <= 1

Cells whose point estimate is at or below 1 are not tested, since "is RPC > 1" is not a
question there. Benjamini-Hochberg at q = 0.10 runs over the tested cells only -- with
~1500 tests, ~75 clear at alpha = 0.05 by chance, so raw counts alone mean little.

Built by `.claude/scripts/tmp_rpc_gt1_significance.py`. 1000 draws, 42 of 62 members,
`n_months` equal to the full record, i.e. MEMBERS ONLY, no time resampling.

## Result: usually not significant for lambda, usually significant for rho

| variable | metric | RPC>1 on | of those p<0.05 | after FDR | median p | median 5th pct |
|---|---|---|---|---|---|---|
| SLP 2-4 | **lambda** | 55.3% | 30.0% | **24.6%** | **0.23** | **0.51** |
| SLP 2-4 | rho | 58.8% | 61.9% | 64.7% | 0.02 | 1.05 |
| TREFHT 2-4 | **lambda** | 56.8% | 40.9% | **38.6%** | **0.11** | **0.93** |
| TREFHT 2-4 | rho | 54.7% | 71.9% | 75.2% | 0.00 | 1.07 |
| PRECT 2-4 | **lambda** | 51.5% | 25.8% | **21.8%** | **0.22** | **0.34** |
| PRECT 2-4 | rho | 57.1% | 61.2% | 63.2% | 0.01 | 1.10 |

Both metrics put a little over half the area above 1, so the POINT ESTIMATES do not
separate them. The bootstrap does:

* the TYPICAL cell above 1 is not significant under lambda -- median one-sided p of
  0.11-0.23, and a median bootstrap 5th percentile of 0.34-0.93, i.e. BELOW 1;
* the typical cell above 1 IS significant under rho -- median p of 0.00-0.02 and a
  median 5th percentile of 1.05-1.10, i.e. ABOVE 1;
* by area, rho's excess survives on 63-75% of its own >1 area against lambda's 22-39%,
  a factor of 2-3 in every variable.

## The qualification, which is not small

22-39% of lambda's >1 area survives FDR correction, against 5% expected by chance. That
is far too much to call noise. The defensible claim is therefore

> where the lambda RPC exceeds 1, the excess is usually not resolvable given
> finite-ensemble sampling error, and it is resolvable over 2-3x less area than for rho

and NOT "the lambda RPC is never significantly above 1". The surviving cells are
spatially coherent (see `maps/`), concentrated in the tropics for SLP and PRECT, which
argues they are real structure rather than scattered multiple-comparison residue.

## Which error bar this is

Members only: `n_members = 42` of 62, `n_months` = the whole record, so
`time_resampling: false` in the provenance. The interval covers the finite-ENSEMBLE
error -- how much RPC moves had a different 42 members been run -- and not the
finite-RECORD error from having ~50 start dates.

That makes the test OPTIMISTIC, in the useful direction: adding time resampling widens
the intervals and can only shrink the significant fraction. So 22-39% is an upper bound
on how much of lambda's >1 area is real, and the "usually not significant" conclusion is
the conservative one.

`maps/<tag>.png`: row 1 lambda, row 2 rho; columns are the RPC point estimate, the
one-sided p on the tested cells, and the ns / p<.05 / FDR verdict.
