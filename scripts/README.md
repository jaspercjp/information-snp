# scripts

Everything in `figures/jugaad-fd-optbins/` is produced by the scripts here.

| file | role | lines |
|---|---|---|
| `jugaad.py` | the measures: FD and Knuth bin rules, plug-in MI, the jugaad λ map (parallel over rows), leave-one-out means, Pearson ρ | 190 |
| `data.py` | paths (`$SNP_DATA`, `$SNP_OUT`), the case lists, `load(var, case)` | 25 |
| `run.py` | point estimates for one case, every binning configuration -> `point_<VAR>_<case>.npz` | 46 |
| `boot.py` | the Eade et al. (2014) bootstrap, modes `fd`, `knuth`, `lmloo`, `olmloo` -> `boot_<mode>_<VAR>_<case>.npz` | 71 |
| `tables.py` | `jugaad_fd_optbins_table` and `jugaad_lam_terms` (md + json), and the area-summary helpers | 166 |
| `plots.py` | every figure and its json (plotting, not counted) | |
| `run_all.sh` | submits the whole pipeline to Slurm: point (array) + boot (array) -> figures | |
| `data-acquisition/` | raw downloads to the per-case input arrays (`$SNP_DATA/cases/`); see its README | |
| `thesis-old/` | the original thesis code: `InfoTheoryMetrics.py`, `DataInventory.py`, and `info_rpc_parallel.py`, a parallel `info_RPC` with bitwise-identical output | |
| `tests/test_jugaad.py` | `jugaad.maps` and `info_rpc_parallel.info_RPC` against the thesis `info_RPC`, bitwise | |

The five analysis modules come to 498 lines (347 without docstrings, comments and blank lines).

## Reproduce

```bash
PY=/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/SNP_env/bin/python
# 1. inputs (once): fetch, build cubes, write $SNP_DATA/cases/<VAR>_<case>.npz
#    -- see data-acquisition/README.md
# 2. everything else, as Slurm jobs (point estimates, 26 bootstraps, then tables + figures)
bash scripts/run_all.sh
# or by hand, inside an allocation:
$PY scripts/run.py SLP 2-4
$PY scripts/boot.py SLP 2-4 fd          # also knuth (rho_vs_lambda), lmloo, olmloo (production rows)
$PY scripts/tables.py && $PY scripts/plots.py
$PY scripts/tests/test_jugaad.py
```

`$SNP_DATA` defaults to `<repo>/data`, `$SNP_OUT` to `$SCRATCH/snp_jugaad`. `jugaad.maps` uses
`$SLURM_CPUS_ON_NODE` worker processes. The bootstrap is seeded per case, so reruns on the
same CPU generation are bitwise identical; across Sherlock's node generations (`-C CPU_GEN:MLN`
vs `SIE`/`GEN`) float64 results differ in the last bits (about 1e-15 in I, 1e-8 at most in a
Knuth λ), which no figure shows. The longest single job (FD λ, TREFHT s1961) takes about 4 h
on 16 cores.

## Cases

`data.CASES`: SLP CMIP5 decadal1961 (the thesis ensemble), SLP and TREFHT DCPP-A s1961 (raw and
month-of-year anomalies), SLP, PRECT and TREFHT DCPP-A lead 2-4. The production figures use
lead 2-4 and the decadal rows SLP s1961, PRECT s1978 and TREFHT s1961 (`data.PRODUCTION`).
