# scripts

Every figure in `figures/{main-text,supp-info}/` except the hand-saved `thesis-*` ones is produced by the scripts here (see `figures/README.md`).

| file | role | lines |
|---|---|---|
| `jugaad.py` | the measures: FD and Knuth bin rules, plug-in MI, the jugaad λ map (parallel over rows), leave-one-out means, Pearson ρ | 190 |
| `data.py` | paths (`$SNP_DATA`, `$SNP_OUT`), the case lists, `load(var, case)` | 25 |
| `run.py` | point estimates for one case, every binning configuration -> `point_<VAR>_<case>.npz` | 46 |
| `boot.py` | the Eade et al. (2014) bootstrap, modes `fd`, `knuth`, `lmloo`, `olmloo` -> `boot_<mode>_<VAR>_<case>.npz` | 71 |
| `tables.py` | `jugaad_fd_optbins_table` and `jugaad_lam_terms` (md + json), and the area-summary helpers | 166 |
| `plots.py` | every map and binning figure, and its json (plotting, not counted) | |
| `style.py` | font (Helvetica Neue if present, else TeX Gyre Heros), bold panel titles, and `save`: main-text or supp-info, png + pdf | |
| `gaussian_demo.py` | main-text `gaussian-array-lambda-rho`: (a) λ_m vs ρ_m on Gaussian members with correlation r; (b, c) λ_o vs ρ_o for a fixed ensemble against o = √a g(x) + noise, g = x and eˣ; supp-info `gaussian-array-rm-sensitivity`, panel (b) at member signal shares R_M = 0.2, 0.4, 0.6, 0.8; and `gaussian-array-heterogeneous-rm`, panel (b) with a signal share R_n per member (all 0.5 / 0.1-0.9 / half 0.1, half 0.9) | |
| `member_lam_m.py` | supp-info `lambda-m-member-rpc` + `figures/data/lambda-m-member.json`: λ_m with one member as pseudo-observation (λ(I(s′₋ₙ; f_k)), the λ_o construction) against production λ_m, and the RPC > 1 area, for the six production cases; flags the one-bin (λ = 0) cells | |
| `member_rho_m.py` | supp-info `rho-m-member-rpc-all[-lead2-4]` + `figures/data/rho-m-member-all.json`: the ρ counterpart, every member, unpooled as production: ρ_m^(k) = corr(s₋ₖ, f_k) (mean over k = production ρ_m), RPC_k = ρ_o / ρ_m^(k) | |
| `run_member_lam_m.sh` | submits `member_lam_m.py --members all`: one Slurm array task per production case, then the summary (json + figures `lambda-m-member-rpc-all[-lead2-4]`) | |
| `siegert_demo.py` | supp-info `siegert-model-ensemble-size`: λ and ρ (o and m) and both RPCs under the Siegert et al. (2016) model, β = 0.1 and 0.9, against ensemble size N, with the analytic values | |
| `flattened_hist.py` | main-text `flattened-histograms`: pooled s, o, f of the decadal rows against Gaussians, log density | |
| `run_all.sh` | submits the whole pipeline to Slurm: point (array) + boot (array) -> figures | |
| `data-acquisition/` | raw downloads to the per-case input arrays (`$SNP_DATA/cases/`); see its README | |
| `thesis-old/` | the original thesis code: `InfoTheoryMetrics.py`, `DataInventory.py`, `info_rpc_parallel.py`, a parallel `info_RPC` with bitwise-identical output, and `gaussian_array_metrics.py`, the thesis Gaussian-array figure (`supp-info/thesis-gaussian-array`) | |
| `tests/test_jugaad.py` | `jugaad.maps` and `info_rpc_parallel.info_RPC` against the thesis `info_RPC`, bitwise | |
| `rho_m_definitions.py` | supp-info `rho-m-definitions[-lead2-4]`: ρ_m^(1) = sd(s)/√⟨var f_n⟩ against ρ_m^(2) = ⟨corr(f_n, s₋ₙ)⟩ | |
| `lam_m_definitions.py`, `run_lam_m_definitions.sh` | supp-info `lambda-m-definitions[-by-model][-lead2-4]`: bias-corrected per-member ⟨λ(s₋ₙ; f_n)⟩ against pooled λ(s₋ₙ~; f~), all members and each model alone (λ_o here is the tiled jugaad one) | |
| `member_models.py` | `$SNP_OUT/member_models.json`: each member's model, checked bitwise against the case files | |
| `lam_m_gap.py` | supp-info `lambda-m-gap-gaussian`: both λ_m estimators (and swapped-ingredient variants) against the known Gaussian answer | |
| `exchangeability.py` | independent of `jugaad.py`: member exchangeability from fixed-edge binned (or pooled-rank copula) MI, I_cond, size-matched I_pool_T, Δ, I(M; X, Y), a within-time label-permutation null; `obs_mi` with g on its T samples only (never tiled) | |
| `exchangeability_check.py` | its synthetic validation (size, bias, power) and runs on the DJF NAO (SLP 2-4) and the lead 2-4 grids -> `figures/data/exchangeability-*.json`; supp-info `exchangeability-lead2-4` and main-text `members-not-exchangeable` | |
| `tests/test_exchangeability.py` | the construction rules of `exchangeability.py`: LOO, fixed edges, pairs kept under permutation, vectorised = loop, seeding, g never tiled | |

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
