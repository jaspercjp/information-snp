#!/bin/bash
# lam_m_definitions.py (lambda_m per-member vs pooled, bias corrected): one Slurm array task per case,
# then a summary job (json + figures) after all six succeed. Returns at once.
#
#   SNP_DATA=<dir with cases/> bash scripts/run_lam_m_definitions.sh
#
set -euo pipefail

export SNP_SCRIPTS=${SNP_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
HERE=$SNP_SCRIPTS
PY=${SNP_PY:-/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/SNP_env/bin/python}
export SNP_OUT=${SNP_OUT:-$SCRATCH/snp_jugaad} MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.mplconfig}
CASES=(SLP:2-4 PRECT:2-4 TREFHT:2-4 SLP:s1961 PRECT:s1978 TREFHT:s1961)

if [[ ${1:-} == task ]]; then                      # inside a job
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    case $2 in
        member)  exec "$PY" "$HERE/lam_m_definitions.py" --case "${CASES[$SLURM_ARRAY_TASK_ID]}" ;;
        summary) exec "$PY" "$HERE/lam_m_definitions.py" --summarize ;;
    esac
fi

mkdir -p "$SNP_OUT/logs" "$MPLCONFIGDIR"
JOB=(-p "${SNP_PARTITION:-serc}" -o "$SNP_OUT/logs/%x_%A_%a.out" --parsable)
m=$(sbatch "${JOB[@]}" -J lam_m_def -c 16 --mem=32G -t 04:00:00 --array=0-$((${#CASES[@]} - 1)) "$0" task member)
s=$(sbatch "${JOB[@]}" -J lam_m_def_summary -c 2 --mem=16G -t 00:30:00 -d "afterok:$m" "$0" task summary)
echo "submitted member=$m summary=$s; logs in $SNP_OUT/logs"
