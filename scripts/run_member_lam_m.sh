#!/bin/bash
# member_lam_m.py for every member of the six production cases: one Slurm array task per case,
# then a summary job (json + figures) after all six succeed. Returns at once.
#
#   SNP_DATA=<dir with cases/> bash scripts/run_member_lam_m.sh
#
#   member   member_lam_m.py --members all --case VAR:case   (array 0-5, ~1 h for TREFHT s1961)
#   summary  member_lam_m.py --members all --summarize       (afterok)
# npz to $SNP_OUT (default $SCRATCH/snp_jugaad), logs to $SNP_OUT/logs; figures/data and
# figures/supp-info get lambda-m-member-all.json and lambda-m-member-rpc-all[-lead2-4].
set -euo pipefail

export SNP_SCRIPTS=${SNP_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
HERE=$SNP_SCRIPTS
PY=${SNP_PY:-/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/SNP_env/bin/python}
export SNP_OUT=${SNP_OUT:-$SCRATCH/snp_jugaad} MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.mplconfig}
CASES=(SLP:2-4 PRECT:2-4 TREFHT:2-4 SLP:s1961 PRECT:s1978 TREFHT:s1961)

if [[ ${1:-} == task ]]; then                      # inside a job
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    case $2 in
        member)  exec "$PY" "$HERE/member_lam_m.py" --members all --case "${CASES[$SLURM_ARRAY_TASK_ID]}" ;;
        summary) exec "$PY" "$HERE/member_lam_m.py" --members all --summarize ;;
    esac
fi

mkdir -p "$SNP_OUT/logs" "$MPLCONFIGDIR"
JOB=(-p "${SNP_PARTITION:-serc}" -o "$SNP_OUT/logs/%x_%A_%a.out" --parsable)
m=$(sbatch "${JOB[@]}" -J member_lam_m -c 16 --mem=24G -t 03:00:00 --array=0-$((${#CASES[@]} - 1)) "$0" task member)
s=$(sbatch "${JOB[@]}" -J member_lam_m_summary -c 2 --mem=8G -t 00:30:00 -d "afterok:$m" "$0" task summary)
echo "submitted member=$m summary=$s; logs in $SNP_OUT/logs"
