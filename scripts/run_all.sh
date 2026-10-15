#!/bin/bash
# Reproduce the map and binning figures (figures/{main-text,supp-info}): submits three Slurm jobs.
#
#   bash scripts/run_all.sh          # after data-acquisition/make_cases.py has written $SNP_DATA/cases
#
#   point    run.py for the 8 cases                     (array, minutes each)
#   boot     boot.py, 9 x fd + 5 x knuth + 6 x lmloo + 6 x olmloo   (array, up to ~6 h each at 16 cores)
#   figures  tables.py + plots.py, after both succeed   (afterok dependency)
# Outputs go to $SNP_OUT (default $SCRATCH/snp_jugaad), figures to figures/{main-text,supp-info}/{png,pdf}, json to figures/data.
set -euo pipefail

# sbatch runs a spooled copy of this file, so the jobs find scripts/ through SNP_SCRIPTS
export SNP_SCRIPTS=${SNP_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
HERE=$SNP_SCRIPTS
PY=${SNP_PY:-/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/SNP_env/bin/python}
export SNP_OUT=${SNP_OUT:-$SCRATCH/snp_jugaad} MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.mplconfig}

POINT=(SLP:cmip5 SLP:s1961 TREFHT:s1961 SLP:s1961moy TREFHT:s1961moy SLP:2-4 PRECT:2-4 TREFHT:2-4)
BOOT=("${POINT[@]/%/:fd}" PRECT:s1978:fd SLP:2-4:knuth PRECT:2-4:knuth TREFHT:2-4:knuth SLP:s1961:knuth
      TREFHT:s1961:knuth)
for m in lmloo olmloo; do
    BOOT+=(SLP:2-4:$m PRECT:2-4:$m TREFHT:2-4:$m SLP:s1961:$m PRECT:s1978:$m TREFHT:s1961:$m)
done

if [[ ${1:-} == task ]]; then                      # inside a job: one array element
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    case $2 in
        point) IFS=: read -r var c <<< "${POINT[$SLURM_ARRAY_TASK_ID]}"
               exec "$PY" "$HERE/run.py" "$var" "$c" ;;
        boot)  IFS=: read -r var c mode <<< "${BOOT[$SLURM_ARRAY_TASK_ID]}"
               exec "$PY" "$HERE/boot.py" "$var" "$c" "$mode" ;;
        figures) "$PY" "$HERE/tables.py" && exec "$PY" "$HERE/plots.py" ;;
    esac
fi

mkdir -p "$SNP_OUT/logs" "$MPLCONFIGDIR"
JOB=(-p "${SNP_PARTITION:-serc}" -c 16 --mem=48G -o "$SNP_OUT/logs/%x_%A_%a.out" --parsable)
p=$(sbatch "${JOB[@]}" -J point -t 01:00:00 --array=0-$((${#POINT[@]} - 1)) "$0" task point)
b=$(sbatch "${JOB[@]}" -J boot -t 08:00:00 --array=0-$((${#BOOT[@]} - 1)) "$0" task boot)
f=$(sbatch "${JOB[@]}" -J figures -t 00:30:00 -d "afterok:$p:$b" "$0" task figures)
echo "submitted point=$p boot=$b figures=$f; logs in $SNP_OUT/logs"
