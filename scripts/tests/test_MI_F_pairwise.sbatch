#!/bin/bash
#SBATCH --job-name=mi_pairwise
#SBATCH --partition=normal
# Measured: ladder 25 core-min + real DCPP cube (N=101, J=48, 13.45M KSG calls)
# 121 core-min => ~2.5 min on 64 cores. 30 min is a wide margin.
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -euo pipefail

PY=/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/SNP_env/bin/python
cd "$OAK/JasperChen/Research/SNP/information-snp/scripts"

# joblib memmaps the rank cube to shared memory; keep BLAS out of the workers' way.
export JOBLIB_TEMP_FOLDER="${L_SCRATCH:-/tmp}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "host $(hostname)  cpus $SLURM_CPUS_PER_TASK"
srun "$PY" test_MI_F_pairwise.py --real --n-jobs "$SLURM_CPUS_PER_TASK"
