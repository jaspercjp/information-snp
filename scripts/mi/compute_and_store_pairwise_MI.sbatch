#!/bin/bash
#SBATCH --job-name=pairwise_MI
#SBATCH -p serc
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Thread scaling saturates near 6x -- the kernel is memory-bandwidth-bound -- so 16 cores
# is the sweet spot and 64 would mostly idle. Measured sweep is ~15 min; 2 h is margin.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -uo pipefail

# The kernel is single-threaded numpy on purpose; joblib does the parallelism. Letting
# BLAS also spawn threads oversubscribes the node and makes it slower, not faster.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY=/oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/SNP_env/bin/python
cd "$OAK/JasperChen/Research/SNP/information-snp/scripts"

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK}"
srun "$PY" compute_and_store_pairwise_MI.py --n-jobs "${SLURM_CPUS_PER_TASK}" "$@"
