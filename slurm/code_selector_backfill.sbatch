#!/usr/bin/env bash
# Frozen code selector backfill: one pair per CPU array task.
#SBATCH --job-name=code-sel-backfill
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/scratch/code-sel-backfill-%A_%a.out
#SBATCH --error=/path/to/scratch/code-sel-backfill-%A_%a.err
set -euo pipefail
cd "${1:?immutable snapshot required}"
test -f SOURCE_COMMIT.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
export HF_HOME=/path/to/scratch/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
/path/to/project/.venv-20260917/bin/python scripts/code_selector_backfill.py \
    --index "${SLURM_ARRAY_TASK_ID:?}" --out-dir "${2:?new output root required}"
