#!/usr/bin/env bash
# Immutable source snapshot, frozen selectors, one model pair per array task.
# sbatch --array=0-67%4 slurm/gv_tables_selectors.sbatch SOURCE_SNAPSHOT OUTPUT_ROOT
#SBATCH --job-name=gv-table-sel
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/scratch/gv-table-sel-%A_%a.out
#SBATCH --error=/path/to/scratch/gv-table-sel-%A_%a.err
set -euo pipefail
SOURCE="${1:?source snapshot required}"
OUT="${2:?output root required}"
cd "$SOURCE"
test -f SOURCE_COMMIT.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/path/to/scratch/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
/path/to/project/.venv-20260917/bin/python scripts/gv_table_selector_fit.py \
  --missing-only --index "$SLURM_ARRAY_TASK_ID" --out-dir "$OUT"
