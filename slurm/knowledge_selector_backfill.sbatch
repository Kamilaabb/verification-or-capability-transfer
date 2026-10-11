#!/usr/bin/env bash
# --array=45,48,53,55,57,60,63,66,67,69,72,78,79%4
# Fixed indices in gv_table_selector_fit.tasks(False); exactly the old 13 knowledge pairs.
# sbatch slurm/knowledge_selector_backfill.sbatch SOURCE_SNAPSHOT NEW_OUTPUT_ROOT
#SBATCH --job-name=mc-sel-backfill
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/scratch/mc-sel-backfill-%A_%a.out
#SBATCH --error=/path/to/scratch/mc-sel-backfill-%A_%a.err
set -euo pipefail
SOURCE="${1:?immutable snapshot required}"
OUT="${2:?new output root required}"
cd "$SOURCE"
test -f SOURCE_COMMIT.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME=/path/to/scratch/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONUTF8=1
# Intentionally no --missing-only: that flag would exclude all thirteen target pairs.
/path/to/project/.venv-20260917/bin/python scripts/gv_table_selector_fit.py \
  --index "$SLURM_ARRAY_TASK_ID" --out-dir "$OUT"
