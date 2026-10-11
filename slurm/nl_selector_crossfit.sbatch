#!/usr/bin/env bash
#SBATCH --job-name=nl-selector-oof
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/path/to/scratch/nl-selector-%A_%a.out
#SBATCH --error=/path/to/scratch/nl-selector-%A_%a.err
set -euo pipefail
cd "${1:?immutable snapshot required}"
test -f SOURCE_COMMIT.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
PY=/path/to/project/.venv-20260917/bin/python
"$PY" scripts/nl_selector_crossfit.py --validate-registration
"$PY" scripts/nl_selector_crossfit.py --index "$SLURM_ARRAY_TASK_ID" --out-dir "${2:?fresh root required}"
