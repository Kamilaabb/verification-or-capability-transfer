#!/usr/bin/env bash
#SBATCH --job-name=code-sel-verify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/scratch/code-sel-verify-%j.out
#SBATCH --error=/path/to/scratch/code-sel-verify-%j.err
set -euo pipefail
cd "${1:?immutable snapshot required}"
test -f SOURCE_COMMIT.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
/path/to/project/.venv-20260917/bin/python scripts/code_selector_backfill.py \
    --verify-root "${2:?completed output root required}" --out-dir "${3:?new validation output required}"
