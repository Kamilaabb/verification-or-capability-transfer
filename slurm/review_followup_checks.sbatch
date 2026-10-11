#!/usr/bin/env bash
# Run only from an immutable snapshot of a committed revision (scripts/review_followup_checks.py).
#SBATCH --job-name=review-followup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/scratch/review-followup-%j.out
#SBATCH --error=/path/to/scratch/review-followup-%j.err
set -euo pipefail
cd "${1:?immutable snapshot required}"
test -f SOURCE_COMMIT.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
PY=/path/to/project/.venv-20260917/bin/python
"$PY" scripts/review_followup_checks.py --db "${2:?results database path}" \
  --artifacts /path/to/project/artifacts --out "${3:?new output path}"
