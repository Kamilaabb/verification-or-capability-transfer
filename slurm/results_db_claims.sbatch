#!/usr/bin/env bash
# Run only after the immutable snapshot's required checks have passed.
#SBATCH --job-name=results-db-claims
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/scratch/results-db-claims-%j.out
#SBATCH --error=/path/to/scratch/results-db-claims-%j.err
set -euo pipefail
cd "${1:?immutable snapshot required}"
test -f SOURCE_COMMIT.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
PY=/path/to/project/.venv-20260917/bin/python
case "${2:?build or claims}" in
  build) "$PY" scripts/build_results_db.py --out-dir "${3:?new results database path}" ;;
  claims) "$PY" scripts/rebuild_results_claims.py --db "${3:?results database path}" --out "${4:?new aggregate output path}" ;;
  *) echo 'expected build or claims' >&2; exit 2 ;;
esac
