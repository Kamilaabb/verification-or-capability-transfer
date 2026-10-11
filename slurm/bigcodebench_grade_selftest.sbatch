#!/usr/bin/env bash
# Check the grading adapter's contract mapping before any research run uses it.
#
# --mem must stay well above max_as_limit (30 GB). The child's rlimit is looser than
# the cgroup otherwise, so the cgroup OOM killer fires first: grading stays correct but
# Slurm marks the job FAILED, and a FAILED job cannot be registered as validated.
# That is what happened to job 65081190 at --mem=16G.
#
# The infinite-loop case burns its full timeout (~242s observed, not the configured
# 60s), so budget wall clock by the former.
#
# NEVER pipe the apptainer call's stdout: untrusted_check's forked children inherit it
# and a downstream reader never sees EOF (job 65076891 idled 3.5 hours that way).

#SBATCH --job-name=bcb-grade-selftest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/project/bcb-grade-selftest-%j.out
#SBATCH --error=/path/to/project/bcb-grade-selftest-%j.err

set -uo pipefail

PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
SIF="$SCRATCH_ROOT/containers/bigcodebench-evaluate.sif"
WORK="$SCRATCH_ROOT/bcb-grade-selftest"

cd "$PROJECT_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is dirty; commit and pull the exact code before running." >&2
  exit 1
fi

module purge
module load gcc/11.5.0
module load apptainer/1.4.1

mkdir -p "$WORK"
# The ground-truth job already warmed this cache; offline execution depends on it.
cp -a "$SCRATCH_ROOT/bcb-gt/hf" "$WORK/hf" 2>/dev/null || true
cp "$PROJECT_ROOT/scripts/bigcodebench_grade.py" "$WORK/bigcodebench_grade.py"
cp "$PROJECT_ROOT/scripts/bigcodebench_grade_selftest.py" "$WORK/selftest.py"

echo "=============== selftest (offline) ==============="
timeout 1500 apptainer exec --net --network=none --cleanenv \
  --bind "$WORK:/work" \
  --env HF_HOME=/work/hf --env HF_DATASETS_CACHE=/work/hf/datasets \
  --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
  --env PYTHONPATH=/work --env TMPDIR=/tmp \
  "$SIF" python3 /work/selftest.py \
  > "$WORK/selftest.txt" 2>&1
echo "selftest exit: $?"
pkill -f "/work/selftest.py" 2>/dev/null || true
cat "$WORK/selftest.txt"

echo DONE
