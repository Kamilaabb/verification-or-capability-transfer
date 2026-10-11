#!/usr/bin/env bash
# Bring a model into the stack at an immutable commit.
#
# Usage: sbatch slurm/model_snapshot.sbatch <HF_REPO_ID> [BRANCH]
#
# Generalises coder_model_snapshot.sbatch, which hardcodes the c005 generator. Gate 1
# measured G and V agreeing far more in the code domain than in the math one — kappa
# 0.787 across all rows against 0.290 — and c003's "first model pair" note registered
# the remedy in advance: bring in a cross-family coder model when complementarity is
# insufficient. Screening candidates means downloading several, so the job takes the
# repo as an argument.
#
# Setup, not a research run. It writes only to the scratch model tree; the pin is the
# committed table in the prepare script, not this job's output.

#SBATCH --job-name=model-snapshot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/project/model-snapshot-%j.out
#SBATCH --error=/path/to/project/model-snapshot-%j.err

set -uo pipefail

REPO_ID="${1:?usage: sbatch model_snapshot.sbatch <HF_REPO_ID> [BRANCH]}"
BRANCH="${2:-main}"

PROJECT_ROOT=/path/to/project

module purge
module load python

cd "$PROJECT_ROOT"
source .venv-20260917/bin/activate

export HF_HOME=/path/to/scratch/huggingface
export HF_DATASETS_CACHE=/path/to/scratch/huggingface/datasets

echo "=============== existing snapshots ==============="
du -sh /path/to/scratch/authority-allocation/models/* 2>/dev/null
df -h /path/to/scratch | tail -1

echo
echo "=============== download $REPO_ID @ $BRANCH ==============="
python scripts/freeze_model_snapshot.py --repo-id "$REPO_ID" --branch "$BRANCH"
echo "exit: $?"

echo DONE
