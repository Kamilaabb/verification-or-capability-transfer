#!/usr/bin/env bash
# Grade, analyze and archive one BigCodeBench review run (c110, c111) after
# slurm/bigcodebench_review.sbatch has written all 832 reviews and their evidence.
# Usage: sbatch -p cpu slurm/bigcodebench_review_grade.sbatch <c110-c114|c148|c149|c199|c278-c302> <RUN_STEM>
# Submit with --dependency=afterany:<review job>:<its resume guard>; the job checks for itself
# that the review and evidence are complete, and stops otherwise. Only the reviewer's outputs
# are graded; the reviewed model's grades are the archived ones copied as g_grades.jsonl.

#SBATCH --job-name=bcb-rgrade
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/project/bcb-rgrade-%j.out
#SBATCH --error=/path/to/project/bcb-rgrade-%j.err

set -euo pipefail

CONDITION_ID="${1:?usage: sbatch bigcodebench_review_grade.sbatch <c110-c114|c148|c149|c199|c278-c302> <RUN_STEM>}"
RUN_STEM="${2:?usage: sbatch bigcodebench_review_grade.sbatch <c110-c114|c148|c149|c199|c278-c302> <RUN_STEM>}"
PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
SIF="$SCRATCH_ROOT/containers/bigcodebench-evaluate.sif"
SIF_SHA=286b4e68f3604864ef7c10c48da83bba554b986fcd1f40ce6c092b695a410ad9
WORK_ROOT="$SCRATCH_ROOT/runs/$RUN_STEM"
GRADE_WORK="$WORK_ROOT/grading"
RUNNER=(python "$PROJECT_ROOT/scripts/run_bigcodebench_review.py" --condition-id "$CONDITION_ID")
CONDITION="$PROJECT_ROOT/conditions/$CONDITION_ID/condition.yaml"
EXPECTED_ROWS=832
WORKERS=12
RECORDS="$WORK_ROOT/records/review.jsonl"
EVIDENCE="$WORK_ROOT/records/scalar_evidence.jsonl"

cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"
module purge
module load python
source .venv-20260917/bin/activate

test "$(wc -l < "$RECORDS")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$WORK_ROOT/records/native_logit_trace.jsonl")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$EVIDENCE")" -eq "$EXPECTED_ROWS"
RESOLVED_NOW="$SCRATCH_ROOT/runs/.bcb-rgrade-resolved-${SLURM_JOB_ID:-$$}.yaml"
python - "$CONDITION" "$RESOLVED_NOW" <<'PY'
from authority_allocation.condition_resolution import resolve_condition
from pathlib import Path
import sys, yaml
resolved = resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())
Path(sys.argv[2]).write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
PY
cmp -s "$RESOLVED_NOW" "$WORK_ROOT/condition.resolved.yaml" \
  || { echo "Condition differs from the one this run started under." >&2; exit 1; }
rm -f "$RESOLVED_NOW"
"${RUNNER[@]}" check-runnable

rm -rf "$GRADE_WORK"
mkdir -p "$GRADE_WORK"
"${RUNNER[@]}" grade-requests --records "$RECORDS" --output "$GRADE_WORK/requests.jsonl"
cp -a "$SCRATCH_ROOT/bcb-gt/hf" "$GRADE_WORK/hf"
cp "$PROJECT_ROOT/scripts/bigcodebench_grade.py" "$GRADE_WORK/bigcodebench_grade.py"
echo "$SIF_SHA  $SIF" | sha256sum -c -

module load gcc/11.5.0
module load apptainer/1.4.1
apptainer exec --net --network=none --cleanenv \
  --bind "$GRADE_WORK:/work" \
  --env HF_HOME=/work/hf --env HF_DATASETS_CACHE=/work/hf/datasets \
  --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
  --env PYTHONPATH=/work --env TMPDIR=/tmp \
  "$SIF" python3 /work/bigcodebench_grade.py \
  --requests /work/requests.jsonl --output /work/results.jsonl --workers "$WORKERS" \
  > "$GRADE_WORK/grade.log" 2>&1
pkill -f "/work/bigcodebench_grade.py" 2>/dev/null || true
tail -8 "$GRADE_WORK/grade.log"
rm -rf "$GRADE_WORK/hf"

"${RUNNER[@]}" grade --records "$RECORDS" --results "$GRADE_WORK/results.jsonl" \
  --g-grades "$WORK_ROOT/records/g_grades.jsonl" --output "$WORK_ROOT/records/grades.jsonl"
"${RUNNER[@]}" analyze --run-id "${RUN_STEM}__analysis__r015" --records "$RECORDS" \
  --evidence "$EVIDENCE" --grades "$WORK_ROOT/records/grades.jsonl" \
  --g-records "$WORK_ROOT/records/g_pass1.jsonl" --g-grades "$WORK_ROOT/records/g_grades.jsonl" \
  --expected-rows "$EXPECTED_ROWS" --output "$WORK_ROOT/reports/analysis.json"

python scripts/validate_records.py --bundle-dir "$WORK_ROOT"
(cd "$WORK_ROOT" && sha256sum condition.resolved.yaml records/*.jsonl reports/* grading/results.jsonl \
  > checksums.sha256)
ARCHIVE="$PROJECT_ROOT/artifacts/$RUN_STEM"
test ! -e "$ARCHIVE"
mkdir "$ARCHIVE"
cp -a "$WORK_ROOT/." "$ARCHIVE/"
(cd "$ARCHIVE" && sha256sum -c --quiet checksums.sha256)
cat "$ARCHIVE/reports/summary.md"
echo "$CONDITION_ID complete: $ARCHIVE"
