#!/usr/bin/env bash
# Grade, analyze and archive one BigCodeBench solo run (c100-c109) after
# slurm/bigcodebench_solo.sbatch has written all 832 records and their evidence.
# Usage: sbatch -p cpu slurm/bigcodebench_solo_grade.sbatch <c100-c109> <RUN_STEM>
# Submit with --dependency=afterany:<generation job>:<its resume guard>; the job checks for
# itself that generation and evidence are complete, and stops otherwise.
#
# The grader runs inside the pinned evaluate container with networking disabled
# (protocols/authority_v6/grader_bigcodebench_v1.yaml); memory must exceed its 30 GB
# address-space limit per program.

#SBATCH --job-name=bcb-grade
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/project/bcb-grade-%j.out
#SBATCH --error=/path/to/project/bcb-grade-%j.err

set -euo pipefail

CONDITION_ID="${1:?usage: sbatch bigcodebench_solo_grade.sbatch <c100-c109> <RUN_STEM>}"
RUN_STEM="${2:?usage: sbatch bigcodebench_solo_grade.sbatch <c100-c109> <RUN_STEM>}"
PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
SIF="$SCRATCH_ROOT/containers/bigcodebench-evaluate.sif"
SIF_SHA=286b4e68f3604864ef7c10c48da83bba554b986fcd1f40ce6c092b695a410ad9
WORK_ROOT="$SCRATCH_ROOT/runs/$RUN_STEM"
GRADE_WORK="$WORK_ROOT/grading"
RUNNER=(python "$PROJECT_ROOT/scripts/run_bigcodebench_solo.py" --condition-id "$CONDITION_ID")
CONDITION="$PROJECT_ROOT/conditions/$CONDITION_ID/condition.yaml"
EXPECTED_ROWS=832
WORKERS=12
RECORDS="$WORK_ROOT/records/pass1.jsonl"
EVIDENCE="$WORK_ROOT/records/scalar_evidence.jsonl"

cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"
module purge
module load python
source .venv-20260917/bin/activate

# The run must be complete and still under the condition it started with.
test "$(wc -l < "$RECORDS")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$WORK_ROOT/records/native_logit_trace.jsonl")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$EVIDENCE")" -eq "$EXPECTED_ROWS"
RESOLVED_NOW="$SCRATCH_ROOT/runs/.bcb-grade-resolved-${SLURM_JOB_ID:-$$}.yaml"
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
  --output "$WORK_ROOT/records/grades.jsonl"
"${RUNNER[@]}" analyze --run-id "${RUN_STEM}__analysis__r015" --records "$RECORDS" \
  --evidence "$EVIDENCE" --grades "$WORK_ROOT/records/grades.jsonl" \
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
