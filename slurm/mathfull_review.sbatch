#!/usr/bin/env bash
# The GV system on mathfull dev1000. c024: Qwen3-8B reviews c022's Qwen3-4B with reviewer
# prompt v1; c027: Qwen3-8B reviews c025's Qwen3-4B with the frozen review prompt v2;
# c028: Qwen3-4B reviews c026's Qwen3-8B with the same frozen prompt. c042/c043 are c027/c028
# on the sealed test manifest, reviewing c040/c041 (docs/plan/test_mathfull_v2.md).
# Usage: sbatch slurm/mathfull_review.sbatch <CONDITION_ID> <RUN_STEM> [i/K | merge/K]
#
# Sharding works as in mathfull_solo.sbatch. A shard job reads G's records straight from the
# G archive and writes only its review rows under runs/<RUN_STEM>__shards/i; the merge job
# copies G's records once, checks them against the archive, and merges the review rows.
#
# The G source archive is the condition's generator_source.artifact_path. Resumable like
# mathfull_solo.sbatch: the same RUN_STEM continues the run after checking the resolved
# condition and the copied G records are unchanged, and reconciling review rows with their
# native traces. A job that finds all rows reviewed starts no model server.

#SBATCH --job-name=mathfull-review
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/project/mathfull-review-%j.out
#SBATCH --error=/path/to/project/mathfull-review-%j.err

set -euo pipefail

CONDITION_ID="${1:?usage: sbatch mathfull_review.sbatch <CONDITION_ID> <RUN_STEM> [i/K | merge/K]}"
RUN_STEM="${2:?usage: sbatch mathfull_review.sbatch <CONDITION_ID> <RUN_STEM> [i/K | merge/K]}"
SHARD="${3:-}"
SHARD_INDEX=""; SHARD_COUNT=""
if [[ -n "$SHARD" ]]; then
  [[ "$SHARD" =~ ^(merge|[0-9]+)/([0-9]+)$ ]] || { echo "bad shard spec $SHARD" >&2; exit 2; }
  SHARD_INDEX="${BASH_REMATCH[1]}"; SHARD_COUNT="${BASH_REMATCH[2]}"
fi
IS_SHARD=""; [[ -n "$SHARD_INDEX" && "$SHARD_INDEX" != merge ]] && IS_SHARD=1
case "$CONDITION_ID" in
  c024|c027|c042|c048|c051|c052|c091|c095|c154|c156|c214|c234|c216|c236|c248|c251|c253|c256|c260|c386|c390|c391) SERVED=qwen3-8b ;;
  c090|c094|c158|c162|c211|c231|c219|c239) SERVED=mistral-small-24b ;;
  c159|c160|c164|c212|c232|c250|c252|c264|c268) SERVED=qwen3-32b ;;
  c028|c043|c061|c065|c166|c210|c230|c261|c249|c262|c266|c387) SERVED=qwen3-4b ;;
  c044|c058|c150|c152|c163|c167|c213|c233|c221|c241|c222) SERVED=qwen2.5-3b ;;
  c060|c064|c155|c157|c161|c218|c238|c220|c240|c224|c242) SERVED=qwen2.5-7b ;;
  c045|c047|c049|c059|c225|c243|c227|c245|c229|c247) SERVED=llama-3.2-3b ;;
  c046|c050|c053|c151|c153|c165|c215|c235|c223|c228|c246|c257) SERVED=ministral-8b ;;
  c217|c237|c226|c244|c258) SERVED=llama-3.1-8b ;;
  c263|c267|c265|c269) SERVED=qwen2.5-72b ;;
  *) echo "unknown condition $CONDITION_ID" >&2; exit 2 ;;
esac
PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
SHARD_ROOT="$SCRATCH_ROOT/runs/${RUN_STEM}__shards"
if [[ -n "$IS_SHARD" ]]; then WORK_ROOT="$SHARD_ROOT/$SHARD_INDEX"; else WORK_ROOT="$SCRATCH_ROOT/runs/$RUN_STEM"; fi
CONTAINER="$SCRATCH_ROOT/containers/vllm-openai-v0.29.0-rebuilt-20260914.sif"
RUNNER=(python "$PROJECT_ROOT/scripts/run_mathfull_review.py" --condition-id "$CONDITION_ID")
CONDITION="$PROJECT_ROOT/conditions/$CONDITION_ID/condition.yaml"
# Serving memory only, not part of the condition. A reviewer prompt carries the problem and
# G's full solution plus up to 4096 new tokens, so the window stays at 16384 by default.
MAX_MODEL_LEN="${MATHFULL_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${MATHFULL_GPU_MEMORY_UTILIZATION:-0.90}"
# MATHFULL_TP=2 serves a model too large for one GPU over two (Qwen2.5-72B as reviewer, on the
# dedicated 2-H100 allocation); the capture processor writes from tensor-parallel rank 0 only.
TP="${MATHFULL_TP:-1}"
SIDECAR="$WORK_ROOT/logs/vllm_capture_sidecar_${SLURM_JOB_ID:-local}.jsonl"

cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"

module purge
module load python
source .venv-20260917/bin/activate
export HF_HOME=/path/to/scratch/huggingface
export HF_DATASETS_CACHE=/path/to/scratch/huggingface/datasets
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
python -m unittest discover -s tests
"${RUNNER[@]}" check-runnable
# The manifest, its hash and its size come from the condition and the dataset registration.
MANIFEST_INFO=$("${RUNNER[@]}" manifest-info)
read -r MANIFEST MANIFEST_SHA EXPECTED_ROWS <<< "$MANIFEST_INFO"
echo "$MANIFEST_SHA  $MANIFEST" | sha256sum -c -
FULL_MANIFEST="$MANIFEST"
bash -n slurm/mathfull_review.sbatch
G_SOURCE=$(python - "$CONDITION" <<'PY'
import sys
from pathlib import Path
from authority_allocation.condition_resolution import resolve_condition
print(resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())["generator_source"]["artifact_path"])
PY
)
test -d "$G_SOURCE"
(cd "$G_SOURCE" && sha256sum -c --quiet checksums.sha256)
test "$(wc -l < "$G_SOURCE/records/pass1.jsonl")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$G_SOURCE/records/native_logit_trace.jsonl")" -eq "$EXPECTED_ROWS"

MODEL_PATH=$(python - "$CONDITION" <<'PY'
import sys
from pathlib import Path
from authority_allocation.condition_resolution import resolve_condition
print(resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())["model"]["server_path"])
PY
)
test -d "$MODEL_PATH"

RESOLVED_NOW="$SCRATCH_ROOT/runs/.mathfull-review-resolved-${SLURM_JOB_ID:-$$}.yaml"
python - "$CONDITION" "$RESOLVED_NOW" <<'PY'
from authority_allocation.condition_resolution import resolve_condition
from pathlib import Path
import sys, yaml
resolved = resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())
Path(sys.argv[2]).write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
PY

if [[ -e "$WORK_ROOT" ]]; then
  if ! cmp -s "$RESOLVED_NOW" "$WORK_ROOT/condition.resolved.yaml"; then
    echo "Condition differs from the one this run started under; refusing to resume." >&2
    exit 1
  fi
  if [[ -z "$IS_SHARD" ]]; then
    for pair in "pass1.jsonl g_pass1.jsonl" "native_logit_trace.jsonl g_native_logit_trace.jsonl" \
                "grades.jsonl g_grades.jsonl"; do
      set -- $pair
      cmp -s "$G_SOURCE/records/$1" "$WORK_ROOT/records/$2" \
        || { echo "Copied G record $2 differs from the archive; refusing to resume." >&2; exit 1; }
    done
  fi
  echo "Resuming $RUN_STEM"
  "${RUNNER[@]}" reconcile "$WORK_ROOT/records/review.jsonl" "$WORK_ROOT/records/native_logit_trace.jsonl"
else
  mkdir -p "$WORK_ROOT/records" "$WORK_ROOT/reports" "$WORK_ROOT/logs"
  if [[ -z "$IS_SHARD" ]]; then
    cp "$G_SOURCE/records/pass1.jsonl" "$WORK_ROOT/records/g_pass1.jsonl"
    cp "$G_SOURCE/records/native_logit_trace.jsonl" "$WORK_ROOT/records/g_native_logit_trace.jsonl"
    cp "$G_SOURCE/records/grades.jsonl" "$WORK_ROOT/records/g_grades.jsonl"
  fi
  cp "$RESOLVED_NOW" "$WORK_ROOT/condition.resolved.yaml"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK_ROOT/started_at_utc"
fi
{
  echo "slurm_job_id=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-}" \
       "node=$(hostname) started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "nvidia-smi failed"
} >> "$WORK_ROOT/logs/execution_hardware.txt"

count_rows() { if [[ -f "$1" ]]; then wc -l < "$1"; else echo 0; fi; }
# A shard reviews its block reading G straight from the archive; the full run uses its copy.
if [[ -n "$IS_SHARD" ]]; then G_RECORDS="$G_SOURCE/records/pass1.jsonl"; else G_RECORDS="$WORK_ROOT/records/g_pass1.jsonl"; fi
if [[ -n "$IS_SHARD" ]]; then
  MANIFEST="$WORK_ROOT/manifest_shard.jsonl"
  EXPECTED_ROWS=$("${RUNNER[@]}" shard-manifest "$FULL_MANIFEST" "$SHARD_INDEX" "$SHARD_COUNT" "$MANIFEST")
  echo "Shard $SHARD_INDEX/$SHARD_COUNT: $EXPECTED_ROWS rows"
elif [[ "$SHARD_INDEX" == merge ]] \
     && (( $(count_rows "$WORK_ROOT/records/review.jsonl") != EXPECTED_ROWS )); then
  for (( i = 0; i < SHARD_COUNT; i++ )); do
    cmp -s "$RESOLVED_NOW" "$SHARD_ROOT/$i/condition.resolved.yaml" \
      || { echo "shard $i ran under a different condition; refusing to merge" >&2; exit 1; }
    cp "$SHARD_ROOT/$i/logs/execution_hardware.txt" "$WORK_ROOT/logs/shard_${i}_execution_hardware.txt"
  done
  "${RUNNER[@]}" merge-shards "$FULL_MANIFEST" "$SHARD_ROOT" "$SHARD_COUNT" "$WORK_ROOT/records" \
    review.jsonl native_logit_trace.jsonl
fi
rm -f "$RESOLVED_NOW"

DONE_ROWS=$(count_rows "$WORK_ROOT/records/review.jsonl")
DONE_TRACES=$(count_rows "$WORK_ROOT/records/native_logit_trace.jsonl")
if (( DONE_ROWS == EXPECTED_ROWS && DONE_TRACES == EXPECTED_ROWS )); then
  echo "All $EXPECTED_ROWS rows already reviewed; skipping the model server."
elif [[ "$SHARD_INDEX" == merge ]]; then
  echo "merged records hold $DONE_ROWS/$DONE_TRACES rows, not $EXPECTED_ROWS" >&2; exit 1
else
module load gcc/11.5.0
module load apptainer/1.4.1
export APPTAINER_CACHEDIR="$SCRATCH_ROOT/apptainer-cache"
export APPTAINER_TMPDIR="$SCRATCH_ROOT/apptainer-tmp"
mkdir -p "$SCRATCH_ROOT/vllm-cache/triton"

VLLM_PID=""
stop_server() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"; wait "$VLLM_PID" || true
  fi
}
trap stop_server EXIT

# --max-num-batched-tokens equals --max-model-len so every prompt is prefilled in one
# step; a chunked prefill stops the logit-capture processor (c018, job 65120295).
# Each job serves on its own free port. With a fixed 8000, c026's resume landed on the node
# already running c027, its server failed to bind, and its requests reached c027's server
# (jobs 65138329 and 65138194 both failed on a missing capture).
PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
API="http://127.0.0.1:$PORT/v1"
apptainer exec --cleanenv --nv \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env VLLM_CACHE_ROOT=/mnt/vllm-cache --env TRITON_CACHE_DIR=/mnt/vllm-cache/triton \
  --env PYTHONPATH="$PROJECT_ROOT/src" \
  --env C014_CAPTURE_PATH="$SIDECAR" \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT:ro" \
  --bind "$MODEL_PATH:/mnt/model:ro" \
  --bind "$SCRATCH_ROOT/vllm-cache:/mnt/vllm-cache" \
  --bind "$WORK_ROOT:$WORK_ROOT" \
  "$CONTAINER" vllm serve /mnt/model \
  --served-model-name "$SERVED" --dtype bfloat16 --logprobs-mode raw_logprobs \
  --logits-processors authority_allocation.vllm_logit_capture:C014LogitCapture \
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_MODEL_LEN" --max-num-seqs 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enforce-eager --tensor-parallel-size "$TP" \
  --host 127.0.0.1 --port "$PORT" > "$WORK_ROOT/logs/server_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
deadline=$((SECONDS + 900))
while (( SECONDS < deadline )); do
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -80 "$WORK_ROOT/logs/server_${SLURM_JOB_ID:-local}.log"; exit 1; }
  if python -c 'import sys; from urllib import request; r=request.urlopen(sys.argv[1] + "/models", timeout=2); raise SystemExit(0 if r.status==200 else 1)' "$API" 2>/dev/null; then
    break
  fi
  sleep 5
done
python -c 'import sys; from urllib import request; request.urlopen(sys.argv[1] + "/models", timeout=2)' "$API"
# The server that answered must be this job's own.
kill -0 "$VLLM_PID"

"${RUNNER[@]}" smoke --run-id "${RUN_STEM}__smoke" --api-base "$API" --served-model "$SERVED"
"${RUNNER[@]}" review --run-id "${RUN_STEM}__review__p01" \
  --api-base "$API" --served-model "$SERVED" --manifest "$MANIFEST" \
  --expected-rows "$EXPECTED_ROWS" --g-records "$G_RECORDS" \
  --output "$WORK_ROOT/records/review.jsonl" \
  --capture-sidecar "$SIDECAR" \
  --logit-output "$WORK_ROOT/records/native_logit_trace.jsonl" \
  --logit-run-id "${RUN_STEM}__same-forward-logits__p01"
stop_server
VLLM_PID=""
fi

test "$(wc -l < "$WORK_ROOT/records/review.jsonl")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$WORK_ROOT/records/native_logit_trace.jsonl")" -eq "$EXPECTED_ROWS"
if [[ -n "$IS_SHARD" ]]; then
  echo "$CONDITION_ID shard $SHARD_INDEX/$SHARD_COUNT reviewed: $WORK_ROOT"
  exit 0
fi

# Evidence is materialized before this process reads gold answers for grading.
rm -f "$WORK_ROOT/records/scalar_evidence.jsonl" "$WORK_ROOT/records/grades.jsonl"
"${RUNNER[@]}" evidence --run-id "${RUN_STEM}__evidence__e01" \
  --pass1-records "$WORK_ROOT/records/review.jsonl" \
  --logit-traces "$WORK_ROOT/records/native_logit_trace.jsonl" \
  --output "$WORK_ROOT/records/scalar_evidence.jsonl"
"${RUNNER[@]}" grade --review-records "$WORK_ROOT/records/review.jsonl" \
  --g-records "$WORK_ROOT/records/g_pass1.jsonl" --output "$WORK_ROOT/records/grades.jsonl"
"${RUNNER[@]}" analyze --run-id "${RUN_STEM}__analysis__r012" \
  --review-records "$WORK_ROOT/records/review.jsonl" \
  --g-records "$WORK_ROOT/records/g_pass1.jsonl" \
  --g-grades "$WORK_ROOT/records/g_grades.jsonl" \
  --grades "$WORK_ROOT/records/grades.jsonl" \
  --expected-rows "$EXPECTED_ROWS" --output "$WORK_ROOT/reports/analysis.json"

python scripts/validate_records.py --bundle-dir "$WORK_ROOT"
(cd "$WORK_ROOT" && sha256sum condition.resolved.yaml records/*.jsonl reports/* > checksums.sha256)
ARCHIVE="$PROJECT_ROOT/artifacts/$RUN_STEM"
test ! -e "$ARCHIVE"
mkdir "$ARCHIVE"
cp -a "$WORK_ROOT/." "$ARCHIVE/"
(cd "$ARCHIVE" && sha256sum -c checksums.sha256)
cat "$ARCHIVE/reports/summary.md"
echo "$CONDITION_ID complete: $ARCHIVE"
