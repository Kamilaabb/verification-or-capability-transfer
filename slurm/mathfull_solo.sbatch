#!/usr/bin/env bash
# Solo pass 1 on mathfull dev1000, 4096 tokens: G (Qwen3-4B) is c022 or c025, V (Qwen3-8B)
# is c023 or c026; c029-c032 are Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct
# and Ministral-8B-Instruct-2410 alone. c025, c026 and c029-c032 use the frozen solve prompt v2.
# c040/c041 are c025/c026 on the sealed test manifest (docs/plan/test_mathfull_v2.md).
# Usage: sbatch slurm/mathfull_solo.sbatch <CONDITION_ID> <RUN_STEM> [i/K | merge/K]
#
# Sharding: with "i/K" the job only generates shard i of K (a contiguous block of the
# manifest) under runs/<RUN_STEM>__shards/i and stops; every row keeps the seed, record id
# and capture id a single job would give it. With "merge/K" the job starts no model server:
# it checks that every shard ran under the identical resolved condition and holds exactly
# its block, concatenates the records in manifest order under runs/<RUN_STEM>, and then
# runs evidence, grading, analysis and archiving as an unsharded run would.
#
# Resumable. Submitting again with the same RUN_STEM continues that run: the resolved
# condition must be identical to the one it started under, pass1 and native-trace records
# are first reconciled to the rows complete in both, and generation skips what is done.
# Each job writes its own capture sidecar, so a stale capture from a killed job can never
# be paired with a regenerated row. Evidence, grading and archiving run only once all
# 1,000 rows exist; a job that finds them all generated starts no model server, so it can
# run on the cpu partition (sbatch -p cpu --gpus=0 ...).

#SBATCH --job-name=mathfull-solo
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/project/mathfull-solo-%j.out
#SBATCH --error=/path/to/project/mathfull-solo-%j.err

set -euo pipefail

CONDITION_ID="${1:?usage: sbatch mathfull_solo.sbatch <CONDITION_ID> <RUN_STEM> [i/K | merge/K]}"
RUN_STEM="${2:?usage: sbatch mathfull_solo.sbatch <CONDITION_ID> <RUN_STEM> [i/K | merge/K]}"
SHARD="${3:-}"
SHARD_INDEX=""; SHARD_COUNT=""
if [[ -n "$SHARD" ]]; then
  [[ "$SHARD" =~ ^(merge|[0-9]+)/([0-9]+)$ ]] || { echo "bad shard spec $SHARD" >&2; exit 2; }
  SHARD_INDEX="${BASH_REMATCH[1]}"; SHARD_COUNT="${BASH_REMATCH[2]}"
fi
case "$CONDITION_ID" in
  c022|c025|c040|c062|c200|c259) SERVED=qwen3-4b ;;
  c023|c026|c041|c092|c201|c255) SERVED=qwen3-8b ;;
  c029|c056|c208) SERVED=llama-3.2-3b ;;
  c030|c057|c202) SERVED=qwen2.5-3b ;;
  c031|c063|c203) SERVED=qwen2.5-7b ;;
  c066|c098|c209) SERVED=qwen2.5-72b ;;
  c032|c096|c204|c254) SERVED=ministral-8b ;;
  c054|c093|c205) SERVED=mistral-small-24b ;;
  c055|c097|c206) SERVED=qwen3-32b ;;
  c067|c068|c207) SERVED=llama-3.1-8b ;;
  *) echo "unknown condition $CONDITION_ID" >&2; exit 2 ;;
esac
PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
SHARD_ROOT="$SCRATCH_ROOT/runs/${RUN_STEM}__shards"
if [[ -n "$SHARD_INDEX" && "$SHARD_INDEX" != merge ]]; then
  WORK_ROOT="$SHARD_ROOT/$SHARD_INDEX"
else
  WORK_ROOT="$SCRATCH_ROOT/runs/$RUN_STEM"
fi
CONTAINER="$SCRATCH_ROOT/containers/vllm-openai-v0.29.0-rebuilt-20260914.sif"
RUNNER=(python "$PROJECT_ROOT/scripts/run_mathfull_solo.py" --condition-id "$CONDITION_ID")
CONDITION="$PROJECT_ROOT/conditions/$CONDITION_ID/condition.yaml"
# Serving memory only, not part of the condition. On a 20 GB MIG slice set
# MATHFULL_MAX_MODEL_LEN=6144: every request needs at most its prompt plus 4096 tokens.
MAX_MODEL_LEN="${MATHFULL_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${MATHFULL_GPU_MEMORY_UTILIZATION:-0.90}"
# MATHFULL_TP=2 serves a model too large for one GPU over two (submit with --gpus=2); the
# capture processor writes from tensor-parallel rank 0 only.
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
bash -n slurm/mathfull_solo.sbatch

# The conditions are overlays, so the model block exists only after resolution.
MODEL_PATH=$(python - "$CONDITION" <<'PY'
import sys
from pathlib import Path
from authority_allocation.condition_resolution import resolve_condition
print(resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())["model"]["server_path"])
PY
)
test -d "$MODEL_PATH"

RESOLVED_NOW="$SCRATCH_ROOT/runs/.mathfull-resolved-${SLURM_JOB_ID:-$$}.yaml"
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
  echo "Resuming $RUN_STEM"
  "${RUNNER[@]}" reconcile "$WORK_ROOT/records/pass1.jsonl" \
    "$WORK_ROOT/records/native_logit_trace.jsonl"
else
  mkdir -p "$WORK_ROOT/records" "$WORK_ROOT/reports" "$WORK_ROOT/logs"
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
if [[ -n "$SHARD_INDEX" && "$SHARD_INDEX" != merge ]]; then
  MANIFEST="$WORK_ROOT/manifest_shard.jsonl"
  EXPECTED_ROWS=$("${RUNNER[@]}" shard-manifest "$FULL_MANIFEST" "$SHARD_INDEX" "$SHARD_COUNT" "$MANIFEST")
  echo "Shard $SHARD_INDEX/$SHARD_COUNT: $EXPECTED_ROWS rows"
elif [[ "$SHARD_INDEX" == merge ]] \
     && (( $(count_rows "$WORK_ROOT/records/pass1.jsonl") != EXPECTED_ROWS )); then
  for (( i = 0; i < SHARD_COUNT; i++ )); do
    cmp -s "$RESOLVED_NOW" "$SHARD_ROOT/$i/condition.resolved.yaml" \
      || { echo "shard $i ran under a different condition; refusing to merge" >&2; exit 1; }
    cp "$SHARD_ROOT/$i/logs/execution_hardware.txt" "$WORK_ROOT/logs/shard_${i}_execution_hardware.txt"
  done
  "${RUNNER[@]}" merge-shards "$FULL_MANIFEST" "$SHARD_ROOT" "$SHARD_COUNT" "$WORK_ROOT/records" \
    pass1.jsonl native_logit_trace.jsonl
fi
rm -f "$RESOLVED_NOW"

DONE_ROWS=$(count_rows "$WORK_ROOT/records/pass1.jsonl")
DONE_TRACES=$(count_rows "$WORK_ROOT/records/native_logit_trace.jsonl")
if (( DONE_ROWS == EXPECTED_ROWS && DONE_TRACES == EXPECTED_ROWS )); then
  echo "All $EXPECTED_ROWS rows already generated; skipping the model server."
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
"${RUNNER[@]}" generate --run-id "${RUN_STEM}__pass1__p01" \
  --api-base "$API" --served-model "$SERVED" --manifest "$MANIFEST" \
  --expected-rows "$EXPECTED_ROWS" \
  --output "$WORK_ROOT/records/pass1.jsonl" \
  --capture-sidecar "$SIDECAR" \
  --logit-output "$WORK_ROOT/records/native_logit_trace.jsonl" \
  --logit-run-id "${RUN_STEM}__same-forward-logits__p01"
stop_server
VLLM_PID=""
fi

test "$(wc -l < "$WORK_ROOT/records/pass1.jsonl")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$WORK_ROOT/records/native_logit_trace.jsonl")" -eq "$EXPECTED_ROWS"
if [[ -n "$SHARD_INDEX" && "$SHARD_INDEX" != merge ]]; then
  echo "$CONDITION_ID shard $SHARD_INDEX/$SHARD_COUNT generated: $WORK_ROOT"
  exit 0
fi

# Evidence is materialized before this process reads gold answers for grading. The
# post-generation stages are rewritten from scratch so a resume never appends to them.
rm -f "$WORK_ROOT/records/scalar_evidence.jsonl" "$WORK_ROOT/records/grades.jsonl"
"${RUNNER[@]}" evidence --run-id "${RUN_STEM}__evidence__e01" \
  --pass1-records "$WORK_ROOT/records/pass1.jsonl" \
  --logit-traces "$WORK_ROOT/records/native_logit_trace.jsonl" \
  --output "$WORK_ROOT/records/scalar_evidence.jsonl"
"${RUNNER[@]}" grade --pass1-records "$WORK_ROOT/records/pass1.jsonl" \
  --output "$WORK_ROOT/records/grades.jsonl"
"${RUNNER[@]}" analyze --run-id "${RUN_STEM}__analysis__r012" \
  --pass1-records "$WORK_ROOT/records/pass1.jsonl" \
  --evidence "$WORK_ROOT/records/scalar_evidence.jsonl" \
  --grades "$WORK_ROOT/records/grades.jsonl" \
  --expected-rows "$EXPECTED_ROWS" \
  --output "$WORK_ROOT/reports/analysis.json"

python scripts/validate_records.py --bundle-dir "$WORK_ROOT"
(cd "$WORK_ROOT" && sha256sum condition.resolved.yaml records/*.jsonl reports/* > checksums.sha256)
ARCHIVE="$PROJECT_ROOT/artifacts/$RUN_STEM"
test ! -e "$ARCHIVE"
mkdir "$ARCHIVE"
cp -a "$WORK_ROOT/." "$ARCHIVE/"
(cd "$ARCHIVE" && sha256sum -c checksums.sha256)
cat "$ARCHIVE/reports/summary.md"
echo "$CONDITION_ID complete: $ARCHIVE"
