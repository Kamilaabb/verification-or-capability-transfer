#!/usr/bin/env bash
# BigCodeBench code study, each model alone on the 832 development rows
# (docs/plan/code_bigcodebench_study.md): c100-c109. GPU part only: generation with
# same-forward capture, then scalar evidence. Grading and analysis run afterwards on the cpu
# partition in slurm/bigcodebench_solo_grade.sbatch, so no GPU waits on the test harness.
# Usage: sbatch slurm/bigcodebench_solo.sbatch <c100-c109> <RUN_STEM>
#   c106 (Mistral-Small-24B) and c107 (Qwen3-32B) need an 80 GB H100: submit with -p gpu-h100.
#
# Resumable like mmlu_pro.sbatch: the same RUN_STEM continues the run after checking the
# resolved condition is unchanged and reconciling the records with their native traces. A job
# that finds all rows generated starts no model server. The serving context comes from the
# resolved condition, never from this file.

#SBATCH --job-name=bcb-solo
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/project/bcb-solo-%j.out
#SBATCH --error=/path/to/project/bcb-solo-%j.err

set -euo pipefail

CONDITION_ID="${1:?usage: sbatch bigcodebench_solo.sbatch <c100-c109> <RUN_STEM>}"
RUN_STEM="${2:?usage: sbatch bigcodebench_solo.sbatch <c100-c109> <RUN_STEM>}"
case "$CONDITION_ID" in
  c100) SERVED=qwen3-4b ;;
  c101) SERVED=qwen3-8b ;;
  c102) SERVED=llama-3.2-3b ;;
  c103) SERVED=qwen2.5-3b ;;
  c104) SERVED=qwen2.5-7b ;;
  c105) SERVED=ministral-8b ;;
  c106) SERVED=mistral-small-24b ;;
  c107) SERVED=qwen3-32b ;;
  c108) SERVED=qwen2.5-coder-7b ;;
  c109) SERVED=seed-coder-8b ;;
  *) echo "unknown condition $CONDITION_ID" >&2; exit 2 ;;
esac
PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
WORK_ROOT="$SCRATCH_ROOT/runs/$RUN_STEM"
CONTAINER="$SCRATCH_ROOT/containers/vllm-openai-v0.29.0-rebuilt-20260914.sif"
RUNNER=(python "$PROJECT_ROOT/scripts/run_bigcodebench_solo.py" --condition-id "$CONDITION_ID")
CONDITION="$PROJECT_ROOT/conditions/$CONDITION_ID/condition.yaml"
MANIFEST="$PROJECT_ROOT/data/manifests/bigcodebench-b74c0d0-instruct-s20260917/development.jsonl"
MANIFEST_SHA=5fc23a137b20357ca254f4a73fc9d72d4bca24adb05ca03b947a92d4917b4a30
EXPECTED_ROWS=832
GPU_MEMORY_UTILIZATION="${BCB_GPU_MEMORY_UTILIZATION:-0.90}"
SIDECAR="$WORK_ROOT/logs/vllm_capture_sidecar_${SLURM_JOB_ID:-local}.jsonl"
RECORDS="$WORK_ROOT/records/pass1.jsonl"
TRACES="$WORK_ROOT/records/native_logit_trace.jsonl"

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
echo "$MANIFEST_SHA  $MANIFEST" | sha256sum -c -
bash -n slurm/bigcodebench_solo.sbatch

read -r MODEL_PATH MAX_MODEL_LEN < <(python - "$CONDITION" <<'PY'
import sys
from pathlib import Path
from authority_allocation.condition_resolution import resolve_condition
m = resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())["model"]
print(m["server_path"], m["serving"]["max_model_len"])
PY
)
test -d "$MODEL_PATH"

RESOLVED_NOW="$SCRATCH_ROOT/runs/.bcb-solo-resolved-${SLURM_JOB_ID:-$$}.yaml"
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
  "${RUNNER[@]}" reconcile "$RECORDS" "$TRACES"
else
  mkdir -p "$WORK_ROOT/records" "$WORK_ROOT/reports" "$WORK_ROOT/logs"
  cp "$RESOLVED_NOW" "$WORK_ROOT/condition.resolved.yaml"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK_ROOT/started_at_utc"
fi
rm -f "$RESOLVED_NOW"
{
  echo "slurm_job_id=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-}" \
       "node=$(hostname) started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "nvidia-smi failed"
} >> "$WORK_ROOT/logs/execution_hardware.txt"

count_rows() { if [[ -f "$1" ]]; then wc -l < "$1"; else echo 0; fi; }
if (( $(count_rows "$RECORDS") == EXPECTED_ROWS && $(count_rows "$TRACES") == EXPECTED_ROWS )); then
  echo "All $EXPECTED_ROWS rows already generated; skipping the model server."
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
# step; a chunked prefill stops the logit-capture processor (c018).
# Each job serves on its own free port (see mathfull_solo.sbatch).
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
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enforce-eager \
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

"${RUNNER[@]}" smoke --run-id "${RUN_STEM}__smoke" --api-base "$API" --served-model "$SERVED" \
  --capture-sidecar "$SIDECAR"
"${RUNNER[@]}" generate --run-id "${RUN_STEM}__pass1__p01" \
  --api-base "$API" --served-model "$SERVED" --manifest "$MANIFEST" \
  --expected-rows "$EXPECTED_ROWS" --output "$RECORDS" --capture-sidecar "$SIDECAR" \
  --logit-output "$TRACES" --logit-run-id "${RUN_STEM}__same-forward-logits__p01"
stop_server
VLLM_PID=""
fi

test "$(wc -l < "$RECORDS")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$TRACES")" -eq "$EXPECTED_ROWS"

# Evidence is materialized before anything reads a test result. Rewritten from scratch so a
# resume never appends to it.
rm -f "$WORK_ROOT/records/scalar_evidence.jsonl"
"${RUNNER[@]}" evidence --run-id "${RUN_STEM}__evidence__e01" --records "$RECORDS" \
  --logit-traces "$TRACES" --output "$WORK_ROOT/records/scalar_evidence.jsonl"
test "$(wc -l < "$WORK_ROOT/records/scalar_evidence.jsonl")" -eq "$EXPECTED_ROWS"
echo "$CONDITION_ID generation and evidence complete: $WORK_ROOT (grade with bigcodebench_solo_grade.sbatch)"
