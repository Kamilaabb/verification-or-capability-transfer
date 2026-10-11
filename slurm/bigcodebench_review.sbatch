#!/usr/bin/env bash
# GV on BigCodeBench development (docs/plan/code_bigcodebench_study.md), pair Qwen3-4B x
# Ministral-8B: c110 Ministral-8B reviews c100, c111 Qwen3-4B reviews c105. GPU part only:
# review with same-forward capture, then scalar evidence. Grading and analysis run afterwards
# on the cpu partition in slurm/bigcodebench_review_grade.sbatch.
# c112 Qwen3-8B reviews c100, c113 Qwen3-4B reviews c101 (the math/knowledge pair, plan section 10);
# c114 replaces c113 with the empty-token-bytes repair.
# c199 Qwen3-8B reviews c102, c148 Llama-3.2-3B reviews c101 (the large-gap pair, decided 2026-10-01).
# c149/c278 Qwen2.5-Coder-7B x Qwen2.5-7B, c279/c280 Seed-Coder-8B x Qwen3-8B (plan section 13, 2026-10-02).
# c281-c300: the stratified code pairs (plan section 14, 2026-10-02); 24B/32B reviewers need an H100.
# c301 replaces c279, and c302 replaces c297, with the empty-token-bytes repair.
# Usage: sbatch slurm/bigcodebench_review.sbatch <c110-c114|c148|c149|c199|c278-c302> <RUN_STEM>
#
# Resumable: the same RUN_STEM continues after checking the resolved condition and the copied
# source records are unchanged and reconciling the records with their native traces.

#SBATCH --job-name=bcb-review
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/project/bcb-review-%j.out
#SBATCH --error=/path/to/project/bcb-review-%j.err

set -euo pipefail

CONDITION_ID="${1:?usage: sbatch bigcodebench_review.sbatch <c110-c114|c148|c149|c199|c278-c302> <RUN_STEM>}"
RUN_STEM="${2:?usage: sbatch bigcodebench_review.sbatch <c110-c114|c148|c149|c199|c278-c302> <RUN_STEM>}"
case "$CONDITION_ID" in
  c110) SERVED=ministral-8b ;;
  c111) SERVED=qwen3-4b ;;
  c112) SERVED=qwen3-8b ;;
  c113) SERVED=qwen3-4b ;;
  c114) SERVED=qwen3-4b ;;
  c148) SERVED=llama-3.2-3b ;;
  c199) SERVED=qwen3-8b ;;
  c149) SERVED=qwen2.5-coder-7b ;;
  c278) SERVED=qwen2.5-7b ;;
  c279) SERVED=seed-coder-8b ;;
  c280) SERVED=qwen3-8b ;;
  c281) SERVED=mistral-small-24b ;;
  c282) SERVED=qwen3-32b ;;
  c283) SERVED=mistral-small-24b ;;
  c284) SERVED=seed-coder-8b ;;
  c285) SERVED=ministral-8b ;;
  c286) SERVED=qwen2.5-coder-7b ;;
  c287) SERVED=qwen2.5-3b ;;
  c288) SERVED=qwen2.5-7b ;;
  c289) SERVED=llama-3.2-3b ;;
  c290) SERVED=qwen2.5-7b ;;
  c291) SERVED=mistral-small-24b ;;
  c292) SERVED=qwen3-8b ;;
  c293) SERVED=mistral-small-24b ;;
  c294) SERVED=ministral-8b ;;
  c295) SERVED=qwen3-32b ;;
  c296) SERVED=qwen2.5-7b ;;
  c297) SERVED=seed-coder-8b ;;
  c298) SERVED=llama-3.2-3b ;;
  c299) SERVED=qwen3-4b ;;
  c300) SERVED=qwen2.5-7b ;;
  c301) SERVED=seed-coder-8b ;;
  c302) SERVED=seed-coder-8b ;;
  *) echo "unknown condition $CONDITION_ID" >&2; exit 2 ;;
esac
PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
WORK_ROOT="$SCRATCH_ROOT/runs/$RUN_STEM"
CONTAINER="$SCRATCH_ROOT/containers/vllm-openai-v0.29.0-rebuilt-20260914.sif"
RUNNER=(python "$PROJECT_ROOT/scripts/run_bigcodebench_review.py" --condition-id "$CONDITION_ID")
CONDITION="$PROJECT_ROOT/conditions/$CONDITION_ID/condition.yaml"
MANIFEST="$PROJECT_ROOT/data/manifests/bigcodebench-b74c0d0-instruct-s20260917/development.jsonl"
MANIFEST_SHA=5fc23a137b20357ca254f4a73fc9d72d4bca24adb05ca03b947a92d4917b4a30
EXPECTED_ROWS=832
GPU_MEMORY_UTILIZATION="${BCB_GPU_MEMORY_UTILIZATION:-0.90}"
SIDECAR="$WORK_ROOT/logs/vllm_capture_sidecar_${SLURM_JOB_ID:-local}.jsonl"
RECORDS="$WORK_ROOT/records/review.jsonl"
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
bash -n slurm/bigcodebench_review.sbatch

read -r MODEL_PATH MAX_MODEL_LEN G_SOURCE < <(python - "$CONDITION" <<'PY'
import sys
from pathlib import Path
from authority_allocation.condition_resolution import resolve_condition
c = resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())
print(c["model"]["server_path"], c["model"]["serving"]["max_model_len"], c["generator_source"]["artifact_path"])
PY
)
test -d "$MODEL_PATH"
test -d "$G_SOURCE"
(cd "$G_SOURCE" && sha256sum -c --quiet checksums.sha256)
test "$(wc -l < "$G_SOURCE/records/pass1.jsonl")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$G_SOURCE/records/native_logit_trace.jsonl")" -eq "$EXPECTED_ROWS"

RESOLVED_NOW="$SCRATCH_ROOT/runs/.bcb-review-resolved-${SLURM_JOB_ID:-$$}.yaml"
python - "$CONDITION" "$RESOLVED_NOW" <<'PY'
from authority_allocation.condition_resolution import resolve_condition
from pathlib import Path
import sys, yaml
resolved = resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())
Path(sys.argv[2]).write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
PY

G_COPIES=("pass1.jsonl g_pass1.jsonl" "native_logit_trace.jsonl g_native_logit_trace.jsonl" \
          "grades.jsonl g_grades.jsonl")
if [[ -e "$WORK_ROOT" ]]; then
  if ! cmp -s "$RESOLVED_NOW" "$WORK_ROOT/condition.resolved.yaml"; then
    echo "Condition differs from the one this run started under; refusing to resume." >&2
    exit 1
  fi
  for pair in "${G_COPIES[@]}"; do
    set -- $pair
    cmp -s "$G_SOURCE/records/$1" "$WORK_ROOT/records/$2" \
      || { echo "Copied source record $2 differs from the archive; refusing to resume." >&2; exit 1; }
  done
  echo "Resuming $RUN_STEM"
  "${RUNNER[@]}" reconcile "$RECORDS" "$TRACES"
else
  mkdir -p "$WORK_ROOT/records" "$WORK_ROOT/reports" "$WORK_ROOT/logs"
  for pair in "${G_COPIES[@]}"; do
    set -- $pair
    cp "$G_SOURCE/records/$1" "$WORK_ROOT/records/$2"
  done
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
  echo "All $EXPECTED_ROWS rows already reviewed; skipping the model server."
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
kill -0 "$VLLM_PID"

"${RUNNER[@]}" smoke --run-id "${RUN_STEM}__smoke" --api-base "$API" --served-model "$SERVED" \
  --capture-sidecar "$SIDECAR"
"${RUNNER[@]}" review --run-id "${RUN_STEM}__review__p01" \
  --api-base "$API" --served-model "$SERVED" --manifest "$MANIFEST" \
  --expected-rows "$EXPECTED_ROWS" --g-records "$WORK_ROOT/records/g_pass1.jsonl" \
  --output "$RECORDS" --capture-sidecar "$SIDECAR" \
  --logit-output "$TRACES" --logit-run-id "${RUN_STEM}__same-forward-logits__p01"
stop_server
VLLM_PID=""
fi

test "$(wc -l < "$RECORDS")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$TRACES")" -eq "$EXPECTED_ROWS"

rm -f "$WORK_ROOT/records/scalar_evidence.jsonl"
"${RUNNER[@]}" evidence --run-id "${RUN_STEM}__evidence__e01" --records "$RECORDS" \
  --logit-traces "$TRACES" --output "$WORK_ROOT/records/scalar_evidence.jsonl"
test "$(wc -l < "$WORK_ROOT/records/scalar_evidence.jsonl")" -eq "$EXPECTED_ROWS"
echo "$CONDITION_ID review and evidence complete: $WORK_ROOT (grade with bigcodebench_review_grade.sbatch)"
