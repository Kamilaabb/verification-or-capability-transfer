#!/usr/bin/env bash
# Natural-language-reasoning domain (docs/plan/nl_reasoning_study.md): ReClor validation, FOLIO v2
# validation and GPQA Diamond pooled into one 901-row manifest, one pre-chosen pair.
#   c192 Llama-3.2-3B alone, c193 Qwen3-8B alone, c194 Qwen3-8B reviewing c192, c195 Llama-3.2-3B reviewing c193;
#   c196 Qwen3-4B alone, c197 Qwen3-8B reviewing c196, c198 Qwen3-4B reviewing c193;
#   c303-c307 solo runs of Qwen2.5-3B, Qwen2.5-7B, Ministral-8B, Mistral-Small-24B, Qwen3-32B;
#   c308-c315 reviews of four more pairs, c316-c359 reviews of the other 22 pairs (see scripts/run_nl_reasoning.py).
# Usage: sbatch slurm/nl_reasoning.sbatch <c192-c198|c303-c359> <RUN_STEM>
#
# This is slurm/mmlu_pro.sbatch with only the condition table, the runner (scripts/run_nl_reasoning.py,
# which reuses the MMLU-Pro runner's code with this dataset) and the manifest directory changed.
# Resumable the same way: the same RUN_STEM continues the run after checking the resolved condition
# (and, for a review, the copied source records) are unchanged.

#SBATCH --job-name=nl-reason
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/project/nl-reason-%j.out
#SBATCH --error=/path/to/project/nl-reason-%j.err

set -euo pipefail

CONDITION_ID="${1:?usage: sbatch nl_reasoning.sbatch <c192-c198|c303-c359> <RUN_STEM>}"
RUN_STEM="${2:?usage: sbatch nl_reasoning.sbatch <c192-c198|c303-c359> <RUN_STEM>}"
case "$CONDITION_ID" in
  c192) SERVED=llama-3.2-3b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c193) SERVED=qwen3-8b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c194) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c195) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c196) SERVED=qwen3-4b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c197) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c198) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c303) SERVED=qwen2.5-3b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c304) SERVED=qwen2.5-7b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c305) SERVED=ministral-8b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c306) SERVED=mistral-small-24b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c307) SERVED=qwen3-32b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c308) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c309) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c310) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c311) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c312) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c313) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c314) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c315) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c316) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c317) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c318) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c319) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c320) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c321) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c322) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c323) SERVED=ministral-8b; ROLE=review; RECORDS=review.jsonl ;;
  c324) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c325) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c326) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c327) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c328) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c329) SERVED=ministral-8b; ROLE=review; RECORDS=review.jsonl ;;
  c330) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c331) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c332) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c333) SERVED=ministral-8b; ROLE=review; RECORDS=review.jsonl ;;
  c334) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c335) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c336) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c337) SERVED=ministral-8b; ROLE=review; RECORDS=review.jsonl ;;
  c338) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c339) SERVED=ministral-8b; ROLE=review; RECORDS=review.jsonl ;;
  c340) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c341) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c342) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c343) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c344) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c345) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c346) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c347) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c348) SERVED=ministral-8b; ROLE=review; RECORDS=review.jsonl ;;
  c349) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c350) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c351) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c352) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c353) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c354) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c355) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c356) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c357) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c358) SERVED=ministral-8b; ROLE=review; RECORDS=review.jsonl ;;
  c359) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  *) echo "unknown condition $CONDITION_ID" >&2; exit 2 ;;
esac
PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
WORK_ROOT="$SCRATCH_ROOT/runs/$RUN_STEM"
CONTAINER="$SCRATCH_ROOT/containers/vllm-openai-v0.29.0-rebuilt-20260914.sif"
RUNNER=(python "$PROJECT_ROOT/scripts/run_nl_reasoning.py" --condition-id "$CONDITION_ID")
CONDITION="$PROJECT_ROOT/conditions/$CONDITION_ID/condition.yaml"
MANIFEST_DIR="$PROJECT_ROOT/data/manifests/nlreason-gpqad-reclorval-foliov2-s20261001"
GPU_MEMORY_UTILIZATION="${MMLU_GPU_MEMORY_UTILIZATION:-0.90}"
# MMLU_TP=2 serves a model too large for one GPU over two (submit with --gpus=2); the
# capture processor writes from tensor-parallel rank 0 only.
TP="${MMLU_TP:-1}"
SIDECAR="$WORK_ROOT/logs/vllm_capture_sidecar_${SLURM_JOB_ID:-local}.jsonl"

cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"

module purge
module load python
source .venv-20260917/bin/activate
export HF_HOME=/path/to/scratch/huggingface
export HF_DATASETS_CACHE=/path/to/scratch/huggingface/datasets
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
# The manifest, its hash and row count come from the resolved condition (eval, 901 rows).
read -r MANIFEST_NAME MANIFEST_SHA EXPECTED_ROWS < <(python - "$CONDITION" <<'PY'
import sys
from pathlib import Path
from authority_allocation.condition_resolution import resolve_condition
d = resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())["dataset"]
print(d["manifest"], d["manifest_sha256"], d["manifest_rows"])
PY
)
MANIFEST="$MANIFEST_DIR/$MANIFEST_NAME.jsonl"
python -m unittest discover -s tests
"${RUNNER[@]}" check-runnable
echo "$MANIFEST_SHA  $MANIFEST" | sha256sum -c -
bash -n slurm/nl_reasoning.sbatch

# Model path and serving context, from the resolved condition only.
read -r MODEL_PATH MAX_MODEL_LEN < <(python - "$CONDITION" <<'PY'
import sys
from pathlib import Path
from authority_allocation.condition_resolution import resolve_condition
m = resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())["model"]
print(m["server_path"], m["serving"]["max_model_len"])
PY
)
test -d "$MODEL_PATH"

if [[ "$ROLE" == review ]]; then
  read -r G_SOURCE < <(python - "$CONDITION" <<'PY'
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
fi

RESOLVED_NOW="$SCRATCH_ROOT/runs/.nl-reason-resolved-${SLURM_JOB_ID:-$$}.yaml"
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
  if [[ "$ROLE" == review ]]; then
    for pair in "${G_COPIES[@]}"; do
      set -- $pair
      cmp -s "$G_SOURCE/records/$1" "$WORK_ROOT/records/$2" \
        || { echo "Copied G record $2 differs from the archive; refusing to resume." >&2; exit 1; }
    done
  fi
  echo "Resuming $RUN_STEM"
  "${RUNNER[@]}" reconcile "$WORK_ROOT/records/$RECORDS" "$WORK_ROOT/records/native_logit_trace.jsonl"
else
  mkdir -p "$WORK_ROOT/records" "$WORK_ROOT/reports" "$WORK_ROOT/logs"
  if [[ "$ROLE" == review ]]; then
    for pair in "${G_COPIES[@]}"; do
      set -- $pair
      cp "$G_SOURCE/records/$1" "$WORK_ROOT/records/$2"
    done
  fi
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
DONE_ROWS=$(count_rows "$WORK_ROOT/records/$RECORDS")
DONE_TRACES=$(count_rows "$WORK_ROOT/records/native_logit_trace.jsonl")
if (( DONE_ROWS == EXPECTED_ROWS && DONE_TRACES == EXPECTED_ROWS )); then
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

# --max-num-batched-tokens equals --max-model-len so every prompt is
# prefilled in one step; a chunked prefill stops the logit-capture processor (c018).
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

"${RUNNER[@]}" smoke --run-id "${RUN_STEM}__smoke" --api-base "$API" --served-model "$SERVED" \
  --capture-sidecar "$SIDECAR"
if [[ "$ROLE" == review ]]; then
  "${RUNNER[@]}" review --run-id "${RUN_STEM}__review__p01" \
    --api-base "$API" --served-model "$SERVED" --manifest "$MANIFEST" \
    --expected-rows "$EXPECTED_ROWS" --g-records "$WORK_ROOT/records/g_pass1.jsonl" \
    --output "$WORK_ROOT/records/review.jsonl" --capture-sidecar "$SIDECAR" \
    --logit-output "$WORK_ROOT/records/native_logit_trace.jsonl" \
    --logit-run-id "${RUN_STEM}__same-forward-logits__p01"
else
  "${RUNNER[@]}" generate --run-id "${RUN_STEM}__pass1__p01" \
    --api-base "$API" --served-model "$SERVED" --manifest "$MANIFEST" \
    --expected-rows "$EXPECTED_ROWS" \
    --output "$WORK_ROOT/records/pass1.jsonl" --capture-sidecar "$SIDECAR" \
    --logit-output "$WORK_ROOT/records/native_logit_trace.jsonl" \
    --logit-run-id "${RUN_STEM}__same-forward-logits__p01"
fi
stop_server
VLLM_PID=""
fi

test "$(wc -l < "$WORK_ROOT/records/$RECORDS")" -eq "$EXPECTED_ROWS"
test "$(wc -l < "$WORK_ROOT/records/native_logit_trace.jsonl")" -eq "$EXPECTED_ROWS"

# Evidence is materialized before this process reads reference answers for grading. The
# post-generation stages are rewritten from scratch so a resume never appends to them.
rm -f "$WORK_ROOT/records/scalar_evidence.jsonl" "$WORK_ROOT/records/grades.jsonl"
"${RUNNER[@]}" evidence --run-id "${RUN_STEM}__evidence__e01" \
  --records "$WORK_ROOT/records/$RECORDS" \
  --logit-traces "$WORK_ROOT/records/native_logit_trace.jsonl" \
  --output "$WORK_ROOT/records/scalar_evidence.jsonl"
G_ARGS=()
if [[ "$ROLE" == review ]]; then
  G_ARGS=(--g-records "$WORK_ROOT/records/g_pass1.jsonl")
fi
"${RUNNER[@]}" grade --records "$WORK_ROOT/records/$RECORDS" ${G_ARGS[@]+"${G_ARGS[@]}"} \
  --output "$WORK_ROOT/records/grades.jsonl"
if [[ "$ROLE" == review ]]; then
  G_ARGS+=(--g-grades "$WORK_ROOT/records/g_grades.jsonl")
fi
"${RUNNER[@]}" analyze --run-id "${RUN_STEM}__analysis__r014" \
  --records "$WORK_ROOT/records/$RECORDS" ${G_ARGS[@]+"${G_ARGS[@]}"} \
  --evidence "$WORK_ROOT/records/scalar_evidence.jsonl" \
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
