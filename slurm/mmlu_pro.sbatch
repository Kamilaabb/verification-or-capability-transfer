#!/usr/bin/env bash
# MMLU-Pro knowledge study on dev1000 (docs/plan/mmlu_pro_study.md): c070 Qwen3-4B alone,
# c071 Qwen3-8B alone, c072 Qwen3-8B reviewing c070, c073 Qwen3-4B reviewing c071.
# c074-c079: the other mathfull models alone (Llama-3.2-3B, Qwen2.5-3B, Qwen2.5-7B, Ministral-8B,
# Mistral-Small-24B, Qwen3-32B), each on one H100 as in mathfull_solo.sbatch.
# c080: Qwen2.5-72B-Instruct alone, over two H100s:
#   MMLU_TP=2 sbatch --gpus=2 --mem=200G -p gpu-h100 slurm/mmlu_pro.sbatch c080 <RUN_STEM>
# c081-c084: c070-c073 on the test1500 prefix of the sealed test (plan section 10).
# c085/c086: Qwen2.5-7B and Qwen3-4B review each other on dev1000 (plan section 11).
# c087/c088: Qwen3-8B and Mistral-Small-24B review each other on dev1000 (plan section 12).
# c089: Mistral-Small-24B alone on test1500, a dev/test stability check (plan section 12).
# c120-c125: Llama-3.2-3B, Qwen2.5-3B, Qwen2.5-7B, Ministral-8B, Qwen3-32B and Qwen2.5-72B alone on
#   test1500 (plan section 13); c125 over two H100s like c080.
# c126-c135: development reviews and c136-c147 their test1500 runs, the stratified pairs (plan section 15).
# c170-c175: development reviews and c176-c183 test1500 reviews, the pairs with a strong model (plan section 16).
# c184-c187: development reviews and c188-c191 test1500 reviews, the strong/strong cell (plan section 16);
#   c186 and c190 (Qwen2.5-72B reviewing) over two H100s like c080.
# c270-c273: development reviews and c274-c277 test1500 reviews, the original Qwen2.5-72B pairs
#   (plan section 20); c270, c272, c274, c276 (Qwen2.5-72B reviewing) over two H100s like c080.
# Usage: sbatch slurm/mmlu_pro.sbatch <c070-c089|c120-c147|c170-c191|c270-c277> <RUN_STEM>
#
# Resumable like mathfull_solo.sbatch: the same RUN_STEM continues the run after checking the
# resolved condition (and, for a review, the copied source records) are unchanged and
# reconciling the records with their native traces. Each job writes its own capture sidecar.
# A job that finds all rows generated starts no model server, so it can finish on the cpu
# partition (sbatch -p cpu --gpus=0 ...). The serving context comes from the resolved
# condition, never from this file.

#SBATCH --job-name=mmlu-pro
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/project/mmlu-pro-%j.out
#SBATCH --error=/path/to/project/mmlu-pro-%j.err

set -euo pipefail

CONDITION_ID="${1:?usage: sbatch mmlu_pro.sbatch <c070-c089|c120-c147|c170-c191|c270-c277> <RUN_STEM>}"
RUN_STEM="${2:?usage: sbatch mmlu_pro.sbatch <c070-c089|c120-c147|c170-c191|c270-c277> <RUN_STEM>}"
case "$CONDITION_ID" in
  c070) SERVED=qwen3-4b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c071) SERVED=qwen3-8b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c072) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c073) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c074) SERVED=llama-3.2-3b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c075) SERVED=qwen2.5-3b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c076) SERVED=qwen2.5-7b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c077) SERVED=ministral-8b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c078) SERVED=mistral-small-24b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c079) SERVED=qwen3-32b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c080) SERVED=qwen2.5-72b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c081) SERVED=qwen3-4b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c082) SERVED=qwen3-8b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c083) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c084) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c085) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c086) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c087) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c088) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c089) SERVED=mistral-small-24b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c120) SERVED=llama-3.2-3b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c121) SERVED=qwen2.5-3b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c122) SERVED=qwen2.5-7b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c123) SERVED=ministral-8b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c124) SERVED=qwen3-32b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c125) SERVED=qwen2.5-72b; ROLE=solve; RECORDS=pass1.jsonl ;;
  c126) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c127) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c128) SERVED=ministral-8b; ROLE=review; RECORDS=review.jsonl ;;
  c129) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c130) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c131) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c132) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c133) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c134) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c135) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c136) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c137) SERVED=qwen3-4b; ROLE=review; RECORDS=review.jsonl ;;
  c138) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c139) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c140) SERVED=ministral-8b; ROLE=review; RECORDS=review.jsonl ;;
  c141) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c142) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c143) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c144) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c145) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c146) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c147) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c392) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c393) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c388) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c389) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c170) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c171) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c172) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c173) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c174) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c175) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c176) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c177) SERVED=qwen3-8b; ROLE=review; RECORDS=review.jsonl ;;
  c178) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c179) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c180) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c181) SERVED=qwen2.5-3b; ROLE=review; RECORDS=review.jsonl ;;
  c182) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c183) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c184) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c185) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c186) SERVED=qwen2.5-72b; ROLE=review; RECORDS=review.jsonl ;;
  c187) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c188) SERVED=qwen3-32b; ROLE=review; RECORDS=review.jsonl ;;
  c189) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c190) SERVED=qwen2.5-72b; ROLE=review; RECORDS=review.jsonl ;;
  c191) SERVED=mistral-small-24b; ROLE=review; RECORDS=review.jsonl ;;
  c270) SERVED=qwen2.5-72b; ROLE=review; RECORDS=review.jsonl ;;
  c271) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c272) SERVED=qwen2.5-72b; ROLE=review; RECORDS=review.jsonl ;;
  c273) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  c274) SERVED=qwen2.5-72b; ROLE=review; RECORDS=review.jsonl ;;
  c275) SERVED=qwen2.5-7b; ROLE=review; RECORDS=review.jsonl ;;
  c276) SERVED=qwen2.5-72b; ROLE=review; RECORDS=review.jsonl ;;
  c277) SERVED=llama-3.2-3b; ROLE=review; RECORDS=review.jsonl ;;
  *) echo "unknown condition $CONDITION_ID" >&2; exit 2 ;;
esac
PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
WORK_ROOT="$SCRATCH_ROOT/runs/$RUN_STEM"
CONTAINER="$SCRATCH_ROOT/containers/vllm-openai-v0.29.0-rebuilt-20260914.sif"
RUNNER=(python "$PROJECT_ROOT/scripts/run_mmlu_pro.py" --condition-id "$CONDITION_ID")
CONDITION="$PROJECT_ROOT/conditions/$CONDITION_ID/condition.yaml"
MANIFEST_DIR="$PROJECT_ROOT/data/manifests/mmlupro-rb189ec7-s20260927"
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
# The manifest, its hash and row count come from the resolved condition (dev1000 or test1500).
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
bash -n slurm/mmlu_pro.sbatch

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

RESOLVED_NOW="$SCRATCH_ROOT/runs/.mmlu-pro-resolved-${SLURM_JOB_ID:-$$}.yaml"
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
