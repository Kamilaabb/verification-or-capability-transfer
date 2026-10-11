#!/bin/bash
# Thinking-mode sanity check (docs/plan/thinking_check_test500.md): Qwen3-8B with thinking on, alone and
# reviewing c068 / c097, on the first 500 test1500 math questions. Serves the frozen c092 snapshot without
# the logit-capture processor, with concurrent requests and vLLM's qwen3 reasoning parser, then grades and
# compares with the thinking-off runs. Resumes from rows already written under the same RUN_STEM.
#
#   sbatch slurm/thinking_check_test500.sbatch <RUN_STEM>
#
#SBATCH --job-name=thinking-check
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/project/thinking-check-%j.out
#SBATCH --error=/path/to/project/thinking-check-%j.err

set -euo pipefail

RUN_STEM="${1:?usage: sbatch thinking_check_test500.sbatch <RUN_STEM>}"
PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
WORK_ROOT="$SCRATCH_ROOT/runs/$RUN_STEM"
CONTAINER="$SCRATCH_ROOT/containers/vllm-openai-v0.29.0-rebuilt-20260914.sif"
SERVED=qwen3-8b
# The model's maximum window; a review prompt plus 32,768 new tokens fits in it.
MAX_MODEL_LEN=40960

cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"
COMMIT=$(git rev-parse HEAD)

module purge
module load python
source .venv-20260917/bin/activate
export HF_HOME=/path/to/scratch/huggingface
export HF_DATASETS_CACHE=/path/to/scratch/huggingface/datasets
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
bash -n slurm/thinking_check_test500.sbatch

# The manifest is c236's, checked against its registered hash; the G archives pass their checksums.
MANIFEST_INFO=$(python scripts/run_mathfull_review.py --condition-id c236 manifest-info)
read -r MANIFEST MANIFEST_SHA EXPECTED_ROWS <<< "$MANIFEST_INFO"
echo "$MANIFEST_SHA  $MANIFEST" | sha256sum -c -
for CID in c236 c253; do
  G_SOURCE=$(python - "conditions/$CID/condition.yaml" <<'PY'
import sys
from pathlib import Path
from authority_allocation.condition_resolution import resolve_condition
print(resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())["generator_source"]["artifact_path"])
PY
)
  (cd "$G_SOURCE" && sha256sum -c --quiet checksums.sha256)
done
MODEL_PATH=$(python - conditions/c092/condition.yaml <<'PY'
import sys
from pathlib import Path
from authority_allocation.condition_resolution import resolve_condition
print(resolve_condition(Path(sys.argv[1]), repository_root=Path.cwd())["model"]["server_path"])
PY
)
test -d "$MODEL_PATH"

mkdir -p "$WORK_ROOT/records" "$WORK_ROOT/logs"
{
  echo "commit=$COMMIT slurm_job_id=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-}" \
       "node=$(hostname) started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi failed"
} >> "$WORK_ROOT/logs/execution_hardware.txt"
# GPUs before Ampere (the gpu-rtx Quadro RTX 8000, Turing) have no bfloat16: serve in float16 there and say so
# on every row it writes.
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if (( ${COMPUTE_CAP%%.*} < 8 )); then
  DTYPE=float16
  SERVER_LABEL="vllm-cluster-float16-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' -)"
else
  DTYPE=bfloat16
  SERVER_LABEL=vllm-cluster
fi
echo "compute_cap=$COMPUTE_CAP dtype=$DTYPE server_label=$SERVER_LABEL" >> "$WORK_ROOT/logs/execution_hardware.txt"

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

PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
API="http://127.0.0.1:$PORT/v1"
apptainer exec --cleanenv --nv \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env VLLM_CACHE_ROOT=/mnt/vllm-cache --env TRITON_CACHE_DIR=/mnt/vllm-cache/triton \
  --bind "$MODEL_PATH:/mnt/model:ro" \
  --bind "$SCRATCH_ROOT/vllm-cache:/mnt/vllm-cache" \
  "$CONTAINER" vllm serve /mnt/model \
  --served-model-name "$SERVED" --dtype "$DTYPE" --reasoning-parser qwen3 \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 96 --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port "$PORT" > "$WORK_ROOT/logs/server_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -80 "$WORK_ROOT/logs/server_${SLURM_JOB_ID:-local}.log"; exit 1; }
  if python -c 'import sys; from urllib import request; r=request.urlopen(sys.argv[1] + "/models", timeout=2); raise SystemExit(0 if r.status==200 else 1)' "$API" 2>/dev/null; then
    break
  fi
  sleep 5
done
python -c 'import sys; from urllib import request; request.urlopen(sys.argv[1] + "/models", timeout=2)' "$API"
kill -0 "$VLLM_PID"

python scripts/thinking_check_test500.py generate --api-base "$API" --served-model "$SERVED" \
  --records "$WORK_ROOT/records" --concurrency 96 --server-label "$SERVER_LABEL"
stop_server

OUT=/path/to/scratch/claims/$(date -u +%Y%m%d)_thinking_check_test500_${SLURM_JOB_ID:-local}
python scripts/thinking_check_test500.py analyze --records "$WORK_ROOT/records" \
  --artifacts "$PROJECT_ROOT/artifacts" --out "$OUT"
(cd "$WORK_ROOT" && sha256sum records/*.jsonl > checksums.sha256)
echo "thinking check complete: $WORK_ROOT; report $OUT/reports/summary.md"
