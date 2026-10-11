#!/usr/bin/env bash
# Start only the currently authorized fixed sweep, on the network-enabled login node.
set -euo pipefail
SOURCE="${1:?immutable source required}"
OUT="${2:?new output root required}"
RESUME="${3:-}"
EXTRA=()
if [[ -n "$RESUME" ]]; then EXTRA=(--resume-cache "$RESUME"); fi
test -f "$SOURCE/SOURCE_COMMIT.txt"
test ! -e "$OUT"
mkdir -p "$OUT"
cp "$SOURCE/SOURCE_COMMIT.txt" "$OUT/"
export HF_HOME=/path/to/scratch/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for DOMAIN in math knowledge; do
  nohup /path/to/project/.venv-20260917/bin/python \
    "$SOURCE/scripts/jev_domain_sweep.py" --domain "$DOMAIN" --split test --workers 2 --out "$OUT/$DOMAIN" "${EXTRA[@]}" \
    > "$OUT/$DOMAIN.log" 2>&1 < /dev/null &
  TASK_PID=$!
  printf '%s\n' "$TASK_PID" > "$OUT/$DOMAIN.pid"
  printf '%s pid=%s\n' "$DOMAIN" "$TASK_PID"
done
