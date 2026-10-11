#!/usr/bin/env bash
# JEV requires login-node network; source must be an immutable committed snapshot.
set -euo pipefail
SOURCE="${1:?immutable source required}"
OUT="${2:?new run output required}"
RESUME="${3:-}"
test -f "$SOURCE/SOURCE_COMMIT.txt"
test ! -e "$OUT"
mkdir -p "$OUT"
EXTRA=()
if [[ -n "$RESUME" ]]; then EXTRA=(--resume-cache "$RESUME"); fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup /path/to/project/.venv-20260917/bin/python \
  "$SOURCE/scripts/code_jev_dev832.py" --out "$OUT" "${EXTRA[@]}" \
  > "$OUT/runner.log" 2>&1 < /dev/null &
TASK_PID=$!
printf '%s\n' "$TASK_PID" > "$OUT/runner.pid"
printf 'code_jev pid=%s output=%s\n' "$TASK_PID" "$OUT"
