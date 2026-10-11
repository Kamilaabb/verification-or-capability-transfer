#!/usr/bin/env bash
# Validate the BigCodeBench grader and measure its noise floor.
#
# Runs every canonical solution twice — networked and with the network namespace
# removed — so that both the grader's own failure rate and the set of tasks that
# silently need network access are measured before any research run.
#
# Measured on 2026-09-16 (jobs 65076891 / 65081056):
#   online  1136/1140 = 0.9965
#   offline 1132/1140 = 0.9930
#   network-dependent: BigCodeBench/101, 176, 314, 1012
#   ground truth already failing: BigCodeBench/205, 363, 590, 1005
# Wall clock is about 5 minutes per pass at 12 workers.
#
# NEVER pipe the apptainer call's stdout. untrusted_check forks children that inherit
# stdout, so a downstream `tail` never sees EOF and the job hangs for hours after its
# work has finished (that cost job 65076891 3.5 idle hours on 16 CPUs).

#SBATCH --job-name=bcb-gt
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/project/bcb-gt-%j.out
#SBATCH --error=/path/to/project/bcb-gt-%j.err

set -uo pipefail

PROJECT_ROOT=/path/to/project
SCRATCH_ROOT=/path/to/scratch/authority-allocation
SIF="$SCRATCH_ROOT/containers/bigcodebench-evaluate.sif"
WORK="$SCRATCH_ROOT/bcb-gt"
WORKERS=12

module purge
module load gcc/11.5.0
module load apptainer/1.4.1

mkdir -p "$WORK/hf"
cp "$PROJECT_ROOT/scripts/bigcodebench_ground_truth.py" "$WORK/driver.py"

COMMON=(--cleanenv --bind "$WORK:/work" --env HF_HOME=/work/hf
  --env HF_DATASETS_CACHE=/work/hf/datasets --env TMPDIR=/tmp)
OFFLINE_ENV=(--env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1)

echo "=============== warm the dataset cache (network on) ==============="
apptainer exec "${COMMON[@]}" "$SIF" python3 -c '
from bigcodebench.data import get_bigcodebench
problems = get_bigcodebench(subset="full")
print("tasks:", len(problems))
' > "$WORK/warm.txt" 2>&1
echo "warm exit: $?"
cat "$WORK/warm.txt"

echo
echo "=============== ONLINE ==============="
timeout 7200 apptainer exec "${COMMON[@]}" "$SIF" \
  python3 /work/driver.py --output /work/gt_online.jsonl --workers "$WORKERS" \
  > "$WORK/online.txt" 2>&1
echo "online exit: $?"
tail -8 "$WORK/online.txt"
pkill -f "/work/driver.py" 2>/dev/null || true

echo
echo "=============== OFFLINE (--net --network=none) ==============="
timeout 7200 apptainer exec --net --network=none "${COMMON[@]}" "${OFFLINE_ENV[@]}" "$SIF" \
  python3 /work/driver.py --output /work/gt_offline.jsonl --workers "$WORKERS" \
  > "$WORK/offline.txt" 2>&1
echo "offline exit: $?"
tail -8 "$WORK/offline.txt"
pkill -f "/work/driver.py" 2>/dev/null || true

echo
echo "=============== EXCLUSIONS ==============="
apptainer exec "${COMMON[@]}" "${OFFLINE_ENV[@]}" "$SIF" python3 -c '
import json

def load(path):
    return {row["task_id"]: row for row in (json.loads(line) for line in open(path))}

online = load("/work/gt_online.jsonl")
offline = load("/work/gt_offline.jsonl")
online_pass = sum(1 for row in online.values() if row["status"] == "pass")
offline_pass = sum(1 for row in offline.values() if row["status"] == "pass")
print("online  pass: {0}/{1}".format(online_pass, len(online)))
print("offline pass: {0}/{1}".format(offline_pass, len(offline)))

gt_failing = sorted(t for t in online if online[t]["status"] != "pass")
network_dependent = sorted(
    t for t in online
    if online[t]["status"] == "pass" and offline[t]["status"] != "pass"
)
excluded = sorted(set(gt_failing) | set(network_dependent))
print("ground truth failing :", gt_failing)
print("network dependent    :", network_dependent)
print("excluded total       : {0}/{1}".format(len(excluded), len(online)))
print("usable               : {0}".format(len(online) - len(excluded)))

with open("/work/excluded_tasks.json", "w") as handle:
    json.dump(
        {
            "ground_truth_failing": gt_failing,
            "network_dependent": network_dependent,
            "excluded": excluded,
        },
        handle,
        indent=1,
    )
print("wrote /work/excluded_tasks.json")
' > "$WORK/exclusions.txt" 2>&1
cat "$WORK/exclusions.txt"

echo DONE
