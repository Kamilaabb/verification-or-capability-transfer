#!/usr/bin/env bash
# Build the BigCodeBench execution sandbox and verify it is usable as a grader.
#
# The upstream repository is archived and the image is only published under a moving
# `latest` tag, so the reproducibility anchor is the SHA256 of the SIF this job
# produces, not the upstream tag. Record that hash in the dataset config.
#
# Measured on 2026-09-16 (job 65075855): 9110036480 bytes,
# sha256 286b4e68f3604864ef7c10c48da83bba554b986fcd1f40ce6c092b695a410ad9

#SBATCH --job-name=bcb-image
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/project/bcb-image-%j.out
#SBATCH --error=/path/to/project/bcb-image-%j.err

set -uo pipefail

SCRATCH_ROOT=/path/to/scratch/authority-allocation
TARGET="$SCRATCH_ROOT/containers/bigcodebench-evaluate.sif"

module purge
module load gcc/11.5.0
module load apptainer/1.4.1

export APPTAINER_CACHEDIR="$SCRATCH_ROOT/apptainer-cache"
export APPTAINER_TMPDIR="$SCRATCH_ROOT/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$SCRATCH_ROOT/containers"

if [[ -f "$TARGET" ]]; then
  echo "Image already present; remove it explicitly to rebuild:"
  ls -la "$TARGET"
  exit 0
fi

apptainer build "$TARGET" docker://bigcodebench/bigcodebench-evaluate:latest

echo "=== pin ==="
ls -la "$TARGET"
sha256sum "$TARGET"

echo
echo "=== interpreter and package ==="
apptainer exec --cleanenv "$TARGET" python3 -c '
import sys, os, inspect, bigcodebench
print("python", sys.version.split()[0])
print("bigcodebench", getattr(bigcodebench, "__version__", "?"))
print("package", os.path.dirname(inspect.getfile(bigcodebench)))
'

echo
echo "=== eval dependencies ==="
apptainer exec --cleanenv "$TARGET" python3 -c '
import importlib
names = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn", "django",
         "flask", "cv2", "geopandas", "shapely", "tensorflow", "torch", "nltk",
         "sympy", "bs4", "requests", "PIL", "statsmodels"]
missing = []
for name in names:
    try:
        module = importlib.import_module(name)
        print("  OK      {0:14s} {1}".format(name, getattr(module, "__version__", "?")))
    except Exception as exc:
        missing.append(name)
        print("  MISSING {0:14s} {1}".format(name, type(exc).__name__))
print("missing:", len(missing), missing)
'

echo
echo "=== network isolation (must report isolated) ==="
apptainer exec --cleanenv --net --network=none "$TARGET" python3 -c '
import socket
socket.setdefaulttimeout(4)
try:
    print("  REACHABLE", socket.gethostbyname("pypi.org"), "-- ISOLATION FAILED")
except Exception as exc:
    print("  isolated (" + type(exc).__name__ + ")")
'

echo DONE
