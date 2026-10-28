#!/usr/bin/env bash
# SHR engine install script (run by the SRBench build).
#
# The engine is a set of flat .py files that import each other at top level.
# To keep the site-packages top-level namespace clean while making the flat
# imports work, this script:
#   1. places the engine in the standalone dir $CONDA_PREFIX/shr_engine/
#   2. writes a .pth file into site-packages containing that dir, so Python
#      adds it to sys.path at startup and flat imports such as
#      `import shr_v7_full` work.
#
# The engine is taken from a local engine/ directory when present (offline /
# self-contained); otherwise it is downloaded from the SHR_URL release.
set -e

# Stable engine source: a GitHub release tarball (flat .py files, at the top
# level or under shr_engine/). Can be overridden with the SHR_URL env var.
SHR_URL="${SHR_URL:-https://github.com/il143/SHR/releases/download/v1.0/shr_engine.tar.gz}"

SP_DIR="$(python -c 'import site; print(site.getsitepackages()[0])')"
ENG_DIR="${CONDA_PREFIX}/shr_engine"
mkdir -p "${ENG_DIR}"

# This script lives at algorithms/shr/install.sh; a local engine lives at
# <repo-root>/engine (../../engine relative to this file).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENG=""
if [ -d "${HERE}/../../engine" ]; then
    LOCAL_ENG="$(cd "${HERE}/../../engine" && pwd)"
fi

if [ -n "${LOCAL_ENG}" ] && [ -f "${LOCAL_ENG}/shr_v7_full.py" ]; then
    echo "Using local engine from ${LOCAL_ENG} (offline, self-contained)."
    cp "${LOCAL_ENG}"/*.py "${ENG_DIR}/"
else
    TMP="$(mktemp -d)"
    echo "Local engine not found; downloading from ${SHR_URL} ..."
    curl -L "${SHR_URL}" -o "${TMP}/shr_engine.tar.gz"
    tar -xzf "${TMP}/shr_engine.tar.gz" -C "${TMP}"
    if [ -d "${TMP}/shr_engine" ]; then
        cp "${TMP}/shr_engine"/*.py "${ENG_DIR}/"
    else
        cp "${TMP}"/*.py "${ENG_DIR}/"
    fi
    rm -rf "${TMP}"
fi

# Key: the .pth adds the engine dir to sys.path (flat imports work)
echo "${ENG_DIR}" > "${SP_DIR}/shr_engine.pth"

echo "SHR engine installed at ${ENG_DIR}"
python -c "import shr_v7_full; print('SHR import OK')"
