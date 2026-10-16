#!/bin/bash
# Pull prebuilt SRBench algorithm images from Docker Hub as Singularity SIFs.
#
# Run this on a node WITH internet access (e.g. the login node); compute
# nodes usually can't reach Docker Hub.
#
# Usage:
#   bash scripts/singularity_pull_all.sh                 # pull all available images
#   bash scripts/singularity_pull_all.sh gplearn feat    # pull specific ones
#
# Env overrides:
#   OUTPUT_DIR           - where .sif files are written
#                          (default: /tatu/container_images/srbench)
#   SINGULARITY_CACHEDIR - docker layer cache (default: /tatu/$USER/singularity_cache)
#   SINGULARITY_TMPDIR   - temp dir during conversion (default: /tatu/$USER/singularity_tmp)
#
# NOTE: images are large (some >10 GB uncompressed). Do NOT let the cache
# default to $HOME or you may fill your home quota.
set -uo pipefail

# --- Pick a container runtime -------------------------------------------------
# "ERROR: pull is only supported for shub URIs" means singularity 2.x, which
# cannot pull from docker://. We need singularity >= 3.x or apptainer.
# Prefer apptainer if installed; otherwise use singularity and check its version.
if command -v apptainer >/dev/null 2>&1; then
    SINGULARITY=apptainer
else
    SINGULARITY=singularity
fi

VER=$(command $SINGULARITY --version 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)+' | head -1)
MAJOR=${VER%%.*}
# if [ -z "${MAJOR}" ] || [ "${MAJOR}" -lt 3 ]; then
#     echo "ERROR: '$SINGULARITY' is version '${VER:-unknown}', but >= 3.0 (or apptainer)"
#     echo "       is required to pull from Docker Hub (docker:// URIs)."
#     echo
#     echo "Look for a newer installation on the cluster, e.g.:"
#     echo "  module avail singularity"
#     echo "  module avail apptainer"
#     echo "  which -a singularity apptainer"
#     echo "then 'module load' it and re-run this script."
#     exit 1
# fi
echo "Using runtime: $SINGULARITY ($VER)"

# Algorithms published on Docker Hub (srbench org), tag :latest.
# Verified 2026-10-16. The remaining repo algorithms are NOT published:
#   brush bsr gp-elite keplearn lightgbm nesymres pantara pir sklearn tpsr udsr xgboost
# (those need a local conda install or a SIF built from a def file).
AVAILABLE=(
    afp
    bingo
    e2et
    eplex
    eql
    feat
    ffx
    geneticengine
    gpgomea
    gplearn
    gpzgd
    itea
    operon
    ps-tree
    pysr
    qlattice
    rils-rols
    tir
)

# OUTPUT_DIR=${OUTPUT_DIR:-/tatu/container_images/srbench}
# export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-/tatu/$USER/singularity_cache}
# export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-/tatu/$USER/singularity_tmp}
OUTPUT_DIR=${OUTPUT_DIR:-/pgeoprj/dxt6/maromba/notebooks/container/output_dir}
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-/pgeoprj/dxt6/maromba/notebooks/container/singularity_cache}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-/pgeoprj/dxt6/maromba/notebooks/container/singularity_tmp}
mkdir -p "$OUTPUT_DIR" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

if [ "$#" -gt 0 ]; then
    ALGS=("$@")
else
    ALGS=("${AVAILABLE[@]}")
fi

FAILED=()
for alg in "${ALGS[@]}"; do
    sif="$OUTPUT_DIR/srbench-${alg}.sif"
    if [ -f "$sif" ]; then
        echo "[skip] $sif already exists"
        continue
    fi
    echo "[pull] docker://srbench/${alg}:latest -> $sif"
    if $SINGULARITY pull "$sif" "docker://srbench/${alg}:latest"; then
        echo "[ok]   $alg"
    else
        echo "[FAIL] $alg"
        FAILED+=("$alg")
    fi
done

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Finished with ${#FAILED[@]} failure(s): ${FAILED[*]}"
else
    echo "All images pulled successfully."
fi
echo "SIF files are in: $OUTPUT_DIR"
echo
echo "Smoke-test one image (run from the srbench repo root):"
echo "  singularity run --pwd /srbench -B \$PWD/experiment:/srbench \\"
echo "      $OUTPUT_DIR/srbench-gplearn.sif bash test.sh"
