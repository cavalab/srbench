#!/bin/bash
# SLURM batch script: run SRBench analyze.py for ONE algorithm inside its
# Apptainer/Singularity container. Submitted by scripts/run_srbench.sh,
# which exports: ALG, SIF, DATASETS, CPUS, RDIR, ANALYZE_ARGS.
# Modeled on run_distributed.slurm.

# Inform user
echo "[srbench] $USER @ $(hostname) | Algorithm: $ALG"
echo "[srbench] $USER @ $(hostname) | Image: $SIF"
echo "[srbench] $USER @ $(hostname) | Datasets: $DATASETS"
echo "[srbench] $USER @ $(hostname) | CWD: $(pwd)"
echo "[srbench] $USER @ $(hostname) | analyze.py args: $ANALYZE_ARGS"

# --- Runtime ------------------------------------------------------------------
if command -v apptainer >/dev/null 2>&1; then
  CONTAINER=apptainer
else
  CONTAINER=singularity
fi
echo "[srbench] Using runtime: $CONTAINER ($($CONTAINER --version 2>/dev/null))"

# Keep per-job cache/tmp off $HOME (quota) and on local/scratch disk.
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/tmp/$USER/apptainer_cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/tmp/$USER/apptainer_tmp}
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
export SINGULARITY_TMPDIR=$APPTAINER_TMPDIR
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# --- Bind mounts ----------------------------------------------------------------
# experiment/ -> /srbench  (the image's WORKDIR; results are written here)
# datasets  -> /pmlb/datasets (read-only)
# /pgeoprj,/tatu kept for consistency with your other jobs.
BINDS="$PWD/experiment:/srbench,$DATASETS:/pmlb/datasets:ro,/pgeoprj,/tatu"

# --- Run ------------------------------------------------------------------------
# `run` (not `exec`) so the image entrypoint activates the conda base env,
# matching `docker compose run <alg> ...`.
$CONTAINER run --pwd /srbench -B "$BINDS" "$SIF" \
  python analyze.py /pmlb/datasets \
    --local \
    -ml "$ALG" \
    -n_jobs "$CPUS" \
    -results "$RDIR" \
    $ANALYZE_ARGS

echo "[srbench] $USER @ $(hostname) | Done ($ALG). Results in experiment/$RDIR"
