#!/bin/bash
# Submit one SLURM job per SRBench algorithm, each running analyze.py
# inside its Apptainer/Singularity container. Modeled on run_distributed.sh.
#
# Usage (from the srbench repo root):
#   bash scripts/run_srbench.sh -a gplearn -d /path/to/pmlb/datasets
#   bash scripts/run_srbench.sh -a gplearn,feat -d /path/to/pmlb/datasets -- -n_trials 10 -time_limit 9:00
#
# Options:
#   -a ALGS     Comma-separated algorithms (default: gplearn). Each must have
#               a SIF at $SIF_DIR/srbench-<alg>.sif (see scripts/singularity_pull_all.sh).
#   -d DIR      PMLB datasets directory (required). Bound read-only into the container.
#   -c CPUS     CPUs per task / parallel jobs per algorithm (default: 8)
#   -t TIME     SLURM walltime, e.g. 24:00:00 (default: 24:00:00)
#   -m MEM      Memory per node, e.g. 32G (default: 32G)
#   -p PART     SLURM partition (default: cpu_batch)
#   -r RDIR     Results dir, relative to experiment/ (default: results)
#   -n          Dry run: print the sbatch commands without submitting
#   -- ARGS     Everything after -- is passed verbatim to analyze.py
#               (e.g. -n_trials 10 -tuned -target_noise 0.01)
set -uo pipefail

CURRENT_DIR=$(pwd)
if [[ $CURRENT_DIR == */scripts ]]; then
  cd ..
fi
echo "[srbench] Running from $(pwd)"

# --- Configuration (override via environment) ---------------------------------
SIF_DIR=${SIF_DIR:-/pgeoprj/dxt6/maromba/notebooks/container/output_dir}
LOG_DIRECTORY=${LOG_DIRECTORY:-logs}
mkdir -p "$LOG_DIRECTORY"

# --- Defaults -----------------------------------------------------------------
ALGS="gplearn"
DATASETS=""
CPUS=8
TIME="24:00:00"
MEM="32G"
PARTITION="cpu_batch"
RDIR="results"
DRYRUN=0
ANALYZE_ARGS=""

# --- Parse arguments ------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -a) ALGS="$2"; shift 2 ;;
    -d) DATASETS="$2"; shift 2 ;;
    -c) CPUS="$2"; shift 2 ;;
    -t) TIME="$2"; shift 2 ;;
    -m) MEM="$2"; shift 2 ;;
    -p) PARTITION="$2"; shift 2 ;;
    -r) RDIR="$2"; shift 2 ;;
    -n) DRYRUN=1; shift ;;
    --) shift; ANALYZE_ARGS="$@"; break ;;
    *) echo "[Error] Unknown option: $1"; exit 1 ;;
  esac
done

if [ -z "$DATASETS" ]; then
  echo "[Error] Missing required -d /path/to/pmlb/datasets"
  exit 1
fi
if [ ! -d "$DATASETS" ]; then
  echo "[Error] Datasets directory not found: $DATASETS"
  exit 1
fi

# --- Submit one job per algorithm ---------------------------------------------
IFS=',' read -ra ALG_LIST <<< "$ALGS"
for ALG in "${ALG_LIST[@]}"; do
  SIF="$SIF_DIR/srbench-${ALG}.sif"
  if [ ! -f "$SIF" ]; then
    echo "[Error] Missing image: $SIF"
    echo "        Pull it first: bash scripts/singularity_pull_all.sh $ALG"
    continue
  fi

  NOW=$(date +"%Y-%m-%d_%H-%M-%S")
  echo "[srbench] Submitting $ALG (cpus=$CPUS, time=$TIME, mem=$MEM, partition=$PARTITION)"
  echo "[srbench]   image:   $SIF"
  echo "[srbench]   analyze: python analyze.py /pmlb/datasets --local -ml $ALG -n_jobs $CPUS -results $RDIR $ANALYZE_ARGS"

  SBATCH_CMD=(
    sbatch
    --job-name="srbench-${ALG}"
    --output="${LOG_DIRECTORY}/srbench-${ALG}-${NOW}.out"
    --error="${LOG_DIRECTORY}/srbench-${ALG}-${NOW}.err"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --time="${TIME}"
    --mem="${MEM}"
    --partition="${PARTITION}"
    --export=ALL,ALG="${ALG}",SIF="${SIF}",DATASETS="${DATASETS}",CPUS="${CPUS}",RDIR="${RDIR}",ANALYZE_ARGS="${ANALYZE_ARGS}"
    scripts/run_srbench.slurm
  )

  if [ "$DRYRUN" -eq 1 ]; then
    echo "[dryrun] ${SBATCH_CMD[*]}"
  else
    "${SBATCH_CMD[@]}"
  fi
done
