#!/bin/bash
# Build and test one algorithm the same way CI does.
#
#   bash local_ci.sh <method-name>
#
# <method-name> must match a directory in both algorithms/ and
# experiment/methods/. Run from the repository root.
set -euo pipefail

SUBNAME="${1:-}"
if [ -z "$SUBNAME" ]; then
    echo "usage: bash local_ci.sh <method-name>" >&2
    exit 1
fi

if [ ! -d "algorithms/${SUBNAME}" ] || [ ! -d "experiment/methods/${SUBNAME}" ]; then
    echo "error: ${SUBNAME} needs a directory in BOTH algorithms/ and experiment/methods/" >&2
    echo "see CONTRIBUTING.md for the expected layout" >&2
    exit 1
fi

echo "==> checking method layout"
python scripts/check_method_layout.py

echo "==> regenerating docker-compose.yml"
bash scripts/make_docker_compose_file.sh

# Algorithm images are FROM srbench/base, so the base image has to exist first.
echo "==> building base image"
docker compose build base

echo "==> building ${SUBNAME}"
docker compose build "${SUBNAME}"

echo "==> testing ${SUBNAME}"
docker compose run --rm "${SUBNAME}" bash test.sh
