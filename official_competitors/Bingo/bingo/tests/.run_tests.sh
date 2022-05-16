#!/usr/bin/env bash

set -e

MPI_EXEC=$(python -c "import mpi4py;import os;filename = next(iter(mpi4py.get_config().items()))[1];print(os.path.dirname(filename)+'/mpiexec');")

$MPI_EXEC -np 3 coverage run --parallel-mode --source=bingo  tests/integration/mpitest_parallel_archipelago.py
coverage combine

pytest tests --cov=bingo --cov-report=term-missing --cov-append
