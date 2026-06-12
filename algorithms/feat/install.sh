#!/bin/bash
set -e

git clone https://github.com/cavalab/feat.git
cd feat

export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

python -m pip install .