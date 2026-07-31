#!/bin/bash
set -e

git clone https://github.com/cavalab/feat.git
cd feat

# Keep the FEAT source and its Shogun ABI aligned with the environment's
# shogun-cpp=6.1.4 dependency.
# git checkout tags/0.5.1

python -m pip install .
