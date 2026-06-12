#!/bin/bash
set -e

git clone https://github.com/cavalab/brush.git
cd brush

# fix version
# git checkout 603a814

# Ensure glog headers are consumed with the expected export/gflags macros.
export CFLAGS="${CFLAGS} -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS"
export CXXFLAGS="${CXXFLAGS} -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS"

python -m pip install .
