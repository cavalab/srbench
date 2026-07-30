#!/bin/bash
set -e

git clone https://github.com/cavalab/brush.git
cd brush

# Ensure glog headers are consumed with the expected export/gflags macros.
export CFLAGS="${CFLAGS} -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS"
export CXXFLAGS="${CXXFLAGS} -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"

python -m pip install .
