#!/bin/bash

PYTHON_SITE=${CONDA_PREFIX}/lib/python`pkg-config --modversion python3`/site-packages
export CC=${CONDA_PREFIX}/bin/gcc
export CXX=${CONDA_PREFIX}/bin/g++

## aria-csv
git clone  https://github.com/AriaFallah/csv-parser csv-parser
mkdir -p ${CONDA_PREFIX}/include/aria-csv
pushd csv-parser
git checkout 544c764d0585c61d4c3bd3a023a825f3d7de1f31
cp parser.hpp ${CONDA_PREFIX}/include/aria-csv/parser.hpp
popd
rm -rf csv-parser

## eve
git clone  https://github.com/jfalcou/eve eve
mkdir -p ${CONDA_PREFIX}/include/eve
pushd eve
git checkout cfcf03be08f99320f39c74d1205d0514e62c3c8e
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DEVE_BUILD_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf eve
cat > ${CONDA_PREFIX}/lib/pkgconfig/eve.pc << EOF
prefix=${CONDA_PREFIX}/include/eve
includedir=${CONDA_PREFIX}/include/eve
libdir=${CONDA_PREFIX}/lib

Name: Eve
Description: Eve - the Expression Vector Engine for C++20
Version: 2022.03.0
Cflags: -I${CONDA_PREFIX}/include/eve
EOF

## vstat
git clone https://github.com/heal-research/vstat.git
pushd vstat
git switch cpp20-eve
git checkout 736fa4802730ac14c2dfaf989a2b9016349c58d3
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf vstat

## pratt-parser
git clone  https://github.com/foolnotion/pratt-parser-calculator.git
pushd pratt-parser-calculator
git checkout a15528b1a9acfe6adefeb41334bce43bdb8d578c
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf pratt-parser-calculator

## fast-float
git clone  https://github.com/fastfloat/fast_float.git
pushd fast_float
git checkout 32d21dcecb404514f94fb58660b8029a4673c2c1
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFASTLOAT_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf fast_float

## robin_hood
git clone  https://github.com/martinus/robin-hood-hashing.git
pushd robin-hood-hashing
git checkout 9145f963d80d6a02f0f96a47758050a89184a3ed
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DRH_STANDALONE_PROJECT=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf robin-hood-hashing

# operon
git clone  https://github.com/heal-research/operon.git
pushd operon
git switch cpp20
git checkout 9d7d410e43d18020df25d6311822be8c3680ac56
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_CLI_PROGRAMS=OFF \
    -DCMAKE_CXX_FLAGS="-march=x86-64 -mavx2 -mfma" \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t operon_operon -- VERBOSE=1
cmake --install build
popd
rm -rf operon

## pyoperon
git clone  https://github.com/heal-research/pyoperon.git
pushd pyoperon
git switch cpp20
git checkout e4ef1047240b2df66555ddd54463ab707863aae6
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=x86-64 -mavx2 -mfma" \
    -DCMAKE_INSTALL_PREFIX=${PYTHON_SITE} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t pyoperon_pyoperon -- VERBOSE=1
cmake --install build
popd
rm -rf pyoperon
