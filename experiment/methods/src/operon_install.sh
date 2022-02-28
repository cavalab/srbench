#!/bin/bash

PYTHON_SITE=${CONDA_PREFIX}/lib/python`pkg-config --modversion python3`/site-packages

## aria-csv
git clone  https://github.com/AriaFallah/csv-parser csv-parser
mkdir -p ${CONDA_PREFIX}/include/aria-csv
pushd csv-parser
git checkout 544c764d0585c61d4c3bd3a023a825f3d7de1f31
cp parser.hpp ${CONDA_PREFIX}/include/aria-csv/parser.hpp
popd
rm -rf csv-parser

## vectorclass
git clone  https://github.com/vectorclass/version2.git vectorclass
mkdir -p ${CONDA_PREFIX}/include/vectorclass
pushd vectorclass
git checkout fee0601edd3c99845f4b7eeb697cff0385c686cb
cp *.h ${CONDA_PREFIX}/include/vectorclass/
popd
rm -rf vectorclass
cat > ${CONDA_PREFIX}/lib/pkgconfig/vectorclass.pc << EOF
prefix=${CONDA_PREFIX}/include/vectorclass
includedir=${CONDA_PREFIX}/include/vectorclass

Name: Vectorclass
Description: C++ class library for using the Single Instruction Multiple Data (SIMD) instructions to improve performance on modern microprocessors with the x86 or x86/64 instruction set.
Version: 2.01.04
Cflags: -I${CONDA_PREFIX}/include/vectorclass
EOF

## vstat
git clone  https://github.com/heal-research/vstat.git
pushd vstat
git checkout 9b48f0d021ec66df122be352ea928b6ceb4bca54
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

## span-lite
git clone  https://github.com/martinmoene/span-lite.git
pushd span-lite
git checkout 8f7935ff4e502ee023990d356d6578b8293eda74
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPAN_LITE_OPT_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf span-lite

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
git checkout d26dd0dcf16acb750da330b5112c63f2528af9a8
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_CLI_PROGRAMS=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t operon_operon
cmake --install build
popd
rm -rf operon

## pyoperon
git clone  https://github.com/heal-research/pyoperon.git
pushd pyoperon
git checkout 1c6eccd3e3fa212ebf611170ca2dfc45714c81de
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PYTHON_SITE}
cmake --build build -j -t pyoperon_pyoperon
cmake --install build
popd
rm -rf pyoperon
