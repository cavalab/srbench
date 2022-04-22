#!/bin/bash

# install ghcup
export BOOTSTRAP_HASKELL_NONINTERACTIVE=1
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | bash
#source "~/.ghcup/env"
export PATH=$PATH:~/.ghcup/bin:~/.cabal/bin 

# install TIR

# remove directory if it exists
if [ -d tir ]; then
    rm -rf tir
fi

git clone https://github.com/folivetti/tir.git

rsync -av tir $CONDA_PREFIX/lib/python3.7/site-packages/ --exclude=".git" --exclude="datasets"
cd tir 
#conda activate srbench
C_INCLUDE_PATH=$CONDA_PREFIX/include LIBRARY_PATH=$CONDA_PREFIX/lib cabal install
cp ~/.cabal/bin/tir-exe ./python/tir
cd python 
pip install .

LD_LIBRARY_PATH=$CONDA_PREFIX/lib tir

#cp $CONDA_PREFIX/lib/libgsl.so bin/libgsl.so.0
