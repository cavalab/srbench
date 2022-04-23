#!/bin/bash

# install ghcup
if [ ! -d ~/.ghcup ]; then 
    export BOOTSTRAP_HASKELL_NONINTERACTIVE=1
    curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | bash
fi    
#source "~/.ghcup/env"
export PATH=$PATH:~/.ghcup/bin:~/.cabal/bin 

# install ITEA

# remove directory if it exists
if [ -d ITEA ]; then
    rm -rf ITEA
fi

git clone https://github.com/folivetti/ITEA.git

cd ITEA 
#conda activate srbench
C_INCLUDE_PATH=$CONDA_PREFIX/include LIBRARY_PATH=$CONDA_PREFIX/lib cabal install
cp ~/.cabal/bin/itea ./python/
cd python 
pip install .

LD_LIBRARY_PATH=$CONDA_PREFIX/lib itea

#cp $CONDA_PREFIX/lib/libgsl.so bin/libgsl.so.0
