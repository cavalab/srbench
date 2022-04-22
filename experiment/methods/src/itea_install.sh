#!/bin/bash

# install ghcup
export BOOTSTRAP_HASKELL_NONINTERACTIVE=1
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | bash
source "~/.ghcup/env"
export PATH=$PATH:~/.ghcup/bin 

# install ITEA

# remove directory if it exists
if [ -d ITEA ]; then
    rm -rf ITEA
fi

git clone https://github.com/folivetti/ITEA.git

rsync -av ITEA $CONDA_PREFIX/lib/python3.7/site-packages/ --exclude=".git" --exclude="datasets"
cd ITEA 
conda activate srbench
C_INCLUDE_PATH=$CONDA_PREFIX/include LIBRARY_PATH=$CONDA_PREFIX/lib cabal install
cp ~/.cabal/bin/itea bin/itea 
LD_LIBRARY_PATH=$CONDA_PREFIX/lib bin/itea

#cp $CONDA_PREFIX/lib/libgsl.so bin/libgsl.so.0
