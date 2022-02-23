#!/bin/bash
# install ITEA

# remove directory if it exists
if [ -d ITEA ]; then
    rm -rf ITEA
fi

git clone https://github.com/folivetti/ITEA.git

#cd ITEA
rsync -av ITEA $CONDA_PREFIX/lib/python3.7/site-packages/ --exclude=".git" --exclude="datasets"
#curl -sSL https://get.haskellstack.org/ | sh
#stack build
#conda activate srbench
#cp $CONDA_PREFIX/lib/libgsl.so bin/libgsl.so.0
