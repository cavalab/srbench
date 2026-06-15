#!/bin/bash

# install smgp
git clone https://github.com/MichalicekPetr/SRBench-SMGPRegressor-Src-Files.git methods/smgp
cd smgp

# install
pip install .
cd ../