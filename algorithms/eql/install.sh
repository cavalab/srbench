#!/bin/bash
set -e

git clone https://github.com/cavalab/eql.git
cd eql

python -m pip install .
