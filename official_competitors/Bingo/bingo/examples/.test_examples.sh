#!/usr/bin/env bash

set -e

python -c "from bingo import symbolic_regression; print('Using %s Backend' % ('c++' if symbolic_regression.ISCPP else 'Python'))"

for i in examples/*.ipynb
do
  echo "Running Notebook: $i"
  jupyter nbconvert --stdout --execute --to python $i > /dev/null
  echo "Success"
  echo ""
done

for i in examples/*.py
do
  echo "Running Script: $i"
  python $i > /dev/null
  echo "Success"
  echo ""
done
