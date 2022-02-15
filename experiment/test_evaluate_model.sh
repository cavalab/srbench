#!/bin/bash

# This bash file mimics the behaviour of this piece
# of python code:
## MLs = [ml.split('/')[-1][:-3] for ml in glob('methods/*.py') if
##        not ml.split('/')[-1][:-3].startswith('_')]

for ml in $(ls methods/*.py | grep -v '__init__.py' | sed 's/^methods\///g' | sed 's/\.py$//g') ; do
    python test_evaluate_model.py --ml="${ml}"
done
