#!/bin/bash
set -e

# Install PySR's Julia-side dependencies into the image. Without this step
# the Python package imports, but fitting fails while initializing Julia and
# SymbolicRegression.jl.
python -c 'import pysr; pysr.install()'
