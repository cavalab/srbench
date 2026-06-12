#!/bin/bash
set -e

git clone https://github.com/cavalab/feat.git
cd feat

python - <<'PY'
from pathlib import Path

path = Path('CMakeLists.txt')
text = path.read_text()
text = text.replace('cmake_minimum_required(VERSION 3.5)', 'cmake_minimum_required(VERSION 3.10)')
path.write_text(text)
PY

python -m pip install .