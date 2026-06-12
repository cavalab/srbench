#!/bin/bash
set -e

git clone https://github.com/cavalab/eql.git
cd eql

python - <<'PY'
from pathlib import Path

setup_py = Path('setup.py')
text = setup_py.read_text()
text = text.replace('packages=find_packages(where="eql")', 'packages=find_packages()')
setup_py.write_text(text)
PY

python -m pip install .
