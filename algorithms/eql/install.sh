git clone https://github.com/cavalab/eql.git
cd eql

# The repository's setup.py historically searched for packages below an
# already-nested `eql` directory, leaving no importable eql package.
python - <<'PY'
from pathlib import Path

setup_py = Path("setup.py")
text = setup_py.read_text()
text = text.replace(
    'packages=find_packages(where="eql")',
    'packages=find_packages()',
)
text = text.replace(
    "packages=find_packages(where='eql')",
    "packages=find_packages()",
)
setup_py.write_text(text)
PY

pip install -e .
