git clone https://github.com/cavalab/eql.git
cd eql

# Install the CPU backend explicitly. The conda environment is useful for
# dependency solving, but JAX was not present in the final runtime image.
python -m pip install --no-cache-dir --upgrade \
    "jax[cpu]>=0.4.26" \
    "jaxlib>=0.4.26" \
    "flax>=0.8.0" \
    "optax>=0.2.0"

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
