#!/bin/bash
set -ex

# Use relative paths because the WORKDIR is already set to /srbench
TARGET_DIR="methods/smgp"

echo "--- DEBUG: Cleaning and preparing $TARGET_DIR ---"
# We only work within the current directory to avoid permission issues
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Clone the repository into the correct relative path
echo "--- DEBUG: Cloning repository ---"
git clone https://github.com/MichalicekPetr/SRBench-SMGPRegressor-Src-Files.git "$TARGET_DIR"

# Verify that files exist
echo "--- DEBUG: Verifying directory content ---"
ls -la "$TARGET_DIR"

# Install the package in editable mode so the framework can import it
echo "--- DEBUG: Installing package ---"
cd "$TARGET_DIR"
pip install -e .

# Test the import
echo "--- DEBUG: Testing module import ---"
python3 -c "import smgp; print('IMPORT SMGP SUCCESSFUL!'); from smgp.regressor import SMGPRegressor; print('IMPORT REGRESSOR SUCCESSFUL!')"

echo "--- DEBUG: Installation complete ---"