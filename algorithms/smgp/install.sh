#!/bin/bash
set -ex

# 1. Define the target directory where SRBench looks
TARGET_DIR="/srbench/methods/smgp"

echo "--- DEBUG: Cleaning and preparing $TARGET_DIR ---"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

# 2. Clone the files directly where the framework needs them
echo "--- DEBUG: Cloning repository ---"
git clone https://github.com/MichalicekPetr/SRBench-SMGPRegressor-Src-Files.git "$TARGET_DIR"

# 3. Diagnostic: Verify file existence
echo "--- DEBUG: Content after cloning ---"
ls -la "$TARGET_DIR"

# 4. Installation in editable mode
# This tells Python to treat this directory as a package
echo "--- DEBUG: Installing package ---"
cd "$TARGET_DIR"
pip install -e .

# 5. Verify import in Python
echo "--- DEBUG: Testing module import ---"
python3 -c "import smgp; print('IMPORT SMGP SUCCESSFUL!'); from smgp.regressor import SMGPRegressor; print('IMPORT REGRESSOR SUCCESSFUL!')"

echo "--- DEBUG: Install script complete ---"