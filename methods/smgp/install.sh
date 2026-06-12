#!/bin/bash

# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "=== Installing SMGP Regressor for SRBench ==="

TARGET_DIR="$SCRIPT_DIR/smgp_src"

# 1. Download or update source codes from GitHub
if [ -d "$TARGET_DIR" ]; then
    echo "Folder smgp_src already exists, updating source code..."
    cd "$TARGET_DIR" && git pull
else
    echo "Downloading source codes from GitHub..."
    git clone https://github.com/MichalicekPetr/SRBench-SMGPRegressor-Src-Files.git "$TARGET_DIR"
fi

# 2. Install Python dependencies using absolute path and user flag
echo "Installing Python libraries from requirements.txt..."
pip install --user -r "$SCRIPT_DIR/requirements.txt"

echo "=== Installation completed successfully ==="