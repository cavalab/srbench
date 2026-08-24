#!/bin/bash
# Install Pantara v7d for SRBench evaluation.
# Pulls torch (>=1.12) and numpy (>=1.21) via the package's install_requires.
set -e

pip install git+https://github.com/Yapock22/pantara.git
