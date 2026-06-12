#!/bin/bash
set -e

# Clone repository
git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales.git
cd NeuralSymbolicRegressionThatScales

# Install package
pip install -e src/

# Download pretrained weights
mkdir -p /srbench_pretrained
curl -L --fail --retry 5 \
     --retry-delay 5 \
     --continue-at - \
     -o https://huggingface.co/TommasoBendinelli/NeuralSymbolicRegressionThatScales/resolve/main/100M.ckpt --output srbench_pretrained/nesymres_100M.ckpt
