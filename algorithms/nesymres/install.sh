# !/bin/bash
set -e

git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales.git

cd NeuralSymbolicRegressionThatScales
pip install -e src/

# NeSymReS loads this checkpoint during import. Download it while building
# the image so tests do not depend on runtime network access.
mkdir -p /srbench_pretrained
curl --location --fail --retry 5 --retry-delay 5 --continue-at - \
    https://huggingface.co/TommasoBendinelli/NeuralSymbolicRegressionThatScales/resolve/main/100M.ckpt \
    --output /srbench_pretrained/nesymres_100M.ckpt
