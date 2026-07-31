# !/bin/bash
set -e

git clone https://github.com/gAldeia/TPSR.git
cd TPSR
pip install -r requirements.txt

touch __init__.py

git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales.git
cd NeuralSymbolicRegressionThatScales
pip install -e src/
pip install lightning==1.9

cd ../..
cp TPSR -r ${CONDA_PREFIX}/bin/tpsr

# TPSR's regressor loads this checkpoint during import. Download it while
# building the image so CI tests do not depend on runtime network access.
mkdir -p /srbench_pretrained
curl --fail --location --retry 5 \
    https://dl.fbaipublicfiles.com/symbolicregression/model1.pt \
    --output /srbench_pretrained/model1.pt
