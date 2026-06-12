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

mkdir -p /pretrained
curl -L https://dl.fbaipublicfiles.com/symbolicregression/model1.pt --output /pretrained/model1.pt
