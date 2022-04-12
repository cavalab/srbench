#install sympytorch
git clone https://github.com/pakamienny/sympytorch.git
cd sympytorch
git checkout rationals
pip install -e .

cd .. 

git clone https://github.com/pakamienny/e2e_transformer.git
pip install -e .

wget -nc https://dl.fbaipublicfiles.com/symbolicregression/model1.pt

