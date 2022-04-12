#install sympytorch
git clone https://github.com/pakamienny/sympytorch.git
cd sympytorch
rm -rf .git
git checkout rationals
pip install -e .

cd .. 
wget -nc https://dl.fbaipublicfiles.com/symbolicregression/model1.pt

git clone https://github.com/pakamienny/e2e_transformer.git
cd e2e_transformer
rm -rf .git
pip install -e .
