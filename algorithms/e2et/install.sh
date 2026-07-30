set -e

# install sympytorch
#git clone https://github.com/pakamienny/sympytorch.git
pip install git+https://github.com/pakamienny/sympytorch.git@rationals

# Downloading pretrained model
# > wget -nc https://dl.fbaipublicfiles.com/symbolicregression/model1.pt
# OR
# > cd /
# > mkdir pretrained
# > curl https://dl.fbaipublicfiles.com/symbolicregression/model1.pt --output /pretrained/model.pt
mkdir -p /pretrained
curl --fail --location --retry 5 \
    https://dl.fbaipublicfiles.com/symbolicregression/model1.pt \
    --output /pretrained/model.pt

pip install git+https://github.com/pakamienny/e2e_transformer.git
