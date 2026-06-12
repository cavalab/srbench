set -e

#install sympytorch
#git clone https://github.com/pakamienny/sympytorch.git
pip install git+https://github.com/pakamienny/sympytorch.git@rationals

# Downloading pretrained model
# > wget -nc https://dl.fbaipublicfiles.com/symbolicregression/model1.pt
# OR
# > cd /
# > mkdir pretrained
# > curl https://dl.fbaipublicfiles.com/symbolicregression/model1.pt --output /pretrained/model.pt
# Save it locally and keep it in an accessible folder, instead of inside the 
# docker image. You need to specify the path to the pretrained models

pip install git+https://github.com/pakamienny/e2e_transformer.git

mkdir -p /pretrained
curl -L https://dl.fbaipublicfiles.com/symbolicregression/model1.pt --output /pretrained/model.pt
