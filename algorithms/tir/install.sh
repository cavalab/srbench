# !/bin/bash
# install ITEA

git clone https://github.com/folivetti/tir.git

cd tir
git checkout 4c0b43758559650e6d65d81fd2974425e103d67c

# WGL NOTE: this is a temp fix until PR https://github.com/folivetti/ITEA/pull/12 is merged
# install ghcup
#export BOOTSTRAP_HASKELL_NONINTERACTIVE=1
#curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | bash
#export PATH=$PATH:~/.ghcup/bin:~/.cabal/bin 

#conda activate srbench
cabal update
cabal install --overwrite-policy=always --installdir=./python && cd python && pip install .
