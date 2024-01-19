# !/bin/bash
# install ITEA

# remove directory if it exists
if [ -d tir ]; then
    rm -rf tir
fi

git clone https://github.com/folivetti/tir.git

cd tir

# WGL NOTE: this is a temp fix until PR https://github.com/folivetti/ITEA/pull/12 is merged
# install ghcup
export BOOTSTRAP_HASKELL_NONINTERACTIVE=1
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | bash
export PATH=$PATH:~/.ghcup/bin:~/.cabal/bin 

#conda activate srbench
cabal install --overwrite-policy=always && cp ~/.cabal/bin/tir ./python/ && cd python && pip install .
