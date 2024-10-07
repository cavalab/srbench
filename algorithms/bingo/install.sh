# install mpi4py
apt-get install -y mpich libmpich-dev
pip install mpi4py>=2.0.0,<4.0

# install Bingo
git clone --recurse-submodules --depth 1 --branch v0.4.1.srbench https://github.com/nasa/bingo.git
cd bingo

# install
pip install .
cd ../
#python -m pytest bingo/tests
