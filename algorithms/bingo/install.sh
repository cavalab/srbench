# install Bingo
git clone --recurse-submodules --depth 1 --branch v0.4.1.srbench https://github.com/nasa/bingo.git
cd bingo

# install
pip install .
cd ../
#python -m pytest bingo/tests
