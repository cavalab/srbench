# install Bingo
git clone --recurse-submodules https://github.com/nasa/bingo.git
cd bingo
git checkout srbench3

# install
pip install .
cd ../
#python -m pytest bingo/tests
