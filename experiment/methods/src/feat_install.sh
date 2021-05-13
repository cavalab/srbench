#install feat

# remove directory if it exists
if [ -d "feat" ]; then 
    rm -rf feat
fi

git clone http://github.com/lacava/feat

cd feat

./configure
export SHOGUN_LIB=$CONDA_PREFIX/lib/
export SHOGUN_DIR=$CONDA_PREFIX/include/
export EIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3/
./install y
