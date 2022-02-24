#install feat

# remove directory if it exists
if [ -d "feat" ]; then 
    rm -rf feat
fi

git clone  https://github.com/cavalab/feat.git
cd feat

# fix version
git checkout 1f9c43b60eb26bb0d7381cb3a4ebd6243f7355dd

./configure
export SHOGUN_LIB=$CONDA_PREFIX/lib/
export SHOGUN_DIR=$CONDA_PREFIX/include/
export EIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3/
./install n
cp ../build/libfeat_lib.so $CONDA_PREFIX/lib/
cd python
python setup.py install
