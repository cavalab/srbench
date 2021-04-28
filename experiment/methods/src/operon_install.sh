# # Installing operon
# export CC=gcc-9
# export CXX=gcc-9
# # Dependencies
# # - TBB - installed via conda
# # - Eigen- installed via conda
# # - Ceres- installed via conda
# # - {fmt}- installed via conda

# # remove directory if it exists
if [ -d operon ]; then
    rm -rf operon
fi

git clone https://github.com/heal-research/operon
cd operon
# fix version
git checkout 015d420944a64353a37e0493ae9be74c645b4198

# run cmake
rm -rf build
mkdir build; cd build;
SOURCE_DATE_EPOCH=`date +%s` cmake .. -DCMAKE_BUILD_TYPE=Release  -DBUILD_PYBIND=ON -DUSE_OPENLIBM=ON -DUSE_SINGLE_PRECISION=ON -DCERES_TINY_SOLVER=ON -DPython3_FIND_VIRTUALENV=ONLY #-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX/lib/python3.7/site-packages/ #-DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python -DPython3_LIBRARY=$CONDA_PREFIX/lib/ -DPython3_INCLUDE_DIR=$CONDA_PREFIX/include/ 
#-DPython3_ROOT_DIR=$CONDA_PREFIX
# -DPython3_FIND_STRATEGY=LOCATION 
# build
make VERBOSE=1 -j pyoperon

# install
make install
