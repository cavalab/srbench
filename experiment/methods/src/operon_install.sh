# Installing operon
export CC=gcc-9
export CXX=gcc-9
# Dependencies 
# - TBB - installed via conda
# - Eigen- installed via conda
# - Ceres- installed via conda
# - {fmt}- installed via conda

# remove directory if it exists
if [ -d operon ]; then 
    rm -rf operon
fi

git clone https://github.com/heal-research/operon
cd operon
# fix version
git checkout 0a7a90139c38ed3d3c3b7a71560f6d7329f8530a

# run cmake
mkdir build; cd build; 
cmake .. -DCMAKE_BUILD_TYPE=Release  -DBUILD_PYBIND=ON -DUSE_OPENLIBM=ON -DUSE_SINGLE_PRECISION=ON -DCERES_TINY_SOLVER=ON 

# build
make VERBOSE=1 -j pyoperon

# install
make install
