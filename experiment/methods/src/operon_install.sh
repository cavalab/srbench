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
git checkout c537a45b5ff5f98b78821244a1fc657e2f081cb0

# run cmake
mkdir build; cd build;
SOURCE_DATE_EPOCH=`date +%s` cmake .. -DCMAKE_BUILD_TYPE=Release  -DBUILD_PYBIND=ON -DUSE_OPENLIBM=ON -DUSE_SINGLE_PRECISION=ON -DCERES_TINY_SOLVER=ON

# build
make VERBOSE=1 -j pyoperon

# install
make install
