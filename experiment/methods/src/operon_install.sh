# Installing operon
export CC=gcc-9
export CXX=gcc-9
# Dependencies 
# - TBB - installed via conda
# - Eigen- installed via conda
# - Ceres- installed via conda
# - {fmt}- installed via conda

git clone https://github.com/heal-research/operon
cd operon
git checkout 1b08b7ffb65edb2a347e08a32595ccc1f078a882

# run cmake
mkdir build; cd build; 
cmake .. -DCMAKE_BUILD_TYPE=Release  -DBUILD_PYBIND=ON  

#install
make VERBOSE=1
