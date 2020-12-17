# Installing operon
# Dependencies - installed via conda
# - oneTBB
# - Eigen
# - Ceres
# - {fmt}

git clone https://github.com/heal-research/operon
cd operon
mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release
make . VERBOSE=1



