set -euxo pipefail

# remove directory if it exists
if [ -d "MultiGPG" ]; then
    rm -rf MultiGPG
fi

git clone https://github.com/matigekunstintelligentie/MultiGPG.git

cd MultiGPG || exit
# ! don't forget to update this
git checkout dd6954fe53e0532101789db7db7951d95231931e

# doesn't seem to work and adds bloat...
# make

# 1. build .so
mkdir -p build
cd build || exit
cmake -S ../ -B . -DCMAKE_PREFIX_PATH=$(CONDA_PREFIX) -DCMAKE_BUILD_TYPE=Release
make

# build python package
cd .. || exit
cp -r src/pypkg ..
mv build/_pb_mgpg*.so ../pypkg/pymgpg/_pb_mgpg.so
cd ../pypkg
python setup.py install

# cleanup
cd .. || exit
rm -rf MultiGPG
