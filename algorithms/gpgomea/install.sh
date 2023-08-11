#!/bin/bash
# install GOMEA

# remove directory if it exists
if [ -d "GP-GOMEA" ]; then 
    rm -rf GP-GOMEA
fi

git clone  https://github.com/marcovirgolin/GP-GOMEA 
cd GP-GOMEA
#fix version
git checkout 6a92cb671c2772002b60df621a513d8b4df57887

# use correct python version
LIBNAME=$(echo $CONDA_PREFIX/lib/libboost_python*.so)
PYVERSION=${LIBNAME##$CONDA_PREFIX/lib/libboost_python}
PYVERSION=${PYVERSION%.so}
LBOOST="lboost_python$PYVERSION"
LNP="lboost_numpy$PYVERSION"

echo "PYVERSION: ${PYVERSION}"
echo "LBOOST: $LBOOST"
echo "LNUMPY: $LNP"

sed -i "s/lboost_python37/$LBOOST/" Makefile-variables.mk
sed -i "s/lboost_numpy37/$LNP/" Makefile-variables.mk

# add extra flags to varables
echo "EXTRA_FLAGS=-I ${CONDA_PREFIX}/include" >> Makefile-variables.mk
echo "EXTRA_LIB=-L ${CONDA_PREFIX}/lib" >> Makefile-variables.mk
# check
tail Makefile-variables.mk

# append EXTRA_FLAGS and CSSFLAGS
sed -i 's/^CXXFLAGS.*/& \$(EXTRA_FLAGS)/' Makefile-Python_Release.mk
sed -i 's/LIB_BOOST_NUMPY.*/& \$(EXTRA_LIB)/' Makefile-Python_Release.mk
#check
cat Makefile-Python_Release.mk | grep EXTRA_FLAGS
cat Makefile-Python_Release.mk | grep EXTRA_LIB

#todo: add CONDA_PREFIX to Make-file
make 
# copy the .so library into the python package
cp dist/Python_Release/GNU-Linux/gpgomea pyGPGOMEA/gpgomea.so

pip install .
