#!/bin/bash
# install deep-symbolic-regression

if [ -d deep-symbolic-regression ] ; then
    rm -rf deep-symbolic-regression
fi

git clone --quiet https://github.com/lacava/deep-symbolic-regression 

cd deep-symbolic-regression

pip install -r requirements.txt
# export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS" # Needed on Mac to prevent fatal error: 'numpy/arrayobject.h' file not found
pip install -e ./dsr # Install DSR package
