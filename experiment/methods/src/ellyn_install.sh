#!/bin/bash
#install ellyn

# remove directory if it exists
if [ -d "ellyn" ]; then 
    rm -rf ellyn
fi

git clone http://github.com/cavalab/ellyn

cd ellyn
# fix version
#git checkout df61c19 

python setup.py install
