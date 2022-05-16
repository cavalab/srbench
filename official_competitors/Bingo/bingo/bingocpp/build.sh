mkdir -p build
cd build 
PYEXECUTABLE=`which python`
cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE:FILEPATH=$PYEXECUTABLE ..
make VERBOSE=1 -j
make gtest
cd ..
