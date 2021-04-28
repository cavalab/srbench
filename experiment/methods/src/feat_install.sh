#install feat

# remove directory if it exists
if [ -d "feat" ]; then 
    rm -rf feat
fi

git clone http://github.com/lacava/feat

cd feat

./configure
./install y
