# install deep-symbolic-regression

if [ -d deep-symbolic-regression ] ; then
    rm -rf deep-symbolic-regression
fi

git clone https://github.com/lacava/deep-symbolic-regression 

cd deep-symbolic-regression

pip install ./dsr # Install DSR package

