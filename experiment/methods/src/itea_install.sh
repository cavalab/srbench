# install ITEA

# remove directory if it exists
if [ -d ITEA ]; then
    rm -rf ITEA
fi

git clone https://github.com/folivetti/ITEA.git

cd ITEA
#curl -sSL https://get.haskellstack.org/ | sh
#stack build
LIBGSL=$(ldconfig -p | grep "libgsl.so " | tr ' ' '\n' | grep /)
#cp ~/.conda/envs/srbench/libgsl.so bin/libgsl.so.0
cp $LIBGSL $LIBGSL.0

