# install ITEA

# remove directory if it exists
if [ -d ITEA ]; then
    rm -rf ITEA
fi

git clone https://github.com/folivetti/ITEA.git

cd ITEA
curl -sSL https://get.haskellstack.org/ | sh 
echo "stack build"
stack build --allow-different-user 
