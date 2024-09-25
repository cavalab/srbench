# remove directory if it exists
if [ -d "gpg" ]; then
    rm -rf gpg
fi

git clone https://github.com/matigekunstintelligentie/MultiGPG.git

cd gpg
# TODO update this...
git checkout 804484f0eff7d12d6ac99e623087dcec3c8b7300
make
