# install conda environment
conda env create -f environment.yml

# move to methods folder
cd experiment/methods/src/

# install all methods
for install_file in $(ls *.sh) ; do
	./install_file


