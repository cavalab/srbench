# install conda environment
conda env create -f environment.yml

eval "$(conda shell.bash hook)"

conda activate regression-benchmarks
conda info 

# install methods
echo "installing GP methods..."

# move to methods folder
cd experiment/methods/src/

# install all methods
for install_file in $(ls *.sh) ; do
    bash $install_file
done
	# bash $install_file
