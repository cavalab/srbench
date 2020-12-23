# install conda environment
conda env create -f environment.yml

eval "$(conda shell.bash hook)"

conda activate regression-benchmarks
conda info 

# install methods
echo "///////////////////////////"
echo "installing GP methods..."
echo "///////////////////////////"

# move to methods folder
cd experiment/methods/src/

# install all methods
for install_file in $(ls *.sh) ; do
    echo "///////////////////////////"
    echo "Running $install_file"
    echo "///////////////////////////"

    bash $install_file

    echo "///////////////////////////"
    echo "Finished $install_file"
    echo "///////////////////////////"
done
