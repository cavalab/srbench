# install conda environment
conda env create -f environment.yml

eval "$(conda shell.bash hook)"

conda init bash
conda activate srbench
conda info 

