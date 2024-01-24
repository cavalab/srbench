# install base srbench environment
conda env create -f base_environment.yml

eval "$(conda shell.bash hook)"

conda init bash
conda activate srbench
conda info 

