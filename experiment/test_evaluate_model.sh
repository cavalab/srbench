# activate conda env
eval "$(conda shell.bash hook)"

conda init bash
conda activate srbench
conda info 

# This bash file mimics the behaviour of this piece
# of python code:
## MLs = [ml.split('/')[-1][:-3] for ml in glob('methods/*.py') if
##        not ml.split('/')[-1][:-3].startswith('_')]

cd ../experiment 

for ml in $(ls methods/*.py | grep -v '__init__.py' | sed 's/^methods\///g' | sed 's/\.py$//g') ; do
    python test_evaluate_model.py --ml="${ml}"
done
