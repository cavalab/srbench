SUBNAME=$1
SUBFOLDER=submission/$SUBNAME 
SUBENV=srcomp-$SUBNAME 
# update base env
mamba env update -n srcomp -f environment.yml 

# install method
cd $SUBFOLDER
pwd
echo "Installing dependencies for ${SUBNAME}"
echo "Copying base environment"
# conda create --name $SUBENV --clone srcomp
echo "Installing conda dependencies"
mamba env update -n $SUBENV -f environment.yml

eval "$(conda shell.bash hook)"
conda init bash
conda activate $SUBENV
if test -f "install.sh" ; then
echo "running install.sh..."
bash install.sh
else
echo "::warning::No install.sh file found in ${SUBFOLDER}. Assuming the method is a conda package specified in environment.yml."
fi

# Copy files and environment
cd ../../
mkdir -p experiment/methods/$SUBNAME
cp $SUBFOLDER/regressor.py experiment/methods/$SUBNAME/
cp $SUBFOLDER/metadata.yml experiment/methods/$SUBNAME/
touch experiment/methods/$SUBNAME/__init__.py

# export env
echo "Exporting environment"
conda env export -n $SUBENV > $SUBFOLDER/environment.lock.yml

# Test Method
cd experiment
pwd
ls
conda activate $SUBENV 
python -m pytest -v test_submission.py --ml $SUBNAME

# Store Competitor
cd ..
rsync -avz --exclude=".git" submission/$SUBNAME official_competitors/
# rm -rf submission/$SUBNAME
