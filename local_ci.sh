SUBNAME=$1
SUBFOLDER="algorithms/$1"
echo "testing $SUBFOLDER"
# SUBFOLDER=official_competitors/$SUBNAME 
SUBENV=srbench-$SUBNAME 
# update base env
# mamba env update -n srbench -f environment.yml 

# install method
cd $SUBFOLDER
pwd
echo "Installing dependencies for ${SUBNAME}"
echo "........................................"
echo "Copying base environment"
echo "........................................"
conda create --name $SUBENV --clone srbench
if [ -e environment.yml ] ; then 
    echo "Installing conda dependencies"
    echo "........................................"
    mamba env update -n $SUBENV -f environment.yml
fi
if [ -e requirements.txt ] ; then 
    echo "Installing pip dependencies"
    echo "........................................"
    mamba run -n $SUBENV pip install -r requirements.txt
fi

eval "$(conda shell.bash hook)"
conda init bash
conda activate $SUBENV
if test -f "install.sh" ; then
echo "running install.sh..."
echo "........................................"
bash install.sh
else
echo "::warning::No install.sh file found in ${SUBFOLDER}. Assuming the method is a conda package specified in environment.yml."
fi

# Copy files and environment
echo "Copying files and environment to experiment/methods ..."
echo "........................................"
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
echo "activating conda env $SUBENV..."
echo "........................................"
conda activate $SUBENV 
conda env list 
conda info 
python -m pytest -v test_algorithm.py --ml $SUBNAME
python -m pytest -v test_population.py --ml $SUBNAME

# Store Competitor
# cd ..
# rsync -avz --exclude=".git" submission/$SUBNAME official_competitors/
# rm -rf submission/$SUBNAME
