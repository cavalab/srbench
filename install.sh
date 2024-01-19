# note: make sure conda srbench environment is installed 
set -e

if (($#==1)); #check if number of arguments is 1 
then
    subnames=($1)
else
    subnames=$(ls algorithms/)
fi

failed=()
succeeded=()

# install methods
echo "////////////////////////////////////////"
echo "installing SR methods..."
echo ${subnames}
echo "////////////////////////////////////////"

# install all methods
for SUBNAME in ${subnames[@]} ; do

    echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
    echo ".................... Installing $SUBNAME ..."

    SUBFOLDER=algorithms/$SUBNAME
    SUBENV=srbench-$SUBNAME
    # install method
    # cd $SUBFOLDER
    pwd
    echo "Installing dependencies for ${SUBNAME}"
    echo "........................................"
    echo "Making environment"
    echo "........................................"
    # conda create --name $SUBENV --clone srbench
    mamba env create --name $SUBENV -f base_environment.yml -f ${SUBFOLDER}/environment.yml
    # if test -f "environment.yml" ; then
    #     echo "Update alg env from environment.yml"
    #     echo "........................................"
    #     mamba env update -n $SUBENV -f ${SUBFOLDER}/environment.yml
    # fi

    if test -f "requirements.txt" ; then
        echo "Update alg env from requirements.txt"
        echo "........................................"
        mamba run -n srbench-${SUBENV} pip install -r ${SUBFOLDER}/requirements.txt
    fi

    eval "$(conda shell.bash hook)"
    conda init bash
    conda activate $SUBENV
    cd $SUBFOLDER
    if test -f "install.sh" ; then
        echo "running install.sh..."
        echo "........................................"
        bash install.sh
    else
        echo "::warning::No install.sh file found in ${SUBFOLDER}."
        echo " Assuming the method is a conda package specified in environment.yml."
    fi
    cd ../../

    # Copy files and environment
    # echo "Copying files and environment to experiment/methods ..."
    # echo "........................................"
    # cd ../../
    mkdir -p experiment/methods/$SUBNAME
    cp $SUBFOLDER/regressor.py experiment/methods/$SUBNAME/
    cp $SUBFOLDER/metadata.yml experiment/methods/$SUBNAME/
    touch experiment/methods/$SUBNAME/__init__.py

    # export env
    echo "Exporting environment"
    # conda env export -n $SUBENV > $SUBFOLDER/environment.lock.yml
    conda env export -n $SUBENV > $SUBFOLDER/environment.lock.yml
    echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
done
