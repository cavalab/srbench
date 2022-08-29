# note: make sure conda srcomp environment is installed 
set -e

if (($#==1)); #check if number of arguments is 1 
then
    subnames=($1)
else
    subnames=(
    # Bingo
    # E2ET
    # HROCH
    # PS-Tree
    # QLattice
    # TaylorGP
    # eql
    # geneticengine
    # gpzgd
    # nsga-dcgp
    # operon
    # pysr
    # uDSR
    )
fi

failed=()
succeeded=()

# install methods
echo "////////////////////////////////////////"
echo "installing SR methods..."
echo "////////////////////////////////////////"

# install all methods
for SUBNAME in ${subnames[@]} ; do

    echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
    echo ".................... Installing $SUBNAME ..."

    SUBFOLDER=official_competitors/$SUBNAME
    SUBENV=srcomp-$SUBNAME
    # install method
    cd $SUBFOLDER
    pwd
    echo "Installing dependencies for ${SUBNAME}"
    echo "........................................"
    echo "Copying base environment"
    echo "........................................"
    conda create --name $SUBENV --clone srcomp
    echo "Installing conda dependencies"
    echo "........................................"
    mamba env update -n $SUBENV -f environment.yml

    eval "$(conda shell.bash hook)"
    conda init bash
    conda activate $SUBENV
    if test -f "install.sh" ; then
        echo "running install.sh..."
        echo "........................................"
        bash install.sh
        else
        echo "::warning::No install.sh file found in ${SUBFOLDER}."
        echo " Assuming the method is a conda package specified in environment.yml."
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
    echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
done
