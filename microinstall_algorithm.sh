# install one algorithm with micromamba, located in directory passed as $1
set -e


SUBNAME=$(basename $1)
SUBFOLDER="$(dirname $1)/${SUBNAME}"
SUBENV="base"

echo "SUBNAME: ${SUBNAME} ; SUBFOLDER: ${SUBFOLDER}"



echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
echo ".................... Installing $SUBNAME ..."

echo "........................................"
echo "Creating environment"
echo "........................................"
add_base_env=true
if test -f "${SUBFOLDER}/environment.lock" ; then 
    echo "using ${SUBFOLDER}/environment.lock"
    micromamba install -n base -y -f ${SUBFOLDER}/environment.lock
    add_base_env=false
elif test -f "${SUBFOLDER}/environment.yml" ; then 
    echo "using ${SUBFOLDER}/environment.yml ... "
    micromamba install -n base -y -f ${SUBFOLDER}/environment.yml
fi
if test -f "${SUBFOLDER}/requirements.txt" ; then
    echo "Update alg env from requirements.txt"
    echo "........................................"
    micromamba install -n base -y -c conda-forge pip
    pip install -r ${SUBFOLDER}/requirements.txt
    pip cache purge
fi

if $add_base_env ; then
    micromamba install -n base -y -f base_environment.yml
fi

if test -f "${SUBFOLDER}/install.sh" ; then
    echo "running install.sh..."
    echo "........................................"
    cd $SUBFOLDER
    micromamba run -n base bash install.sh
    cd -
else
    echo "::warning::No install.sh file found in ${SUBFOLDER}."
    echo " Assuming the method is a conda package specified in environment.yml."
fi

# export env
echo "Exporting environment"
echo "........................................"
# conda env export > $SUBFOLDER/environment.lock.yml
micromamba env export --explicit --md5 > $SUBFOLDER/environment.lock
echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
