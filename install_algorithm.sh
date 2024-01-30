# install one algorithm, located in directory passed as $1
# note: make sure conda srbench environment is installed 
set -e

# script to read yaml
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}


SUBNAME=$(basename $1)
SUBFOLDER="$(dirname $1)/${SUBNAME}"
SUBENV="srbench-$SUBNAME"

echo "SUBNAME: ${SUBNAME} ; SUBFOLDER: ${SUBFOLDER}"

install_base() {
    # install base srbench environment if it doesn't exist
    if conda info --envs | grep srbench | grep -v "srbench-"; then 
        echo "existing base srbench environment (not installing)"; 
    else 
        echo "installing base srbench environment"

        mamba env create -f base_environment.yml

        eval "$(conda shell.bash hook)"

        conda init bash
    fi
}


echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
echo ".................... Installing $SUBNAME ..."


build_clone_base_env="yes"
########################################
# read yaml
eval $(parse_yaml $SUBFOLDER/metadata.yml)
########################################

echo "build_clone_base_env: ${build_clone_base_env}"
if [ ${build_clone_base_env} == "yes" ] ; then

    echo "........................................"
    echo "Cloning base environment"
    echo "........................................"

    if conda info --envs | grep -q $SUBENV ; then 
        echo "not cloning base because ${SUBENV} already exists"
    else
        install_base
        conda create --name $SUBENV --clone srbench
        # update from base environment
        if test -f "${SUBFOLDER}/environment.yml" ; then
            echo "Update alg env from environment.yml"
            echo "........................................"
            mamba env update -n $SUBENV -f ${SUBFOLDER}/environment.yml
        fi
    fi
else

    echo "........................................"
    echo "Creating environment ${SUBENV} from scratch"
    echo "........................................"
    mamba create -n $SUBENV -f ${SUBFOLDER}/environment.yml
fi


if test -f "${SUBFOLDER}/requirements.txt" ; then
    echo "Update alg env from requirements.txt"
    echo "........................................"
    mamba run -n ${SUBENV} pip install -r ${SUBFOLDER}/requirements.txt
fi

cd $SUBFOLDER
if test -f "install.sh" ; then
    echo "running install.sh..."
    echo "........................................"
    mamba run -n $SUBENV bash install.sh
else
    echo "::warning::No install.sh file found in ${SUBFOLDER}."
    echo " Assuming the method is a conda package specified in environment.yml."
fi
cd -


# export env
echo "Exporting environment"
echo "........................................"
# conda env export -n $SUBENV > $SUBFOLDER/environment.lock.yml
conda env export -n $SUBENV > $SUBFOLDER/environment.lock.yml
conda env export -n $SUBENV 
echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
