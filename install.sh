# note: make sure conda srbench environment is installed 
set -e

if (($#>=1)); #check if number of arguments is 1 
then
    subnames=($1)
else
    subnames=$(ls algorithms/)
fi

failed=()
succeeded=()

# install methods
echo "////////////////////////////////////////"
echo "installing these SR methods:"
echo ${subnames}
echo "////////////////////////////////////////"

# install all methods
for SUBNAME in ${subnames[@]} ; do

    bash install_algorithm.sh algorithms/$SUBNAME 
    # Copy files and environment
    bash scripts/copy_algorithm_files.sh $SUBNAME

    # export env
    echo "Exporting environment"
    echo "........................................"
    # conda env export -n $SUBENV > $SUBFOLDER/environment.lock.yml
    conda env export -n $SUBENV > $SUBFOLDER/environment.lock.yml
    conda env export -n $SUBENV 
    echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
done
