# note: make sure conda environment is installed 
# and activated before running install. 
# see configure.sh
# or run this as: 
#   conda run -n srbench bash install.sh

failed=()
succeeded=()

# install methods
echo "////////////////////////////////////////"
echo "installing SR methods from scripts..."
echo "////////////////////////////////////////"

# move to methods folder
cd experiment/methods/src/

# install all methods
for install_file in $(ls itea*.sh) ; do
    echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
    echo ".................... Running $install_file..."

    name=${install_file%.sh}
    # Option to redirect logs:
    # bash ${install_file} > "${name}.log" 2> "${name}.err"
    bash ${install_file} 

    if [ $? -gt 0 ];
    then
        # failed+=("${install_file}")
        failed+=("$name")
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! $install_file FAILED"
    else
        succeeded+=("$install_file")
        echo "........................................ $install_file complete"
    fi
    echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
done

echo "${#succeeded[@]} successful installs:"
for s in ${succeeded[@]} ; do
    echo "  "$s
done
echo "${#failed[@]} failed installs:"
for f in ${failed[@]} ; do
    echo "  "$f
done

if [ ${#failed[@]} -gt 0 ] ; then
    exit 1
else
    echo "All installations completed successfully."
fi
