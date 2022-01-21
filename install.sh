# note: make sure conda environment is installed 
# before running install. see configure.sh
conda activate srbench
conda info

failed=()

# install methods
echo "///////////////////////////"
echo "installing GP methods..."
echo "///////////////////////////"

ml=$1

# Get install scripts:
install_files=$(python get_install_script.py $ml)

# move to methods folder
cd experiment/methods/src/


# install methods for this algorithm:
for install_file in "${install_files[@]}"; do
    echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
    echo "Running $install_file"
    echo "////////////////////////////////////////////////////////////////////////////////"

    bash $install_file

    if [ $? -gt 0 ]
    then
        failed+=($install_file)
    fi

    echo "////////////////////////////////////////////////////////////////////////////////"
    echo "Finished $install_file"
    echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
done

if [ ${#failed[@]} -gt 0 ] ; then
    echo "${#failed[@]} installs failed: ${failed}"
else
    echo "All installations completed successfully."
fi
