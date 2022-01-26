#!/bin/bash
# note: make sure conda environment is installed 
# before running install. see configure.sh
conda activate srbench
conda info

failed=()

# install methods
echo "///////////////////////////"
echo "installing GP methods..."
echo "///////////////////////////"

# move to methods folder
cd experiment/methods/src/

# install all methods
for install_file in $(ls *.sh) ; do
    echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
    echo "Running $install_file"
    echo "////////////////////////////////////////////////////////////////////////////////"

    # Run install_file in same env:
    source $install_file

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
