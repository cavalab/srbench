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
for install_file in $(ls *.sh) ; do
    echo "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
    echo "Running $install_file..."

    # Run install_file in same env:
    bash $install_file 2>"${install_file}.err" 1>"${install_file}.log"
    # echo $code
    # echo $msg
    # if [ -n "$msg" ]; 
    if [ $? -gt 0 ];
    then
        failed+=($install_file)
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "$install_file FAILED"
    else
        succeeded+=($install_file)
        echo "$install_file complete"
        echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    fi
done

if [ ${#failed[@]} -gt 0 ] ; then
    echo "vvvvvvvvvvvvvvv failures vvvvvvvvvvvvvvv"
    # n=0
    for f in ${!failed[@]} ; do
        echo "---------- $f ----------"
        echo "---- stdout: "
        cat  "${f}.log"
        echo "---- stderr: "
        echo "${f}.err"
        echo "----------------------------------------"
    done

    echo "${#succeeded[@]} successful installs:"
    for s in ${!succeeded[@]} ; do
        echo "  "$s
    done
    echo "${#failed[@]} failed installs:"
    for f in ${!failed[@]} ; do
        echo "  "$f
    done
    exit 1
else
    echo "All installations completed successfully."
fi
