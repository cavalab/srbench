# note: make sure conda environment is installed 
# and activated before running install. 
# see configure.sh
# or run this as: 
#   conda run -n srbench bash install.sh

failed=()
succeeded=()
declare -A failures=()

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
    msg=$(bash $install_file 2>&1 1>/dev/null)
    # echo $code
    # echo $msg
    if [ -n "$msg" ]; 
    then
        failed+=($install_file)
        failures[$install_file]="$msg"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "$install_file FAILED"
    else
        succeeded+=($install_file)
        echo "////////////////////////////////////////////////////////////////////////////////"
        echo "Finished $install_file"
        echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    fi
done

if [ ${#failed[@]} -gt 0 ] ; then
    echo "vvvvvvvvvvvvvvv failures vvvvvvvvvvvvvvv"
    # n=0
    for f in ${!failures[@]} ; do
        echo "---------- $f ----------"
        echo "${failures[$f]}"
        echo "----------------------------------------"
    done

    echo "${#succeeded[@]} successful installs:"
    for s in ${!succeeded[@]} ; do
        echo "  "$s
    done
    echo "${#failed[@]} failed installs:"
    for f in ${!failures[@]} ; do
        echo "  "$f
    done
    exit 1
else
    echo "All installations completed successfully."
fi
