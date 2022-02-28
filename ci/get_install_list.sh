# generate json install list for ci
cd experiment/methods/src/
install_list="{\"install_file\":["
# install all methods
for install_file in $(ls *.sh) ; do
    echo $install_file
    install_list="${install_list}\"${install_file}\"," 
done
cd ../../../ci/
install_list="${install_list%,}]}"
echo $install_list > installs.json
