# make a Dockerfile for building each method in parallel
# move to methods folder
path=experiment/methods/src/
cd $path
files=$(ls *.sh)
cd ../../../
i=0
target_list="{\"target\":["
COMBO="\n#combine installations\nFROM base as final\n"

dockerfile="Dockerfile"
cp Dockerfile.base Dockerfile

for install_file in ${files[@]} ; do
    
    # dockerfile="dockerfile.${install_file%*\.sh}"
    name="${install_file%*\.sh}"

    TXT="FROM base as $name\nWORKDIR /srbench/$path\nRUN /srbench/$path$install_file\nWORKDIR /srbench/"
    echo -e "$TXT" >> "$dockerfile"

    # COMBO="${COMBO}\nFROM lacava/srbench:${dockerfile}"
    COMBO="${COMBO}\nCOPY --from=$name /opt/conda/envs/srbench/ /opt/conda/envs/srbench/"

    target_list="${target_list}\"${name}\"," 

    ((i++))
done
target_list="${target_list%,}]}"
echo $target_list > ci/docker_targets.json
echo "ci/docker_targets.json:"
cat ci/docker_targets.json

# changed to hardcoding this
# echo -e $COMBO >> Dockerfile
echo "Dockerfile:"
cat Dockerfile
