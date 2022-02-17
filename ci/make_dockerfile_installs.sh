# make a Dockerfile for building each method in parallel
# move to methods folder
path=experiment/methods/src/
cd $path
files=$(ls *.sh)
cd ../../../
i=0
file_list="{\"dockerfile\":["
COMBO="#combine installations"
for install_file in ${files[@]} ; do

    TXT="FROM lacava/srbench:base\nWORKDIR /srbench/$path\nRUN /srbench/$path$install_file\nWORKDIR /srbench/"
    echo -e "$TXT" > "Dockerfile.$i"

    COMBO="${COMBO}\nFROM Dockerfile.${i}"

    file_list="${file_list}\"Dockerfile.${i}\"," 

    ((i++))
done
file_list="${file_list%,}]}"
echo $file_list > ci/docker_files.json
cat ci/docker_files.json
echo -e $COMBO > Dockerfile.combo
cat Dockerfile.combo
