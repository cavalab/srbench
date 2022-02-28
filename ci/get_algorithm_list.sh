# generate json algorithm list for testing
alg_list="{\"ml\":["
cd experiment/methods/
for ml in $(ls *.py | grep -v '__init__.py' | sed 's/\.py$//g') ; do
    echo $ml
    alg_list="${alg_list}\"${ml}\"," 
done
cd ../../
alg_list="${alg_list%,}]}"
echo $alg_list > ci/algs.json
cat ci/algs.json


alg_list="{\"ml\":["
cd experiment/methods/tuned/
for ml in $(ls *.py | grep -v '__init__.py' | sed 's/\.py$//g') ; do
    echo $ml
    alg_list="${alg_list}\"tuned.${ml}\"," 
done
alg_list="${alg_list%,}]}"
cd ../../../
echo $alg_list > ci/algs-tuned.json
cat ci/algs-tuned.json
