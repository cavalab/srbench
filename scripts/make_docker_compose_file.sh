#!/bin/bash
# Make the dockerfile, including all algorithms in algorithms/
# run from repo root directory as 
#   bash scripts/make_dockerfile.sh

cat Dockerfile-template > autoDockerfile
cat <<EOF > docker-compose.yml
version: '3'

services:
  base:
    image: srbench/base
    build:
      dockerfile: baseDockerfile
EOF

algorithms=$(ls algorithms/)
# algorithms=(
#     gplearn
#     ffx
# )

for alg in ${algorithms[@]} ; do
    # allow user to specify their own Dockerfile. 
    # otherwise use the default one (argDockerfile)
    if test -f "./algorithms/${alg}/Dockerfile" ; then
      dockerfile="./algorithms/${alg}/Dockerfile" 
    else
      dockerfile="argDockerfile"
    fi

    cat <<EOF >> docker-compose.yml
  ${alg}:
    build:
      dockerfile: ${dockerfile}
      args:
        ALGORITHM: ${alg}
    container_name: "srbench/${alg}"
    depends_on:
      - base
    volumes:
      - ./experiment:/srbench
  
EOF
done
