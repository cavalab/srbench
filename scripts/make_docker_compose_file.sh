#!/bin/bash
# Make the dockerfile, including all algorithms in algorithms/
# run from repo root directory as 
#   bash scripts/make_dockerfile.sh

cat <<EOF > docker-compose.yml
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
    # otherwise use the default one (alg-Dockerfile)
    if test -f "./algorithms/${alg}/Dockerfile" ; then
      dockerfile="./algorithms/${alg}/Dockerfile" 
    else
      dockerfile="alg-Dockerfile"
    fi

    cat <<EOF >> docker-compose.yml
  ${alg}:
    image: srbench/${alg}
    container_name: srbench-${alg}
    build:
      dockerfile: ${dockerfile}
      args:
        ALGORITHM: ${alg}
    depends_on:
      - base
    volumes:
      - ./experiment:/srbench
EOF
done
