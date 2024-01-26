#!/bin/bash
# Make the dockerfile, including all algorithms in algorithms/
# run from repo root directory as 
#   bash scripts/make_dockerfile.sh

cat Dockerfile-template > autoDockerfile
cat <<EOF > docker-compose.yml
version: '3'

services:
EOF

algorithms=$(ls algorithms/)

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
            context: .
            dockerfile: ${dockerfile}
            args:
                ALG: ${alg}
        container_name: "srbench-${alg}"
        stdin_open: true
        tty: true
        volumes:
          - ./:/srbench
        network_mode: host
        deploy:
          resources:
            reservations:
              devices:
                - capabilities: [gpu]
EOF
done
