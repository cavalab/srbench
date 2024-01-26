#!/bin/bash
# Make the dockerfile, including all algorithms in algorithms/
# run from repo root directory as 
#   bash scripts/make_dockerfile.sh

cat Dockerfile-template > autoDockerfile

algorithms=$(ls algorithms/)

for alg in ${algorithms[@]} ; do
    cat <<EOF >> autoDockerfile

FROM base as ${alg}
RUN bash install.sh  ${alg}
EOF
done

cat <<EOF >> autoDockerfile

FROM base as final
EOF

for alg in ${algorithms[@]} ; do
    cat <<EOF >> autoDockerfile
COPY --from=${alg} /opt/conda/envs/${alg} /opt/conda/envs/
RUN bash scripts/copy_algorithm_files.sh ${alg}

EOF
done
