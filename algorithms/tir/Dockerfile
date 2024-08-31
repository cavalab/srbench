FROM srbench/base AS build
################################################################################

USER root

RUN apt update && apt install -y \
    curl \
    build-essential \
    libgmp3-dev \
    libblas-dev \
    liblapack-dev \
    libgsl-dev \
    ghc \
    cabal-install \
    && \
    rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
# Install env
ARG ALGORITHM
ENV ALGORITHM="tir"
WORKDIR "/install"
COPY --chown=$MAMBA_USER:$MAMBA_USER base_environment.yml install_algorithm.sh .
COPY --chown=$MAMBA_USER:$MAMBA_USER ./algorithms/${ALGORITHM} ./${ALGORITHM}
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN bash install_algorithm.sh $ALGORITHM \
    && micromamba clean --all --yes
WORKDIR "/srbench"
USER root

