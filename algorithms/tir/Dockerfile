################################################################################
# Notes: this image is large and many improvements are possible. 
# Sources:
# - https://uwekorn.com/2021/03/01/deploying-conda-environments-in-docker-how-to-do-it-right.html
# - https://pythonspeed.com/articles/conda-docker-image-size/
# micromamba is failing for PySR, so sticking with mambaforge for now.
# FROM --platform=linux/amd64 mambaorg/micromamba:0.21.2 as build
FROM condaforge/mambaforge:4.11.0-2 as base
################################################################################
# Nvidia code ##################################################################
################################################################################
ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"
################################################################################
ENV http_proxy="http://proxy.tch.harvard.edu:3128"
ENV https_proxy=http://proxy.tch.harvard.edu:3128
ENV HTTPS_PROXY=http://proxy.tch.harvard.edu:3128
ENV HTTP_PROXY=http://proxy.tch.harvard.edu:3128
# Install base packages.
USER root

ARG DEBIAN_FRONTEND=noninteractive

#RUN echo "deb http://ftp.debian.org/debian experimental main contrib non-free" >> /etc/apt/sources.list.d/experimental.list
RUN apt update && apt install -y \
    default-jdk \
    rsync \
    # bzip2 \
    # ca-certificates \
    curl \
    # git \
    # wget \
    build-essential \
    vim \
    git \
    libgmp3-dev \
    libblas-dev \
    liblapack-dev \
    libgsl-dev \
    ghc \
    cabal-install \
    jq && \
    rm -rf /var/lib/apt/lists/*

# Install env
################################################################################
USER $MAMBA_USER
SHELL ["/bin/bash", "-c"]
COPY . .
RUN mamba env create -f base_environment.yml
# RUN bash install.sh
RUN bash install.sh  afp
RUN bash install.sh  bingo
RUN bash install.sh  e2et
RUN bash install.sh  eplex
RUN bash install.sh  eql
RUN bash install.sh  feat
RUN bash install.sh  ffx
RUN bash install.sh  geneticengine
RUN bash install.sh  gpgomea
RUN bash install.sh  gplearn
RUN bash install.sh  gpzgd
RUN bash install.sh  itea
RUN bash install.sh  operon
RUN bash install.sh  ps-tree
RUN bash install.sh  pysr
RUN bash install.sh  qlattice
RUN bash install.sh  rils-rols
USER root
RUN apt install -y libgmp3-dev
USER $MAMBA_USER
RUN bash install.sh  tir
