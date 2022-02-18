FROM --platform=linux/amd64 mambaorg/micromamba:0.21.2 as build
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

# Install base packages.
USER root

RUN apt update && apt install -y \
    default-jdk \
    bzip2 \
    ca-certificates \
    curl \
    git \
    wget \
    vim \
    jq && \
    rm -rf /var/lib/apt/lists/*

# Install env
FROM build as build-mamba
USER $MAMBA_USER
WORKDIR /srbench/
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /srbench/environment.yml
RUN micromamba create -y -f /srbench/environment.yml \
    && micromamba clean --all --yes
ENV CONDA_PREFIX $MAMBA_ROOT_PREFIX
# conda is currently only needed for PySR
# RUN micromamba install -y --name base -c conda-forge conda
# ENV PATH=$PATH:/opt/conda/bin
 # RUN echo 'export PATH=$PATH:/opt/conda/bin' >> ~/.bashrc

SHELL ["micromamba", "run", "-n", "srbench", "/bin/bash", "-c"]

# Always run inside srbench:
# RUN source ~/.bashrc && conda init bash
# RUN echo "conda activate srbench" >> ~/.bashrc

# Copy remaining files and install
FROM build-mamba as base
RUN ls
RUN pwd
COPY --chown=$MAMBA_USER:$MAMBA_USER . /srbench/
# COPY --chown=$MAMBA_USER:$MAMBA_USER /opt/conda/ /opt/conda
# RUN source ~/.bashrc && source install.sh
# RUN bash configure.sh
# COPY --chown=$MAMBA_USER:$MAMBA_USER . .
# CMD ["/bin/bash", "-c"]
CMD echo "Hello from the base image."
ENTRYPOINT ["micromamba", "run", "-n", "srbench"]
FROM base as dsr_install
WORKDIR /srbench/experiment/methods/src/
RUN /srbench/experiment/methods/src/dsr_install.sh
WORKDIR /srbench/
FROM base as ellyn_install
WORKDIR /srbench/experiment/methods/src/
RUN /srbench/experiment/methods/src/ellyn_install.sh
WORKDIR /srbench/
FROM base as feat_install
WORKDIR /srbench/experiment/methods/src/
RUN /srbench/experiment/methods/src/feat_install.sh
WORKDIR /srbench/
FROM base as gpgomea_install
WORKDIR /srbench/experiment/methods/src/
RUN /srbench/experiment/methods/src/gpgomea_install.sh
WORKDIR /srbench/
FROM base as gsgp_install
WORKDIR /srbench/experiment/methods/src/
RUN /srbench/experiment/methods/src/gsgp_install.sh
WORKDIR /srbench/
FROM base as itea_install
WORKDIR /srbench/experiment/methods/src/
RUN /srbench/experiment/methods/src/itea_install.sh
WORKDIR /srbench/
FROM base as operon_install
WORKDIR /srbench/experiment/methods/src/
RUN /srbench/experiment/methods/src/operon_install.sh
WORKDIR /srbench/
FROM base as pysr_install
WORKDIR /srbench/experiment/methods/src/
RUN /srbench/experiment/methods/src/pysr_install.sh
WORKDIR /srbench/

#combine installations
FROM base as final

COPY --from=dsr_install /opt/conda/envs/srbench/ /opt/conda/envs/srbench/
COPY --from=ellyn_install /opt/conda/envs/srbench/ /opt/conda/envs/srbench/
COPY --from=feat_install /opt/conda/envs/srbench/ /opt/conda/envs/srbench/
COPY --from=gpgomea_install /opt/conda/envs/srbench/ /opt/conda/envs/srbench/
COPY --from=gsgp_install /opt/conda/envs/srbench/ /opt/conda/envs/srbench/
COPY --from=itea_install /opt/conda/envs/srbench/ /opt/conda/envs/srbench/
COPY --from=operon_install /opt/conda/envs/srbench/ /opt/conda/envs/srbench/
COPY --from=pysr_install /opt/conda/envs/srbench/ /opt/conda/envs/srbench/
