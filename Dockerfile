FROM --platform=linux/amd64 mambaorg/micromamba:0.19.1

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
USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba env create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Install conda, so that install.sh files can be run without
# modification.
RUN micromamba install -y --name base -c conda-forge conda
ENV PATH=$PATH:/opt/conda/bin
RUN echo 'export PATH=$PATH:/opt/conda/bin' >> ~/.bashrc

# Always run inside srbench:
RUN conda init bash
RUN echo "conda activate srbench" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Copy remaining files and install
COPY  --chown=$MAMBA_USER:$MAMBA_USER . .
RUN ./install.sh
CMD ["/bin/bash", "--login"]
