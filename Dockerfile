FROM --platform=linux/amd64 mambaorg/micromamba:0.19.1

COPY environment.yml /tmp/environment.yml
RUN micromamba env create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Install base packages.
USER root
RUN apt update && apt install -y \
    default-jdk \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    wget \
    vim \
    build-essential \
    jq && \
    rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
RUN micromamba install -y --name base -c conda-forge conda
ENV PATH=$PATH:/opt/conda/bin

# Always run inside srbench:
RUN conda init bash
RUN echo "conda activate srbench" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

COPY . .
RUN ./install.sh
CMD ["/bin/bash", "--login"]