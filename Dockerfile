FROM --platform=linux/amd64 mambaorg/micromamba:0.19.1

COPY environment.yml /tmp/environment.yml
RUN micromamba env create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Install base packages.
# RUN apt-get update --fix-missing && apt-get install -y \
#     default-jdk \
#     bzip2 \
#     ca-certificates \
#     curl \
#     gcc \
#     git \
#     wget \
#     vim \
#     build-essential \
#     jq && \
#     rm -rf /var/lib/apt/lists/*

# SHELL ["conda", "run", "-n", "srbench", "/bin/bash", "-c"]
# RUN ./install.sh
