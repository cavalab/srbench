FROM continuumio/anaconda3:2021.05

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /

# Make this dir to install JDK
RUN mkdir -p /usr/share/man/man1

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    default-jdk \
    bzip2 \
    ca-certificates \
    curl \
    git \
    wget \
    vim \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages.
RUN pip install --upgrade pip

# Checkout the latest version as of August 24th, 2021
WORKDIR /opt/app/
RUN git clone https://github.com/foolnotion/srbench.git
WORKDIR /opt/app/srbench/

RUN conda update conda -y
RUN bash configure.sh
SHELL ["conda", "run", "-n", "srbench", "/bin/bash", "-c"]
RUN bash install.sh
