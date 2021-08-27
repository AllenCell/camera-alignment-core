# This provides a build environment for this repository on Jenkins.
# Once/if this project no longer uses Jenkins, this file can be safely deleted.

FROM docker-virtual.artifactory.corp.alleninstitute.org/aics-ubuntu20.04:2020_10_12-32

ARG USER=jenkins
ARG GROUP=jenkins
ARG UID=305
ARG GID=305

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    sudo \
    software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip

RUN /usr/sbin/groupadd -g ${GID} ${GROUP} && \
    /usr/sbin/useradd --gid ${GID} -G sudo -N --shell /bin/bash --create-home --uid ${UID} ${USER}
