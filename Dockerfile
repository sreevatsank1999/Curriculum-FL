# FROM jupyter/scipy-notebook:python-3.9.12
FROM nvcr.io/nvidia/pytorch:22.05-py3

ARG USER=torch
ARG USERDIR=/home/${USER}

RUN adduser --gecos "" ${USER}
RUN passwd -d ${USER}

RUN apt -y update && \
    DEBIAN_FRONTEND=noninteractive apt -y install \
    build-essential \
    software-properties-common \
    gnupg2 \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    ffmpeg \
    curl \
    patchelf \
    libglfw3 \  
    libglfw3-dev \
    cmake \
    sudo \
    zlib1g \
    zlib1g-dev 

RUN pip3 install --upgrade pip

RUN pip3 install \
    numpy 

RUN pip3 install \
    medmnist 

# RUN conda install \
# #     pytorch \
#     torchvision 
#     torchaudio \
#     cudatoolkit=10.2 -c pytorch

# RUN pip3 install \
#     opencv-python

RUN mkdir -p ${USERDIR}/.vscode-server && chown ${USER}:${USER} ${USERDIR}/.vscode-server
VOLUME ${USERDIR}/.vscode-server

EXPOSE 8888

USER ${USER}
