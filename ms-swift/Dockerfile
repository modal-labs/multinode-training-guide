# Megatron 0.14.1, PyTorch 2.8, CUDA 12.8, FA 2.8


# CUDA 12.8.1 devel on Ubuntu 22.04 (NCCL 2.25.1, cuBLAS 12.8.4)
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11.11

ENV TZ=America/New_York
ENV arch=x86_64
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ninja-build \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
    lzma \
    xz-utils \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11.11 from source for faster execution
RUN cd /tmp && \
    wget -q https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-lto --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/Python-${PYTHON_VERSION}* && \
    ln -sf /usr/local/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

ENV PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV PYTHON_PIP_VERSION=23.0.1
ENV PYTHON_SETUPTOOLS_VERSION=65.5.1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    numpy \
    pybind11 \
    packaging \
    ninja \
    pydantic \
    sentencepiece \
    protobuf

# PyTorch 2.8.0 + CUDA 12.8
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir psutil

# Flash Attention 2.8.1 (compiled from source)
RUN pip install --no-cache-dir flash-attn==2.8.1 --no-build-isolation

# Megatron-Core 0.14.1
RUN pip install --no-cache-dir megatron-core==0.14.1

# MS-Swift 3.10.3
RUN pip install --no-cache-dir ms-swift==3.10.3

# Additional packages commonly in the image
RUN pip install --no-cache-dir \
    # Install transformers with GLM-4 MoE support, 4.57.3
    transformers==4.57.3 \
    datasets \
    accelerate \
    tensorboard \
    wandb

# Layer 100 (~86 MB): NVM / Node.js
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install --lts

# Layers 101-105: environment variables
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
ENV VLLM_USE_MODELSCOPE=True
ENV LMDEPLOY_USE_MODELSCOPE=True
ENV MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope/hub
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

WORKDIR /workspace

CMD ["/bin/bash"]