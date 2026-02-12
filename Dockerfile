FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# ======================
# Base system
# ======================
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    vim \
    tmux \
    unzip \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ======================
# Python 3.9
# ======================
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3-setuptools \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip \
 && python -m pip install --upgrade pip setuptools wheel

# ======================
# CUDA / Torch env
# ======================
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV FORCE_CUDA=1

# ======================
# PyTorch 1.10 + CUDA 11.3
# ======================
RUN pip install \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# ======================
# Python dependencies
# ======================
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
