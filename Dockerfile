## Build time
# Use the specified Python runtime as a parent image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS build

# Set the working directory in the container
WORKDIR /usr/src/app

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0 12.0+PTX" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV DEBIAN_FRONTEND=noninteractive
RUN echo "deb http://archive.ubuntu.com/ubuntu jammy main universe restricted multiverse\n\
    deb http://archive.ubuntu.com/ubuntu jammy-updates main universe restricted multiverse\n\
    deb http://archive.ubuntu.com/ubuntu jammy-security main universe restricted multiverse" > /etc/apt/sources.list

# Install basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    ca-certificates \
    curl \
    git \
    apt-transport-https \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Delete nvidia apt list and Install required packages
RUN DEBIAN_FRONTEND=noninteractive \
    && rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    && apt-get -yq update \
    && apt-get install --no-install-recommends -yq \
    build-essential \
    ninja-build \
    gcc-11 \
    apt-transport-https \
    ca-certificates \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup virtual env
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install any needed packages specified in requirements.txt
COPY req.txt .
COPY gradio_image_prompter-0.1.0-py3-none-any.whl .
RUN --mount=type=cache,id=pip,target=/root/.cache \
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,id=pip,target=/root/.cache \
    pip install mmcv==2.2.0+pt2.7.0cu128 \
    --extra-index-url https://miropsota.github.io/torch_packages_builder
RUN --mount=type=cache,id=pip,target=/root/.cache \
    pip install -r req.txt


WORKDIR /usr/src/app/models/GeCo/ops
COPY Deformable-DETR/models/ops .

# Run the setup script and the test script
RUN CC=/usr/bin/gcc-11 python3 setup.py build && \
    pip install .

# Pre-download ResNet50 backbone (used by AODC encoder)
RUN python3 -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"

## Runtime
# Use the specified Python runtime as a parent image
FROM ubuntu:22.04

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0 12.0+PTX" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:/home/user/.local/bin:$PATH" \
    HOME=/home/user

RUN DEBIAN_FRONTEND=noninteractive apt-get -yq update && apt-get install --no-install-recommends -yq \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN useradd -m -u 1000 user && chown -R user /app

COPY --from=build --chown=user /opt/venv /opt/venv

# Copy pre-downloaded model weights (ResNet50 for AODC)
COPY --from=build --chown=user /root/.cache/torch /home/user/.cache/torch

COPY --chown=user CNTQG_multitrain_ca44.pth .
COPY --chown=user Deformable-DETR Deformable-DETR
COPY --chown=user Deformable-DETR/models/ops models/ops
COPY --chown=user configs configs
COPY --chown=user utils utils
COPY --chown=user models models
COPY --chown=user sam2 sam2
COPY --chown=user AODC AODC
COPY --chown=user api_server.py ./
COPY --chown=user inference_point.py ./
COPY --chown=user demo_gradio.py ./

USER user
# Expose the port the API server will run on
EXPOSE 7860
# Default command to run the FastAPI server
CMD ["python3", "api_server.py"]
