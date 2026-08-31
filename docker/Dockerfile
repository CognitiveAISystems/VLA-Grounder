ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    XDG_RUNTIME_DIR=/tmp/runtime-root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libgles2 \
    libglib2.0-0 \
    libglvnd0 \
    libgomp1 \
    libsm6 \
    libvulkan1 \
    libxext6 \
    libxrender1 \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/runtime-root \
    && chmod 700 /tmp/runtime-root \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN mkdir -p /opt/nvidia/vulkan/icd.d /opt/nvidia/egl_vendor.d \
    && curl --fail --location \
        https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json \
        --output /opt/nvidia/vulkan/icd.d/nvidia_icd.json \
    && curl --fail --location \
        https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json \
        --output /opt/nvidia/egl_vendor.d/10_nvidia.json

ENV VK_ICD_FILENAMES=/opt/nvidia/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/opt/nvidia/vulkan/icd.d/nvidia_icd.json \
    __EGL_VENDOR_LIBRARY_FILENAMES=/opt/nvidia/egl_vendor.d/10_nvidia.json \
    VLA_DATA_DIR=/workspace/data \
    LOCAL_RANK=0 \
    RANK=0 \
    WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=29500 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace
CMD ["sleep", "infinity"]
