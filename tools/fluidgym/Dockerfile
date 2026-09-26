FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev git \
    build-essential ninja-build && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir torch==2.9.* --index-url https://download.pytorch.org/whl/cu128
COPY . /opt/fluidgym
# GB10 is sm_121: same-major SASS compat means 12.0 kernels run; build both.
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN cd /opt/fluidgym && python3 -m pip install --no-cache-dir --no-build-isolation .
RUN python3 -m pip install --no-cache-dir platformdirs gymnasium pandas "matplotlib>=3.9,<3.10" seaborn "huggingface_hub>=1.0.0" scikit-image imageio
