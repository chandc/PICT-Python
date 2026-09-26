# 2.5D unstructured LES solver on an NVIDIA GPU (A100 / H100 / any CUDA 12 device, x86-64 or aarch64).
# The only GPU dependency is CuPy; the one raw CUDA kernel (src/ucuda.py) compiles at first use (NVRTC).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /work
CMD ["bash"]
