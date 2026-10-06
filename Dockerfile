# Multi-stage llama.cpp server image. CUDA_ARCH build arg: 0 = CPU-only,
# 86 = RTX 3060 (rog3060), 120 = RTX 5060 Ti Blackwell (rtx5060ti).
# CUDA 12.8.1 is required for sm_120.
#
# Runtime registers compat libs via ld.so.conf
# so libcuda.so.1 resolves even without a full driver mount.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

# Default to CPU (0); override via --build-arg CUDA_ARCH=120 (rtx5060ti) or 86 (rog3060).
ARG CUDA_ARCH=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake ninja-build ccache build-essential \
    libopenblas-dev libomp-dev pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /llama.cpp

WORKDIR /llama.cpp

# CUDA only if CUDA_ARCH != 0. BLAS/OpenMP kept for hybrid/MoE. BUILD_TESTING=OFF.
# ccache -z/-s for stats (persisted via the ./ccache-llama bind mount).
RUN ccache -z && \
    echo "=== Building with CUDA_ARCH=${CUDA_ARCH} (CPU fallback if 0) ===" && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_SERVER=ON \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        $(if [ "${CUDA_ARCH}" = "0" ]; then echo "-DGGML_CUDA=OFF"; else echo "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"; fi) \
    && cmake --build build --config Release --target llama-server -j$(nproc) && \
    ccache -s

RUN mkdir -p /output/bin && \
    cp build/bin/llama-server /output/bin/llama-server && \
    ls -la /output/bin

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 libomp5 libgomp1 libcurl4 ca-certificates \
    && echo "/usr/local/cuda-12.8/compat" > /etc/ld.so.conf.d/cuda-compat.conf \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

COPY --from=builder /output/bin/llama-server /usr/local/bin/llama-server

RUN ldconfig

ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "18080", "-c", "4096", "--n-gpu-layers", "0", "--threads", "4"]
