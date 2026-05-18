# Trimmed from Ollama Dockerfile. Updated for hardware profiles: supports CUDA via
# build arg CUDA_ARCH (0=CPU-only fallback, no CUDA dep; 86 for RTX 3060). Builder uses
# nvidia/cuda:12.5.1-devel-ubuntu22.04. Runtime ensures compat libs via ld.so.conf for
# libcuda.so.1 (fixes shared library error even without full GPU passthrough/CDI).
# Profile (RAM_GB/VRAM_GB) drives flags via Makefile. Multi-stage with /output.

FROM nvidia/cuda:12.5.1-devel-ubuntu22.04 AS builder

# Default to CPU (0); override via --build-arg CUDA_ARCH=86 for GPU (rog3060 profile).
# Conditional CMake prevents unnecessary CUDA linkage in CPU builds.
ARG CUDA_ARCH=0

# Deps for both CPU/GPU (Ollama-inspired + CUDA toolkit in base; pkg-config, libssl for server).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake ninja-build ccache build-essential \
    libopenblas-dev libomp-dev pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp (depth 1, Ollama-style).
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /llama.cpp

WORKDIR /llama.cpp

# Conditional build using command substitution for CMake flags (avoids shell var expansion/PID/$$ issues in Docker sh -c).
# CUDA only if CUDA_ARCH != 0. Retains BLAS/OpenMP for hybrid/MoE performance. BUILD_TESTING=OFF
# skips tests. ccache -z/-s for stats (persisted via volume). Echo shows active profile.
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

# Prepare artifacts (Ollama-style copy to ensure paths exist across CPU/GPU builds).
RUN mkdir -p /output/bin && \
    cp build/bin/llama-server /output/bin/llama-server && \
    ls -la /output/bin

FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04

# Runtime libs + CUDA compat config to ensure libcuda.so.1 (and symlinks) are found by ld.so
# (resolves "cannot open shared object file" error; compat dir already in base image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 libomp5 libgomp1 libcurl4 ca-certificates \
    && echo "/usr/local/cuda-12.5/compat" > /etc/ld.so.conf.d/cuda-compat.conf \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Copy server binary (Ollama-style).
COPY --from=builder /output/bin/llama-server /usr/local/bin/llama-server

# Ensure all libs (incl. CUDA compat) are registered.
RUN ldconfig

# Server-only (Ollama-style ENTRYPOINT/CMD with proper defaults; overridden by Makefile for profile).
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "18080", "-c", "4096", "--n-gpu-layers", "0", "--threads", "4"]

