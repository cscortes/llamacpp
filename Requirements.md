# Requirements.md - llama.cpp GPU (CUDA) Setup

**Permanent Requirements** for this project (as of May 3, 2026). These are non-negotiable for reliable, repeatable builds and high-performance inference. All have been implemented in `Dockerfile`, `Makefile`, named volumes, and Git Bash workflows. See `LessonsLearned.md` for history of issues resolved to reach this state.

## 1. Terminal (Permanent)
- **Must use Git Bash** (`C:\Program Files\Git\bin\bash.exe`).
  - All `make`, `podman`, and debug commands are executed via `& "C:\Program Files\Git\bin\bash.exe" -c "cd 'C:/Users/MrLui/Code/llamacpp' && command..."`.
  - PowerShell is incompatible with Makefile syntax, volume path parsing (`/mnt/c...;C:` errors), and WSL/Podman integration.
- Run `make help` in Git Bash for current targets.

## 2. Hardware (Permanent)
- ROG RTX 3060 laptop (or equivalent) with profile support: RAM_GB=40 (for mlock, no-mmap, large ctx), VRAM_GB=6 (tuned n-gpu-layers, aggressive KV cache per video for 256k ctx and 17+ t/s on MoE models like DeepSeek).
- NVIDIA GPU with CUDA compute capability ≥ 7.5 (sm_86 for 3060 via profile CUDA_ARCH).
- Windows 10/11 with WSL2 + sufficient RAM (profile sets Podman machine memory).

## 3. Software & Environment (Permanent)
- **Hardware Profile**: Activate with `HARDWARE_PROFILE=rog3060` (or default). Controls RAM vs VRAM tradeoffs, video flags (--no-mmap, --mlock, KV quant, MoE offload), CUDA build, Podman resources.
- **Podman** (with GPU/CDI support via profile RUN_CAPS and reset):
  - NVIDIA drivers, CUDA on WSL, nvidia-container-toolkit in machine.
  - Profile adds --cap-add=IPC_LOCK --ipc=host for mlock.
- **Git Bash** mandatory.
- CUDA 12.5+ via nvidia/cuda images (profile selects arch).
- At least 30GB+ free disk (devel image, models, cache).

## 4. Caching (Permanent Requirement)
- **ccache via named volume `ccache-llama`**:
  - Persists **permanently** across `make clean`, image rebuilds, container restarts, and Podman machine resets.
  - `make clean` removes **only the image**; use `make clean-cache` to reset cache (rarely needed).
  - First build: 10-20+ min (CUDA kernel compilation).
  - Subsequent builds: near-instant (cache hit rates >90% for source changes).
  - Dockerfile runs `ccache -F 0 && ccache -z && ... && ccache -s` in builder; stats printed on every build.
- Volume created automatically in `make build`.

## 5. Build Requirements (Permanent)
- `make build`: Uses `nvidia/cuda:12.5.1-devel-ubuntu22.04` (apt + `ninja-build`, `ccache`, OpenBLAS dev, CUDA toolkit).
  - Flags: `-DLLAMA_CUDA=ON`, `-DLLAMA_SERVER=ON`, `-DCMAKE_CUDA_ARCHITECTURES=75` (adjust for your GPU), `-G Ninja`, Release, OpenBLAS.
  - Multi-stage with `/output` artifact prep for reliable COPY (avoids "no such file or directory" for `build/lib/` in static builds).
  - `--volume ccache-llama:/root/.ccache`.
- Models: Run `make getmodels` (downloads Phi-3.5, Qwen2.5-Coder-7B, DeepSeek-Coder-V2-Lite to `./models/`).
- Image tagged `llamacpp:latest`.

## 6. Runtime Requirements (Permanent)
- `make cli` / `make server`: 
  - `--gpus all`, volume mount for models (`-v ./models:/models`).
  - Default: Qwen2.5-Coder-7B (`-c 4096 -n 256`).
  - Server: OpenAI-compatible API at `http://localhost:8080/v1/chat/completions`, `--host 0.0.0.0`.
- `make stop` to clean up server container.
- CUDA runtime image (`nvidia/cuda:12.5.1-runtime-ubuntu22.04`) + `libopenblas0`, `libomp5`, `ldconfig`.

## 7. Other Permanent Practices
- **Never remove the `ccache-llama` volume** unless intentionally resetting cache.
- Always run in Git Bash.
- Update `CMAKE_CUDA_ARCHITECTURES` in Dockerfile for your specific GPU (or use `native`/`all-major`).
- Monitor with `podman logs llamacpp-server`, `nvidia-smi` inside container, or ccache stats.
- For production: Add `.dockerignore`, healthchecks, resource limits, and consider flash-attention or higher quant models.
- Dependencies in `Dockerfile` (no host installs needed beyond NVIDIA drivers and Podman).

## Validation
- `make build && make server` should succeed without volume/CDI/manifest/COPY/CMake errors.
- Cache hits confirmed via `ccache -s` output.
- GPU utilization visible in logs/server responses (5-20x faster than CPU).

This document is the single source of truth for permanent requirements. Update it alongside code changes. See `Dockerfile`, `Makefile`, `LessonsLearned.md`, and `CHANGES.md` for implementation details.

---
*Created per user request. Consolidates all permanent elements (ccache volume, Git Bash, GPU CUDA 12.5.1, named volumes, build flags, models, terminal, clean semantics) into one reference. No more scattered notes.*
