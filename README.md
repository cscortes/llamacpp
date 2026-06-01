# llama.cpp GPU/CUDA with CPU Fallback Docker Build

This repository provides a **multi-stage Dockerfile** supporting both NVIDIA CUDA GPU acceleration (via hardware profile) **and CPU fallback**. The build is now conditional on `CUDA_ARCH` (0 = no CUDA linkage; 86 for RTX 3060). Fixes libcuda.so.1 shared library error.

## Why These Dependencies? (GPU + CPU Edition)

### Builder Stage (`nvidia/cuda:12.5.1-devel-ubuntu22.04`)
- Full CUDA toolkit + nvcc **only when enabled** (conditional `-DGGML_CUDA=ON`).
- **build-essential, cmake, git, ccache**: Core build tools + compiler cache (volume-mounted for fast rebuilds).
- **libopenblas-dev, libomp-dev, pkg-config, libssl-dev**: BLAS/OpenMP for hybrid/CPU, server HTTPS support.
- Optimizations: Ninja, conditional CUDA arch, BUILD_TESTING=OFF.

CMake now uses shell logic for CPU-only vs GPU builds based on profile. ccache reduces rebuild time dramatically.

### Runtime Stage (`nvidia/cuda:12.5.1-runtime-ubuntu22.04`)
- CUDA runtime libs (cudart, compat including libcuda.so.1 via ld.so.conf.d) + BLAS/OpenMP.
- Explicit `/etc/ld.so.conf.d/cuda-compat.conf` + `ldconfig` ensures libs resolve without NVIDIA driver mount (fixes previous library errors).
- **libopenblas0 libomp5 libgomp1 libcurl4 ca-certificates**: Full runtime deps.

Image supports both profiles; ~1.5-3GB. See `Dockerfile` (conditional build) and `LessonsLearned.md` (May 15 section) for history and libcuda fix.

**Note**: For pure CPU without nvidia base, could switch runtime but current unified approach avoids base oscillation anti-pattern.

## Why Multi-Stage Dockerfile? (GPU)

- **Size Reduction**: Builder (CUDA devel ~several GB + build tools/sources) is discarded. Runtime uses lightweight `nvidia/cuda-runtime` (~1-2GB total).
- **Security/Optimization**: No compilers in final image. CUDA libs only where needed for inference.
- **Cache Efficiency**: ccache volume + layer caching speeds repeated `make build`.
- **GPU Passthrough**: `--gpus all` (or CDI) enables NVIDIA GPU in container.

## Step-by-Step Setup with Hardware Profile (RAM/VRAM Configuration)

1. **Prerequisites**
   - NVIDIA RTX 3060 (or similar) with CUDA-on-WSL drivers installed on the Windows host (see https://developer.nvidia.com/cuda/wsl). Verify with `nvidia-smi` in PowerShell — must show the GPU before anything else works.
   - Podman installed and initialised (`podman machine start`).
   - nvidia-container-toolkit installed inside the Podman VM with a CDI spec generated — run `make setup-gpu` after first `make reset` (see GPU Passthrough Setup below).
   - **Git Bash only** (`C:\Program Files\Git\bin\bash.exe` — PowerShell breaks Makefile heredocs, `||`, and variable expansion).
   - 30GB+ free disk (models ~13GB, devel image, ccache).
   - Edit `Makefile` profile block if your RAM/VRAM differs (default rog3060 uses RAM_GB=40, VRAM_GB=6).

2. **Configure Hardware Profile (RAM vs VRAM)**
   - Open `Makefile`.
   - In the `HARDWARE_PROFILE` block (after MODEL_SHORT), set:
     - `RAM_GB`: System RAM for --mlock (locks model in RAM), --no-mmap (full load to avoid disk I/O), and Podman machine memory limit. Use ~80% of your total (e.g. 32-36 for 40GB system to avoid OOM).
     - `VRAM_GB`: GPU VRAM for --n-gpu-layers, MoE expert offload (`--n-cpu-moe`), and KV cache quantization aggressiveness. Lower VRAM = more aggressive quant (q4_0 for keys/values) and more experts on CPU per video.
     - Derived vars (THREADS, N_GPU_LAYERS, VIDEO_OPT_FLAGS, PODMAN_RAM_MB, RUN_CAPS) auto-adjust for video tricks (17+ t/s, 256k ctx on 6GB VRAM).
   - Save. Use `HARDWARE_PROFILE=rog3060` (or your custom name) for all commands.

3. **GPU Passthrough Setup** (required for rog3060 profile; must repeat after every `make reset`)

   GPU access inside the Podman container requires four layers to be working in order:

   | # | Layer | What's needed | How to verify |
   |---|---|---|---|
   | 1 | Windows host | CUDA-on-WSL driver installed | `nvidia-smi` in PowerShell |
   | 2 | Podman VM | nvidia-container-toolkit + CDI spec | `make setup-gpu` |
   | 3 | Podman machine | Restart after toolkit install | `podman machine stop && podman machine start` |
   | 4 | Makefile | `--device nvidia.com/gpu=all` in RUN_CAPS | already set for rog3060 profile |

   ```bash
   # Step 1 — verify Windows host sees the GPU (PowerShell, not Git Bash):
   nvidia-smi                               # must show RTX 3060 before proceeding

   # Step 2 — install toolkit and generate CDI spec inside the Podman VM:
   make setup-gpu                           # installs nvidia-container-toolkit, writes /etc/cdi/nvidia.yaml

   # Step 3 — REQUIRED: restart Podman machine so CDI config takes effect:
   podman machine stop && podman machine start

   # Step 4 — start server with GPU flags and verify:
   HARDWARE_PROFILE=rog3060 make server MODEL_SHORT=qwen2
   make logs | grep -E "CUDA|cuda|n_gpu_layers|device"   # look for "found 1 CUDA devices"
   make live-stats                          # GPU access shows YES (/dev/dxg — WSL passthrough mode)
   ```

   > **After every `make reset`**: the Podman VM is destroyed and re-created. Re-run `make setup-gpu` then `podman machine stop && podman machine start` before starting the server.

   > **WSL note**: In WSL mode the GPU device is `/dev/dxg` (DirectX Graphics), not `/dev/nvidia0`. `make live-stats` reports "WSL passthrough mode" when this is working correctly.

4. **Build and Run** (always clean+build after Dockerfile changes)
   ```bash
   make getmodels                                       # Download models (~13GB; run once)
   make reset                                           # Init Podman machine (Git Bash only)
   make setup-gpu                                       # Install GPU toolkit in Podman VM
   podman machine stop && podman machine start          # Restart so CDI takes effect
   HARDWARE_PROFILE=rog3060 make build                  # Compile with CUDA_ARCH=86
   HARDWARE_PROFILE=rog3060 make server MODEL_SHORT=qwen2
   make logs | grep -E "CUDA|cuda|n_gpu_layers|device" # Verify GPU initialised
   make live-stats                                      # Full status: GPU, model, host URL
   make win-forward                                     # Map VM IP to localhost (if needed)
   # CPU fallback (no CUDA, no toolkit needed):
   # make build && make server MODEL_SHORT=qwen2
   ```

5. **Monitoring and Tuning**
   - `make logs` (now succeeds without libcuda.so.1 error; watch for CUDA detection, BLAS fallback, "listening on http://0.0.0.0:18080", tokens/sec).
   - `nvidia-smi` inside container (if GPU profile + toolkit) or host to monitor VRAM/layers offloaded.
   - `make ccache-stats`, `make test`, `podman logs --tail 50 llamacpp-server`.
   - Adjust profile in Makefile for your hardware (RAM_GB vs VRAM_GB tradeoffs for mlock/no-mmap/KV quant/MoE).
   - For custom: Add ifeq in HARDWARE_PROFILE block; rebuild after changes.
   - See LessonsLearned.md May 15 section for full verification sequence and permanent lessons.

**Server API (OpenAI-compatible):**
```bash
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deep", "messages": [{"role": "user", "content": "Write Python SQL code"}]}'
```

**Expected Performance (rog3060 profile on your hardware)**: 17+ tokens/sec with MoE models, 256k context via video tricks (experts on CPU RAM, compute on GPU, turbo KV quant, mlock for stability). Default profile falls back to CPU.

**Example for Lower-Spec Hardware (RAM=20GB, VRAM=5.5GB)**
- Edit the `HARDWARE_PROFILE` block in `Makefile` (add or modify an entry):
  ```
  ifeq ($(HARDWARE_PROFILE),lowspec)
    CUDA_ARCH = 75
    THREADS = 8
    N_GPU_LAYERS = 20
    RAM_GB = 20
    VRAM_GB = 5.5
    PODMAN_RAM_MB = 16384
    VIDEO_OPT_FLAGS = --no-mmap --mlock --n-cpu-moe 41 --cache-type-k q4_0 --cache-type-v q8_0
    RUN_CAPS = --cap-add=IPC_LOCK --ipc=host
  endif
  ```
- Commands (uses profile to tune for lower RAM/VRAM; more aggressive KV quant, fewer GPU layers):
  ```bash
  make clean                  # Removes image/server (cache preserved)
  HARDWARE_PROFILE=lowspec make build   # Builds with adjusted CUDA arch
  HARDWARE_PROFILE=lowspec make server MODEL_SHORT=deep   # Or use custom target
  make test                   # API test
  make logs                   # View output (check layers, tokens/sec, VRAM)
  ```

See `LessonsLearned.md` (Hardware Profile section) for troubleshooting, full verification, and permanent practices. `make help` for targets.
## Makefile Targets

Run `make info` for the full structured reference. Key targets:

| Target | Description |
|---|---|
| `make getmodels` | Download all 3 GGUF models (~13 GB) |
| `make reset` | Initialise/reinitialise Podman + WSL (Git Bash only) |
| `make setup-gpu` | Install nvidia-container-toolkit + generate CDI spec in Podman VM |
| `make build` | Build the container image (use `HARDWARE_PROFILE=rog3060` for CUDA) |
| `make server` | Start the API server (use `MODEL_SHORT=qwen2\|phi\|deep`) |
| `make live-stats` | Show container, process, model, GPU access/VRAM, and host URL |
| `make stop` | Stop the server container |
| `make build-extension` | Build the VS Code Copilot extension |
| `make info` | Full target reference with primary/secondary sections |

## Models (./models/)

| Short | Model | Size | Strengths |
|-------|-------|------|-----------|
| phi | Phi-3.5-mini-instruct-q4_K_M.gguf | ~2.5GB | Fast C/Python/SQL |
| qwen2 (default) | Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf | ~4.7GB | Top coding (all langs) |
| deep | DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf | ~5.5GB | Advanced code gen |

## Optimizations
- Parallel build (`-j$(nproc)`).
- Stripped binary (`strip --strip-all`).
- ccache stats printed post-build.

## Notes
- **GPU access** uses `--device nvidia.com/gpu=all` (CDI) rather than `--gpus all`. Requires `make setup-gpu` + `podman machine stop && podman machine start` after every `make reset`.
- **WSL GPU device**: In WSL mode the GPU appears as `/dev/dxg` (not `/dev/nvidia0`). `make live-stats` reports "WSL passthrough mode" when working. Use `make logs | grep -i cuda` to confirm CUDA initialised inside the container.
- **Full GPU setup sequence**: `make reset` → `make setup-gpu` → `podman machine stop && podman machine start` → `HARDWARE_PROFILE=rog3060 make build` → `HARDWARE_PROFILE=rog3060 make server` → `make logs` → `make live-stats`.
- **If `make live-stats` shows GPU access NO after all steps**: the Podman machine restart was likely skipped. Stop the server, restart the machine, and start the server again.
- Tested with coding models on CUDA (RTX 3060, CUDA_ARCH=86). Update the profile block in `Makefile` for other GPUs.

## Features
- Multi-stage (~250MB image).
- ccache for fast rebuilds.
- OpenBLAS/OpenMP CPU accel.
- Pre-configured coding models (7B-class).
- Makefile-driven (no Docker CLI needed).

## Customization
- **GPU (CUDA):** Add `-DLLAMA_CUDA=ON` to CMakeLists in Dockerfile builder stage.
- **Other models:** Edit `getmodels` curl URLs.
- **Ctx size:** Edit `-c 4096` in targets.
- See [llama.cpp docs](https://github.com/ggerganov/llama.cpp).

See `CHANGES.md` for version history.