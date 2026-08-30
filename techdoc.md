# Technical reference

How this Podman wrapper builds and runs llama.cpp. For first-run commands and profile names, see [README.md](README.md). Build history and failed approaches live in [LessonsLearned.md](LessonsLearned.md).

## Image layout

The [Dockerfile](Dockerfile) is multi-stage so the devel toolkit never ships in the runtime image.

| Stage | Image | Role |
|---|---|---|
| Builder | `nvidia/cuda:12.8.1-devel-ubuntu22.04` | nvcc, cmake, ninja, ccache, clone and compile `llama-server` |
| Runtime | `nvidia/cuda:12.8.1-runtime-ubuntu22.04` | CUDA runtime + OpenBLAS/OpenMP + the stripped server binary |

Builder apt packages: `ca-certificates`, `git`, `cmake`, `ninja-build`, `ccache`, `build-essential`, `libopenblas-dev`, `libomp-dev`, `pkg-config`, `libssl-dev`.

Runtime apt packages: `libopenblas0`, `libomp5`, `libgomp1`, `libcurl4`, `ca-certificates`.

`/etc/ld.so.conf.d/cuda-compat.conf` points at `/usr/local/cuda-12.8/compat` and `ldconfig` runs so `libcuda.so.1` resolves even without a full host-driver mount. CUDA **12.8.1** is required for Blackwell (`sm_120` / RTX 50-series). 12.5 cannot emit those kernels.

The builder copies `llama-server` to `/output/bin` so the runtime `COPY --from=builder` path is stable for both CPU and GPU builds (`-DBUILD_SHARED_LIBS=OFF` does not produce a `build/lib/` tree).

A CPU-only Ubuntu runtime would shrink the image further. A single CUDA base for both profiles avoids swapping bases between CPU and GPU builds.

## CMake flags

Passed in the builder `RUN`. `CUDA_ARCH` is a Docker build arg from the Makefile profile (`0`, `86`, or `120`).

```
cmake -B build -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_SERVER=ON
  -DGGML_BLAS=ON
  -DGGML_BLAS_VENDOR=OpenBLAS
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_TESTING=OFF
  # CUDA_ARCH=0  → -DGGML_CUDA=OFF
  # otherwise    → -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
cmake --build build --config Release --target llama-server -j$(nproc)
```

| `CUDA_ARCH` | Profile | Meaning |
|---|---|---|
| `0` | `default` | No CUDA linkage; faster CPU-only compile |
| `86` | `rog3060` | Ampere (RTX 3060) |
| `120` | `rtx5060ti` | Blackwell (RTX 5060 Ti, compute 12.0) |

llama.cpp is cloned shallow from `main` at build time. Only the `llama-server` target is built. ENTRYPOINT is `llama-server`; the Makefile overrides CMD with profile flags.

## ccache

`make build` depends on `./ccache-llama` (gitignored). Make creates that directory if it is missing, then bind-mounts it at `/root/.ccache`. Podman on Windows treats `--volume name:path` as a host path, so a named volume fails with `faccessat .../ccache-llama`. The local directory survives `make clean`, `make reset`, and `make prune`. First CUDA compile is 10–20+ minutes; later builds should be mostly cache hits (`make ccache-stats`). `make clean-cache` deletes `./ccache-llama`.

## Hardware profiles

Set on the make command: `HARDWARE_PROFILE=<name> make <target>`. One-shots (`make rtx5060ti`, `make rog3060`) set the profile for build/server; `make rtx5060ti` also applies it to `reset` so the VM gets 20 GB.

Variables in the Makefile `ifeq ($(HARDWARE_PROFILE),...)` block:

| Variable | Role |
|---|---|
| `CUDA_ARCH` | Passed to `podman build --build-arg`. `0` disables CUDA. |
| `THREADS` | `--threads` for llama-server |
| `N_GPU_LAYERS` | `--n-gpu-layers`. `99` = all layers. `0` = CPU. |
| `RAM_GB` | Host RAM budget (documentation + reset echo). Used conceptually for mlock / no-mmap / large ctx. |
| `VRAM_GB` | GPU memory budget (documentation + layer/quant choices) |
| `PODMAN_RAM_MB` | `podman machine init --memory` during `make reset` |
| `CONTEXT_SIZE` | `-c` context length |
| `VIDEO_OPT_FLAGS` | Extra server flags (`--no-mmap`, `--parallel`, optional `--mlock`, MoE, KV quant) |
| `RUN_CAPS` | `podman run` caps. GPU profiles: `--cap-add=IPC_LOCK --ipc=host --device nvidia.com/gpu=all` |

Built-in values:

| Profile | `CUDA_ARCH` | `N_GPU_LAYERS` | `VRAM_GB` | `PODMAN_RAM_MB` | `CONTEXT_SIZE` |
|---|---|---|---|---|---|
| `default` | 0 | 0 | 0 | 16384 | 4096 |
| `rtx5060ti` | 120 | 99 | 16 | 20480 | 16384 |
| `rog3060` | 86 | 22 | 6 | 32768 | 16384 |

`rog3060` assumes ~40 GB host RAM. `rtx5060ti` assumes ~31 GB host RAM (20 GB VM). Do not give the VM more RAM than the host can spare.

### Adding a profile

Copy an `ifeq` / `else ifeq` block in the Makefile. Example (older GPU, tighter RAM):

```
ifeq ($(HARDWARE_PROFILE),lowspec)
  CUDA_ARCH = 75
  THREADS = 8
  N_GPU_LAYERS = 20
  RAM_GB = 20
  VRAM_GB = 5.5
  PODMAN_RAM_MB = 16384
  CONTEXT_SIZE = 8192
  VIDEO_OPT_FLAGS = --no-mmap --mlock --n-cpu-moe 41 --cache-type-k q4_0 --cache-type-v q8_0
  RUN_CAPS = --cap-add=IPC_LOCK --ipc=host --device nvidia.com/gpu=all
endif
```

`--cache-type-k/v` needs Flash Attention compiled in (`-DLLAMA_FLASH_ATTN=ON` / `--flash-attn`). The current Dockerfile does not enable it; default fp16 KV cache is used.

Then:

```bash
make clean
HARDWARE_PROFILE=lowspec make reset
make setup-gpu
HARDWARE_PROFILE=lowspec make build
HARDWARE_PROFILE=lowspec make server MODEL_SHORT=deep
```

`RAM_GB`: use ~80% of host RAM when sizing `PODMAN_RAM_MB`. Lower `VRAM_GB` means fewer GPU layers and more aggressive KV/MoE offload.

## GPU passthrough (Windows + Podman WSL)

Four layers, in order:

| # | Layer | What | Verify |
|---|---|---|---|
| 1 | Windows host | CUDA-on-WSL driver | `nvidia-smi` in PowerShell |
| 2 | Podman VM | nvidia-container-toolkit + CDI spec | `make setup-gpu` writes `/etc/cdi/nvidia.yaml` |
| 3 | Podman machine | Restart so CDI is loaded | `setup-gpu` already restarts; or `podman machine stop && podman machine start` |
| 4 | `podman run` | `--device nvidia.com/gpu=all` in `RUN_CAPS` | set on GPU profiles |

This project uses CDI (`--device nvidia.com/gpu=all`), not `--gpus all`.

In WSL the GPU node is `/dev/dxg` (DirectX Graphics), not `/dev/nvidia0`. `make live-stats` reports "WSL passthrough mode" when `/dev/dxg` is visible in the container.

### `make reset` wipes the toolkit

`reset` runs `podman machine rm` and `podman machine init`. That destroys VM state, including nvidia-container-toolkit and `/etc/cdi`. After every reset, run `make setup-gpu` before a GPU `make server`. One-shots (`make rtx5060ti`, `make rog3060`) already do this.

If `live-stats` shows GPU access NO after a full setup, the machine was usually not restarted after toolkit install. `make stop`, `podman machine stop && podman machine start`, then start the server again with the same `HARDWARE_PROFILE`.

## Windows networking

Podman on Windows (slirp4netns/WSL) often publishes on the VM IPv4 address, not `localhost`, and may bind IPv6-only.

- `make vm-ip` prints the machine IPv4 (fallback documented as `172.26.156.205`).
- `make win-forward` (Administrator) adds a `netsh interface portproxy` from `0.0.0.0:18080` to that VM IP.
- `make test` hits `http://localhost:18080/v1/models`.

If localhost fails, use `http://<vm-ip>:18080` or re-run `win-forward` as Admin.

## Models

`make getmodels` downloads into `./models/` (gitignored) and skips files that already exist. The container mounts `./models` as `/models`.

The original trio was picked for an **RTX 3060 (6GB VRAM)**: 7B Q4 and DeepSeek-Coder-V2-Lite **Q3** so weights plus a modest KV cache still fit. On an **RTX 5060 Ti (16GB)** those files work but underuse the GPU. Prefer a 14B Q4, or a higher-quality 7B quant.

### Files the Makefile knows

| Short (`MODEL_SHORT`) | File | Size | Aimed at |
|---|---|---|---|
| `phi` | `Phi-3.5-mini-instruct-q4_K_M.gguf` | ~2.5 GB | 3060 / CPU, fastest coding |
| `qwen2` (default) | `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` | ~4.7 GB | 3060 coding |
| `deep` | `DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf` | ~5.5 GB | 3060 stretch coding |
| `qwen14` | `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` | ~9 GB | 5060 Ti coding (daily) |
| `deep5` | `DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf` | ~12 GB | 5060 Ti coding (stretch) |
| `chat3` | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` | ~2 GB | 3060 language |
| `chat7` | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | ~4.7 GB | 3060 / 5060 Ti language (fast) |
| `chat14` | `Qwen2.5-14B-Instruct-Q4_K_M.gguf` | ~9 GB | 5060 Ti language (daily) |
| `chat14q6` | `Qwen2.5-14B-Instruct-Q6_K.gguf` | ~12 GB | 5060 Ti language (stretch) |
| `aya8` | `aya-expanse-8b-Q4_K_M.gguf` | ~5 GB | Multilingual (23 languages; 3060 stretch / 5060 easy). CC-BY-NC. |

llama-server does **not** do text-to-image generation.

Downloads: `make getmodels` / `getmodels-5060ti` / `getmodels-deep5` (coding), `getmodels-lang` / `getmodels-lang-5060ti` / `getmodels-chat14q6` (language), `getmodels-aya` (multilingual). Profile one-shots still pull the daily coding file only (`qwen14` or the 3060 trio).

Sources: [microsoft/Phi-3.5-mini-instruct-GGUF](https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF), [bartowski/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF), [bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF](https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF), [bartowski/Qwen2.5-Coder-14B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF), [bartowski/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF), [bartowski/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF), [bartowski/Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF), [bartowski/aya-expanse-8b-GGUF](https://huggingface.co/bartowski/aya-expanse-8b-GGUF) (upstream [CohereForAI/aya-expanse-8b](https://huggingface.co/CohereForAI/aya-expanse-8b), CC-BY-NC).

`chat14q6` must stay above `chat14`, `deep5` above `deep`, and `qwen14` above `qwen` in the Makefile `ifeq` chain (`findstring` matches prefixes).

### VRAM budget (16GB, `rtx5060ti`)

Rough split with `N_GPU_LAYERS=99` and `CONTEXT_SIZE=16384`:

| Weights | KV cache (16k, fp16) | Fits 16GB? |
|---|---|---|
| 7B Q4 (~4.7 GB) | small | Yes, lots of headroom (underuses the card) |
| 7B Q6 / Q8 (~6–8 GB) | small | Yes — quality upgrade if you stay at 7B |
| 14B Q4 (~9 GB) | moderate | Yes — recommended |
| 14B Q5 / Q6 (~10–12 GB) | moderate | Tight; drop context if OOM |
| 32B Q4 (~18 GB) | — | No |

On 6GB (`rog3060`), stay with the `getmodels` trio. Lite Q3 exists because Lite Q4 plus 22 GPU layers and context would OOM.

Add another file by dropping a `.gguf` in `./models/`, adding a `MODEL_SHORT` `ifeq` (more specific names first), and an optional `getmodels-*` curl.

## Verification

After `make server` (with a GPU profile):

```bash
make logs | grep -E "CUDA|cuda|n_gpu_layers|device"
make live-stats
```

Look for CUDA device detection in logs and `GPU access : YES` (`/dev/dxg`) in live-stats. `API check : FAIL` usually means run `make win-forward`.

Host `nvidia-smi` shows VRAM on Windows. Inside the container, WSL often exposes `nvidia-smi` under `/usr/lib/wsl/drivers` rather than `PATH`; `live-stats` searches both.

## VS Code extension

`make build-extension` requires **Node.js 18+** (`npm install` + `tsc` in `vscode-llamacpp/`). Default endpoint is `http://localhost:18080`. Open `vscode-llamacpp/` and press F5. Not required to run the server.

## Other docs

- [README.md](README.md) — quickstart
- [LessonsLearned.md](LessonsLearned.md) — historical failures (manifest tags, CDI, libcuda, CPU-fallback anti-pattern)
- [Requirements.md](Requirements.md) — older permanent-requirements note (some ports/flags are stale vs the Makefile)
- [CHANGES.md](CHANGES.md) — changelog
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — upstream
