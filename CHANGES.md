# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-05-03
### Added
- Full **GPU (CUDA)** support: Switched to official `nvidia/cuda:12.5.1-devel` (builder) and `runtime` (final) Ubuntu-based images.
- Proper `-DLLAMA_CUDA=ON`, CUDA architectures, apt-based deps, artifact prep for multi-stage.
- Restored `--gpus all` in Makefile cli/server targets.
- Updated README.md, Makefile help, and `LessonsLearned.md` with GPU setup, troubleshooting (manifest unknown, CDI errors on Windows Podman), and migration from CPU Fedora version.

### Changed
- Dockerfile completely reworked for CUDA compatibility (fixed previous dnf/Ubuntu mismatch and COPY issues).
- Image now ~2GB+ but provides massive inference speedup on NVIDIA GPUs.
- ccache volume still supported for fast rebuilds.

### Fixed
- Base image tag (used valid 12.5.1 to avoid "manifest unknown").
- Build and runtime consistency for CUDA libs.
- Server/CLI now leverage GPU by default.

## [0.5.0] - 2026-05-14
### Added
- **Hardware profile system** in `Makefile` (after MODEL_SHORT block): `HARDWARE_PROFILE=rog3060` (or default).
- RAM_GB (system memory for --mlock/--no-mmap/Podman --memory) vs VRAM_GB (GPU offload, n-gpu-layers, KV cache quant, MoE expert pinning per video) vars.
- Video optimizations integrated: --no-mmap, --mlock, tuned layers/MoE, KV cache types (q4_0/q8_0), higher context.
- Convenience `rog3060` target, dynamic build/reset/server targets, updated help.
- Dockerfile now supports CUDA via ARG CUDA_ARCH (default 0 for CPU fallback; 86 for RTX 3060), switched to nvidia/cuda-devel builder.

### Changed
- reset now uses profile PODMAN_RAM_MB (32GB for rog3060).
- server uses profile flags/caps/THREADS/N_GPU_LAYERS (overrides Dockerfile CMD).
- build passes CUDA_ARCH arg.
- Docs updated with profile usage, RAM vs VRAM explanation, expected performance.

## [0.6.0] - 2026-05-15
### Added
- Conditional CUDA build in Dockerfile (`if CUDA_ARCH != 0` then `-DGGML_CUDA=ON` + arch; else OFF). Defaults to CPU fallback.
- Runtime ld.so.conf.d/cuda-compat.conf + ldconfig to register `/usr/local/cuda-12.5/compat` libs.

### Fixed
- `llama-server: error while loading shared libraries: libcuda.so.1: cannot open shared object file` (exit 127 on container start).
- Hardcoded CUDA flags prevented true CPU fallback; now profile-driven (CUDA_ARCH=0 for default).
- Updated all docs with coordinated edits.

### Changed
- Dockerfile header/comments, ARG default=0, cmake logic, runtime stage for compat libs.
- README and LessonsLearned reflect CPU/GPU duality and new verification steps.

## [Unreleased]
- (future changes here)

## [0.3.0] - 2026-05-03
### Added
- `make cli` / `make server` targets with `MODEL_SHORT` (phi/qwen2/deep) & `PORT` vars.
- `llama-server` binary to Docker image (OpenAI-compatible API).
- Comprehensive Makefile documentation (header, comments, best practices).

### Fixed
- Makefile formatting/corruption from prior edits.

## [0.2.0] - 2026-05-03
- Added \`make getmodels\` target to download 3 coding-optimized GGUF models to \`./models/\`: Qwen2.5-Coder-7B-Instruct-Q4_K_M, Phi-3.5-mini-instruct-q4_K_M, DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.
- Updated \`make help\` to document new target.

## [0.1.0] - 2026-05-03
- Initial release: Multi-stage Podman/Dockerfile for llama.cpp (CPU-only with OpenBLAS, ccache).
- Makefile targets: build, clean, run, help.