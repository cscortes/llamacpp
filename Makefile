# ======================================================================================
# Podman/Docker multi-stage build for llama.cpp (GPU/CUDA).
# Features: ccache, OpenBLAS, CUDA acceleration, llama-cli/server, coding GGUF models.
# Base: nvidia/cuda:12.8.1 (devel + runtime). Podman networking (slirp4netns/WSL) often binds IPv6-only on Windows.
#
# Quick Start (Windows, Git Bash):
#   make rtx5060ti            # RTX 5060 Ti: 14B, GPU toolkit, build, server
#   make rog3060              # RTX 3060: 7B set, GPU toolkit, build, server
#   make live-stats && make test
# CPU: make getmodels && make reset && make build && make server
# See README.md for profiles; techdoc.md for internals.
#
# Models: coding phi/qwen2/deep/qwen14/deep5 | language chat3/chat7/chat14/chat14q6 | aya8
# Vars: MODEL_SHORT=... PORT=18080

#
# Server: http://localhost:18080 or http://172.26.156.205:18080/v1/chat/completions
# Stop: make stop
# Clean: make clean
# Note: Podman VM IP (172.26.156.205) used for reliable access. win-forward creates netsh proxy.
# ======================================================================================

# Force bash (Git Bash on Windows; required for reset/heredoc/||).
SHELL := bash
# Podman ssh passes UserKnownHostsFile=NUL. Git Bash/MSYS treats that as ./NUL
# (a real file of SSH host keys). Disable MSYS path conversion so NUL stays the discard device.
export MSYS_NO_PATHCONV := 1
export MSYS2_ARG_CONV_EXCL := *

.PHONY: build clean reset run help info getmodels getmodels-5060ti getmodels-deep5 getmodels-lang getmodels-lang-5060ti getmodels-chat14q6 getmodels-aya list-models cli server stop prune build-extension live-stats setup-gpu rog3060 rog3060-build rog3060-server restart-podman rtx5060ti rtx5060ti-reset rtx5060ti-build rtx5060ti-server
.DEFAULT_GOAL := info

# Model short names (more specific findstring matches first: chat14q6 before chat14, deep5 before deep).
#   Coding:     phi, qwen2, deep, qwen14, deep5
#   Language:   chat3, chat7, chat14, chat14q6, aya8 (23-language)
MODEL_SHORT ?= qwen2
PORT ?= 18080
DOWNLOAD_HINT := make getmodels

ifeq ($(findstring phi,$(MODEL_SHORT)),phi)
MODEL_FILE = Phi-3.5-mini-instruct-q4_K_M.gguf
DOWNLOAD_HINT = make getmodels
else ifeq ($(findstring qwen14,$(MODEL_SHORT)),qwen14)
MODEL_FILE = Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf
DOWNLOAD_HINT = make getmodels-5060ti
else ifeq ($(findstring chat14q6,$(MODEL_SHORT)),chat14q6)
MODEL_FILE = Qwen2.5-14B-Instruct-Q6_K.gguf
DOWNLOAD_HINT = make getmodels-chat14q6
else ifeq ($(findstring chat14,$(MODEL_SHORT)),chat14)
MODEL_FILE = Qwen2.5-14B-Instruct-Q4_K_M.gguf
DOWNLOAD_HINT = make getmodels-lang-5060ti
else ifeq ($(findstring chat7,$(MODEL_SHORT)),chat7)
MODEL_FILE = Qwen2.5-7B-Instruct-Q4_K_M.gguf
DOWNLOAD_HINT = make getmodels-lang
else ifeq ($(findstring chat3,$(MODEL_SHORT)),chat3)
MODEL_FILE = Qwen2.5-3B-Instruct-Q4_K_M.gguf
DOWNLOAD_HINT = make getmodels-lang
else ifeq ($(findstring aya8,$(MODEL_SHORT)),aya8)
MODEL_FILE = aya-expanse-8b-Q4_K_M.gguf
DOWNLOAD_HINT = make getmodels-aya
else ifeq ($(findstring qwen,$(MODEL_SHORT)),qwen)
MODEL_FILE = Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
DOWNLOAD_HINT = make getmodels
else ifeq ($(findstring deep5,$(MODEL_SHORT)),deep5)
MODEL_FILE = DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf
DOWNLOAD_HINT = make getmodels-deep5
else ifeq ($(findstring deep,$(MODEL_SHORT)),deep)
MODEL_FILE = DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf
DOWNLOAD_HINT = make getmodels
else
MODEL_FILE = Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
DOWNLOAD_HINT = make getmodels
endif

# Hardware profile (HARDWARE_PROFILE=rtx5060ti|rog3060 make ...).
# Controls RAM_GB (system memory, mlock, no-mmap, Podman --memory) vs VRAM_GB
# (n-gpu-layers, MoE experts to CPU, KV cache quant). Extensible: copy an ifeq block.
HARDWARE_PROFILE ?= default
ifeq ($(HARDWARE_PROFILE),rtx5060ti)
  CUDA_ARCH = 120
  THREADS = 12
  # 16GB VRAM holds a 7B Q4 plus large context; 99 = offload all layers.
  N_GPU_LAYERS = 99
  RAM_GB = 24
  VRAM_GB = 16
  # 20GB VM leaves ~11GB for Windows on a 31GB host.
  PODMAN_RAM_MB = 20480
  CONTEXT_SIZE = 16384
  VIDEO_OPT_FLAGS = --no-mmap --parallel 1
  RUN_CAPS = --cap-add=IPC_LOCK --ipc=host --device nvidia.com/gpu=all
else ifeq ($(HARDWARE_PROFILE),rog3060)
  CUDA_ARCH = 86
  THREADS = 12
  # 20 layers on GPU leaves ~1.8GB headroom for KV cache + compute on 6GB VRAM laptop.
  # Raise toward 28 (all layers) only after confirming the model loads without OOM.
  # 28 does not fit on 6GB VRAM with 8192 context, but 26 does with some headroom (see logs/live-stats).
  N_GPU_LAYERS = 22
  RAM_GB = 40
  VRAM_GB = 6
  PODMAN_RAM_MB = 32768
  CONTEXT_SIZE = 16384
  # cache-type-k/v quantization requires Flash Attention (--flash-attn / -DLLAMA_FLASH_ATTN=ON).
  VIDEO_OPT_FLAGS = --no-mmap --parallel 1
  # --device nvidia.com/gpu=all requires nvidia-container-toolkit + CDI in the Podman VM.
  RUN_CAPS = --cap-add=IPC_LOCK --ipc=host --device nvidia.com/gpu=all
else
  # Default: conservative CPU-only (matches prior server target)
  CUDA_ARCH = 0
  THREADS = 4
  N_GPU_LAYERS = 0
  RAM_GB = 16
  VRAM_GB = 0
  PODMAN_RAM_MB = 16384
  CONTEXT_SIZE = 4096
  VIDEO_OPT_FLAGS =
  RUN_CAPS =
endif


IMAGE_NAME := llamacpp
TAG := latest
# Bind-mount dir for ccache. Podman on Windows treats --volume name:path as a host
# path, so a named volume fails with faccessat .../ccache-llama. Make creates this.
CCACHE_DIR := ccache-llama

$(CCACHE_DIR):
	mkdir -p $@

# ======================================================================================
# Full Podman/WSL reset (profile-driven RAM_GB via PODMAN_RAM_MB; 32GB+ for rog3060).
# Uses Git Bash ONLY. Preserves ccache-llama volume. Updated for hardware profile
# (RAM_GB for container memory limit; see profile block for RAM vs VRAM details).
# GPU/CDI: See LessonsLearned.md (now includes profile activation).
# ======================================================================================
reset:
	@echo "=== Full Podman/WSL reset (profile RAM: $(PODMAN_RAM_MB)MB, $(RAM_GB)GB system) ==="
	@echo "WARNING: Run from Git Bash only (pwsh/make caused prior exit 1 on heredoc)."
	wsl --shutdown 2>/dev/null || true
	podman machine rm -f 2>/dev/null || true
	@mkdir -p "$$HOME/.config/containers"
	@printf '[storage]\ndriver = "vfs"\nrunroot = "/run/containers/storage"\ngraphroot = "/var/lib/containers/storage"\n' > "$$HOME/.config/containers/storage.conf"
	@echo "Created clean storage.conf with vfs driver."
	podman machine init --memory $(PODMAN_RAM_MB) --cpus 8 --disk-size 100 || echo "Note: init may need manual follow-up (podman machine ls)"
	podman machine start
	podman system connection list
	@echo "Podman reset complete. ccache-llama volume preserved. Run 'make build' next."
	@echo "For full GPU: Follow exact steps in LessonsLearned.md (drivers + toolkit in VM)."

# ======================================================================================
# Build the GPU (CUDA) Podman image (note: current Dockerfile is server-only with --n-gpu-layers 0).
# Bind-mounts ./ccache-llama (created if missing). Run `make reset` (Git Bash) first.
# ======================================================================================
build: $(CCACHE_DIR)
	@echo "=== Building with profile $(HARDWARE_PROFILE) (CUDA_ARCH=$(CUDA_ARCH), RAM_GB=$(RAM_GB)) ==="
	@echo "First build takes 10-20+ minutes (CUDA kernels if GPU profile). Later builds use ccache."
	podman build --pull=newer \
		--volume ./$(CCACHE_DIR):/root/.ccache \
		--build-arg CUDA_ARCH=$(CUDA_ARCH) \
		--tag $(IMAGE_NAME):$(TAG) \
		--tag localhost/$(IMAGE_NAME):$(TAG) \
		--file Dockerfile .
	@echo ""
	@echo "Build successful! Image '$(IMAGE_NAME):$(TAG)' (and localhost/ variant) is ready (profile: $(HARDWARE_PROFILE))."
	@echo "Run 'make ccache-stats' or 'HARDWARE_PROFILE=rtx5060ti make server'."
	@echo "(./$(CCACHE_DIR) persists across cleans, resets, and prune.)"

# ======================================================================================
# Remove image + stop/rm server container (cache preserved). Use `make clean-cache` or `make prune` for more.
# Dash prefix + SHELL=bash ensures robustness (updated for Git Bash requirement).
# ======================================================================================
clean:
	-podman stop llamacpp-server 2>/dev/null || true
	-podman rm -f llamacpp-server 2>/dev/null || true
	-podman rmi -f $(IMAGE_NAME):$(TAG) localhost/$(IMAGE_NAME):$(TAG) 2>/dev/null || true
	@echo "Clean complete: image and server container removed (./$(CCACHE_DIR) preserved)."

clean-cache:
	-rm -rf $(CCACHE_DIR)
	-podman volume rm -f ccache-llama 2>/dev/null || true
	@echo "ccache directory removed. Run 'make build' to recreate and repopulate cache."

ccache-stats: $(CCACHE_DIR)
	@echo "=== ccache statistics (./$(CCACHE_DIR)) ==="
	-podman run --rm --volume ./$(CCACHE_DIR):/root/.ccache nvidia/cuda:12.8.1-devel-ubuntu22.04 \
		bash -c "apt-get update -qq && apt-get install -y -qq ccache && ccache -s" 2>/dev/null || echo "Cache not initialized yet (run 'make build' first)."

# ======================================================================================
# Test container (llama-cli --help)
# ======================================================================================
run:
	podman run --rm -it localhost/$(IMAGE_NAME):$(TAG) --help

# Health check: GOOD if /v1/models answers on 127.0.0.1 or the Podman VM IP.
# BROKEN if the container is down or the API never comes up. localhost miss
# alone is not BROKEN (normal on Windows/WSL). Waits while the model loads.
test:
	@echo "=== make test: is the server good or broken? ==="
	@if ! podman ps --format '{{.Names}}' | grep -q llamacpp-server 2>/dev/null; then \
		echo ""; \
		echo "RESULT: BROKEN  -- container llamacpp-server is not running"; \
		if podman ps -a --format '{{.Names}}' | grep -q llamacpp-server 2>/dev/null; then \
			echo "  It started then exited. Last logs:"; \
			podman logs --tail 20 llamacpp-server 2>&1 | sed 's/^/    /'; \
			echo "  If those say 'No such file', download the GGUF first (make list-models)."; \
		else \
			echo "  Fix: make rtx5060ti   or   make rog3060   or   HARDWARE_PROFILE=... make server"; \
		fi; \
		echo ""; \
		exit 1; \
	fi
	@echo "  Container : running"
	@VM_IP=$$(podman machine ssh "ip -4 addr show eth0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}'" 2>/dev/null || true); \
	OK_URL=; LOCAL=no; \
	i=1; \
	while [ $$i -le 30 ]; do \
		if curl -sf --max-time 3 "http://127.0.0.1:$(PORT)/v1/models" >/dev/null 2>&1; then \
			OK_URL="http://127.0.0.1:$(PORT)"; LOCAL=yes; break; \
		fi; \
		if [ -n "$$VM_IP" ] && curl -sf --max-time 3 "http://$$VM_IP:$(PORT)/v1/models" >/dev/null 2>&1; then \
			OK_URL="http://$$VM_IP:$(PORT)"; break; \
		fi; \
		if [ $$i -eq 1 ]; then echo "  API       : waiting for model load (up to ~90s)"; fi; \
		sleep 3; \
		i=$$((i + 1)); \
	done; \
	if [ -z "$$OK_URL" ]; then \
		echo "  localhost : no"; \
		echo "  VM IP     : $${VM_IP:-unknown}  -- no response"; \
		echo ""; \
		echo "RESULT: BROKEN  -- container is up but /v1/models never answered"; \
		echo "  make logs     (need 'listening on' and no CUDA/OOM errors)"; \
		echo "  make live-stats"; \
		echo ""; \
		exit 1; \
	fi; \
	if [ "$$LOCAL" = yes ]; then echo "  localhost : yes  ($$OK_URL)"; \
	else echo "  localhost : no   (expected on Windows/WSL -- not a failure)"; fi; \
	if [ -n "$$VM_IP" ]; then echo "  VM IP     : yes  (http://$$VM_IP:$(PORT))"; fi; \
	echo ""; \
	echo "RESULT: GOOD  -- API is up at $$OK_URL"; \
	echo "  curl $$OK_URL/v1/models"; \
	echo "  curl $$OK_URL/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen14\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"

# ======================================================================================
# Show all targets/usage (updated for SHELL=bash, clean enhancements, and Git Bash requirement)
# ======================================================================================
help:
	@echo "Available targets:"
	@echo "  make reset        - Full Podman/WSL reset (profile-driven RAM)"
	@echo "  make build        - Build with profile (CUDA_ARCH from VRAM_GB)"
	@echo "  make clean        - Remove image + server container (cache preserved)"
	@echo "  make clean-cache  - Delete ./ccache-llama (forces full recompile)"
	@echo "  make ccache-stats - Show ccache statistics"
	@echo "  make server       - Start HTTP server/API (uses profile)"
	@echo "  make rtx5060ti    - Full RTX 5060 Ti GPU workflow (14B, reset, setup-gpu, build, server)"
	@echo "  make rog3060      - Full ROG 3060 GPU workflow (getmodels, reset, setup-gpu, restart, build, server)"
	@echo "  make test         - Health check: RESULT GOOD or BROKEN (API on localhost or VM IP)"
	@echo "  make logs         - Follow server logs"
	@echo "  make live-stats   - Container, process, model, memory, host URL"
	@echo "  make stop         - Stop/remove server"
	@echo "  make win-forward  - Map VM IP to localhost (admin PowerShell)"
	@echo "  make vm-ip        - Show Podman VM IP"
	@echo "  make prune        - Clean unrelated Podman resources"
	@echo "  make list-models       - List MODEL_SHORT names and files in ./models"
	@echo "  make getmodels         - Download 3060-sized models (phi, qwen2, deep)"
	@echo "  make getmodels-5060ti  - Download Qwen2.5-Coder-14B Q4 (~9GB) for 16GB VRAM"
	@echo "  make getmodels-deep5   - Coding stretch: DeepSeek Lite Q5 (~12GB, 16GB VRAM)"
	@echo "  make getmodels-lang    - Language: Qwen2.5 3B+7B Instruct (3060)"
	@echo "  make getmodels-lang-5060ti - Language: Qwen2.5 14B Instruct Q4 (16GB)"
	@echo "  make getmodels-chat14q6 - Language stretch: Qwen2.5 14B Instruct Q6 (~12GB)"
	@echo "  make getmodels-aya     - Multilingual: Aya Expanse 8B Q4 (~5GB, 23 languages)"
	@echo "  make build-extension   - Build the VS Code extension (requires Node.js 18+)"
	@echo "  make info              - Full structured target reference (primary + secondary)"
	@echo "  make help              - Show this compact list"
	@echo ""
	@echo "NOTE: Use Git Bash for all targets (SHELL=bash). pwsh causes parse errors."
	@echo "HARDWARE_PROFILE=rtx5060ti|rog3060|default controls RAM_GB vs VRAM_GB + video flags."
	@echo "Override: HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=deep PORT=18080"
	@echo "See README.md (quickstart) and techdoc.md (profile variables)."

# ======================================================================================
# info: structured reference of all targets, split into primary and secondary.
# Primary = normal workflow. Secondary = diagnostics, hardware, and cleanup.
# ======================================================================================
info:
	@echo ""
	@echo "============================== llama.cpp + VS Code =============================="
	@echo ""
	@echo "Models : coding  phi | qwen2 | deep | qwen14 | deep5"
	@echo "       : language chat3 | chat7 | chat14 | chat14q6 | aya8 (23 languages)"
	@echo "API    : http://localhost:$(PORT)/v1/chat/completions  (OpenAI-compatible)"
	@echo "Profile: HARDWARE_PROFILE=rtx5060ti  (RTX 5060 Ti 16GB, CUDA arch 120, all GPU layers)"
	@echo "         HARDWARE_PROFILE=rog3060    (RTX 3060 6GB, CUDA arch 86, 22 GPU layers)"
	@echo "         HARDWARE_PROFILE=default    (CPU-only, conservative, no GPU)"
	@echo ""
	@echo "-------------------------------- Primary Targets --------------------------------"
	@echo ""
	@echo "  make list-models                    List shorts and ./models"
	@echo "  make getmodels                      Coding 3060 set (phi, qwen2, deep)"
	@echo "  make getmodels-5060ti               Coding 14B Q4 (MODEL_SHORT=qwen14)"
	@echo "  make getmodels-deep5                Coding stretch DeepSeek Lite Q5 (deep5)"
	@echo "  make getmodels-lang                 Language 3B+7B Instruct"
	@echo "  make getmodels-lang-5060ti          Language 14B Instruct Q4 (chat14)"
	@echo "  make getmodels-chat14q6             Language stretch 14B Instruct Q6 (chat14q6)"
	@echo "  make getmodels-aya                  Multilingual Aya Expanse 8B (aya8, 23 languages)"
	@echo "  make reset                          Init/reinit Podman + WSL (Git Bash only)"
	@echo "  make build                          Build the llama.cpp container image"
	@echo "  make server  [MODEL_SHORT=qwen2]    Start API server on port $(PORT) (background)"
	@echo "  make test                           Verify server is responding"
	@echo "  make stop                           Stop and remove the server container"
	@echo "  make clean                          Remove image + container (keeps ccache)"
	@echo "  make build-extension                Build the VS Code extension (Copilot Chat + model picker + inline)"
	@echo ""
	@echo "  Typical first-time flow (GPU, this PC):"
	@echo "    make rtx5060ti                   (ends with make test)"
	@echo "  CPU-only:"
	@echo "    make getmodels && make reset && make build && make server && make test"
	@echo ""
	@echo "------------------------------- Secondary Targets -------------------------------"
	@echo ""
	@echo "  make setup-gpu                      Install nvidia-container-toolkit + CDI in Podman VM"
	@echo "  make live-stats                     Container state, process, model, memory, host URL"
	@echo "  make rtx5060ti                      Full RTX 5060 Ti GPU setup workflow"
	@echo "  make rog3060                        Full ROG 3060 GPU setup workflow"
	@echo "  make win-forward  [PORT=$(PORT)]        Proxy VM IP to localhost (run as Admin)"
	@echo "  make vm-ip                          Print current Podman VM IP address"
	@echo "  make logs                           Stream server container logs"
	@echo "  make ccache-stats                   Show compiler cache hit rate and size"
	@echo "  make clean-cache                    Delete ./ccache-llama (forces full recompile)"
	@echo "  make prune                          Remove unused Podman resources (safe)"
	@echo "  make run                            Run container with --help (smoke test)"
	@echo "  make info                           Show this reference"
	@echo "  make help                           Show compact target list"
	@echo ""
	@echo "  Variable overrides (example):"
	@echo "    HARDWARE_PROFILE=rtx5060ti MODEL_SHORT=deep PORT=18080 make server"
	@echo ""
	@echo "================================================================================"

# Full first-run: 14B model, VM RAM, GPU toolkit/CDI, CUDA build, server.
# Do not replace this with bare `HARDWARE_PROFILE=... make build && make server`
# — that skips setup-gpu and fails with "unresolvable CDI devices nvidia.com/gpu=all".
rtx5060ti: getmodels-5060ti rtx5060ti-reset setup-gpu restart-podman rtx5060ti-build rtx5060ti-server
	@$(MAKE) test
	@echo ""
	@echo "RTX 5060 Ti setup complete (qwen14). Optional: make live-stats"
	@echo ""

rtx5060ti-reset:
	HARDWARE_PROFILE=rtx5060ti make reset

rtx5060ti-build:
	HARDWARE_PROFILE=rtx5060ti make build

rtx5060ti-server:
	HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=qwen14

rog3060: getmodels reset setup-gpu restart-podman rog3060-build rog3060-server
	@$(MAKE) test
	@echo ""
	@echo "ROG 3060 setup complete (qwen2). Optional: make live-stats"
	@echo ""

restart-podman:
	podman machine stop && podman machine start

rog3060-build:
	HARDWARE_PROFILE=rog3060 make build

rog3060-server:
	HARDWARE_PROFILE=rog3060 make server MODEL_SHORT=qwen2

# ======================================================================================
# List MODEL_SHORT names and what is already in ./models
# ======================================================================================
list-models:
	@echo "Select with:  HARDWARE_PROFILE=<profile> make server MODEL_SHORT=<short>"
	@echo ""
	@echo "  Coding"
	@echo "  short   file                                              download                 VRAM"
	@echo "  phi     Phi-3.5-mini-instruct-q4_K_M.gguf                 make getmodels           6GB+"
	@echo "  qwen2   Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf             make getmodels           6GB+"
	@echo "  deep    DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf       make getmodels           6GB"
	@echo "  qwen14  Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf            make getmodels-5060ti    16GB"
	@echo "  deep5   DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf       make getmodels-deep5     16GB"
	@echo ""
	@echo "  Language (general instruct, not coder)"
	@echo "  chat3    Qwen2.5-3B-Instruct-Q4_K_M.gguf                  make getmodels-lang      6GB+"
	@echo "  chat7    Qwen2.5-7B-Instruct-Q4_K_M.gguf                  make getmodels-lang      6GB+"
	@echo "  chat14   Qwen2.5-14B-Instruct-Q4_K_M.gguf                 make getmodels-lang-5060ti  16GB"
	@echo "  chat14q6 Qwen2.5-14B-Instruct-Q6_K.gguf                   make getmodels-chat14q6  16GB"
	@echo "  aya8     aya-expanse-8b-Q4_K_M.gguf                       make getmodels-aya       6GB+"
	@echo ""
	@echo "On disk (./models):"
	@if ls models/*.gguf >/dev/null 2>&1; then \
		ls -lh models/*.gguf | awk '{printf "  %s  %s\n", $$5, $$9}'; \
	else \
		echo "  (empty -- run a getmodels* target)"; \
	fi
	@echo ""
	@echo "Example:  HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=chat14"
	@echo "Then:     make test"

# ======================================================================================
# Download 3 coding GGUF models to ./models (~13GB)
# ======================================================================================
getmodels:
	mkdir -p models
	@echo "Checking model files in models/ ..."
	@if [ -f models/Phi-3.5-mini-instruct-q4_K_M.gguf ]; then \
		echo "Skipping Phi-3.5-mini model (already present)."; \
	else \
		echo "Downloading Phi-3.5-mini model..."; \
		curl -L -o models/Phi-3.5-mini-instruct-q4_K_M.gguf https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-q4_K_M.gguf; \
	fi
	@if [ -f models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf ]; then \
		echo "Skipping Qwen2 model (already present)."; \
	else \
		echo "Downloading Qwen2 model..."; \
		curl -L -o models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf; \
	fi
	@if [ -f models/DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf ]; then \
		echo "Skipping DeepSeek model (already present)."; \
	else \
		echo "Downloading DeepSeek model..."; \
		curl -L -o models/DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf; \
	fi
	@echo "Models ready in models/!"
	@echo "These are sized for 6GB VRAM (RTX 3060). On a 5060 Ti (16GB), also run: make getmodels-5060ti"

# 14B Q4 (~9GB) fits 16GB VRAM with 16k context. Also pulled by make rtx5060ti.
getmodels-5060ti:
	mkdir -p models
	@echo "Checking 5060 Ti model in models/ ..."
	@if [ -f models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf ]; then \
		echo "Skipping Qwen2.5-Coder-14B Q4 (already present)."; \
	else \
		echo "Downloading Qwen2.5-Coder-14B-Instruct Q4_K_M (~9GB)..."; \
		curl -L -o models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf; \
	fi
	@echo "5060 Ti daily coding model ready. Start with:"
	@echo "  HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=qwen14"
	@echo "Optional stretch: make getmodels-deep5  (MODEL_SHORT=deep5)"

# DeepSeek Lite Q5 (~12GB). Same model as `deep` at a quant that is worth using on 16GB.
getmodels-deep5:
	mkdir -p models
	@echo "Checking DeepSeek Lite Q5 in models/ ..."
	@if [ -f models/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf ]; then echo "Skipping deep5 (already present)."; \
	else echo "Downloading DeepSeek-Coder-V2-Lite-Instruct Q5_K_M (~12GB)..."; \
		curl -L -o models/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf; fi
	@echo "Start with: HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=deep5"

# Language (general instruct). 3060: 3B+7B. 5060 Ti: 14B Q4 (daily) / Q6 (stretch).
getmodels-lang:
	mkdir -p models
	@echo "Checking language models (3060-sized) in models/ ..."
	@if [ -f models/Qwen2.5-3B-Instruct-Q4_K_M.gguf ]; then echo "Skipping chat3 (already present)."; \
	else echo "Downloading Qwen2.5-3B-Instruct Q4_K_M (~2GB)..."; \
		curl -L -o models/Qwen2.5-3B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf; fi
	@if [ -f models/Qwen2.5-7B-Instruct-Q4_K_M.gguf ]; then echo "Skipping chat7 (already present)."; \
	else echo "Downloading Qwen2.5-7B-Instruct Q4_K_M (~4.7GB)..."; \
		curl -L -o models/Qwen2.5-7B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf; fi
	@echo "Language models ready. 3060: MODEL_SHORT=chat3 or chat7. 5060 Ti can use those too, or make getmodels-lang-5060ti."

getmodels-lang-5060ti:
	mkdir -p models
	@echo "Checking 5060 Ti language model in models/ ..."
	@if [ -f models/Qwen2.5-14B-Instruct-Q4_K_M.gguf ]; then echo "Skipping chat14 (already present)."; \
	else echo "Downloading Qwen2.5-14B-Instruct Q4_K_M (~9GB)..."; \
		curl -L -o models/Qwen2.5-14B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf; fi
	@echo "Start with: HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=chat14"
	@echo "Optional stretch: make getmodels-chat14q6  (MODEL_SHORT=chat14q6)"

getmodels-chat14q6:
	mkdir -p models
	@echo "Checking 14B Instruct Q6 in models/ ..."
	@if [ -f models/Qwen2.5-14B-Instruct-Q6_K.gguf ]; then echo "Skipping chat14q6 (already present)."; \
	else echo "Downloading Qwen2.5-14B-Instruct Q6_K (~12GB)..."; \
		curl -L -o models/Qwen2.5-14B-Instruct-Q6_K.gguf https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q6_K.gguf; fi
	@echo "Start with: HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=chat14q6"

# Multilingual (23 languages). CC-BY-NC. ~5GB — 3060 stretch, comfortable on 5060 Ti.
getmodels-aya:
	mkdir -p models
	@echo "Checking Aya Expanse 8B Q4 in models/ ..."
	@if [ -f models/aya-expanse-8b-Q4_K_M.gguf ]; then echo "Skipping aya8 (already present)."; \
	else echo "Downloading aya-expanse-8b Q4_K_M (~5GB)..."; \
		curl -L -o models/aya-expanse-8b-Q4_K_M.gguf https://huggingface.co/bartowski/aya-expanse-8b-GGUF/resolve/main/aya-expanse-8b-Q4_K_M.gguf; fi
	@echo "Start with: HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=aya8"
	@echo "  or:       HARDWARE_PROFILE=rog3060 make server MODEL_SHORT=aya8"
	@echo "License: CC-BY-NC (non-commercial). 23 languages — see README."

# ======================================================================================
# Server-only mode (per request). CLI target retained but deprecated.
# ======================================================================================
cli:
	@echo "CLI disabled (server-only image). Use 'make server' or run manually."
	@exit 1

# ======================================================================================
# Start background API server with hardware profile (RAM_GB/VRAM_GB driven).
# Uses VIDEO_OPT_FLAGS (video tricks: no-mmap, mlock, MoE, KV quant), RUN_CAPS,
# THREADS, N_GPU_LAYERS. Overrides Dockerfile CMD. Activate with HARDWARE_PROFILE=rog3060.
# ======================================================================================
server:
	@missing=0; \
	if [ ! -f models/$(MODEL_FILE) ] || [ "$$(wc -c < models/$(MODEL_FILE) 2>/dev/null || echo 0)" -lt 1000000 ]; then \
		echo "Missing or incomplete: models/$(MODEL_FILE)"; missing=1; \
	fi; \
	if [ $$missing -eq 1 ]; then \
		echo "Download first: $(DOWNLOAD_HINT)"; \
		echo "Then: HARDWARE_PROFILE=$(HARDWARE_PROFILE) make server MODEL_SHORT=$(MODEL_SHORT)"; \
		exit 1; \
	fi
	-podman rm -f llamacpp-server
	podman run -d --name llamacpp-server \
		--pull=never \
		$(RUN_CAPS) \
		-p $(PORT):$(PORT) \
		-v ./models:/models \
		localhost/$(IMAGE_NAME):$(TAG) \
		-m /models/$(MODEL_FILE) \
		--host 0.0.0.0 \
		--port $(PORT) \
		-c $(CONTEXT_SIZE) \
		--n-gpu-layers $(N_GPU_LAYERS) \
		--threads $(THREADS) \
		$(VIDEO_OPT_FLAGS)

# ======================================================================================
# View server logs (useful for debugging startup, model loading, or port binding)
# ======================================================================================
logs:
	-podman logs --tail 100 llamacpp-server 2>&1 || echo "No logs (container may not be running). Try 'make server' first."

# ======================================================================================
# live-stats: container state, llama-server process, model, memory, GPU layers/access/
# VRAM (with VM-host fallback and CDI warning), and public host URL.
# ======================================================================================
live-stats:
	@echo ""
	@echo "=== llama.cpp Live Status ==="
	@echo ""
	@if ! podman ps --format '{{.Names}}' | grep -q '^llamacpp-server$$'; then \
		echo "  Container : NOT running"; \
		echo ""; \
		echo "  Start with: make server"; \
		echo ""; \
		exit 0; \
	fi; \
	STATUS=$$(podman ps --filter "name=^llamacpp-server$$" --format "{{.Status}}" 2>/dev/null); \
	echo "  Container : $$STATUS"; \
	MODEL=$$(podman inspect llamacpp-server \
		--format '{{range .Config.Cmd}}{{.}} {{end}}' 2>/dev/null \
		| tr ' ' '\n' | grep '\.gguf$$' | xargs basename 2>/dev/null); \
	[ -n "$$MODEL" ] && echo "  Model     : $$MODEL" || echo "  Model     : (unknown)"; \
	PORTMAP=$$(podman port llamacpp-server 2>/dev/null); \
	[ -n "$$PORTMAP" ] && echo "  Port map  : $$PORTMAP" || true; \
	MEM=$$(podman stats --no-stream --format "{{.MemUsage}}" llamacpp-server 2>/dev/null); \
	[ -n "$$MEM" ] && echo "  Memory    : $$MEM" || true; \
	echo ""; \
	PID=$$(podman exec llamacpp-server sh -c \
		'for p in /proc/[0-9]*/cmdline; do pid=$${p%/cmdline}; pid=$${pid##*/}; \
		cat "$$p" 2>/dev/null | tr "\000" " " | grep -q llama-server \
		&& echo $$pid && break; done' 2>/dev/null); \
	if [ -z "$$PID" ]; then \
		echo "  Process   : llama-server NOT running  (still starting up?)"; \
		echo "  Follow logs: make logs"; \
		echo ""; \
		exit 0; \
	fi; \
	echo "  Process   : llama-server  (PID $$PID)"; \
	N_LAYERS=$$(podman inspect llamacpp-server \
		--format '{{range .Config.Cmd}}{{.}} {{end}}' 2>/dev/null \
		| grep -oP '(?<=--n-gpu-layers )\d+'); \
	N_LAYERS=$${N_LAYERS:-0}; \
	if [ "$$N_LAYERS" = "0" ]; then \
		echo "  GPU layers : 0  (CPU-only - rebuild with HARDWARE_PROFILE=rtx5060ti or rog3060 for GPU)"; \
	else \
		echo "  GPU layers : $$N_LAYERS offloaded to GPU"; \
	fi; \
	GPU_DEVS=$$(podman exec llamacpp-server sh -c \
		'ls /dev/nvidia[0-9] /dev/dxg 2>/dev/null | wc -l' 2>/dev/null); \
	GPU_DEVS=$${GPU_DEVS:-0}; \
	DXG=$$(podman exec llamacpp-server sh -c \
		'[ -e /dev/dxg ] && echo wsl || echo native' 2>/dev/null); \
	if [ "$$GPU_DEVS" -gt 0 ]; then \
		[ "$$DXG" = "wsl" ] \
			&& echo "  GPU access : YES  (/dev/dxg - WSL passthrough mode)" \
			|| echo "  GPU access : YES  ($$GPU_DEVS /dev/nvidia device(s))"; \
		WSL_SMI=$$(podman exec llamacpp-server sh -c \
			'find /usr/lib/wsl/drivers -name nvidia-smi 2>/dev/null | head -1' 2>/dev/null); \
		SMIOUT=$$(podman exec llamacpp-server sh -c \
			"$${WSL_SMI:-nvidia-smi} --query-gpu=name,memory.used,memory.total,utilization.gpu \
			--format=csv,noheader,nounits 2>/dev/null | head -1" 2>/dev/null); \
		if [ -n "$$SMIOUT" ]; then \
			GNAME=$$(echo "$$SMIOUT" | cut -d, -f1 | sed 's/^ *//;s/ *$$//'); \
			MU=$$(echo "$$SMIOUT"    | cut -d, -f2 | tr -d ' '); \
			MT=$$(echo "$$SMIOUT"    | cut -d, -f3 | tr -d ' '); \
			GU=$$(echo "$$SMIOUT"    | cut -d, -f4 | tr -d ' '); \
			echo "  GPU       : $$GNAME"; \
			echo "  GPU VRAM  : $${MU} MiB / $${MT} MiB  ($${GU}% util)"; \
		fi; \
	else \
		echo "  GPU access : NO  (/dev/nvidia* and /dev/dxg not found in container)"; \
		[ "$$N_LAYERS" != "0" ] && \
			echo "               WARNING: $$N_LAYERS layers requested but no GPU visible - try: podman machine stop && podman machine start" || true; \
		VM_SMI=$$(podman machine ssh \
			"find /usr/lib/wsl/drivers -name nvidia-smi 2>/dev/null | head -1" 2>/dev/null); \
		VMGPU=$$(podman machine ssh \
			"$${VM_SMI:-nvidia-smi} --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1" \
			2>/dev/null); \
		if [ -n "$$VMGPU" ]; then \
			VGNAME=$$(echo "$$VMGPU" | cut -d, -f1 | sed 's/^ *//;s/ *$$//'); \
			VMU=$$(echo "$$VMGPU" | cut -d, -f2 | tr -d ' '); \
			VMT=$$(echo "$$VMGPU" | cut -d, -f3 | tr -d ' '); \
			echo "  GPU (VM)  : $$VGNAME - $${VMU}/$${VMT} MiB  (present in VM but NOT reaching container)"; \
		else \
			echo "  GPU (VM)  : not reachable - run 'make setup-gpu' if not done, then restart Podman machine"; \
		fi; \
	fi; \
	echo ""; \
	VM_IP=$$(podman machine ssh \
		"ip -4 addr show eth0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}'" \
		2>/dev/null || true); \
	if [ -n "$$VM_IP" ]; then \
		echo "  Host URL  : http://$$VM_IP:$(PORT)"; \
		echo "  Endpoint  : http://$$VM_IP:$(PORT)/v1/chat/completions"; \
	else \
		echo "  Host URL  : http://127.0.0.1:$(PORT)"; \
	fi; \
	echo ""; \
	if curl -sf --max-time 3 "http://127.0.0.1:$(PORT)/v1/models" >/dev/null 2>&1 \
		|| { [ -n "$$VM_IP" ] && curl -sf --max-time 3 "http://$$VM_IP:$(PORT)/v1/models" >/dev/null 2>&1; }; then \
		echo "  API check : PASS"; \
	else \
		echo "  API check : not ready yet (model still loading — run: make test)"; \
	fi; \
	echo ""

# ======================================================================================
# Stop/remove server container
# ======================================================================================
stop:
	-podman stop llamacpp-server
	-podman rm llamacpp-server
	@echo "Server stopped."

# ======================================================================================
# Build the VS Code extension (vscode-llamacpp/) using npm + tsc.
# Requires Node.js 18+. Outputs compiled JS to vscode-llamacpp/out/.
# Proposed API (model picker) is enabled at runtime via launch.json --enable-proposed-api;
# no type-definition download needed because the provider calls use (vscode.lm as any).
# After building: open vscode-llamacpp/ in VS Code and press F5.
# ======================================================================================
build-extension:
	@echo "=== Building VS Code extension (vscode-llamacpp/) ==="
	@command -v node >/dev/null 2>&1 || { echo "ERROR: Node.js not found. Install Node.js 18+ from https://nodejs.org"; exit 1; }
	@node --version | awk -F'[v.]' '{if ($$2+0 < 18) { print "ERROR: Node.js 18+ required (found " $$0 "). Download from https://nodejs.org"; exit 1 } }' || exit 1
	@echo "Node: $$(node --version)  |  npm: $$(npm --version)"
	@echo ""
	@echo "--- Step 1/2: Installing dependencies ---"
	@cd vscode-llamacpp && npm install || { \
		echo ""; \
		echo "FAILED: npm install failed. Check errors above."; \
		echo "  Hint: confirm package.json exists in vscode-llamacpp/ and npm registry is reachable."; \
		exit 1; \
	}
	@echo ""
	@echo "--- Step 2/2: Compiling TypeScript ---"
	@cd vscode-llamacpp && npm run compile || { \
		echo ""; \
		echo "FAILED: TypeScript compilation failed. Fix the errors above, then re-run 'make build-extension'."; \
		exit 1; \
	}
	@echo ""
	@echo "SUCCESS: Extension built to vscode-llamacpp/out/"
	@echo ""
	@echo "  To launch:     Open vscode-llamacpp/ in VS Code, press F5 (Extension Development Host)"
	@echo "  @llama chat:   Type @llama <question> in Copilot Chat"
	@echo "  Model picker:  Open Copilot Chat model dropdown → select phi / qwen2 / deep"
	@echo "  Switch model:  Ctrl+Shift+P -> 'llama.cpp: Switch Model'"
	@echo "  Inline:        Copilot ghost-text is disabled; llama.cpp handles all completions"
	@echo "  Package:       npm install -g @vscode/vsce && cd vscode-llamacpp && vsce package"

# ======================================================================================
# setup-gpu: installs nvidia-container-toolkit inside the Podman VM and generates the
# CDI spec that allows 'podman run --device nvidia.com/gpu=all' to work.
# Must be re-run after every 'make reset' (podman machine rm destroys the VM state).
# Requires: NVIDIA CUDA-on-WSL drivers installed on the Windows host first.
# ======================================================================================
setup-gpu:
	@echo "=== Installing nvidia-container-toolkit in Podman VM ==="
	@echo "Requires NVIDIA CUDA-on-WSL drivers on Windows host (run nvidia-smi in PowerShell first)."
	@echo ""
	@podman machine ssh ' \
		set -e; \
		sudo mkdir -p /etc/cdi; \
		if command -v dnf >/dev/null 2>&1; then \
			echo "--- Fedora/RHEL detected (dnf) ---"; \
			curl -sL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
				| sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null; \
			sudo dnf install -y nvidia-container-toolkit; \
		elif command -v apt-get >/dev/null 2>&1; then \
			echo "--- Ubuntu/Debian detected (apt-get) ---"; \
			sudo mkdir -p /usr/share/keyrings /etc/apt/sources.list.d; \
			curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
				| sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg; \
			curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
				| sed "s|deb https://|deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://|g" \
				| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null; \
			sudo apt-get update -qq && sudo apt-get install -y nvidia-container-toolkit; \
		else \
			echo "ERROR: neither dnf nor apt-get found in Podman VM"; exit 1; \
		fi; \
		sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml; \
		echo "CDI spec written to /etc/cdi/nvidia.yaml"; \
		set +e; \
		NVIDIA_SMI=$$(find /usr/lib/wsl/drivers -name "nvidia-smi" 2>/dev/null | head -1); \
		if [ -n "$$NVIDIA_SMI" ]; then \
			"$$NVIDIA_SMI" --query-gpu=name,driver_version --format=csv,noheader; \
		elif command -v nvidia-smi >/dev/null 2>&1; then \
			nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; \
		else \
			echo "nvidia-smi not in PATH (normal in WSL mode - GPU confirmed via CDI spec)"; \
		fi; \
	' || { \
		echo ""; \
		echo "FAILED: see errors above."; \
		echo "  Common causes:"; \
		echo "  1. Podman machine not running (run: podman machine start)"; \
		echo "  2. Network issue in VM (test: podman machine ssh curl -s https://nvidia.github.io)"; \
		echo "  3. NVIDIA CUDA-on-WSL driver not on Windows host (verify: nvidia-smi in PowerShell)"; \
		exit 1; \
	}
	@echo ""
	@echo "SUCCESS: toolkit installed and CDI spec generated."
	@echo "Restarting Podman machine so CDI config takes effect..."
	podman machine stop
	podman machine start
	@echo ""
	@echo "Podman machine restarted. Run: HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=qwen2"
	@echo "Then: make live-stats  (GPU access should show YES)"

vm-ip:
	@podman machine ssh "ip -4 addr show eth0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}'" 2>/dev/null || (echo "ERROR: could not read Podman VM IP" && exit 1)

# Maps VM IP -> localhost. Git Bash expands $vars inside double quotes, so the
# PowerShell script is single-quoted; $$ becomes $ for PowerShell after Make.
win-forward:
	@echo "=== Setting Windows localhost forwarding for port $(PORT) ==="
	@VM_IP=$$(podman machine ssh "ip -4 addr show eth0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}'" 2>/dev/null || true); \
	if [ -z "$$VM_IP" ]; then echo "ERROR: could not read Podman VM IP."; exit 1; fi; \
	echo "Using VM IP: $$VM_IP"; \
	echo "Without Admin you can already use: http://$$VM_IP:$(PORT)"; \
	powershell.exe -NoProfile -Command '$$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator); if (-not $$IsAdmin) { Write-Host "ERROR: Run Git Bash as Administrator for win-forward." -ForegroundColor Red; exit 1 }; $$ErrorActionPreference = "SilentlyContinue"; netsh interface portproxy delete v4tov4 listenport=$(PORT) listenaddress=0.0.0.0 | Out-Null; netsh interface portproxy add v4tov4 listenport=$(PORT) listenaddress=0.0.0.0 connectport=$(PORT) connectaddress='"$$VM_IP"'; netsh interface portproxy show v4tov4; Write-Host "Port forwarding active. Test: curl http://127.0.0.1:$(PORT)/v1/models" -ForegroundColor Green' \
	|| echo "win-forward failed. Use http://$$VM_IP:$(PORT) (no Admin needed)."

# ======================================================================================

# ======================================================================================
# Prune unused resources (preserves llamacpp image, ccache-llama volume, llamacpp-server).
# Safe for bash; run after clean if more disk space needed.
# ======================================================================================
prune:
	@echo "=== Pruning all Podman resources NOT associated with this llamacpp project ==="
	-podman machine start 2>/dev/null || true
	-podman rm -f $$(podman ps -a -q --filter "name!=llamacpp-server") 2>/dev/null || true
	-podman rmi -f $$(podman images -q --filter "reference!=llamacpp" --filter "reference!=localhost/llamacpp") 2>/dev/null || true
	-podman volume prune -f
	-podman network prune -f
	@echo "Prune complete."
	@echo "Preserved: llamacpp image + ccache-llama volume."
	@echo "Run 'make build' if image removed. Use make clean-cache for ccache."
# ======================================================================================
# (Old duplicate test target removed; bash-compatible version is used.)

