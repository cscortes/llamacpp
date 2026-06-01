# ======================================================================================
# Podman/Docker multi-stage build for llama.cpp (GPU/CUDA).
# Features: ccache, OpenBLAS, CUDA acceleration, llama-cli/server, coding GGUF models.
# Base: nvidia/cuda:12.5.1 (devel + runtime). Podman networking (slirp4netns/WSL) often binds IPv6-only on Windows.
#
# Quick Start (Windows):
#   make getmodels
#   make reset   # MUST use Git Bash
#   make build
#   make server PORT=18080   # Uses localhost:18080 (via win-forward if needed)
#   make win-forward          # Maps VM IP to Windows localhost
#   make test
#
# Models: phi (Phi-3.5-mini), qwen2 (Qwen2.5-Coder-7B), deep (DeepSeek-Coder-V2-Lite)
# Vars: MODEL_SHORT=[phi|qwen2|deep] PORT=18080
#
# Server: http://localhost:18080 or http://172.26.156.205:18080/v1/chat/completions
# Stop: make stop
# Clean: make clean
# Note: Podman VM IP (172.26.156.205) used for reliable access. win-forward creates netsh proxy.
# ======================================================================================

# Force bash (Git Bash on Windows; required for reset/heredoc/||). Windows targets (test/win-forward) updated for compatibility.
SHELL := bash

.PHONY: build clean reset run help info getmodels cli server stop prune build-extension live-stats setup-gpu
.DEFAULT_GOAL := info

# Model short names (case-insensitive partial match):
#   phi     -> Phi-3.5-mini-instruct-q4_K_M.gguf (~2.5GB)
#   qwen2   -> Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf (~4.7GB, default)
#   deep    -> DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf (~5.5GB)
MODEL_SHORT ?= qwen2
PORT ?= 18080

ifeq ($(findstring phi,$(MODEL_SHORT)),phi)
MODEL_FILE = Phi-3.5-mini-instruct-q4_K_M.gguf
else ifeq ($(findstring qwen,$(MODEL_SHORT)),qwen)
MODEL_FILE = Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
else ifeq ($(findstring deep,$(MODEL_SHORT)),deep)
MODEL_FILE = DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf
else
MODEL_FILE = Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
endif

# Hardware profile (activatable via HARDWARE_PROFILE=rog3060 make ...).
# Controls RAM_GB (system memory, mlock, no-mmap, Podman --memory) vs VRAM_GB
# (n-gpu-layers, MoE experts to CPU, KV cache quant for VRAM savings per video).
# Mirrors MODEL_SHORT pattern; extensible for other hardware.
HARDWARE_PROFILE ?= default
ifeq ($(HARDWARE_PROFILE),rog3060)
  CUDA_ARCH = 86
  THREADS = 12
  # 20 layers on GPU leaves ~1.8GB headroom for KV cache + compute on 6GB VRAM laptop.
  # Raise toward 28 (all layers) only after confirming the model loads without OOM.
  N_GPU_LAYERS = 10
  RAM_GB = 40
  VRAM_GB = 6
  PODMAN_RAM_MB = 32768
  # 8192 context fits within the ~1.8GB headroom left by 20 GPU layers.
  CONTEXT_SIZE = 8192
  # cache-type-k/v quantization requires Flash Attention (--flash-attn / -DLLAMA_FLASH_ATTN=ON).
  # Removed until FA is compiled in; default fp16 KV cache works fine at 10 GPU layers + 8192 ctx.
  VIDEO_OPT_FLAGS = --no-mmap
  # --device nvidia.com/gpu=all requires nvidia-container-toolkit + CDI in the Podman VM.
  # Run 'make setup-gpu' then 'podman machine stop && podman machine start' if this fails.
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
# Mounts permanent `ccache-llama` volume. Run `make reset` (Git Bash) first for WSL/Podman stability.
# ======================================================================================
build:
	@echo "=== Building with profile $(HARDWARE_PROFILE) (CUDA_ARCH=$(CUDA_ARCH), RAM_GB=$(RAM_GB)) ==="
	@echo "First build takes 10-20+ minutes (CUDA kernels if rog3060). Later builds use ccache."
	-podman volume create ccache-llama 2>/dev/null || true
	podman build --pull=newer \
		--volume ccache-llama:/root/.ccache \
		--build-arg CUDA_ARCH=$(CUDA_ARCH) \
		--tag $(IMAGE_NAME):$(TAG) \
		--tag localhost/$(IMAGE_NAME):$(TAG) \
		--file Dockerfile .
	@echo ""
	@echo "Build successful! Image '$(IMAGE_NAME):$(TAG)' (and localhost/ variant) is ready (profile: $(HARDWARE_PROFILE))."
	@echo "Run 'make ccache-stats' or 'HARDWARE_PROFILE=rog3060 make server'."
	@echo "(ccache volume persists across cleans, resets, and prune.)"

# ======================================================================================
# Remove image + stop/rm server container (cache preserved). Use `make clean-cache` or `make prune` for more.
# Dash prefix + SHELL=bash ensures robustness (updated for Git Bash requirement).
# ======================================================================================
clean:
	-podman stop llamacpp-server 2>/dev/null || true
	-podman rm -f llamacpp-server 2>/dev/null || true
	-podman rmi -f $(IMAGE_NAME):$(TAG) localhost/$(IMAGE_NAME):$(TAG) 2>/dev/null || true
	@echo "Clean complete: image and server container removed (ccache-llama volume preserved)."

clean-cache:
	-podman volume rm -f ccache-llama 2>/dev/null || true
	@echo "ccache volume removed. Run 'make build' to recreate and repopulate cache."

ccache-stats:
	@echo "=== ccache statistics (permanent volume) ==="
	-podman run --rm --volume ccache-llama:/root/.ccache nvidia/cuda:12.5.1-devel-ubuntu22.04 \
		bash -c "apt-get update -qq && apt-get install -y -qq ccache && ccache -s" 2>/dev/null || echo "Cache not initialized yet (run 'make build' first)."

# ======================================================================================
# Test container (llama-cli --help)
# ======================================================================================
run:
	podman run --rm -it localhost/$(IMAGE_NAME):$(TAG) --help

test:
	@echo "=== Testing server API (requires 'make server' first) ==="
	@podman ps --format '{{.Names}}' | grep -q llamacpp-server 2>/dev/null || (echo "Server not running (check with 'podman ps'). Run 'make server' first." && exit 1)
	@powershell -Command "try { $$r = Invoke-WebRequest -Uri http://localhost:$(PORT)/v1/models -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop; Write-Host 'Server test passed (models endpoint responding).' -ForegroundColor Green } catch { Write-Host 'API test failed. Check make logs, run make win-forward, or use VM IP 172.26.156.205:$(PORT). Restart with make stop && make server.' -ForegroundColor Red; exit 1 }"
	@echo "Full chat test example:"
	@echo "curl http://localhost:$(PORT)/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen2\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"

# ======================================================================================
# Show all targets/usage (updated for SHELL=bash, clean enhancements, and Git Bash requirement)
# ======================================================================================
help:
	@echo "Available targets:"
	@echo "  make reset        - Full Podman/WSL reset (profile-driven RAM)"
	@echo "  make build        - Build with profile (CUDA_ARCH from VRAM_GB)"
	@echo "  make clean        - Remove image + server container (cache preserved)"
	@echo "  make clean-cache  - Reset ccache volume"
	@echo "  make ccache-stats - Show ccache statistics"
	@echo "  make server       - Start HTTP server/API (uses profile)"
	@echo "  make rog3060      - Convenience: run server with rog3060 profile (40GB RAM, 6GB VRAM)"
	@echo "  make test         - Test API (uses localhost after win-forward)"
	@echo "  make logs         - Follow server logs"
	@echo "  make live-stats   - Container, process, model, memory, host URL"
	@echo "  make stop         - Stop/remove server"
	@echo "  make win-forward  - Map VM IP to localhost (admin PowerShell)"
	@echo "  make vm-ip        - Show Podman VM IP"
	@echo "  make prune        - Clean unrelated Podman resources"
	@echo "  make getmodels         - Download models"
	@echo "  make build-extension   - Build the VS Code extension (requires Node.js 18+)"
	@echo "  make info              - Full structured target reference (primary + secondary)"
	@echo "  make help              - Show this compact list"
	@echo ""
	@echo "NOTE: Use Git Bash for all targets (SHELL=bash). pwsh causes parse errors."
	@echo "HARDWARE_PROFILE=rog3060 (or default) controls RAM_GB vs VRAM_GB + video flags."
	@echo "Override: HARDWARE_PROFILE=rog3060 make server MODEL_SHORT=deep PORT=18080"
	@echo "See README.md and profile block for RAM (mlock/no-mmap) vs VRAM (layers/KV) details."

# ======================================================================================
# info: structured reference of all targets, split into primary and secondary.
# Primary = normal workflow. Secondary = diagnostics, hardware, and cleanup.
# ======================================================================================
info:
	@echo ""
	@echo "============================== llama.cpp + VS Code =============================="
	@echo ""
	@echo "Models : phi (Phi-3.5-mini 2.5GB) | qwen2 (Qwen2.5-Coder-7B 4.4GB, default)"
	@echo "       : deep (DeepSeek-Coder-V2-Lite 7.6GB)"
	@echo "API    : http://localhost:$(PORT)/v1/chat/completions  (OpenAI-compatible)"
	@echo "Profile: HARDWARE_PROFILE=rog3060  (RTX 3060, 40GB RAM, GPU layers + video opts)"
	@echo "         HARDWARE_PROFILE=default  (CPU-only, conservative, no GPU)"
	@echo ""
	@echo "-------------------------------- Primary Targets --------------------------------"
	@echo ""
	@echo "  make getmodels                      Download all 3 GGUF coding models (~13 GB)"
	@echo "  make reset                          Init/reinit Podman + WSL (Git Bash only)"
	@echo "  make build                          Build the llama.cpp container image"
	@echo "  make server  [MODEL_SHORT=qwen2]    Start API server on port $(PORT) (background)"
	@echo "  make test                           Verify server is responding"
	@echo "  make stop                           Stop and remove the server container"
	@echo "  make clean                          Remove image + container (keeps ccache)"
	@echo "  make build-extension                Build the VS Code extension (Copilot Chat + model picker + inline)"
	@echo ""
	@echo "  Typical first-time flow:"
	@echo "    make getmodels"
	@echo "    make reset                        (Git Bash; one-time Podman/WSL setup)"
	@echo "    make build"
	@echo "    make server"
	@echo "    make test"
	@echo "    make build-extension"
	@echo ""
	@echo "------------------------------- Secondary Targets -------------------------------"
	@echo ""
	@echo "  make setup-gpu                      Install nvidia-container-toolkit + CDI in Podman VM"
	@echo "  make live-stats                     Container state, process, model, memory, host URL"
	@echo "  make rog3060                        Start server with RTX 3060 / 40 GB profile"
	@echo "  make win-forward  [PORT=$(PORT)]        Proxy VM IP to localhost (run as Admin)"
	@echo "  make vm-ip                          Print current Podman VM IP address"
	@echo "  make logs                           Stream server container logs"
	@echo "  make ccache-stats                   Show compiler cache hit rate and size"
	@echo "  make clean-cache                    Delete ccache volume (forces full recompile)"
	@echo "  make prune                          Remove unused Podman resources (safe)"
	@echo "  make run                            Run container with --help (smoke test)"
	@echo "  make info                           Show this reference"
	@echo "  make help                           Show compact target list"
	@echo ""
	@echo "  Variable overrides (example):"
	@echo "    HARDWARE_PROFILE=rog3060 MODEL_SHORT=deep PORT=18080 make server"
	@echo ""
	@echo "================================================================================"

# Convenience target for this hardware (uses profile vars for RAM/VRAM/video optimizations).
# Uses bash -c to avoid Win32 make path-with-parentheses parsing error in sh -c.
rog3060:
	@bash -c 'HARDWARE_PROFILE=rog3060 make server'

# ======================================================================================
# Download 3 coding GGUF models to ./models (~13GB)
# ======================================================================================
getmodels:
	mkdir -p models
	@echo "Downloading coding models to models/ (this may take 10-30min depending on connection)..."
	curl -L -o models/Phi-3.5-mini-instruct-q4_K_M.gguf https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-q4_K_M.gguf
	curl -L -o models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
	curl -L -o models/DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf
	@echo "Models ready in models/!"

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
		echo "  GPU layers : 0  (CPU-only — rebuild with HARDWARE_PROFILE=rog3060 for GPU)"; \
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
			&& echo "  GPU access : YES  (/dev/dxg — WSL passthrough mode)" \
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
		else \
			echo "  GPU VRAM  : (nvidia-smi not in container — check make logs for CUDA init)"; \
		fi; \
	else \
		echo "  GPU access : NO  (/dev/nvidia* and /dev/dxg not found in container)"; \
		[ "$$N_LAYERS" != "0" ] && \
			echo "               WARNING: $$N_LAYERS layers requested but no GPU visible — try: podman machine stop && podman machine start" || true; \
		VM_SMI=$$(podman machine ssh \
			"find /usr/lib/wsl/drivers -name nvidia-smi 2>/dev/null | head -1" 2>/dev/null); \
		VMGPU=$$(podman machine ssh \
			"$${VM_SMI:-nvidia-smi} --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1" \
			2>/dev/null); \
		if [ -n "$$VMGPU" ]; then \
			VGNAME=$$(echo "$$VMGPU" | cut -d, -f1 | sed 's/^ *//;s/ *$$//'); \
			VMU=$$(echo "$$VMGPU" | cut -d, -f2 | tr -d ' '); \
			VMT=$$(echo "$$VMGPU" | cut -d, -f3 | tr -d ' '); \
			echo "  GPU (VM)  : $$VGNAME — $${VMU}/$${VMT} MiB  (present in VM but NOT reaching container)"; \
		else \
			echo "  GPU (VM)  : not reachable — run 'make setup-gpu' if not done, then restart Podman machine"; \
		fi; \
	fi; \
	echo ""; \
	VM_IP=$$(podman machine ssh \
		"ip -4 addr show eth0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}'" \
		2>/dev/null || echo "172.26.156.205"); \
	echo "  Host URL  : http://$$VM_IP:$(PORT)"; \
	echo "  Endpoint  : http://$$VM_IP:$(PORT)/v1/chat/completions"; \
	echo ""; \
	if curl -sf --max-time 3 "http://$$VM_IP:$(PORT)/v1/models" >/dev/null 2>&1; then \
		echo "  API check : PASS"; \
	else \
		echo "  API check : FAIL  (run: make win-forward)"; \
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
	@echo "  Package:       npm install -g @vscode/vsce && vsce package --allow-proposed-api"

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
			echo "nvidia-smi not in PATH (normal in WSL mode — GPU confirmed via CDI spec)"; \
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
	@echo "Run: HARDWARE_PROFILE=rog3060 make server MODEL_SHORT=qwen2"
	@echo "Then: make live-stats  (GPU access should show YES)"

vm-ip:
	@echo "Podman VM IP:"
	@podman machine ssh "ip -4 addr show eth0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}'" 2>/dev/null || echo "172.26.156.205"

win-forward:
	@echo "=== Setting Windows localhost forwarding for port $(PORT) ==="
	@VM_IP=$$(podman machine ssh "ip -4 addr show eth0 | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}'" 2>/dev/null || echo "172.26.156.205"); \
	echo "Using VM IP: $$VM_IP"; \
	powershell -Command "netsh interface portproxy delete v4tov4 listenport=$(PORT) listenaddress=0.0.0.0 2> nul; netsh interface portproxy add v4tov4 listenport=$(PORT) listenaddress=0.0.0.0 connectport=$(PORT) connectaddress='$$VM_IP'; netsh interface portproxy show v4tov4 listenport=$(PORT); Write-Host 'Port forwarding active. Test with: curl http://localhost:$(PORT)/v1/models' -ForegroundColor Green" || echo "Run PowerShell as Administrator for netsh."
	@echo "(If netsh fails, run terminal as Administrator or use VM IP directly: http://172.26.156.205:$(PORT)")

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

