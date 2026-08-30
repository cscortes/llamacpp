# llama.cpp local server (Podman)

Windows wrapper that builds [llama.cpp](https://github.com/ggerganov/llama.cpp) in a container and runs an OpenAI-compatible API on port **18080**. All commands go through the Makefile. Internals (CUDA images, CMake, CDI, profile variables) are in [techdoc.md](techdoc.md).

## Prerequisites

- **Git Bash** (`C:\Program Files\Git\bin\bash.exe`). PowerShell breaks this Makefile.
- **Podman** installed. The profile targets start the machine for you.
- For GPU: host `nvidia-smi` must show your NVIDIA GPU ([CUDA on WSL](https://developer.nvidia.com/cuda/wsl)).
- Disk: ~30 GB for a 3060 coding setup (models ~13 GB plus images and ccache). On a **5060 Ti**, ~40 GB for the daily 14B (~9 GB plus images and ccache). Stretch files (`deep5`, `chat14q6`, ~12 GB each) and language Q4 (`chat14` ~9 GB) are extra and are **not** downloaded by the profile targets.

You do not install CMake or CUDA on Windows. Those live in the container.

## First run — use a profile target

From **Git Bash**, in this repo, run **one** of these. They download models, size the Podman VM, install the GPU toolkit (CDI), build, and start the server. That is the path that works.

```bash
make rtx5060ti    # RTX 5060 Ti — 14B Q4 (qwen14), CUDA arch 120
make rog3060      # RTX 3060 laptop — 7B set (qwen2), CUDA arch 86
```

Then confirm it yourself:

```bash
make test
```

`make test` is the health check. It prints **RESULT: GOOD** or **RESULT: BROKEN**.

- **GOOD** — `/v1/models` answered. Use the URL it prints (often `http://172.x.x.x:18080` on Windows, not localhost).
- **BROKEN** — the container is down, or the API never came up. If it started then exited, `make test` prints the last logs. A common cause is a missing GGUF: `podman run -d` still prints a container ID, then llama-server dies with `No such file or directory`. Download the model first (`make list-models` shows which `getmodels-*` target), then `make server` again. Use `make logs` / `make live-stats` for CUDA or OOM errors.

The profile target also runs `make test` at the end. Re-run `make test` any time you want a yes/no. A missing localhost bind is **not** BROKEN on Podman/Windows.

| Target | Hardware | Model | VM RAM |
|---|---|---|---|
| `make rtx5060ti` | RTX 5060 Ti 16GB | `qwen14` (14B Q4) | 20 GB |
| `make rog3060` | RTX 3060 6GB | `qwen2` (7B Q4) | 32 GB (needs ~40 GB host RAM) |
| CPU | no NVIDIA | `make getmodels && make reset && make build && make server` | 16 GB |

First GPU build takes 10–20+ minutes. `make` with no arguments only prints help.

Do **not** start with `HARDWARE_PROFILE=... make build` and `make server` alone. Those skip `setup-gpu`. The server then dies with `unresolvable CDI devices nvidia.com/gpu=all` and it looks like the project is broken. After a profile target has run once, you can use `HARDWARE_PROFILE` for rebuilds and daily `make server` (see below).

Optional: `make win-forward` from Admin Git Bash if you want `127.0.0.1` as well as the VM IP.

## List and select a model

`make rtx5060ti` / `make rog3060` only download the **coding** set (`qwen14` or `phi` / `qwen2` / `deep`). Language files are separate. If they are not in `./models/`, **do not** run `make server` yet.

**Always download first, then start:**

```bash
make list-models          # shorts, download target, and what is already on disk
make getmodels-lang-5060ti
make stop
HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=chat14
make test
```

`make server` checks `./models/` before `podman run`. Missing or tiny (failed) files print `Download first: make getmodels-…` and exit. If you skip that check on an older Makefile, `podman run -d` still prints a container ID, the process exits immediately (`No such file or directory`), and `make test` reports **BROKEN — container is not running**.

This server does **not** generate images. Coding and language models are text in, text out.

Each profile has **three** coding picks (fast / daily / stretch). Language on 6GB only has room for two sizes; 16GB gets a third (higher-quality 14B).

### RTX 3060 (6GB) — pick one

**Coding** (`make getmodels`)

| Short | Use when | Pros | Cons |
|---|---|---|---|
| `phi` | Tiny / CPU / just trying the API | Fastest load, smallest file (~2.5 GB) | Weakest coder; short answers |
| `qwen2` | Daily coding on 6GB | Best 7B instruct coder; default | Mid size (~4.7 GB); not as smart as 14B |
| `deep` | Harder code, fill-in-the-middle | Different architecture (MoE Lite); strong on code | Q3 to fit 6GB — mushier than Q5; slower; stretch VRAM |

**Language** (`make getmodels-lang`, plus `make getmodels-aya` for multilingual)

| Short | Use when | Pros | Cons |
|---|---|---|---|
| `chat3` | Quick chat, titles, simple Q&A | Smallest (~2 GB), snappy | Shallow reasoning; English/Chinese-leaning |
| `chat7` | Daily assistant on 6GB | Real 7B Instruct quality | Still not a 14B; ~4.7 GB |
| `aya8` | Non-English or mixed-language chat | Built for **23 languages** (see below) | ~5 GB stretch on 6GB; CC-BY-NC (non-commercial) |

A 14B language model does not fit 6GB with useful context. Qwen `chat*` already speaks several languages (especially Chinese). Use `aya8` when you need even coverage across Arabic, Hindi, Turkish, Ukrainian, and the rest of its 23.

### RTX 5060 Ti (16GB) — pick one

**Coding**

| Short | Use when | Pros | Cons | Download |
|---|---|---|---|---|
| `qwen2` | Fast iteration, long context | Same 7B as 3060; lots of VRAM left for 16k context | Underuses 16GB; weaker than 14B | `make getmodels` |
| `qwen14` | Daily driver (first-run default) | 14B Q4 is the sweet spot on this card | ~9 GB; slower than 7B | `make getmodels-5060ti` |
| `deep5` | Harder code / FIM, and you can spare VRAM | Same DeepSeek Lite as `deep`, but **Q5** — actually worth it on 16GB | ~12 GB; slower; less KV headroom than `qwen14` | `make getmodels-deep5` |

**Language**

| Short | Use when | Pros | Cons | Download |
|---|---|---|---|---|
| `chat7` | Fast chat, drafts | Snappy; same file as 3060 | Weaker writing/reasoning than 14B | `make getmodels-lang` |
| `chat14` | Daily assistant (recommended) | 14B Q4 general Instruct; fits 16k context | ~9 GB; not as sharp as Q6 | `make getmodels-lang-5060ti` |
| `chat14q6` | Best writing/reasoning that still fits | Same 14B as `chat14`, **Q6** (less quant loss) | ~12 GB; slower; 32B Q4 does **not** fit 16GB | `make getmodels-chat14q6` |
| `aya8` | Non-English or mixed-language chat | 23-language specialist; easy fit on 16GB | Weaker English reasoning than `chat14`; CC-BY-NC | `make getmodels-aya` |

```bash
# 5060 Ti language (daily, then optional stretch)
make getmodels-lang-5060ti
HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=chat14
make getmodels-chat14q6
HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=chat14q6

# 5060 Ti coding stretch
make getmodels-deep5
HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=deep5

# 3060 language
make getmodels-lang
HARDWARE_PROFILE=rog3060 make server MODEL_SHORT=chat7

# Multilingual (both profiles) — Arabic, Chinese, French, German, Hindi, Japanese, …
make getmodels-aya
HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=aya8
```

`aya8` is [Aya Expanse 8B](https://huggingface.co/CohereForAI/aya-expanse-8b) (Q4, ~5 GB): Arabic, Chinese (simplified and traditional), Czech, Dutch, English, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, Vietnamese. License is **CC-BY-NC** (non-commercial).

## After the first run

A profile sets CUDA arch, GPU layers, and VM RAM. Pass it on later GPU commands:

```bash
HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=qwen14
HARDWARE_PROFILE=rog3060 make server MODEL_SHORT=qwen2
```

Custom hardware: copy an `ifeq` block in the Makefile. Variables: [techdoc.md](techdoc.md). After every later `make reset`, run `make setup-gpu` again (or re-run the profile target).

| Command | Purpose |
|---|---|
| `make list-models` | List shorts and files in `./models/` |
| `make getmodels-lang` / `getmodels-lang-5060ti` / `getmodels-chat14q6` | Language 3B+7B / 14B Q4 / 14B Q6 |
| `make getmodels-aya` | Multilingual Aya Expanse 8B (`aya8`, 23 languages) |
| `make getmodels-deep5` | 5060 Ti coding stretch (DeepSeek Lite Q5) |
| `HARDWARE_PROFILE=rtx5060ti make server MODEL_SHORT=chat14` | Run a selected model (`qwen14`, `deep5`, `chat14q6`, …) |
| `make stop` | Stop the server container |
| `make logs` | Recent server logs |
| `make live-stats` | Container, GPU, model, host URL |
| `make test` | Health check — prints RESULT: GOOD or BROKEN |
| `make win-forward` | Map VM IP → localhost (Admin Git Bash) |
| `make clean` | Remove image + container (keeps ccache) |
| `make info` | Full target list |

## API

Use the host `make test` or `make live-stats` prints (often `http://172.x.x.x:18080`, not localhost):

```bash
curl http://$(make vm-ip):18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen14", "messages": [{"role": "user", "content": "Hello"}]}'
```

Use `"qwen2"` on a 3060. After `make win-forward` (Admin), `http://127.0.0.1:18080` works too.

## More

- [techdoc.md](techdoc.md) — Dockerfile, profiles, GPU passthrough, models, verification
- `make info` — all Makefile targets
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — upstream
- [CHANGES.md](CHANGES.md) — version history
