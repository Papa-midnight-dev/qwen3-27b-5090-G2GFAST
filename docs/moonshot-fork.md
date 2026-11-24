# Moonshot Fork Notes (`qwen3-27b-5090`)

This repository is a community optimization fork of Masahiro Kiyota's [**`mkyt/qwen3-27b-5090`**](https://github.com/mkyt/qwen3-27b-5090) (MIT License — original copyright and full attribution preserved in [`LICENSE`](../LICENSE)).

All additions below preserve **100% lossless quality** (verified against 9/9 needle-in-a-haystack tests up to 96K, bit-exact GPU vs Host RAM prefix cache checks, structured JSON schema / `llguidance`, tool calling, and all 284 upstream frontend unit tests).

---

## What This Fork Adds

### 1. SM120 Blackwell CUDA & Speculative Decoding
- **Fused NVFP4 quantization epilogues (`iter1`, `iter7`):**
  - Fused NVFP4 quantization directly into `gdn_rec`, `gated_norm`, and `attention_combine`, plus PDL on `quant_fp4`.
  - Fused `skinny_fp4_glu` decode/verify MLP epilogue (`silu(alpha_g * gate) * (alpha_u * up) -> NVFP4 + swizzled scales` directly in CTA SMEM), eliminating 64 kernel launches and `285 MB–1.14 GB/step` of intermediate FP32 HBM traffic (100% bit-identical).
- **Sarathi chunked-prefill interleaving (`iter7` + `PREFILL_DECODE_CHUNK=512`):**
  - Strictly enforces aligned (`kGdnChunk=64`) prefill chunk budgets while decode streams are active, cutting concurrent C=4 32K worst-case ITL stalls from **`544.6 ms → 56.3 ms` (9.7x smoother)** vs stock upstream (`651.6 ms → 56.3 ms` vs unchunked `0.6.x`) and lifting mean stream decode from **`115.6 → 214.0 tok/s` (+85%)**.
- **Hybrid CUDA 13.4 Graph conditional prompt lookup (`iter9`, `Q27_LOOKUP=1`, `Q27_LOOKUP_MIN=8`):**
  - Uses CUDA 13.4 `cudaGraphCondTypeIf` to skip the entire DSpine draft forward pass on verbatim n-gram hits (`~1,012–1,038 tok/s` on edit/copy workloads).
- **Adaptive DSpine 2-path tree verification on FP8 Tensor Cores (`iter10`, `Q27_SPEC_ADAPT_PATHS=2`):**
  - Fixes `SpecTree` RNG index overlap for `kSpecBlock=16` (`kRngWalk=0, kRngAccept=64, kRngResidual=192`), caps `spec_rows` at `kTreeMax=32`, adds a 32-bit register ancestor bitmask to `attention_tc8_kernel`, and enables 2-path trees automatically when acceptance drops below `4.0` while keeping 16-row chains + lookup skip above `4.5`.
- **Low-rank DSpine refinement (`iter11A`, `Q27_DSPINE_LR=1`):**
  - Precomputes static Gram $G = W_4^T W_4$ (`0.5 MiB`) and $U_{\text{head}} = (\text{head} \odot \text{norm}) W_4$ (`128 MiB` BF16 for 128K draft vocab), eliminating all 5120-dim refinement GEMMs and cutting NVFP4 head reads **16x** (`1,710 tok/s` C=8 batched).
- **Zero-PCIe GPU prefix cache + async `POST /v1/prefetch` (`iter2`, `iter11B`):**
  - Reuses the in-VRAM `stage_` checkpoint directly on hot hits (`7.24 ms → 0.23 ms`, **31.5x faster**), and exposes `POST /v1/prefetch` to asynchronously stream a cached checkpoint H2D on `cs_` while the user is still typing or speaking.

### 2. Zero-Lag Serving Stack & Isolation Tools
- **Frontend (`iter4`, `iter5`):** `TCP_NODELAY`, HTTP/1.1 keep-alive chunked SSE, Unix domain socket (`LISTEN_UNIX`), segment-cached tokenizer, O(1) lock-free detokenizer, binary IPC, `ADMIN_LOCK=1` operator guardrail, and in-thinking tool-call parser fallback.
- **[`tools/q27-fwd.c`](../tools/q27-fwd.c):** Zero-fork C `epoll` TCP↔UDS forwarder so the Docker container can run with `--network none` while exposing `127.0.0.1:8000` locally.

### 3. Linux RTX 5090 True NVML Undervolt Tool
- **[`tools/rtx5090-undervolt.py`](../tools/rtx5090-undervolt.py)** + **[`tools/rtx5090-undervolt.service`](../tools/rtx5090-undervolt.service):**
  - True V/F curve undervolt on Linux (no MSI Afterburner required) via `libnvidia-ml.so.1`: shifts the P0 graphics V/F curve upward (`+180 MHz`), flattens the boost ceiling (`2,750–2,800 MHz`, `min=210 MHz` so idle stays P8), caps power at `450 W` (hard-coded `<= 550 W` longevity guardrail), and applies a conservative GDDR7 offset (`+1,000` to `+1,500 MHz`).

---

## Test Rig & Benchmark Methodology

All before/after numbers below were measured apples-to-apples on the **same machine, model checkpoint, KV dtype, and idle single-tenant load** against the **unoptimized upstream `mkyt/qwen3-27b-5090` base** (before any of this fork's `iter1..iter11` optimizations):
- **Hardware:** NVIDIA RTX 5090 (32 GB SM120, driver `615.78.08`, CUDA `13.4`) + AMD Ryzen 9 9950X3D
- **Engine config:** `Huihui-Qwen3.8-27B-abliterated-NVFP4`, `SPEC=dspine` (`tencent/Qwen3.8-27B-DSpine-b16`, 128K draft vocab), `KV_DTYPE=fp8`, `VISION=fp8`, `MAX_TOKENS=131072`, `MAX_SEQS=8`, `VRAM_BUDGET_GIB=25.2`, `STATE_HOST_CACHE=8`
- **Upstream baselines:**
  - **E2E serving table:** stock upstream `v0.5.0` HTTP/Docker baseline vs full `moonshot` (`iter1..iter11`, `--network none` UDS + `q27-fwd`) on the identical `bench_suite.py` workload.
  - **CUDA engine table:** stock upstream `v0.6.1` (`fe41696`, exact fork parent commit) vs full `moonshot` (`iter1..iter11`) in-process on `bench_engine.py`.
- *(Note: upstream `README.md` benchmarks use `juzbox/Qwen3.8-27B-NVFP4-GDN` with `NVFP4 KV` at `400 W`; use the tables below for direct upstream-base → moonshot comparisons on identical settings.)*

### End-to-End Serving (Stock Upstream `v0.5.0` → Moonshot Fork)

| Metric | Upstream Base (`v0.5.0`) | Moonshot Fork | Gain |
|---|---:|---:|---:|
| HTTP keep-alive RTT (p50) | 41.02 ms | **0.13 ms** | **325x faster** |
| Warm 60K follow-up TTFT (p50) | 92.91 ms | **25.07 ms** | **3.7x faster** (`fe`: `52.2 -> 3.1 ms`) |
| Warm 20K follow-up TTFT (p50) | 48.18 ms | **21.03 ms** | **2.3x faster** (`fe`: `20.0 -> 2.2 ms`) |
| Cold 64-token TTFT | 36.73 ms | **20.07 ms** | **1.83x faster** |
| C=4 32K worst-case ITL stall (max / p95) | 544.6 / 482.9 ms | **56.3 / 52.2 ms** | **9.7x smoother** |
| C=4 32K mean per-stream decode | 115.6 tok/s | **214.0 tok/s** | **+85.1%** |
| C=8 concurrent code decode | 1,117.7 tok/s (`135 ms` max ITL) | **1,229.7 tok/s** (`43 ms` max ITL) | **+10.0%**, **3.1x lower tail** |
| C=4 concurrent code decode | 859.2 tok/s (`151 ms` TTFT) | **948.5 tok/s** (`88 ms` TTFT) | **+10.4%**, **1.7x faster TTFT** |
| Short request beside C=4 load (p50 / max TTFT) | 28.5 / 37.7 ms | **21.8 / 24.3 ms** | **1.30x faster** |
| 32K long-context single code decode | 326.2 tok/s | **370.1 tok/s** | **+13.5%** |
| Sampled code decode | 306.2 tok/s | **338.1 tok/s** | **+10.4%** |

### In-Process CUDA Engine (Exact Fork Base `v0.6.1` `fe41696` → Moonshot Fork)

| Metric | Fork Base (`v0.6.1`) | Moonshot Fork | Gain |
|---|---:|---:|---:|
| GPU prefix cache restore (`prefix_begin`, 20K) | 7.21 ms | **0.27 ms** (`0.23 ms` w/ prefetch) | **27x–31x faster** |
| Warm 20K engine TTFT (`+18` new tokens) | 33.95 ms | **19.38 ms** | **1.75x faster** |
| C=8 batched decode | 1,646.3 tok/s | **1,710.3 tok/s** | **+64 tok/s (+3.9%)** |
| C=4 batched decode (fixed8) | 1,136.2 tok/s | **1,182.1 tok/s** | **+46 tok/s (+4.0%)** |
| 32K long-context code decode | 347.2 tok/s | **367.0 tok/s** | **+5.7%** |
| Edit / copy verbatim decode | 1,001.1 tok/s | **1,038.6 tok/s** | **+3.7%** (`12.29 ms/step`) |
| Prose decode (adaptive 2-path tree) | 221.3 tok/s | **229.6 tok/s** | **+3.7%** |

### RTX 5090 450W True NVML Undervolt (Live on Moonshot Fork)

| Metric | Stock 575W / 550W Ref | 450W True Undervolt (`+180 / 2800 / +1500`) | Gain |
|---|---:|---:|---:|
| Single decode (3x512 sustained, SHA-exact) | 406.9 tok/s @ 57°C | **424.8 tok/s peak** (`416.9` avg) @ **53°C** | **+4.4% faster, -125W cap, -4°C** |
| Sustained C=4 burst (2,048 tok, SHA-exact) | — | **1,052.0 tok/s** @ **448W peak / 53°C** | **2.85 tok/J** |

---

## Quick Start: Build & Run

### Build the CUDA engine (`libq27.so`)
```bash
cmake -S engine -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a
ninja -C build
```

### Build the Docker image
```bash
docker build -t qwen3-27b-5090:moonshot .
```

### Optional: `--network none` isolation with `q27-fwd`
```bash
gcc -O2 -Wall -Wextra tools/q27-fwd.c -o q27-fwd
# Run the container with --network none and -e LISTEN_UNIX=/run/q27/api.sock, then forward locally:
./q27-fwd 127.0.0.1:8000 unix:/run/q27-engine/api.sock
```

### Optional: RTX 5090 450W true undervolt on Linux
```bash
python3 tools/rtx5090-undervolt.py --status
sudo python3 tools/rtx5090-undervolt.py --apply --watts 450 --core-offset 180 --max-clock 2800 --mem-offset 1500
# Revert to stock anytime:
sudo python3 tools/rtx5090-undervolt.py --reset
```
See [`tools/README.md`](../tools/README.md) for systemd persistence and safety details.

---

## New Environment Variables (All A/B Toggleable)

| Variable | Default | Purpose |
|---|---:|---|
| `Q27_LOOKUP` | `1` | Enable hybrid CUDA 13.4 Graph conditional prompt-lookup (`0` to disable) |
| `Q27_LOOKUP_MIN` | `8` | Minimum n-gram match length to bypass DSpine on verbatim hits |
| `Q27_SPEC_ADAPT_PATHS` | `2` | DSpine tree paths used when adaptive width drops below `Q27_SPEC_ADAPT_DOWN` (`1` = chains only) |
| `Q27_DSPINE_LR` | `1` | Precomputed low-rank DSpine refinement (`Gram G` + `U_head`, `0` = original path) |
| `PREFILL_DECODE_CHUNK` | `512` | Sarathi prefill chunk budget (64-aligned) when decode streams are active |
| `LISTEN_UNIX` | *(unset)* | Bind the HTTP API on a Unix domain socket path instead of TCP |
| `ADMIN_LOCK` | `0` | Set `1` to block mutating `POST /admin/*` endpoints from autonomous agents |
