# Moonshot Fork Notes (`qwen3-27b-5090`)

This repository is a build-and-run community optimization fork of Masahiro Kiyota's [**`mkyt/qwen3-27b-5090`**](https://github.com/mkyt/qwen3-27b-5090) C++/CUDA inference engine and Python API server (MIT License — original copyright and full attribution preserved in [`LICENSE`](../LICENSE)).

- **Fork base:** upstream `v0.6.2` (`fe41696`, 17 commits behind upstream `v0.8.1` on `main`).
- **What is included:** the modified C++/CUDA engine (`engine/`), Python OpenAI/Anthropic server (`frontend/`), `Dockerfile`, and two standalone Linux utilities (`tools/`). Model weights and DSpine draft weights are **not** included and must be downloaded separately from Hugging Face.
- **Quality verification:** distribution-preserving and quality-verified across 9/9 needle-in-a-haystack tests up to 96K, bit-exact GPU vs Host RAM prefix cache checks, structured JSON schema / `llguidance`, tool calling, math checks, and all 284 upstream frontend unit tests (`284 passed, 16 skipped`). *(Note: because `iter3` extends FP8 tensor-core attention to 33–64-token prompt chunks and `iter10` enables adaptive multi-path DSpine verification, greedy output hashes can differ from upstream `fe41696` — this fork is distribution-preserving, not bit-identical to upstream across all prompts.)*

---

## What This Fork Adds

### 1. SM120 Blackwell CUDA & Speculative Decoding
- **Fused NVFP4 quantization epilogues (`iter1`, `iter7`):**
  - Fused NVFP4 quantization directly into `gdn_rec`, `gated_norm`, and `attention_combine`, plus PDL on `quant_fp4`.
  - Fused `skinny_fp4_glu` decode/verify MLP epilogue (`silu(alpha_g * gate) * (alpha_u * up) -> NVFP4 + swizzled scales` directly in CTA SMEM), eliminating 64 kernel launches and `285 MB–1.14 GB/step` of intermediate FP32 HBM traffic (bit-identical to the unfused epilogue).
- **Sarathi chunked-prefill interleaving (`iter7` + `PREFILL_DECODE_CHUNK=512`):**
  - Strictly enforces aligned (`kGdnChunk=64`) prefill chunk budgets while decode streams are active, cutting concurrent C=4 32K worst-case ITL stalls from **`544.6 ms → 56.3 ms` (9.7x smoother)** vs `v0.5.0` (`651.6 ms → 56.3 ms` vs unchunked `iter5`) while increasing mean per-stream decode from **`115.6 → 214.0 tok/s` (+85%)** at **`+3.6%`** aggregate throughput (`65.6 → 68.0 tok/s`).
- **Hybrid CUDA Graph conditional prompt lookup (`iter9`, `Q27_LOOKUP=1`, `Q27_LOOKUP_MIN=8`):**
  - Uses CUDA Graph conditional nodes (`cudaGraphCondTypeIf`) to skip the DSpine draft forward pass on verbatim n-gram hits (`~1,012–1,039 tok/s` on edit/copy workloads).
- **Adaptive DSpine 2-path tree verification on FP8 Tensor Cores (`iter10`, `Q27_SPEC_ADAPT_PATHS=2`):**
  - Fixes `SpecTree` RNG index overlap for `kSpecBlock=16` (`kRngWalk=0, kRngAccept=64, kRngResidual=192`), caps `spec_rows` at `kTreeMax=32`, adds a 32-bit register ancestor bitmask to `attention_tc8_kernel`, and enables 2-path trees automatically when acceptance drops below `4.0` while keeping 16-row chains + lookup skip above `4.5`.
- **Low-rank DSpine refinement (`iter11A`, `Q27_DSPINE_LR=1`):**
  - Precomputes static Gram $G = W_4^T W_4$ (`0.5 MiB`) and $U_{\text{head}} = (\text{head} \odot \text{norm}) W_4$ (`128 MiB` BF16 for 128K draft vocab), eliminating 5120-dim refinement GEMMs and cutting NVFP4 head reads 16x (~`+0.7%` standalone on C=8; combined `iter1..iter11` stack reaches `1,710 tok/s` C=8 engine).
- **Zero-PCIe GPU prefix cache + async `POST /v1/prefetch` (`iter2`, `iter11B`):**
  - Reuses the in-VRAM `stage_` checkpoint directly on hot hits without a host round-trip, and exposes `POST /v1/prefetch` to asynchronously stage a cached checkpoint H2D on `cs_` while the user is still typing or speaking.

### 2. Serving Stack & Isolation Tools
- **Frontend (`iter4`, `iter5`):** `TCP_NODELAY`, HTTP/1.1 keep-alive chunked SSE, Unix domain socket (`LISTEN_UNIX`), segment-cached tokenizer, O(1) lock-free detokenizer, binary IPC, `ADMIN_LOCK=1` operator guardrail, and in-thinking tool-call parser fallback.
- **[`tools/q27-fwd.c`](../tools/q27-fwd.c):** Zero-fork C `epoll` TCP↔UDS forwarder so the Docker container can run with `--network none` while exposing `127.0.0.1:8000` locally.

### 3. Linux RTX 5090 NVML Undervolt & Memory OC Tool
- **[`tools/rtx5090-undervolt.py`](../tools/rtx5090-undervolt.py)** + **[`tools/rtx5090-undervolt.service`](../tools/rtx5090-undervolt.service):**
  - Native Linux NVML (`libnvidia-ml.so.1`) controller (no MSI Afterburner required): shifts the P0 graphics V/F curve upward (`+180 MHz`), clamps the boost ceiling (`2,750–2,800 MHz`, `min=210 MHz` so idle stays P8), caps power at `450 W` (hard-coded `<= 550 W` guardrail), and applies a GDDR7 memory offset (`+1,000` to `+1,500 MHz`).

---

## Behaviour Changes vs Upstream (`v0.6.2` `fe41696`)

- `Q27_LOOKUP` defaults to **`1`** (upstream default: `0`). Set `Q27_LOOKUP=0` to disable prompt-lookup bypass.
- Adaptive DSpine verification can switch to 2-path trees when acceptance drops below `4.0` (upstream restricted DSpine to single-path). Set `Q27_SPEC_ADAPT_PATHS=1` to force chains only.
- Streaming SSE responses use HTTP/1.1 chunked `Connection: keep-alive` by default instead of closing the socket after every stream.
- Same RNG seed can produce different sampled/greedy outputs than upstream due to `iter3` FP8 TC attention on 33–64-token chunks and `iter10` RNG slot layout changes.

---

## Test Rig & Benchmark Methodology

All measurements below are **single-run results** on the same machine, checkpoint, and KV dtype (no multi-run averaging; small ±3–5% engine deltas can fall within run-to-run variance):
- **Hardware:** NVIDIA RTX 5090 (32 GB SM120, host driver `615.78.08`, host CUDA `13.4`; Docker image builds with CUDA `13.3`) + AMD Ryzen 9 9950X3D
- **Engine config:** `Huihui-Qwen3.8-27B-abliterated-NVFP4`, `SPEC=dspine` (`tencent/Qwen3.8-27B-DSpine-b16`, 128K draft vocab), `KV_DTYPE=fp8`, `VISION=fp8`, `MAX_TOKENS=131072`, `MAX_SEQS=8`, `VRAM_BUDGET_GIB=25.2`, `STATE_HOST_CACHE=8`
- *(Note: upstream `README.md` benchmarks use `juzbox/Qwen3.8-27B-NVFP4-GDN` with `NVFP4 KV` at `400 W`, which is why prose speeds differ between the upstream README table and the FP8-KV tables below.)*

### 1. In-Process CUDA Engine (Exact Fork Base `v0.6.2` `fe41696` → Moonshot Fork)

Measured directly against `libq27.so` (no HTTP proxy in between). *Caveat: the `moonshot` column was recorded just before commit `34e1270` lowered the adaptive tree thresholds from `5.5 / 6.5` to `4.0 / 4.5`; the single-path (`Q27_SPEC_ADAPT_PATHS=1`) column is also shown for comparison.*

| Metric | Fork Base (`v0.6.2` `fe41696`) | Moonshot (2-path) | Moonshot (1-path) | Delta |
|---|---:|---:|---:|---:|
| Warm 20K engine TTFT (`+14` new tok) | 20.20 ms | **17.13 ms** | 16.67 ms | **-15.2%** |
| Warm 20K engine TTFT (`+18` new tok) | 33.95 ms | **19.38 ms** | 19.63 ms | **-42.9% (1.75x)** |
| Warm 20K engine TTFT (`+35` new tok) | 31.37 ms | **22.39 ms** | 22.41 ms | **-28.6%** |
| C=8 batched decode (`fixed8`) | 1,646.3 tok/s | 1,708.8 tok/s | **1,710.3 tok/s** | **+3.9% (+64 tok/s)** |
| C=4 batched decode (`default` / `fixed8`) | 1,147.8 / 1,136.2 | **1,179.2 / 1,182.1** | 1,179.5 / 1,178.5 | **+2.7% / +4.0%** |
| C=4 32K batched decode | 886.0 tok/s | **904.2 tok/s** | 905.0 tok/s | +2.1% |
| 32K long-context code decode | 347.2 tok/s | **367.0 tok/s** | 355.2 tok/s | +5.7% (2-path) / +2.3% (1-path) |
| Edit / copy verbatim decode | 1,001.1 tok/s | **1,038.6 tok/s** | 1,038.2 tok/s | +3.7% |
| Prose decode | 221.3 tok/s | **229.6 tok/s** | 205.6 tok/s | +3.7% (2-path) |
| Short code decode | 344.3 tok/s | 344.4 tok/s | **348.1 tok/s** | flat (+0.0% / +1.1%) |

### 2. End-to-End Serving (`v0.5.0` behind `socat` → Moonshot Fork over UDS + `q27-fwd`)

*Important baseline caveats:*
- The baseline ran official `ghcr.io/mkyt/qwen3-27b-5090:0.5.0` (27 upstream commits before `fe41696`) behind a local `socat` TCP proxy (no `TCP_NODELAY`, forks per connection), so the HTTP keep-alive RTT and frontend-overhead improvements reflect **both** the fork's HTTP/1.1 UDS stack and replacing `socat`.
- Correction (25 Nov 2026): this run used the `4.0 / 4.5` thresholds already (the measured image's library carries them), so the `-23.5%` sampled-reasoning row is not explained by the thresholds. It is one unseeded sample. All workloads — wins and regressions — are shown below.

| Workload | `v0.5.0` + `socat` | Moonshot + `q27-fwd` | Delta |
|---|---:|---:|---:|
| HTTP keep-alive RTT (p50) | 41.02 ms | **0.13 ms** | **325x faster** *(incl. removing `socat`)* |
| Warm 60K follow-up TTFT (p50) | 92.91 ms | **25.07 ms** | **3.7x faster** (`fe`: `52.2 -> 3.1 ms`) |
| Warm 20K follow-up TTFT (p50) | 48.18 ms | **21.03 ms** | **2.3x faster** (`fe`: `20.0 -> 2.2 ms`) |
| Cold 64-token TTFT | 36.73 ms | **20.07 ms** | **1.83x faster** |
| C=4 32K worst-case ITL stall (max / p95) | 544.6 / 482.9 ms | **56.3 / 52.2 ms** | **9.7x smoother** |
| C=4 32K mean stream / aggregate | 115.6 / 65.6 tok/s | **214.0 / 68.0 tok/s** | **+85.1% stream** / **+3.6% agg** |
| C=8 concurrent code (agg / max ITL) | 1,117.7 tok/s / 135.2 ms | **1,229.7 tok/s / 43.0 ms** | **+10.0%**, **3.1x lower tail** |
| C=4 concurrent code (agg / p50 TTFT) | 859.2 tok/s / 151.3 ms | **948.5 tok/s / 87.9 ms** | **+10.4%**, **1.7x faster TTFT** |
| Short request beside C=4 (p50 / max TTFT) | 28.5 / 37.7 ms | **21.8 / 24.3 ms** | **1.30x faster** |
| 32K long-context single code | 326.2 tok/s | **370.1 tok/s** | **+13.5%** |
| Sampled code decode | 306.2 tok/s | **338.1 tok/s** | **+10.4%** |
| Greedy code decode | 339.0 tok/s | 331.4 tok/s | -2.2% |
| Greedy JSON decode | 766.6 tok/s | 754.1 tok/s | -1.6% |
| Greedy edit / copy decode | 1,040.0 tok/s | 1,011.8 tok/s | -2.7% |
| C=4 mixed sampled (agg / max ITL) | 802.5 tok/s / 62.5 ms | 760.5 tok/s / **39.3 ms** | -5.2% agg / **1.6x lower tail** |
| Greedy prose decode | 245.7 tok/s | 219.7 tok/s | -10.6% |
| Sampled reasoning *(one unseeded sample)* | 411.0 tok/s | 314.6 tok/s | -23.5% *(cause not established)* |

### 3. RTX 5090 450W Undervolt + GDDR7 OC (`tools/rtx5090-undervolt.py`)

Measured live on `moonshot` (`3x512` single decode + `2,048`-token C=4 burst, greedy SHA-identical across passes). The speed gain comes from combining the V/F curve offset (`+180 MHz`) with the GDDR7 memory overclock (`+1,000` to `+1,500 MHz`).

| Profile | Single Decode | Power Cap & Temp | Notes |
|---|---:|---:|---|
| 550W capped stock-clocks ref | 406.9 tok/s | 550 W cap / 57°C | Stock V/F & GDDR7 clocks (factory TDP is 575 W) |
| 450W default (`+180 / 2750 / +1000`) | **410.9 tok/s** (`+1.0%`) | **450 W cap / 53°C** (`-100 W` vs ref, `-4°C`) | `1.92 tok/J` single (avg `214 W`) |
| 450W tuned (`+180 / 2800 / +1500`) | **416.9 avg** / **424.8 peak** (`+2.5%` avg) | **450 W cap / 53°C** (`-100 W` vs ref, `-4°C`) | C=4 burst: **1,052 tok/s @ 448 W peak** (`~2.35+ tok/J` at burst power) |

---

## Quick Start: Build & Run

### Build the CUDA engine (`libq27.so`)
```bash
cmake -S engine -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a
ninja -C build
```

### Build the Docker image (from this fork — do not `docker compose pull`)
```bash
docker build -t qwen3-27b-5090:moonshot .
```

### Optional: `--network none` isolation with `q27-fwd`
```bash
gcc -O2 -Wall -Wextra tools/q27-fwd.c -o q27-fwd
# Run the container with --network none and -e LISTEN_UNIX=/run/q27/api.sock, then forward locally:
./q27-fwd 127.0.0.1:8000 unix:/run/q27-engine/api.sock
```

### Optional: RTX 5090 450W undervolt + GDDR7 OC on Linux
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
| `Q27_LOOKUP` | `1` | Enable hybrid CUDA Graph conditional prompt-lookup (`0` to disable / restore upstream default) |
| `Q27_LOOKUP_MIN` | `8` | Minimum n-gram match length to bypass DSpine on verbatim hits |
| `Q27_SPEC_ADAPT_PATHS` | `2` | DSpine tree paths used when adaptive width drops below `Q27_SPEC_ADAPT_DOWN` (`1` = chains only) |
| `Q27_DSPINE_LR` | `1` | Precomputed low-rank DSpine refinement (`Gram G` + `U_head`, `0` = original path) |
| `PREFILL_DECODE_CHUNK` | `512` | Sarathi prefill chunk budget (64-aligned) when decode streams are active |
| `LISTEN_UNIX` | *(unset)* | Bind the HTTP API on a Unix domain socket path instead of TCP |
| `ADMIN_LOCK` | `0` | Set `1` to block mutating `POST /admin/*` endpoints from autonomous agents |
