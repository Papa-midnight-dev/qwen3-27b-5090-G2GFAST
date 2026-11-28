# Guide: Qwen3.8-27B at 600+ tok/s on one RTX 5090

This guide takes you from a clean Linux box with an RTX 5090 to the `g2gfast-0.8.1` setup (28 Nov 2026: **687.5 tok/s** median15 / **729.1** leader-nonce on QUASAR `reasoning-v1`, **466.4 tok/s** median15 on `code-v1`, building on the verified 27 Nov `kit4b-0.8.1` [LocalMaxxing](https://www.localmaxxing.com) runs at 663.9 / 453.0). Each step ends with a check, so you know it worked before you go on.
The results, the method and the checks are in [LOCAL-PATCHES.md](LOCAL-PATCHES.md).

## What makes it fast
- **The engine.** [mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090) is a C++/CUDA engine written for this
  one model on this one card. Most of the speed is its work. This branch adds patches on top.
- **4-bit weights and an 8-bit cache.** NVFP4 weights (about 20 GB) and an FP8 key-value cache let the 27B model and a
  131,072-token context fit in 32 GB.
- **Speculative decoding with DSpine.** A small draft model ([tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16))
  guesses the next tokens. The big model checks all the guesses in one pass and keeps the ones it agrees with. The output
  is the same as without the draft; only the speed changes. On the reasoning prompt the big model keeps about 8.5
  tokens per pass instead of 1.
- **A best-first draft tree (this branch).** For greedy requests (temperature 0) the draft offers its 15 best candidates
  as a tree, not one line of guesses, so more of them survive the check.
- **Copy steps (this branch).** When the model repeats text it has seen (a file it edits, a passage it quotes), the
  engine takes the guesses from that text and checks up to 128 tokens in one pass: file edits run at about 3,600 tok/s.
- **Reasoning effort "medium".** The model's chat template adds an instruction to think at length unless you ask for
  `medium`. Concise thinking is easier for the draft to guess: 664 tok/s at medium against 554 at the default.

## What you need
- An NVIDIA RTX 5090 (32 GB) with a driver new enough for CUDA 13.3 containers.
- Linux with Docker and the NVIDIA Container Toolkit. Check: `docker run --rm --gpus all ubuntu nvidia-smi` shows the card.
- About 30 GB of free disk for the weights and the draft, and room for the image build.
- The Hugging Face CLI (`pip install -U "huggingface_hub[cli]"`, the command is `hf`) and Python 3.
- A power limit you are happy with. We ran at 450 W (`sudo nvidia-smi -pl 450`); the card draws about 390 W while it
  generates.

## Steps

### 1. Get the code
```bash
git clone https://github.com/Papa-midnight-dev/qwen3-27b-5090-G2GFAST
cd qwen3-27b-5090-G2GFAST
```
Check: `git branch --show-current` prints `g2gfast-0.8.1`.

### 2. Download the draft and the weights
The engine never downloads at run time, so download first. The draft must be the revision the engine pins:
```bash
hf download tencent/Qwen3.8-27B-DSpine-b16 --revision 5f54174d301dbca65d9aa4448dc47e56732d9a12 --cache-dir ./hf/hub
hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 --local-dir ./weights
```
For the abliterated weights, download `sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4` into `./weights` instead.
Check: `du -sh hf weights` shows about 4 GB and 20 GB.

### 3. Build the image
```bash
docker build -t q27:g2gfast .
```
This compiles the CUDA code, so it takes a while the first time. Check: `docker images q27` lists `g2gfast`.

### 4. Start the engine
Stop any other program that uses the GPU first (another engine, a game). Then:
```bash
docker run -d --name q27 --gpus all --shm-size=8g --ulimit memlock=-1 -p 127.0.0.1:8000:8000 \
  --env-file bench/localmaxxing/g2gfast.env \
  -v "$PWD/hf":/hf:ro -v "$PWD/weights":/models/weights:ro q27:g2gfast
```
Keep the port on 127.0.0.1: the engine's stats page shows the start of recent requests and has no sign-in.
`bench/localmaxxing/g2gfast.env` holds every setting of branch `g2gfast-0.8.1` (Class A + Class B), each with a comment.
Check: after about a minute, `curl -s 127.0.0.1:8000/health` answers, and `docker logs q27` shows a line that
starts `[q27] kv pool:` (the number of tokens depends on the free GPU memory).

### 5. Send a request
```bash
curl -s 127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b", "max_tokens": 512, "temperature": 0,
  "chat_template_kwargs": {"reasoning_effort": "medium"},
  "messages": [{"role": "user", "content": "Explain speculative decoding in three sentences."}]}'
```
Check: the answer comes back, and its `timings.predicted_per_second` is in the hundreds.

### 6. Measure the LocalMaxxing cells
Get the canonical prompts from the LocalMaxxing CLI, then run 15 fixed variants of each prompt:
```bash
curl -sLO https://raw.githubusercontent.com/LottoLottoLotto/localmaxxing-cli/main/cmd/lmx/canonical_prompts.go
python3 bench/localmaxxing/lmx_canonical.py canonical_prompts.go results --cells medium --save-text
```
It takes about two minutes. Close anything that uses the GPU or the CPU hard while it runs (a browser playing video,
a game): on our box that cost 1-8%. Check: the last lines print a median for each prompt, for example
`reasoning-v1 medium median15 663.9 ... every request within 2%: True`, and the script exits without `[X]`.

### 7. Build a submission
```bash
python3 bench/localmaxxing/make_payload.py results reasoning-v1 QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  4a22293aa6d1026844bc93a8727f1837e48acd89 "$(git rev-parse HEAD)" canonical_prompts.go > payload.json
```
It takes the median run of the 15 and writes the evidence the site checks: the prompt and its hash, a sample of the
output, the engine's own timings and the draft counts. Give the revision of the weights you downloaded:
`head -1 weights/.cache/huggingface/download/config.json.metadata` prints it.

### 8. Check it, then submit it (optional)
Make an account on localmaxxing.com, create an API key, and keep it in a file only you can read. First the dry-run,
which stores nothing:
```bash
curl -s -X POST https://www.localmaxxing.com/api/speed-tests/dry-run -H "Authorization: Bearer $(cat ~/.lmx-key)" \
  -H 'Content-Type: application/json' --data @payload.json
```
Check: the answer has `"valid": true` and `"verified": true`. Then send the same file to
`https://www.localmaxxing.com/api/speed-tests` (the same command without `/dry-run`). You can edit a run for 24 hours.

## When something goes wrong
- **Out of memory at start.** Something else holds GPU memory. Stop it, or lower `VRAM_BUDGET_GIB` in the env file.
- **`... is not in /hf/hub` at start.** The draft is missing or has another revision. Repeat step 2 with `--revision`.
- **Much slower than expected.** Check that the request has `"reasoning_effort": "medium"`, that nothing else uses
  the GPU, and the power limit (`nvidia-smi -q -d POWER`). The first request after a start is slower (warm-up).
- **`[X] ... another client shared the engine`.** Something else sent requests during the measurement. Measure again.
- **The dry-run says not verified.** It lists the issues. A common one is a run with fewer than 256 output tokens.

## Credits
The engine is Masahiro Kiyota's [mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090) (MIT). Edit copy is a port of
[YASAKANI](https://github.com/yasakani-dev/yasakani)'s (Apache-2.0; see NOTICE). The draft is Tencent's DSpine; the weights
are the QUASAR-QAT and sakamakismile NVFP4 packs of Qwen3.8-27B.
