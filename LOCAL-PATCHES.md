# Local patches on qwen3-27b-5090 v0.8.1: branch g2gfast-0.8.1

This branch is upstream [mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090) v0.8.1 (MIT) with local patches,
building directly on branch `kit4b-0.8.1` (`b28fa62`, preserved unchanged for existing LocalMaxxing references).
Upstream did the hard work: the engine, the kernels and the DSpine integration. On top of `kit4b-0.8.1` (copy steps,
greedy best-first draft tree, warm-tail attention kernel), `g2gfast-0.8.1` adds two zero-loss optimization layers
controlled by opt-in environment variables:

1. **Class A — Bit-identical SM120 silicon kernels** (`Q27_ML_L2_HINT=3`, `Q27_ML_FUSE_RN=3`, `Q27_ML_GDN_PREP=2`,
   `Q27_ML_GDN_CS=1`): PTX `L2::evict_first` + `ld.global.cs` on streaming NVFP4 weights and scale factors across all
   `M=1..128`, exact `clusterDim.x=5` fused `reduce+residual+rmsnorm+NVFP4 quant`, zero-spill `gdn_prep_v2`
   (`REG: 207 -> 87`, `STACK: 32 -> 0`), zero-barrier streaming GDN state access, and zero-spill `spec_accept`
   (`STACK: 768 -> 0`). Cuts verified step latency by **-3.63% to -7.47%** with bit-identical greedy and sampled outputs.
2. **Class B — Strictly lossless optimal speculative tree + `kSnap1` + Self-EditCopy** (`Q27_ML_OPT_TREE=1`,
   `Q27_ML_SELF_EC=1`): Hu et al. (ICLR 2025) `Top-(m-1)+Residual OPT (c=1.0)` exact verification paired with strictly
   `F_0`-measurable 4-bin entropy-gated asymmetric BFS trees, a 1-slot `kSnap1` GDN DFS fast path (`REG: 153 -> 117`,
   restoring 2 CTAs/SM across all 48 GDN layers), non-trapping `F_{t-1}` hysteresis (`ot0=0`, on `< 4.0`, off `>= 4.6`),
   and Self-EditCopy for self-generated repetitive spans.

Measured on one RTX 5090 with a 450 W power limit (undervolted), on Linux, on 27 Nov 2026.

## Results: `g2gfast-0.8.1` vs `kit4b-0.8.1` (Huihui, 450 W)

### Mirrored `ABCCBA` window (n=2 per arm, `GREEDY_CHECK=1`, zero foreign requests)
All six runs matched **11 / 11 exact greedy SHA** against `kit4b-0.8.1` and passed `transition-check: SAME`.

| Measure | `kit4b-0.8.1` (n=2) | `g2gfast` Class A (n=2) | `g2gfast` Class A+B (n=2) | Class A vs `kit4b` | Class A+B vs `kit4b` |
|---|---:|---:|---:|---:|---:|
| Decode, code tok/s | 469 (462–476) | **482 (482–482)** | **480 (476–484)** | **+2.8%** | **+2.3%** |
| Decode, JSON tok/s | 729 (709–749) | **737 (736–737)** | **736 (731–742)** | **+1.1%** | **+1.0%** |
| Decode, prose tok/s | 271 (262–280) | **276 (276–276)** | 258 (256–261) | **+1.8%** | -4.8% |
| Reasoning, 1 request tok/s (sampled) | 290 (284–295) | 295 (294–296) | **311 (309–312)** | **+1.7%** | **+7.2%** |
| Reasoning, 1 request, mean of 3 (sampled) | 288 (282–294) | 286 (281–292) | **309 (309–309)** | -0.7% | **+7.3%** |
| **Reasoning, 8 at once tok/s (C=8)** | 1,495 (1,446–1,544) | 1,509 (1,483–1,535) | **1,597 (1,586–1,608)** | **+0.9%** | **+6.8% (+102 tok/s)** |
| YASAKANI long JSONL tok/s | 1,116 (1,098–1,133) | 1,125 (1,117–1,132) | **1,136 (1,135–1,136)** | **+0.8%** | **+1.8%** |
| BLACKFURY 129K warm writing tok/s | 434 (424–444) | **439 (429–450)** | 428 (427–428) | **+1.2%** | -1.4% |
| Transition check (`alone`, s) | 1.43 s (`SAME`) | **1.37 s (`SAME`)** | **1.34 s (`SAME`)** | **-4.2%** | **-6.3%** |
| Greedy parity (`11` prompts) | ref | **11/11 SHA** | **11/11 SHA** | exact | exact |

### Paired kernel step times (Class A vs `kit4b-0.8.1`, 450 W, A/A noise ±0.07%, 100% `same=True`)
| Context / Rows | `kit4b-0.8.1` | `g2gfast` Class A | Delta |
|---|---:|---:|---:|
| `512 / 8r` | 11,316 µs | **10,867–10,906 µs** | **-3.63% to -3.87%** |
| `512 / 16r` | 12,235 µs | **11,740–11,744 µs** | **-4.02%** |
| `512 / 32r` | 14,039 µs | **12,993 µs** | **-7.45%** (-1,046 µs) |
| `512 / 64r` | 16,201 µs | **15,406 µs** | **-4.90%** (-795 µs) |
| `8192 / 8r` | 11,809 µs | **11,350–11,380 µs** | **-3.63% to -3.75%** |
| `8192 / 16r` | 12,746 µs | **12,278–12,294 µs** | **-3.67%** |
| `8192 / 32r` | 14,637 µs | **13,544 µs** | **-7.47%** (-1,093 µs) |
| `8192 / 64r` | 17,240 µs | **16,449 µs** | **-4.59%** (-791 µs) |

### Verification gates on `g2gfast-0.8.1` (Class A+B active)
- **Engine kernel + CPU test suite**: `459 passed, 11 skipped`.
- **Formal exact-distribution shield (`OPT_TREE`)**: 1,000 trials against exact decode target probabilities (`forward_topk`,
  entropy `H = 2.757`, support 19): **`chi2 = 19.87, df = 18, p = 0.3399, impossible = 0`**.
- **Tool-call accuracy gate (`1,000` calls, `T=1.0, top_p=0.95, top_k=20`, thinking low)**: **917 / 1,000 (91.7%)** in 206 s
  (within the `912–926` baseline band).
- **Needles (`32K / 60K / 86K`, 16 per depth)**: **47 / 48** (`16/16`, `16/16`, `15/16` — identical to `kit4b-0.8.1`).
- **Multi-seed sampled eval (`3 seeds × 9 prompts`, `T=1.0, top_k=20, top_p=0.95`)**: Prose **+6.69%**, Reasoning **+15.71%**,
  Code **+10.22% to +21.00%**, JSON **+16.68%** over `kit4b-0.8.1`.

## Baseline Results (`kit4b-0.8.1`, commit `b28fa62`)

### LocalMaxxing canonical prompts (single request, batch 1)
These runs follow the [LocalMaxxing](https://www.localmaxxing.com) rules: a cache-bust nonce line before the canonical
prompt, batch 1, at least 256 output tokens, and evidence the site can check. We report the engine's own
`predicted_per_second`. Our choices within those rules: T 0, 512 output tokens (all of them in the reasoning channel;
the answer channel stays empty), 15 fixed nonce variants of each prompt (the first is the nonce of the site's best
verified reasoning-v1 run) and their median, and thinking on with `reasoning_effort: medium`. The chat template's
default effort (`xhigh`) adds a system instruction; that best run's record has the 290-token prompt that medium gives.
At the default effort (26 Nov): reasoning-v1 554.5 / 553.8 on QUASAR and 504.8 / 505.1 on Huihui (below the site's
best); code-v1 356.2 on QUASAR and 366.0 on Huihui.

| Weights | reasoning-v1 | code-v1 | Mean accepted length (reasoning, code) |
|---|---|---|---|
| QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 | **663.9** (other order 627.3, not used) | **453.0** (454.9) | 8.52, 5.87 |
| sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4 | **630.4** (other order 612.4, not used) | **440.2** (441.8) | 8.11, 5.68 |

The submitted runs (verified by the site, 27 Nov): QUASAR [reasoning-v1](https://www.localmaxxing.com/en/models/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4?run=cmwy44lqy057hpd0134ni7anf), [code-v1](https://www.localmaxxing.com/en/models/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4?run=cmwy452hv057kpd01ls9nbkml); Huihui [reasoning-v1](https://www.localmaxxing.com/en/models/sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4?run=cmwy453ln057npd01vbdlur7e), [code-v1](https://www.localmaxxing.com/en/models/sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4?run=cmwy45530057qpd01co4mqy6m).

One test session (27 Nov, desktop idle), both run orders. Bold = the run submitted to LocalMaxxing: the median run
of 15 in the lower of the run orders where every request's client-side rate agreed with the engine's within 2%.
Exception, decided by us when the data showed it: one code prompt reads +2.0-2.4% (client faster) on Huihui in every
session; a positive offset of up to 3% that repeats in the other order passes, since the engine figure reported is the
lower. "Not used": in that order, consecutive requests read off the engine's rate (QUASAR: 6 requests, 9.3-12.3%;
Huihui: 7 requests, 6.6-11.1%), a sign of possible interference from other programs; their median requests were
within 2%. An earlier records session (desktop in use) used a median-request check only; we tightened the check to
every request before choosing these runs. A first session the night before gave 663.8 / 664.1 and 630.8 / 630.3 on
reasoning-v1, every request within 1.6%. Sessions with the desktop in use gave 0.6-8.5% less, by weights and prompt.

For reference, the site's best verified single-RTX 5090 runs before ours (26 Nov): reasoning-v1 510.7 (NInfer-ext,
QUASAR, DFlash2; engine-timed, 512 tokens, the median of 5 runs) and 439.9 (vLLM 0.32, DFlash2; client-timed by the
site's CLI, 256 tokens); code-v1 354.8 (320 tokens, the median of 5). The site also lists unverified runs that are faster.

### This branch against upstream, the same settings
Upstream 0.8.1 with the same serve settings (KV fp8, the one-term output head, an 8,192-token draft window, DSpine)
writes the same text as this branch: 30 of 30 canonical requests and 11 of 11 fixture requests, on both weights. So
the comparison is paired: the same output, timed on both engines (27 Nov, a second session, both run orders).

| | QUASAR reasoning | QUASAR code | Huihui reasoning | Huihui code |
|---|---|---|---|---|
| This branch, median of 15 | 643 / 650 | 424 / 447 | 627 / 619 | 432 / 433 |
| Upstream, same settings | 627 / 619 | 371 / 367 | 565 / 552 | 385 / 382 |
| Paired gain (median of 30 per-request ratios) | +5% | +13% | +10% | +13% |
| Mean accepted length, this branch / upstream | 8.52 / 8.11 | 5.87 / 4.56 | 8.11 / 7.30 | 5.68 / 4.82 |

Two of those settings are ours, not upstream's defaults: the one-term output head and the 8,192-token draft window.
With its own defaults (two-term head, 2,048-token window), upstream gave 594 / 384 on Huihui reasoning / code in the
idle session; against that, the submitted runs are +6% / +15% (different text, not paired; not measured on QUASAR).
So most of the leaderboard result is upstream's: with DSpine and medium effort, upstream 0.8.1 is already above the
site's best verified reasoning-v1 run. The gain of this branch on these prompts comes from fewer steps for the same
text; the best-first tree is the likely source (not measured on its own).

### Agent work (Huihui unless named; the second session; upstream with the same settings)
| | This branch | Upstream, same settings |
|---|---|---|
| Rewriting a file (YASAKANI edit cases, greedy) | 3,579 tok/s | 1,094 tok/s |
| First token when the same 129K-token request is sent again (cached) | 0.060 s | 0.164 s |
| 8 requests at once, sum of the eight stream rates | 1,425-1,549 tok/s | 1,367-1,401 tok/s |
| 8 requests at once, QUASAR | 1,469-1,516 tok/s | 1,522-1,578 tok/s (upstream ahead) |
| Upstream's sampled decode benchmark, prose (server-default sampling, texts differ) | 280 / 287 tok/s | 252 / 258 tok/s |

The cached request reuses 128,960 of its 129,024 prompt tokens; a cold read of that document takes about 17 s.

### Decode by context length (this branch, Huihui, greedy, median of 8 prompts a point, client-side rate)
| Prompt tokens before the reply | short | 13K | 29K | 58K | 94K | 120K |
|---|---|---|---|---|---|---|
| reasoning (thinking on, medium) | 652 | 550 | 527 | 530 | 449 | 447 |
| code (thinking on, medium) | 450 | 446 | 419 | 397 | 382 | 380 |
| JSON (thinking off) | 803 | 753 | 713 | 633 | 656 | 603 |

## How the outputs were checked
On the measured commit `b28fa62`:
- Greedy output with speculation equals greedy output without it: QUASAR 90 of 90 canonical runs (three effort
  settings) and 11 of 11 fixture requests (edits, JSONL, code, prose, quotes from a 29K-token document, grammar after
  thinking); Huihui 30 of 30 and 11 of 11. It also equals upstream 0.8.1 with the same settings (above).
- Request options (penalties, logit bias, stop strings, short max_tokens), tool calls, top-5 logprobs, 8 requests at
  once against each alone, grammar after thinking and a quote from a 126,516-token document: identical to the
  previous release of this fork.

On earlier commits of this branch: an image request (`a44af24`): identical to the previous release. On `012e4a0`
(the later commits change the batch planner, the copy-step reservation and start-up checks): sampled copy steps
against plain sampling (1,000 samples against 300 plain samples, the rule written first): pass; a tool-call accuracy
gate of 1,000 calls: 931 (the previous release 929); engine kernel tests: 431 passed.

## Settings
The settings measured above (`/etc/q27-engine.env` style):
`SPEC=dspine KV_DTYPE=fp8 VISION=fp8 MAX_TOKENS=131072 MIN_KV_TOKENS=131072 MAX_SEQS=8 VRAM_BUDGET_GIB=25.2
STATE_HOST_CACHE=8 Q27_ACT_A4X2=off Q27_LM_HEAD_A4X2=0 Q27_DRAFT_WINDOW=8192 Q27_COPY=1 Q27_COPY_ROWS=128
Q27_COPY_SAMPLED=1 Q27_EDIT_COPY=1 Q27_SPEC_BFTREE=1 Q27_BFTREE_NODES=16 Q27_BFTREE_TEMP=1.2 Q27_FA_SMALL=1
Q27_TOKEN_CACHE_MB=16`.
- `Q27_LM_HEAD_A4X2=0` (the one-term output head) and `Q27_DRAFT_WINDOW=8192` are upstream settings that differ from
  upstream's defaults. Upstream's default two-term head is closer to the BF16 model (top-1 agreement on QUASAR with
  NVFP4 KV: 87.7% against 86.8%, upstream's `docs/benchmarks.md`) at the same speed (upstream's measurement). On this
  branch (27 Nov, both run orders) the two-term head gave reasoning / code 636 / 443 against 627 / 439 on Huihui and
  659 / 448 against 658 / 453 on QUASAR, with file edits within 0.4%: about the same speed on these greedy prompts. On
  sampled decoding it costs about 12% of the accepted draft tokens on the Huihui weights (upstream's sampled decode
  benchmark: code 410-431 against 497, prose 238-248 against 278-281 tok/s), so this branch keeps the one-term head.
  A 2,048-token draft window cost 8% on file edits.
- Copy steps (`Q27_COPY`, default off): when the output repeats earlier text (a file it edits, a passage it quotes),
  the engine drafts the next tokens from that span and verifies up to 128 rows in one step (64 first, 128 after a step
  that accepted every row). The target model still checks every token.
- `Q27_COPY_SAMPLED` (default off): copy steps for sampled requests (a copied token is accepted with probability p).
- `Q27_EDIT_COPY` (default off): YASAKANI's learned replacements applied to the copied text (a port; see NOTICE).
- `Q27_SPEC_BFTREE` (default off): a best-first tree of 16 nodes (the anchor and 15 DSpine candidates) on greedy
  requests; sampled requests keep chains.
- `Q27_FA_SMALL` (default ON): a short-prompt attention kernel for warm follow-ups.
- `Q27_TOKEN_CACHE_MB` (default 0): the frontend caches encoded text between added tokens.
- `Q27_ML_L2_HINT=3` (default 0): PTX `L2::evict_first` + `ld.global.cs` on streaming NVFP4 weights and scale factors
  across all `M=1..128` (Class A, bit-identical).
- `Q27_ML_FUSE_RN=3` (default 0): exact `clusterDim.x=5` fused `skinny_reduce + residual + rmsnorm + NVFP4 quant`
  (Class A, bit-identical).
- `Q27_ML_GDN_PREP=2` (default 0): zero-spill `gdn_prep_v2` for `T > 8` (`REG: 207 -> 87`, `STACK: 32 -> 0`, Class A).
- `Q27_ML_GDN_CS=1` (default 0): zero-barrier streaming `ld/st.global.cs.f32` on GDN SSM state tiles (Class A).
- `Q27_ML_OPT_TREE=1` (default 0): Hu et al. (ICLR 2025) strictly lossless `Top-(m-1)+Residual OPT` verification with
  4-bin entropy-gated asymmetric 1-slot `kSnap1` BFS trees and non-trapping `F_{t-1}` hysteresis on sampled requests
  (Class B, exact target distribution). Omit for pure bit-identical Class A mode.
- `Q27_ML_SELF_EC=1` (default 0): extends `EditCopy` to match self-generated repetitive spans (`>= 32` tokens back).
- Always active: a frontend fallback that returns a complete tool call written inside unclosed reasoning as a tool call.
- Start-up resets unsupported values and logs them: `Q27_SPEC_PATHS` becomes 2 at most (1 for draft blocks over 8
  rows); two-step rollout becomes 1; the best-first tree stays greedy-only; the batched tree is off.

## Reproduce
1. Get the code: `git clone https://github.com/Papa-midnight-dev/qwen3-27b-5090-G2GFAST && cd qwen3-27b-5090-G2GFAST`
   (the default branch is `g2gfast-0.8.1`; the previous `kit4b-0.8.1` branch is kept unchanged).
2. Download the DSpine draft at the revision the engine pins, and the weights (the engine never downloads at run time):
   `hf download tencent/Qwen3.8-27B-DSpine-b16 --revision 5f54174d301dbca65d9aa4448dc47e56732d9a12 --cache-dir ./hf/hub`
   and `hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 --local-dir ./quasar` (or
   `sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4`). We measured QUASAR revision `4a22293`; it has the same weight
   files as the revision upstream pins (`15d2e47`) and the newer `a07bf63` (only the README differs). Mounting the folder as `WEIGHTS` uses the checkpoint's own output
   head, as measured; upstream's `WEIGHTS=quasar` mode takes another checkpoint's head instead.
3. Build: `docker build -t q27:kit4b .` (it compiles the CUDA code; allow some time).
4. Start: `docker run --gpus all --ulimit memlock=-1 -p 127.0.0.1:8000:8000 -v "$PWD/hf":/hf:ro
   -v "$PWD/quasar":/models/weights:ro -e WEIGHTS=/models/weights` with each setting above as `-e NAME=value`, then
   `q27:kit4b`. Keep the port on 127.0.0.1: the engine's stats page shows the start of recent requests and has no
   sign-in.
5. Measure: `python3 bench/localmaxxing/lmx_canonical.py "$PROMPTS_GO" "$OUT_DIR" --cells medium --save-text` with
   nothing else using the GPU. `$PROMPTS_GO` is `cmd/lmx/canonical_prompts.go` from
   [localmaxxing-cli](https://github.com/LottoLottoLotto/localmaxxing-cli); the script checks its sha256 against the
   site. Expect some spread between sessions: other programs on the GPU or CPU cost us 1-8%.

## Known limits
- The KV pool is 6% smaller than upstream's at the same settings (219,072 against 233,408 tokens; 128-row copy steps
  need more state). The context limit stays 131,072.
- The greedy best-first tree (`Q27_SPEC_BFTREE=1`) applies to greedy requests only; sampled requests use the
  strictly lossless OPT tree (`Q27_ML_OPT_TREE=1`), which trades a small dip on one short-prose benchmark (-4.8%)
  for large sampled reasoning (+7.2%) and C=8 (+6.8%) gains. Omit `Q27_ML_OPT_TREE=1` for pure bit-identical Class A mode.
- With 8 requests at once on QUASAR, upstream was faster in both run orders.
- In a soak of 40 requests, all six grammar-after-thinking requests gave different text batched than alone; the
  previous release of this fork gives the same six (cause not known).

## Credits
Engine: Masahiro Kiyota and contributors, [mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090) (MIT).
Edit copy: a port of [YASAKANI](https://github.com/yasakani-dev/yasakani)'s edit_copy (Apache-2.0; NOTICE).
Draft: [tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16). Weights: the QUASAR-QAT
and sakamakismile (Huihui abliterated) NVFP4 packs of Qwen3.8-27B.
