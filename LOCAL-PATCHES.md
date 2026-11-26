# Local patches on qwen3-27b-5090 v0.8.1: branch kit4b-0.8.1

This branch is upstream [mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090) v0.8.1 (MIT) with local patches.
Upstream did the hard work: the engine, the kernels and the DSpine integration. The patches add copy steps for text the
model repeats, a best-first draft tree for greedy requests, a short-prompt attention kernel and some smaller changes.
Measured on one RTX 5090 with a 450 W power limit (undervolted), on Linux, on 26-27 Nov 2026. The engine measured is
commit `b28fa62` of this branch.

## Results

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
  NVFP4 KV: 87.7% against 86.8%, upstream's `docs/benchmarks.md`) at the same speed (upstream's measurement). This
  branch with the two-term head is not yet measured on these prompts.
- Copy steps (`Q27_COPY`, default off): when the output repeats earlier text (a file it edits, a passage it quotes),
  the engine drafts the next tokens from that span and verifies up to 128 rows in one step (64 first, 128 after a step
  that accepted every row). The target model still checks every token.
- `Q27_COPY_SAMPLED` (default off): copy steps for sampled requests (a copied token is accepted with probability p).
- `Q27_EDIT_COPY` (default off): YASAKANI's learned replacements applied to the copied text (a port; see NOTICE).
- `Q27_SPEC_BFTREE` (default off): a best-first tree of 16 nodes (the anchor and 15 DSpine candidates) on greedy
  requests; sampled requests keep chains.
- `Q27_FA_SMALL` (default ON): a short-prompt attention kernel for warm follow-ups.
- `Q27_TOKEN_CACHE_MB` (default 0): the frontend caches encoded text between added tokens.
- Always active: a frontend fallback that returns a complete tool call written inside unclosed reasoning as a tool call.
- Start-up resets unsupported values and logs them: `Q27_SPEC_PATHS` becomes 2 at most (1 for draft blocks over 8
  rows); two-step rollout becomes 1; the best-first tree stays greedy-only; the batched tree is off.

## Reproduce the LocalMaxxing cells
1. Build: `git clone -b kit4b-0.8.1 https://github.com/Papa-midnight-dev/qwen3-27b-5090-moonshot && cd
   qwen3-27b-5090-moonshot && docker build -t q27:kit4b .`
2. Start: `docker run --gpus all -p 127.0.0.1:8000:8000 -v "$MODEL_DIR":/models/weights:ro -v "$HF_HOME":/hf:ro
   -e WEIGHTS=/models/weights` with each setting above as `-e NAME=value`, then `q27:kit4b`. Keep the port on
   127.0.0.1: the engine's stats page shows the start of recent requests and has no sign-in.
3. Run `python3 bench/localmaxxing/lmx_canonical.py "$PROMPTS_GO" "$OUT_DIR" --cells medium --save-text` with nothing
   else using the GPU. `$PROMPTS_GO` is `cmd/lmx/canonical_prompts.go` from
   [localmaxxing-cli](https://github.com/LottoLottoLotto/localmaxxing-cli); the script checks its sha256 against the site.

## Known limits
- The KV pool is 6% smaller than upstream's at the same settings (219,072 against 233,408 tokens; 128-row copy steps
  need more state). The context limit stays 131,072.
- The draft tree helps greedy requests only.
- With 8 requests at once on QUASAR, upstream was faster in both run orders.
- In a soak of 40 requests, all six grammar-after-thinking requests gave different text batched than alone; the
  previous release of this fork gives the same six (cause not known).

## Credits
Engine: Masahiro Kiyota and contributors, [mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090) (MIT).
Edit copy: a port of [YASAKANI](https://github.com/yasakani-dev/yasakani)'s edit_copy (Apache-2.0; NOTICE).
Draft: [tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16). Weights: the QUASAR-QAT
and sakamakismile (Huihui abliterated) NVFP4 packs of Qwen3.8-27B.
