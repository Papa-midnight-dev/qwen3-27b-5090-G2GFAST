# Local patches on qwen3-27b-5090 v0.8.1: copy steps

This branch is upstream [mkyt/qwen3-27b-5090](https://github.com/mkyt/qwen3-27b-5090) v0.8.1 (MIT) with five patches.
Upstream did all the hard work; these patches add one feature, copy steps, and leave everything else as it was.
Measured on 25 Nov 2026 on one RTX 5090 at a 450 W power limit (undervolted), with an NVFP4 abliterated
Qwen3.8-27B, FP8 KV cache, a 131,072-token context and 8 request slots.

## What copy steps do

When a greedy request (temperature 0) repeats earlier text, such as a file it edits or a passage it quotes, the engine
finds the earlier span (a hash of the last 16 tokens), drafts the next 31 tokens from it, and verifies them in one
32-row step instead of a 16-token DSpine step. The target model still verifies every token, so the output is the
same as without copy steps. When the output leaves the span, the engine goes back to DSpine; a fresh match that fails
at once (repeated structure, not a copy) backs off for 16 to 256 steps.

Turn it on with `Q27_COPY=1` (default 0). `Q27_COPY_ROWS=16|32` (default 32) and `Q27_COPY_MIN` (default 16 tokens)
tune it. 32-row steps need FP8 or NVFP4 KV and the FP8-MMA attention (the default); otherwise copy steps use 16 rows.

## The patches

| Commit | What |
|---|---|
| 9d4b8c7 | DSpine trees at the low adaptive width (`Q27_SPEC_ADAPT_PATHS`, default 1 = off; not recommended yet) |
| 60d5d5e | Copy steps of 16 rows: host-drafted chains from the span the output copies |
| 4344f6e | Copy steps of 32 rows: GDN replay in 16-row tiles, a 32-row GDN chain kernel, FP8-MMA decode attention in 16-row launches with a row offset (copy verifies only), a 32-token step output |
| 80b5a70 | Launch-ahead stays on unless a copy step may follow |
| 2f0a4dc | Review fixes: full commit-path clear, no copy across non-text (image) ids, grammar and pinned sequences clear the copy proposal, the copy lock is checked again after batched steps, `q27_set_step_capacity` (a caller with a 16-token buffer never gets more), draft scratch for 32 rows |

`engine/python/q27lib.py` here sets the step capacity to 32. A caller that does not gets copy steps of at most 16 rows.

## Results (one RTX 5090, 450 W, 25 Nov 2026)

Copy steps on against off, the same build and settings (tok/s while writing, one run each):

| Greedy task | Copy off | Copy on |
|---|---:|---:|
| YASAKANI code edit (4,706 tokens out) | 1,108 | 1,937 |
| YASAKANI JSONL edit (2,853 tokens) | 1,135 | 2,034 |
| Write an edited file again in a follow-up turn | 1,102 | 2,062 |
| Quote a section of a 25K-token document (189 tokens) | 304 | 1,124 |
| Quote 30 lines of a 25K-token document (905 tokens) | 385 | 1,827 |
| YASAKANI long JSONL (8,192 tokens), plain code / JSON / prose | level | level |

Sampled requests are unchanged by these patches.

## How the outputs were checked

- 11 greedy requests (the edits, the quotes, a long JSONL, plain code/JSON/prose, thinking that quotes a passage
  before a JSON answer, an edit cut at 1,000 tokens, a follow-up turn on the cached edit) give the same text and token
  count as the unmodified 0.8.1 library with the same settings, with CUDA graphs, without them (`Q27_NO_GRAPH=1`) and
  without dependent launches (`Q27_NO_PDL=1`).
- A code edit gives the same output alone and with a short second request that joins and ends while it runs.
- 1,000 BFCL tool calls (temperature 1.0): 926 right; needles at 32K-86K: the same misses as without the patches.
- Two GPT-6.2 Astra review rounds and two Fable 5.2 checkpoints; the findings and their fixes are in commit 2f0a4dc.

## Known limits

- Greedy requests only (sampled copy steps are in progress).
- Not tested: a grammar that starts right after a copy step (the fix that clears the copy proposal there was checked
  by reading the code); `Q27_ATTN_TC_SPLITS=1` with copy steps. Do not use `Q27_DBG_ACCEPT` with copy steps or
  `Q27_SKINNY_CHECK` with the one-term head (`Q27_LM_HEAD_A4X2=0`).
- Upstream's in-process tests (`structured_test.py`, `batch_test.py check`, `prefix_cache_test.py`) were not run with
  copy steps on.

## Credits

Copy drafting is not new: YASAKANI checks up to 64 copied tokens in one pass, and TensorFold has "copy drafts". The
design of this version came from two rounds with Gemini 4 Neon and two Fable 5.2 reviews; the code and the tests are
here.
