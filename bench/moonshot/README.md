# Benchmark and check scripts behind the fork's numbers

These are the scripts that produced the numbers in `docs/moonshot-fork.md`, published so the numbers can be
reproduced. They were run on one RTX 5090 between 23 and 24 Nov 2026. Home paths have been replaced by environment
variables; nothing else in the measurement logic changed.

| Script | What it measures | Needs |
|---|---|---|
| `bench_suite.py` | End-to-end through the HTTP API: keep-alive round trip, cold prefill at 64-65K tokens, warm follow-ups, one-stream decode (greedy and sampled), and 4 and 8 requests at once | a running server; `--url`, `--tag`, `--out` |
| `verify_quality.py` | Needles at 32K, 64K and 96K, GPU against host prefix-cache restores, five interleaved 20K sessions, a tool call, a JSON schema and a sum | a running server URL as the first argument |
| `bench_engine.py` | In-process engine decode and warm-turn timing, without HTTP | `WEIGHTS`, `DRAFT_DIR`, a built `libq27.so` (`--lib` or `Q27_LIB`) |
| `profile_engine.py` | Per-kernel step profile (`Q27_PROFILE=1`) | as `bench_engine.py` |
| `sweep_knobs.py` | A sweep of runtime tuning variables | as `bench_engine.py` |

Caveats, so the numbers are read correctly:
- Every published figure is a single run. The sampled rows of `bench_suite.py` have no seed, so one sampled run can
  move by tens of percent from the text alone.
- Optional token logging: set `Q27_USAGE_LOG_DIR` to a folder that holds an `llm_usage_log.py` with a `log_usage`
  function; without it nothing is logged.
