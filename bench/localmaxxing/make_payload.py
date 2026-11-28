#!/usr/bin/env python3
"""Build a LocalMaxxing speed-test submission (JSON) from a lmx_canonical.py record (27 Nov 2026).

It takes the MEDIAN run (by engine predicted_per_second) of one cell of a record made with --save-text: that request's
own prompt, output, engine timings and draft counts. Mean accepted length uses the site's definition: 1 + accepted draft
tokens per decode step = (n - 1) / (n - 1 - accepted). Check the result with the site's dry-run before you submit it
(GUIDE.md, step 8). Nothing is sent from here.

usage: python3 make_payload.py <out dir of lmx_canonical.py> <prompt: reasoning-v1|code-v1> <hfId> <model revision>
                               <engine commit (git rev-parse HEAD)> <canonical_prompts.go> [--cell medium] > payload.json
"""
import argparse, hashlib, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lmx_canonical import prompts  # noqa: E402

ap = argparse.ArgumentParser()
for a in ("out", "prompt", "hf_id", "revision", "commit", "go"):
    ap.add_argument(a)
ap.add_argument("--cell", default="medium")
ap.add_argument("--repo", default="https://github.com/Papa-midnight-dev/qwen3-27b-5090-G2GFAST")
a = ap.parse_args()

rows = [json.loads(l) for l in open(os.path.join(a.out, "lmx.jsonl"))]
cell = [r for r in rows if r["prompt"] == a.prompt and r["cell"] == a.cell]
if not cell or any("reasoning_text" not in r for r in cell):
    sys.exit("no runs with saved text for that prompt and cell: run lmx_canonical.py with --save-text")
if any(r["requests_delta"] != 1 or r["tokens"] < 256 for r in cell):
    sys.exit("a run in this cell shared the engine or was short: measure again")
med = sorted(cell, key=lambda r: r["engine_tps"])[len(cell) // 2]
t, u = med["timings"], med["usage"]
out = med["reasoning_text"] + med["content_text"]
n, acc = u["completion_tokens"], t["draft_n_accepted"]
eng = [r["engine_tps"] for r in cell]
body = {
    "hfId": a.hf_id, "modelRevision": a.revision,
    "hardware": {"hwClass": "DISCRETE_GPU", "gpuName": "RTX 5090", "gpuCount": 1, "vramGb": 32, "os": "Linux"},
    "engineName": "custom", "engineVersion": "0.8.1 + g2gfast patches", "engineRepository": a.repo, "engineCommit": a.commit,
    "quantization": "NVFP4", "backend": "cuda",
    "promptTokens": u["prompt_tokens"], "outputTokens": n, "contextLength": 131072, "batchSize": 1,
    "ttftMs": round(med["ttft_s"] * 1000, 1), "tokSOut": round(t["predicted_per_second"], 1),
    "promptSha256": hashlib.sha256(prompts(a.go)[a.prompt].encode()).hexdigest(),
    "promptSample": (f"[LocalMaxxing cache-bust nonce: {med['nonce']}]\n" + prompts(a.go)[a.prompt])[:2000],
    "outputSha256": hashlib.sha256(out.encode()).hexdigest(), "outputSample": out[:4000],
    "engineTimingsRaw": {"usage": u, "timings": t},
    "notes": (f"Median run of {len(cell)} fixed cache-bust nonces, batch 1, T 0, {n} tokens, reasoning_effort={a.cell}. "
              f"Median {statistics.median(eng):.1f} tok/s, range {min(eng):.1f}-{max(eng):.1f} (engine-reported)."),
    "engineFlags": {
        "kvCacheDtype": "fp8", "prefixCaching": True, "concurrency": 1, "temperature": 0,
        "specDecoding": True, "specMethod": "DSpine", "specDraftModel": "tencent/Qwen3.8-27B-DSpine-b16",
        "specDraftTokens": t["draft_n"], "specAcceptedTokens": acc,
        "specAcceptanceRate": round(acc / t["draft_n"], 4),
        "specMeanAcceptedLength": round((n - 1) / (n - 1 - acc), 3),
        "commandSnippet": "docker run --gpus all --shm-size=8g --ulimit memlock=-1 -p 127.0.0.1:8000:8000 --env-file "
                          "bench/localmaxxing/g2gfast.env -v \"$PWD/hf\":/hf:ro -v \"$PWD/weights\":/models/weights:ro q27:g2gfast"},
}
if t.get("prompt_per_second"):
    body["tokSPrefill"] = round(t["prompt_per_second"], 1)
json.dump(body, sys.stdout, indent=1)
print(f"\n{a.prompt}: tokSOut {body['tokSOut']} (median of {len(cell)})", file=sys.stderr)
