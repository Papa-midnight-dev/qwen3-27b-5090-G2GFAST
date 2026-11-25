#!/usr/bin/env python3
"""Comprehensive latency, wire-bytes, lag, and throughput benchmark suite for q27-engine."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

try:  # optional token-usage hook; set Q27_USAGE_LOG_DIR to a folder that holds llm_usage_log.py
    if os.environ.get("Q27_USAGE_LOG_DIR"):
        sys.path.insert(0, os.environ["Q27_USAGE_LOG_DIR"])
    import llm_usage_log  # noqa: E402
except ImportError:
    llm_usage_log = None
_log_usage = getattr(llm_usage_log, "log_usage", lambda **kw: None)

SCRIPT_NAME = "q27_hillclimb_bench"


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def make_filler(approx_tokens: int, seed: int = 1) -> str:
    """Deterministic varied technical text (~40 Qwen tokens per line) avoiding degenerate repetition."""
    lines = []
    # ~40 tokens per line for this format
    n_lines = max(1, approx_tokens // 40)
    topics = [
        "scheduler latency budget", "NVFP4 block scale quantization", "GDN recurrent state decay",
        "paged attention KV cache", "speculative draft acceptance", "CUDA persistent kernel grid",
        "tensor core MMA pipeline", "ring buffer zero-copy IPC", "prefix checkpoint eviction",
        "HTTP chunked stream framing", "BPE incremental UTF-8 decode", "memory bandwidth saturation",
    ]
    for i in range(n_lines):
        t = topics[(i + seed * 7) % len(topics)]
        v = (i * 2654435761 + seed * 1013904223) & 0xFFFF
        lines.append(f"[{seed:02d}:{i:05d}] Record 0x{v:04x}: measured {t} delta={(i % 97) + 1}.{v % 10}ms across lane {i % 8}.")
    return "\n".join(lines)


CODE_PROMPT = (
    "Write a complete, self-contained C++20 lock-free single-producer single-consumer (SPSC) "
    "ring buffer header supporting batched push/pop, cache-line padding to prevent false sharing, "
    "acquire/release memory ordering, and a small test main() that verifies 1M items. "
    "Include clear comments and edge-case handling."
)

PROSE_PROMPT = (
    "Explain in thorough technical detail how modern GPUs hide global memory latency across "
    "warps, tensor cores, L1/L2 caches, and asynchronous shared-memory copies (TMA / cp.async). "
    "Discuss occupancy trade-offs, register pressure, and bank conflicts."
)

JSON_PROMPT = (
    "Output ONLY a valid JSON object (no markdown fences) describing a 12-service microservices "
    "benchmark topology. Each service entry must have fields: id, name, language, protocol, "
    "port, target_p50_ms, target_p99_ms, memory_mb, dependencies (array of ids), and endpoints "
    "(array of {method, path, payload_bytes})."
)

EDIT_SOURCE = """```python
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class RequestState:
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int = 256
    temperature: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    first_token_at: Optional[float] = None
    output_tokens: List[int] = field(default_factory=list)
    finished: bool = False

class BatchScheduler:
    def __init__(self, max_batch_size: int = 8, max_context: int = 131072):
        self.max_batch_size = max_batch_size
        self.max_context = max_context
        self.waiting: List[RequestState] = []
        self.running: Dict[str, RequestState] = {}
        self.completed: List[RequestState] = []

    def enqueue(self, req: RequestState) -> None:
        if len(req.prompt_tokens) > self.max_context:
            raise ValueError(f"Prompt {len(req.prompt_tokens)} exceeds max_context {self.max_context}")
        self.waiting.append(req)

    def admit(self) -> List[RequestState]:
        admitted = []
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting.pop(0)
            self.running[req.request_id] = req
            admitted.append(req)
        return admitted

    def record_token(self, request_id: str, token_id: int, is_eos: bool = False) -> None:
        req = self.running[request_id]
        if req.first_token_at is None:
            req.first_token_at = time.monotonic()
        req.output_tokens.append(token_id)
        if is_eos or len(req.output_tokens) >= req.max_tokens:
            req.finished = True
            self.completed.append(self.running.pop(request_id))

    def metrics_snapshot(self) -> str:
        return json.dumps({
            "waiting": len(self.waiting),
            "running": len(self.running),
            "completed": len(self.completed),
        })
```"""

EDIT_PROMPT = (
    "Refactor the following Python code to add priority scheduling (an integer `priority: int = 0` "
    "field on `RequestState`, higher priority admitted first with FIFO tie-breaking), track "
    "`prefix_cached_tokens: int = 0`, and add a `cancel(request_id: str) -> bool` method on "
    "`BatchScheduler`. Keep all existing methods, names, docstrings, and logic intact, and "
    "output the COMPLETE updated Python file.\n\n" + EDIT_SOURCE
)

REASONING_PROMPT = (
    "A distributed cache has 4 nodes using consistent hashing with 128 virtual nodes each. "
    "A key lookup takes 0.4ms on a hit, 4.2ms on a miss (database fetch), and 1.1ms for an "
    "inter-node forward. Suppose hit rate is 82%, 15% of requests hit a stale routing table "
    "requiring one forward, and node failure causes 25% of keys to rebalance. Calculate the "
    "expected p50 and mean latency before and during a single-node failure, showing all steps."
)


def detect_model(base_url: str) -> str:
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{base_url}/v1/models")
        r.raise_for_status()
        data = r.json()
        return data["data"][0]["id"]


def bench_http_rtt(base_url: str, n: int = 40) -> dict[str, Any]:
    out = {}
    for mode in ("fresh_conn", "keep_alive"):
        rtts = []
        total_bytes = 0
        if mode == "keep_alive":
            with httpx.Client(timeout=10.0) as c:
                for _ in range(n):
                    t0 = time.perf_counter()
                    r = c.get(f"{base_url}/health")
                    r.raise_for_status()
                    rtts.append((time.perf_counter() - t0) * 1000.0)
                    total_bytes += len(r.content)
        else:
            for _ in range(n):
                t0 = time.perf_counter()
                with httpx.Client(timeout=10.0, headers={"Connection": "close"}) as c:
                    r = c.get(f"{base_url}/health")
                    r.raise_for_status()
                    total_bytes += len(r.content)
                rtts.append((time.perf_counter() - t0) * 1000.0)
        out[mode] = {
            "n": n,
            "p50_ms": round(pct(rtts, 50), 3),
            "p95_ms": round(pct(rtts, 95), 3),
            "p99_ms": round(pct(rtts, 99), 3),
            "mean_ms": round(statistics.mean(rtts), 3),
            "bytes_per_req": round(total_bytes / n, 1),
        }
    # Pure raw TCP fresh-connection RTT (no httpx.Client constructor overhead)
    u = urlparse(base_url)
    host, port = u.hostname or "127.0.0.1", u.port or 80
    req_raw = f"GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode("ascii")
    raw_rtts = []
    raw_bytes = 0
    for _ in range(n):
        t0 = time.perf_counter()
        with socket.create_connection((host, port), timeout=5.0) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.sendall(req_raw)
            chunks = []
            while True:
                d = s.recv(4096)
                if not d:
                    break
                chunks.append(d)
        raw_rtts.append((time.perf_counter() - t0) * 1000.0)
        raw_bytes += sum(len(x) for x in chunks)
    out["fresh_raw_tcp"] = {
        "n": n,
        "p50_ms": round(pct(raw_rtts, 50), 3),
        "p95_ms": round(pct(raw_rtts, 95), 3),
        "p99_ms": round(pct(raw_rtts, 99), 3),
        "mean_ms": round(statistics.mean(raw_rtts), 3),
        "wire_bytes_per_req": round(raw_bytes / n, 1),
    }
    return out


def stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    enable_thinking: bool = False,
    timeout: float = 300.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if temperature == 0.0:
        payload["seed"] = 42
    req_bytes = len(json.dumps(payload).encode("utf-8"))

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=timeout)

    t0 = time.perf_counter()
    first_chunk_t: float | None = None
    first_tok_t: float | None = None
    token_times: list[float] = []
    wire_bytes = 0
    sse_events = 0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}

    try:
        with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            buf = b""
            for raw in resp.iter_raw():
                now = time.perf_counter()
                wire_bytes += len(raw)
                if first_chunk_t is None and raw:
                    first_chunk_t = now
                buf += raw
                while b"\n" in buf:
                    line_b, buf = buf.split(b"\n", 1)
                    line = line_b.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data_b = line[5:].strip()
                    if data_b == b"[DONE]":
                        continue
                    sse_events += 1
                    ev = json.loads(data_b)
                    if "usage" in ev and ev["usage"]:
                        usage = ev["usage"]
                    if "timings" in ev and ev["timings"]:
                        timings = ev["timings"]
                    choices = ev.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        c = delta.get("content")
                        rc = delta.get("reasoning_content")
                        r = delta.get("reasoning")
                        if c:
                            content_parts.append(c)
                            if first_tok_t is None:
                                first_tok_t = now
                            token_times.append(now)
                        elif rc or r:
                            reasoning_parts.append(rc or r)
                            if first_tok_t is None:
                                first_tok_t = now
                            token_times.append(now)
        t_end = time.perf_counter()
    finally:
        if own_client:
            client.close()

    total_ms = (t_end - t0) * 1000.0
    ttft_ms = ((first_tok_t or t_end) - t0) * 1000.0
    itls = [(token_times[i] - token_times[i - 1]) * 1000.0 for i in range(1, len(token_times))]

    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    prompt_ms = float(timings.get("prompt_ms", 0.0)) if timings else 0.0
    pred_ms = float(timings.get("predicted_ms", 0.0)) if timings else 0.0

    gen_s = max(1e-6, (t_end - (first_tok_t or t_end)))
    client_tok_s = (max(0, completion_tokens - 1) / gen_s) if completion_tokens > 1 else 0.0
    engine_tok_s = float(timings.get("predicted_per_second", 0.0)) if timings else (
        completion_tokens / max(1e-6, pred_ms / 1000.0) if pred_ms > 0 else 0.0
    )

    _log_usage(
        script=SCRIPT_NAME,
        model=model,
        endpoint=f"{base_url}/v1/chat/completions",
        usage=usage or {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        latency_ms=int(round(total_ms)),
    )

    full_text = "".join(reasoning_parts) + "\n---\n" + "".join(content_parts)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "total_ms": round(total_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "engine_prompt_ms": round(prompt_ms, 2),
        "frontend_overhead_ms": round(ttft_ms - prompt_ms, 2) if prompt_ms > 0 else None,
        "engine_pred_ms": round(pred_ms, 2),
        "client_tok_s": round(client_tok_s, 2),
        "engine_tok_s": round(engine_tok_s, 2),
        "itl_p50_ms": round(pct(itls, 50), 3),
        "itl_p95_ms": round(pct(itls, 95), 3),
        "itl_p99_ms": round(pct(itls, 99), 3),
        "itl_max_ms": round(max(itls) if itls else 0.0, 3),
        "req_bytes": req_bytes,
        "sse_wire_bytes": wire_bytes,
        "sse_events": sse_events,
        "wire_bytes_per_tok": round(wire_bytes / max(1, completion_tokens), 2),
        "sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest()[:16],
        "content_preview": "".join(content_parts)[:160],
    }


def bench_concurrent(
    base_url: str,
    model: str,
    prompts: list[list[dict[str, Any]]],
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futs = [
            ex.submit(
                stream_chat,
                base_url,
                model,
                msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for msgs in prompts
        ]
        results = [f.result() for f in futs]
    wall_s = time.perf_counter() - t0
    tot_gen = sum(r["completion_tokens"] for r in results)
    tot_wire = sum(r["sse_wire_bytes"] for r in results)
    ttfts = [r["ttft_ms"] for r in results]
    itls_p95 = [r["itl_p95_ms"] for r in results]
    return {
        "concurrency": len(prompts),
        "wall_ms": round(wall_s * 1000.0, 2),
        "total_completion_tokens": tot_gen,
        "agg_tok_s": round(tot_gen / max(1e-6, wall_s), 2),
        "mean_stream_tok_s": round(statistics.mean([r["client_tok_s"] for r in results]), 2),
        "ttft_p50_ms": round(pct(ttfts, 50), 2),
        "ttft_p95_ms": round(pct(ttfts, 95), 2),
        "itl_p95_max_ms": round(max(itls_p95), 3),
        "wire_bytes_per_tok": round(tot_wire / max(1, tot_gen), 2),
        "streams": results,
    }


def bench_short_beside_c4(base_url: str, model: str) -> dict[str, Any]:
    bg_prompts = [
        [{"role": "user", "content": f"[BG {i}] " + CODE_PROMPT}] for i in range(4)
    ]
    short_msg = [{"role": "user", "content": "Reply with exactly one word: READY"}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        bg_futs = [
            ex.submit(stream_chat, base_url, model, m, max_tokens=384, temperature=0.0)
            for m in bg_prompts
        ]
        time.sleep(0.35)
        short_samples = []
        for k in range(3):
            r = stream_chat(
                base_url,
                model,
                [{"role": "user", "content": f"Check {k}: reply with exactly one word: OK"}],
                max_tokens=16,
                temperature=0.0,
            )
            short_samples.append(r["ttft_ms"])
            time.sleep(0.15)
        bg_res = [f.result() for f in bg_futs]
    return {
        "short_ttft_samples_ms": short_samples,
        "short_ttft_p50_ms": round(pct(short_samples, 50), 2),
        "short_ttft_max_ms": round(max(short_samples), 2),
        "bg_agg_tok_s": round(
            sum(r["completion_tokens"] for r in bg_res)
            / max(1e-6, max(r["total_ms"] for r in bg_res) / 1000.0),
            2,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    model = detect_model(args.url)
    print(f"[{args.tag}] Target={args.url} Model={model}", flush=True)

    # Warmup request
    _ = stream_chat(args.url, model, [{"role": "user", "content": "Say hi."}], max_tokens=8, temperature=0.0)

    report: dict[str, Any] = {
        "tag": args.tag,
        "url": args.url,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "http_rtt": {},
        "cold_prefill": {},
        "warm_followup": {},
        "single_decode": {},
        "concurrent": {},
    }

    print("1) HTTP RTT...", flush=True)
    report["http_rtt"] = bench_http_rtt(args.url, n=30 if args.quick else 50)
    print("   ", report["http_rtt"], flush=True)

    print("2) Cold prefill TTFT...", flush=True)
    cold_sizes = [64, 4096, 16384] if args.quick else [64, 4096, 16384, 32768, 65536]
    run_nonce = int(time.time()) % 100000
    for sz in cold_sizes:
        doc = make_filler(sz, seed=run_nonce + sz)
        msgs = [{"role": "user", "content": f"Document (nonce {run_nonce}-{sz}):\n{doc}\n\nSummarize the last record in one sentence."}]
        r = stream_chat(args.url, model, msgs, max_tokens=32, temperature=0.0)
        report["cold_prefill"][f"tok_{sz}"] = r
        print(f"   cold {sz}: prompt={r['prompt_tokens']} cached={r['cached_tokens']} ttft={r['ttft_ms']}ms engine_prompt={r['engine_prompt_ms']}ms fe_oh={r['frontend_overhead_ms']}ms", flush=True)

    print("3) Warm multi-turn cached follow-up TTFT...", flush=True)
    warm_sizes = [20000] if args.quick else [20000, 60000]
    for sz in warm_sizes:
        base_doc = make_filler(sz, seed=run_nonce + 9000 + sz)
        msgs = [
            {"role": "system", "content": "You are a concise engineering assistant. Follow instructions precisely."},
            {"role": "user", "content": f"Reference log ({run_nonce}-{sz}):\n{base_doc}\n\nAcknowledge receipt with OK."},
        ]
        # Prime cache
        r0 = stream_chat(args.url, model, msgs, max_tokens=16, temperature=0.0)
        msgs.append({"role": "assistant", "content": r0["content_preview"] or "OK."})
        turns = []
        n_turns = 3 if args.quick else 5
        for t in range(n_turns):
            msgs.append({"role": "user", "content": f"Turn {t+1}: what is the hex value in record 0000{t}? Reply briefly."})
            rt = stream_chat(args.url, model, msgs, max_tokens=32, temperature=0.0)
            turns.append(rt)
            msgs.append({"role": "assistant", "content": rt["content_preview"] or "Checked."})
            print(f"   warm {sz} turn {t+1}: prompt={rt['prompt_tokens']} cached={rt['cached_tokens']} ttft={rt['ttft_ms']}ms engine_prompt={rt['engine_prompt_ms']}ms fe_oh={rt['frontend_overhead_ms']}ms", flush=True)
        report["warm_followup"][f"ctx_{sz}"] = {
            "prime": r0,
            "turns": turns,
            "ttft_p50_ms": round(pct([x["ttft_ms"] for x in turns], 50), 2),
            "engine_prompt_p50_ms": round(pct([x["engine_prompt_ms"] for x in turns], 50), 2),
            "fe_overhead_p50_ms": round(pct([x["frontend_overhead_ms"] for x in turns if x["frontend_overhead_ms"] is not None], 50), 2),
        }

    print("4) Single-stream decode workloads...", flush=True)
    decode_cases = [
        ("code_greedy", CODE_PROMPT, 512, 0.0, 1.0, False),
        ("code_sampled", CODE_PROMPT, 512, 0.7, 0.95, False),
        ("prose_greedy", PROSE_PROMPT, 512, 0.0, 1.0, False),
        ("json_greedy", JSON_PROMPT, 512, 0.0, 1.0, False),
        ("code_edit_greedy", EDIT_PROMPT, 512, 0.0, 1.0, False),
        ("reasoning_sampled", REASONING_PROMPT, 512, 0.7, 0.95, True),
    ]
    if args.quick:
        decode_cases = [decode_cases[0], decode_cases[2], decode_cases[4], decode_cases[5]]
    for name, prompt, mt, temp, tp, think in decode_cases:
        r = stream_chat(
            args.url, model, [{"role": "user", "content": prompt}],
            max_tokens=mt, temperature=temp, top_p=tp, enable_thinking=think,
        )
        report["single_decode"][name] = r
        print(
            f"   {name}: tok/s={r['client_tok_s']} (eng={r['engine_tok_s']}) gen={r['completion_tokens']} "
            f"ttft={r['ttft_ms']}ms itl_p50={r['itl_p50_ms']}ms p99={r['itl_p99_ms']}ms "
            f"wire={r['wire_bytes_per_tok']}B/tok sha={r['sha256']}",
            flush=True,
        )

    # Long-context decode at 32K
    doc32 = make_filler(32000, seed=run_nonce + 32000)
    r32 = stream_chat(
        args.url, model,
        [{"role": "user", "content": f"Log archive:\n{doc32}\n\n" + CODE_PROMPT}],
        max_tokens=256, temperature=0.0,
    )
    report["single_decode"]["long_ctx_32k_code"] = r32
    print(f"   long_ctx_32k_code: tok/s={r32['client_tok_s']} (eng={r32['engine_tok_s']}) ttft={r32['ttft_ms']}ms itl_p99={r32['itl_p99_ms']}ms", flush=True)

    print("5) Concurrent workloads...", flush=True)
    c4_code = [[{"role": "user", "content": f"[Stream {i}] " + CODE_PROMPT}] for i in range(4)]
    report["concurrent"]["c4_code_greedy"] = bench_concurrent(args.url, model, c4_code, max_tokens=384, temperature=0.0)
    print(f"   c4_code_greedy: agg={report['concurrent']['c4_code_greedy']['agg_tok_s']} tok/s mean={report['concurrent']['c4_code_greedy']['mean_stream_tok_s']} ttft_p95={report['concurrent']['c4_code_greedy']['ttft_p95_ms']}ms", flush=True)

    if not args.quick:
        c4_mixed = [
            [{"role": "user", "content": CODE_PROMPT}],
            [{"role": "user", "content": PROSE_PROMPT}],
            [{"role": "user", "content": JSON_PROMPT}],
            [{"role": "user", "content": EDIT_PROMPT}],
        ]
        report["concurrent"]["c4_mixed_sampled"] = bench_concurrent(args.url, model, c4_mixed, max_tokens=384, temperature=0.7)
        print(f"   c4_mixed_sampled: agg={report['concurrent']['c4_mixed_sampled']['agg_tok_s']} tok/s", flush=True)

        c8_code = [[{"role": "user", "content": f"[Lane {i}] " + CODE_PROMPT}] for i in range(8)]
        report["concurrent"]["c8_code_greedy"] = bench_concurrent(args.url, model, c8_code, max_tokens=256, temperature=0.0)
        print(f"   c8_code_greedy: agg={report['concurrent']['c8_code_greedy']['agg_tok_s']} tok/s", flush=True)

        c4_long = [
            [{"role": "user", "content": f"Context {i}:\n{make_filler(32000, seed=run_nonce + 500 + i)}\n\n" + CODE_PROMPT}]
            for i in range(4)
        ]
        report["concurrent"]["c4_long32k_greedy"] = bench_concurrent(args.url, model, c4_long, max_tokens=192, temperature=0.0)
        print(f"   c4_long32k_greedy: agg={report['concurrent']['c4_long32k_greedy']['agg_tok_s']} tok/s ttft_p95={report['concurrent']['c4_long32k_greedy']['ttft_p95_ms']}ms", flush=True)

        report["concurrent"]["short_beside_c4"] = bench_short_beside_c4(args.url, model)
        print(f"   short_beside_c4: short_ttft_p50={report['concurrent']['short_beside_c4']['short_ttft_p50_ms']}ms max={report['concurrent']['short_beside_c4']['short_ttft_max_ms']}ms", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
