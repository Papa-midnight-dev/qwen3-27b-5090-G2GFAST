#!/usr/bin/env python3
"""Comprehensive quality, exactness, needle-retrieval, multi-session, tool-call, and grammar gate."""
from __future__ import annotations

import json
import os
import sys
import time
import httpx

try:  # optional token-usage hook; set Q27_USAGE_LOG_DIR to a folder that holds llm_usage_log.py
    if os.environ.get("Q27_USAGE_LOG_DIR"):
        sys.path.insert(0, os.environ["Q27_USAGE_LOG_DIR"])
    import llm_usage_log  # noqa: E402
except ImportError:
    llm_usage_log = None
_log_usage = getattr(llm_usage_log, "log_usage", lambda **kw: None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_suite import make_filler  # noqa: E402


def chat(c: httpx.Client, url: str, messages, *, max_tokens=64, temp=0.0, thinking=False, tools=None, response_format=None, stream=False):
    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "seed": 42,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if tools:
        payload["tools"] = tools
    if response_format:
        payload["response_format"] = response_format
    t0 = time.perf_counter()
    if not stream:
        r = c.post(f"{url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        _log_usage(script="q27_hillclimb_verify", model="qwen3.8-27b",
                                endpoint=f"{url}/v1/chat/completions", usage=u,
                                latency_ms=int((time.perf_counter() - t0) * 1000))
        return d
    text = []
    usage = {}
    with c.stream("POST", f"{url}/v1/chat/completions", json={**payload, "stream_options": {"include_usage": True}}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            s = line[5:].strip()
            if s == "[DONE]":
                break
            ev = json.loads(s)
            if ev.get("usage"):
                usage = ev["usage"]
            ch = (ev.get("choices") or [{}])[0].get("delta") or {}
            if ch.get("content"):
                text.append(ch["content"])
    _log_usage(script="q27_hillclimb_verify", model="qwen3.8-27b",
                            endpoint=f"{url}/v1/chat/completions", usage=usage,
                            latency_ms=int((time.perf_counter() - t0) * 1000))
    return {"content": "".join(text), "usage": usage}


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
    with httpx.Client(timeout=300.0) as c:
        # 1. ADMIN_LOCK=1 checks (all 3 locked POSTs -> 403, GETs -> 200)
        print("Gate 1: ADMIN_LOCK & endpoints...", flush=True)
        for p in ("/admin/engine", "/admin/concurrency", "/admin/engine/idle"):
            r = c.post(f"{url}{p}", json={})
            assert r.status_code == 403, f"{p} expected 403 got {r.status_code}"
        for p in ("/health", "/metrics", "/v1/models", "/admin/engine"):
            r = c.get(f"{url}{p}")
            assert r.status_code == 200, f"GET {p} expected 200 got {r.status_code}"
        print("  PASS: ADMIN_LOCK (3x 403) + GET endpoints (4x 200)", flush=True)

        # 2. Needle-in-a-Haystack at 32K, 64K, 96K (start, mid, end) + zero stray <|im_start|>
        print("Gate 2: Needle-in-a-Haystack (32K, 64K, 96K x 3 positions = 9 needles)...", flush=True)
        passed_needles = 0
        total_needles = 0
        for ctx_k in (32000, 64000, 96000):
            for pos_name, frac in (("start", 0.1), ("mid", 0.5), ("end", 0.9)):
                total_needles += 1
                code = f"KESTREL-{ctx_k//1000}-{pos_name.upper()}-{(ctx_k * 13 + int(frac*100)) % 9973:04d}"
                filler_lines = make_filler(ctx_k, seed=ctx_k + int(frac * 100)).splitlines()
                ins = int(len(filler_lines) * frac)
                filler_lines.insert(ins, f"CRITICAL OVERRIDE CODE: The secret launch passphrase is {code}. Remember this exact string.")
                doc = "\n".join(filler_lines)
                d = chat(c, url, [{"role": "user", "content": f"Archive:\n{doc}\n\nWhat is the exact CRITICAL OVERRIDE CODE secret launch passphrase? Reply with ONLY the code."}], max_tokens=32)
                ans = d["choices"][0]["message"]["content"]
                assert "<|im_start|>" not in ans and "<|im_end|>" not in ans, f"Stray special token: {ans!r}"
                ok = code in ans
                if ok:
                    passed_needles += 1
                print(f"  needle {ctx_k//1000:2d}K {pos_name:5s}: prompt={d['usage']['prompt_tokens']} expected={code} got={ans.strip()!r} -> {'PASS' if ok else 'FAIL'}", flush=True)
        assert passed_needles == total_needles, f"Needles: {passed_needles}/{total_needles}"
        print(f"  PASS: {passed_needles}/{total_needles} needles exact, 0 stray tokens!", flush=True)

        # 3. Prefix-Cache Exactness Gate (Cold vs GPU stage_ hit vs Host RAM hit after eviction of stage_)
        print("Gate 3: Prefix-Cache Bit-Exactness (Cold vs GPU stage_ hit vs Host RAM hit)...", flush=True)
        doc_a = make_filler(20000, seed=7771)
        doc_b = make_filler(20000, seed=7772)
        msgs_a_base = [{"role": "user", "content": f"Doc A:\n{doc_a}\n\nSay READY."}]
        r_prime = chat(c, url, msgs_a_base, max_tokens=16)
        ans_prime = r_prime["choices"][0]["message"]["content"]
        msgs_a_q = msgs_a_base + [
            {"role": "assistant", "content": ans_prime},
            {"role": "user", "content": "Explain the pattern in records 00000 to 00005 in 3 sentences."},
        ]
        # GPU stage_ hit (immediate follow-up on Doc A)
        r_gpu = chat(c, url, msgs_a_q, max_tokens=96)
        txt_gpu = r_gpu["choices"][0]["message"]["content"]
        cached_gpu = r_gpu["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert cached_gpu > 15000, f"Expected GPU cache hit, got {cached_gpu}"
        # Now run Doc B to overwrite GPU stage_, forcing Doc A to restore from Host RAM!
        _ = chat(c, url, [{"role": "user", "content": f"Doc B:\n{doc_b}\n\nSay OK."}], max_tokens=16)
        r_host = chat(c, url, msgs_a_q, max_tokens=96)
        txt_host = r_host["choices"][0]["message"]["content"]
        cached_host = r_host["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert cached_host > 15000, f"Expected Host RAM cache hit, got {cached_host}"
        assert txt_gpu == txt_host, f"GPU stage_ vs Host RAM mismatch!\nGPU:\n{txt_gpu}\nHOST:\n{txt_host}"
        print(f"  PASS: GPU stage_ hit ({cached_gpu} cached) == Host RAM hit ({cached_host} cached) EXACT MATCH!", flush=True)

        # 4. Multi-session interleaved cache (5 x 20K sessions)
        print("Gate 4: Multi-Session Interleaved Cache (5 x 20K sessions)...", flush=True)
        sessions = []
        for s in range(5):
            base = [{"role": "user", "content": f"Session {s}:\n{make_filler(18000, seed=8800+s)}\n\nAcknowledge with S{s}."}]
            r0 = chat(c, url, base, max_tokens=12)
            sessions.append(base + [{"role": "assistant", "content": r0["choices"][0]["message"]["content"]}])
        for s in range(5):
            q = sessions[s] + [{"role": "user", "content": f"What is record 0000{s}?"}]
            rt = chat(c, url, q, max_tokens=24)
            pt = rt["usage"]["prompt_tokens"]
            ct = rt["usage"]["prompt_tokens_details"]["cached_tokens"]
            ratio = ct / max(1, pt)
            print(f"  session {s}: cached={ct}/{pt} ({ratio*100:.1f}%)", flush=True)
            assert ratio > 0.95, f"Session {s} cache miss: {ct}/{pt}"
        print("  PASS: All 5 interleaved sessions >95% cached!", flush=True)

        # 5. Tool calling + Structured JSON Schema (llguidance) over keep-alive chunked SSE
        print("Gate 5: Tool Calling + Structured JSON Schema (llguidance)...", flush=True)
        tools = [{
            "type": "function",
            "function": {
                "name": "get_metric",
                "description": "Fetch a system latency metric",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }]
        rt = chat(c, url, [{"role": "user", "content": "Call get_metric with name='ttft_p50'."}], tools=tools, max_tokens=128)
        tcs = rt["choices"][0]["message"].get("tool_calls") or []
        assert len(tcs) == 1 and tcs[0]["function"]["name"] == "get_metric", f"Bad tool call: {rt}"
        assert json.loads(tcs[0]["function"]["arguments"])["name"] == "ttft_p50"

        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "latency_ms": {"type": "number"},
                    },
                    "required": ["status", "latency_ms"],
                    "additionalProperties": False,
                },
            },
        }
        rs = chat(c, url, [{"role": "user", "content": "Return status='fast' and latency_ms=21.5 as JSON."}],
                  response_format=schema, max_tokens=64, stream=True)
        parsed = json.loads(rs["content"])
        assert parsed["status"] == "fast" and abs(parsed["latency_ms"] - 21.5) < 0.01, parsed
        print("  PASS: Tool call + llguidance streaming JSON schema exact!", flush=True)

        # 6. Quick GSM8K-style reasoning check
        print("Gate 6: Math/Reasoning sanity check...", flush=True)
        rm = chat(c, url, [{"role": "user", "content": "A server processes 42 requests per second. Each request takes 15ms of GPU time on average, and 3 requests run in parallel. After 2 minutes, how many total requests were processed? Show brief steps and give the final integer."}], max_tokens=1024, thinking=True)
        mc = rm["choices"][0]["message"]["content"]
        assert "5040" in mc or "5,040" in mc, f"Math check failed: {mc!r}"
        print(f"  PASS: Math reasoning exact (5040)! reasoning_len={len(rm['choices'][0]['message'].get('reasoning_content',''))}", flush=True)

    print("\nALL 6 QUALITY & REGRESSION GATES PASSED 100%!", flush=True)


if __name__ == "__main__":
    main()
