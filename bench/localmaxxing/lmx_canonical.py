#!/usr/bin/env python3
"""LocalMaxxing canonical-prompt cells (26-27 Nov 2026), against a server on port 8000.
The site's method: a cache-bust nonce line ("[LocalMaxxing cache-bust nonce: <16 hex>]") before the canonical prompt,
batch 1, an untimed warmup, timed runs, median; score = the engine's timings.predicted_per_second (tokSOut).
Fixed nonces: the SAME 15 for every setting and run order (the nonce changes the text, and with it the
number of steps: 58 to 84 steps for 512 tokens of one prompt); the first is the
leader's own (7445cc4d6c60bdf0, its run had prompt_n 290). Cells: reasoning-v1 and code-v1 x reasoning effort
medium (primary: no system instruction, matches the leader's 290 prompt tokens), xhigh (the template default: adds a
system instruction) and thinking off; 512 tokens, T 0. Per request: the engine's requests_total must move by exactly 1
(no other client), the client rate must be within 2% of the engine rate (flagged if not), and the output text sha256
is kept for the exactness comparison against a SPEC=none arm. nvidia-smi pmon and the load average are logged.
Run it with nothing else using the GPU or the server. canonical_prompts.go: the LocalMaxxing CLI's
cmd/lmx/canonical_prompts.go (github.com/LottoLottoLotto/localmaxxing-cli); the script checks the prompt hashes.
usage: python3 lmx_canonical.py <canonical_prompts.go> <out dir> [--cells medium,xhigh,off] [--n 15] [--save-text]"""
import argparse, hashlib, json, os, random, re, statistics, subprocess, sys, time, urllib.request

BASE = "http://127.0.0.1:8000"; URL = BASE + "/v1/chat/completions"; MODEL = "qwen3.8-27b"
LEADER_NONCE = "7445cc4d6c60bdf0"
SAVE_TEXT = False  # --save-text: keep the whole output (the submission record needs the output sample)
NONCES = [LEADER_NONCE] + ["%016x" % random.Random(20261126 + i).getrandbits(64) for i in range(1, 15)]
WARM_NONCE = "%016x" % random.Random(1).getrandbits(64)
SITE_SHA = {"reasoning-v1": "9000edaadb16fc77ab78b39917245a03d4d7585c3ba770cb045a47a0e0683445",
            "code-v1": "a737db738bd4e37881d1737f8dc51cae0295eff9ece6551d39826ce8ed43a690"}  # the site's canonicalPrompts sha256 (26 Nov 2026)


def prompts(go_src):
    src = open(go_src).read(); out = {}
    for pid in ("reasoning-v1", "code-v1"):
        blk = src.split('ID:              "%s"' % pid)[1]
        lines = re.findall(r'^\t\t\t"((?:[^"\\]|\\.)*)",$', blk.split('}, "\\n")')[0], re.M)
        out[pid] = "\n".join(bytes(l, "utf-8").decode("unicode_escape") for l in lines)
        if hashlib.sha256(out[pid].encode()).hexdigest() != SITE_SHA[pid]:
            sys.exit(f"{pid}: the prompt text does not match the site's sha256; check canonical_prompts.go")
    return out


def requests_total():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("q27:requests_total "):
                return float(line.split()[-1])
    return None


def run(text, cell, nonce):
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"[LocalMaxxing cache-bust nonce: {nonce}]\n{text}"}],
            "max_tokens": 512, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False} if cell == "off" else {"reasoning_effort": cell}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    n0 = requests_total(); load0 = os.getloadavg()[0]
    t0 = time.perf_counter(); first = last = None; usage = timings = None; reasoning = []; content = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:]); usage = ev.get("usage") or usage; timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("reasoning_content") or d.get("content"):
                    now = time.perf_counter() - t0; first = now if first is None else first; last = now
                    reasoning.append(d.get("reasoning_content") or ""); content.append(d.get("content") or "")
    n1 = requests_total()
    n = (usage or {}).get("completion_tokens", 0); t = timings or {}
    acc = t.get("draft_n_accepted"); rs, cs = "".join(reasoning), "".join(content)
    eng = t.get("predicted_per_second"); cli = (n - 1) / (last - first) if first is not None and last > first else None
    return {"nonce": nonce, "prompt_tokens": (usage or {}).get("prompt_tokens"), "tokens": n,
            "engine_tps": round(eng, 1) if eng else None, "client_tps": round(cli, 1) if cli else None,
            "client_gap": round(cli / eng - 1, 4) if eng and cli else None, "ttft_s": round(first or 0, 3),
            "draft_n": t.get("draft_n"), "draft_n_accepted": acc,
            # the site's mean accepted length: 1 + accepted drafts per decode step (the first token comes from prefill)
            "mean_len": round((n - 1) / (n - 1 - acc), 3) if acc is not None and n - 1 - acc > 0 else None,
            "reasoning_chars": len(rs), "content_chars": len(cs),
            # cmp_sha256: an internal comparison digest (reasoning NUL content), NOT the site's outputSha256 (sha256 of
            # reasoning + content without a separator)
            "cmp_sha256": hashlib.sha256((rs + "\x00" + cs).encode()).hexdigest(), "head": (rs or cs)[:120],
            "requests_delta": None if n0 is None or n1 is None else n1 - n0, "load1": round(load0, 2),
            "timings": t, "usage": usage, **({"reasoning_text": rs, "content_text": cs} if SAVE_TEXT else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("go"); ap.add_argument("out")
    ap.add_argument("--cells", default="medium,xhigh,off"); ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--save-text", action="store_true")
    a = ap.parse_args()
    global SAVE_TEXT; SAVE_TEXT = a.save_text
    os.makedirs(a.out, exist_ok=True)
    pmon = subprocess.Popen(["nvidia-smi", "pmon", "-s", "um", "-d", "1"], stdout=open(os.path.join(a.out, "pmon.txt"), "w"),
                            stderr=subprocess.DEVNULL)
    try:
        P = prompts(a.go); bad = 0
        with open(os.path.join(a.out, "lmx.jsonl"), "a") as fo:
            for pid, text in P.items():
                for cell in a.cells.split(","):
                    run(text, cell, WARM_NONCE)  # untimed warmup
                    rs = [run(text, cell, nz) for nz in NONCES[: a.n]]
                    for r in rs:
                        bad += r["requests_delta"] != 1
                        short = r["tokens"] != 512 or r["client_tps"] is None or r["engine_tps"] is None
                        bad += short  # a short or unmeasured response fails the run
                        if short or abs(r["client_gap"]) > 0.02:
                            print(f"  flag: nonce {r['nonce']} tokens {r['tokens']} engine {r['engine_tps']} "
                                  f"client {r['client_tps']} gap {r['client_gap']}", flush=True)
                        fo.write(json.dumps({"prompt": pid, "cell": cell, **r}) + "\n")
                    eng = [r["engine_tps"] or 0 for r in rs]
                    eligible = all(r["client_gap"] is not None and abs(r["client_gap"]) <= 0.02 for r in rs)
                    print(pid, cell, "median15", statistics.median(eng), "first5", statistics.median(eng[:5]),
                          "leader-nonce", eng[0], "prompt_n", rs[0]["prompt_tokens"],
                          "mean_len", statistics.median([r["mean_len"] or 0 for r in rs]),
                          "requests outside 2%", sum(r["client_gap"] is None or abs(r["client_gap"]) > 0.02 for r in rs),
                          "every request within 2%:", eligible, flush=True)
        if bad:
            print(f"[X] {bad} bad request(s): another client shared the engine, or a response was short or unmeasured")
            sys.exit(1)
    finally:
        pmon.terminate()


if __name__ == "__main__":
    main()
