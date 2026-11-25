#!/usr/bin/env python3
"""Fast, exact in-process CUDA-graph engine benchmark & bit-identity verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MAX_SEQS", "8")
os.environ.setdefault("VRAM_BUDGET_GIB", "25.2")
os.environ.setdefault("STATE_HOST_CACHE", "8")
os.environ.setdefault("MIN_KV_TOKENS", "131072")
os.environ.setdefault("VISION", "fp8")

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "engine", "python"))
sys.path.insert(0, os.path.join(_REPO, "frontend"))

from bench_suite import CODE_PROMPT, EDIT_PROMPT, PROSE_PROMPT, make_filler  # noqa: E402


def tok_hash(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


def run_single(eng, tok, eos_ids: set[int], prompt: str, max_new: int = 384) -> dict:
    ids = tok.prompt_ids([{"role": "user", "content": prompt}], enable_thinking=False)
    eng.reset()
    t0 = time.perf_counter()
    first = eng.spec_begin(ids, temperature=0.0, seed=42)
    t_pf = (time.perf_counter() - t0) * 1000.0
    out = [first]
    t1 = time.perf_counter()
    steps = 0
    while len(out) < max_new:
        toks = eng.spec_step()
        steps += 1
        for x in toks:
            out.append(x)
            if x in eos_ids or len(out) >= max_new:
                break
        if out[-1] in eos_ids:
            break
    dt = time.perf_counter() - t1
    gen = len(out) - 1
    return {
        "prompt_toks": len(ids),
        "prefill_ms": round(t_pf, 2),
        "gen_toks": len(out),
        "steps": steps,
        "tok_per_step": round(gen / max(1, steps), 2),
        "ms_per_step": round(dt * 1000.0 / max(1, steps), 3),
        "decode_tok_s": round(gen / max(1e-6, dt), 2),
        "sha": tok_hash(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("Q27_LIB", os.path.join(_REPO, "build", "libq27.so")))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    os.environ["Q27_LIB"] = args.lib
    if args.profile:
        os.environ["Q27_PROFILE"] = "1"
    else:
        os.environ.pop("Q27_PROFILE", None)

    import q27lib
    import tokenizer as tok_mod

    model_dir = os.environ["WEIGHTS"]  # the target checkpoint folder (NVFP4)
    draft_dir = os.environ["DRAFT_DIR"]  # the DSpine draft snapshot folder

    t0 = time.perf_counter()
    eng = q27lib.Engine(model_dir, max_tokens=131072, kv_dtype="fp8", draft=draft_dir)
    load_s = round(time.perf_counter() - t0, 2)
    eng.cache_grow(6)
    tok = tok_mod.Tokenizer(model_dir)
    eos_ids = set(getattr(tok, "eos_ids", (151645, 151643)))

    # Warmup CUDA graphs (single + C=4)
    _ = run_single(eng, tok, eos_ids, "Say hello in C++.", max_new=32)

    res: dict = {"tag": args.tag, "lib": args.lib, "profile": args.profile, "load_s": load_s, "single": {}, "warm": {}, "batched": {}}

    # 1. Single-stream greedy workloads
    for name, prompt in [("code", CODE_PROMPT), ("prose", PROSE_PROMPT), ("edit", EDIT_PROMPT)]:
        if args.profile:
            eng.profile_dump()
        r = run_single(eng, tok, eos_ids, prompt, max_new=384)
        res["single"][name] = r
        print(f"[{args.tag}] single {name:5s}: {r['decode_tok_s']:6.1f} tok/s | {r['ms_per_step']:6.3f} ms/step | {r['tok_per_step']:4.2f} tok/step | pf={r['prefill_ms']:6.2f}ms | sha={r['sha']}", flush=True)
        if args.profile and name == "code":
            eng.profile_dump()

    # 2. 32K long-context prefill + decode
    doc32 = make_filler(32000, seed=1032)
    r32 = run_single(eng, tok, eos_ids, f"Log archive:\n{doc32}\n\n" + CODE_PROMPT, max_new=256)
    r32["prefill_tok_s"] = round(r32["prompt_toks"] / (r32["prefill_ms"] / 1000.0), 1)
    res["single"]["ctx32k_code"] = r32
    print(f"[{args.tag}] 32K code   : pf={r32['prefill_ms']:7.1f}ms ({r32['prefill_tok_s']:6.0f} tok/s) | dec={r32['decode_tok_s']:6.1f} tok/s ({r32['ms_per_step']:6.3f} ms/step) | sha={r32['sha']}", flush=True)

    # 3. Warm prefix-cache hits (20K + 14 / 18 / 35 toks)
    doc20 = make_filler(20000, seed=1020)
    ids20 = tok.prompt_ids([{"role": "user", "content": doc20}], enable_thinking=False)
    eng.prefix_begin(ids20, spec=True)
    _ = eng.spec_begin(ids20, temperature=0.0, seed=42)
    eng.checkpoint()
    warm_rows = []
    cur = list(ids20)
    for delta_label, suffix in [
        ("plus14", "\nTurn 1: check record 00001."),
        ("plus18", "\nTurn 2: inspect record 00002 and summarize its latency."),
        ("plus35", "\nTurn 3: inspect records 00003, 00004, and 00005, then report the maximum latency value concisely."),
    ]:
        cur = cur + tok.encode(suffix)
        t0 = time.perf_counter()
        reuse = eng.prefix_begin(cur, spec=True)
        t_pb = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        first = eng.spec_begin(cur[reuse:], temperature=0.0, seed=42)
        t_sb = (time.perf_counter() - t1) * 1000.0
        t2 = time.perf_counter()
        eng.checkpoint()
        t_ck = (time.perf_counter() - t2) * 1000.0
        row = {
            "case": delta_label,
            "prompt_toks": len(cur),
            "reuse": reuse,
            "new_toks": len(cur) - reuse,
            "prefix_begin_ms": round(t_pb, 3),
            "spec_begin_ms": round(t_sb, 3),
            "total_ttft_ms": round(t_pb + t_sb, 3),
            "checkpoint_ms": round(t_ck, 3),
            "first_tok": first,
        }
        warm_rows.append(row)
        print(f"[{args.tag}] warm {delta_label:6s}: reuse={reuse}/{len(cur)} (+{row['new_toks']:2d}) | pb={row['prefix_begin_ms']:5.2f}ms + sb={row['spec_begin_ms']:5.2f}ms = {row['total_ttft_ms']:5.2f}ms | ckpt={row['checkpoint_ms']:5.2f}ms | tok={first}", flush=True)
    res["warm"]["ctx20k"] = warm_rows

    # 4. Batched C=4 short (default + ADAPT_BATCH=0) & C=8 short
    for label, nl, adapt in [("c4_default", 4, None), ("c4_fixed8", 4, "0"), ("c8_fixed8", 8, "0")]:
        eng.configure(Q27_SPEC_ADAPT_BATCH=adapt)
        areas = []
        outs = []
        for l in range(nl):
            eng.reset()
            f = eng.spec_begin(
                tok.prompt_ids([{"role": "user", "content": f"[Stream {l}] " + CODE_PROMPT}], enable_thinking=False),
                temperature=0.0, seed=42,
            )
            outs.append([f])
            if l < nl - 1:
                areas.append(eng.seq_park())
        lanes = [None] + areas
        for _ in range(3):
            eng.spec_step_lanes(lanes)
        t0 = time.perf_counter()
        gen = 0
        steps = 0
        while steps < 40:
            step_toks = eng.spec_step_lanes(lanes)
            for z, st in enumerate(step_toks):
                outs[z].extend(st)
                gen += len(st)
            steps += 1
        dt = time.perf_counter() - t0
        for a in areas:
            eng.seq_drop(a)
        b = {
            "lanes": nl,
            "steps": steps,
            "gen_toks": gen,
            "ms_per_step": round(dt * 1000.0 / steps, 3),
            "agg_tok_s": round(gen / max(1e-6, dt), 2),
            "tok_per_step": round(gen / steps, 2),
            "sha": tok_hash([x for o in outs for x in o[:64]]),
        }
        res["batched"][label] = b
        print(f"[{args.tag}] batch {label:10s}: agg={b['agg_tok_s']:7.1f} tok/s | {b['ms_per_step']:6.3f} ms/step | {b['tok_per_step']:5.2f} tok/step | sha={b['sha']}", flush=True)

    # 5. Batched C=4 32K long context
    eng.configure(Q27_SPEC_ADAPT_BATCH="0")
    areas = []
    for l in range(4):
        eng.reset()
        d32 = make_filler(32000, seed=2000 + l)
        _ = eng.spec_begin(
            tok.prompt_ids([{"role": "user", "content": f"Ctx {l}:\n{d32}\n\n" + CODE_PROMPT}], enable_thinking=False),
            temperature=0.0, seed=42,
        )
        if l < 3:
            areas.append(eng.seq_park())
    lanes = [None] + areas
    for _ in range(3):
        eng.spec_step_lanes(lanes)
    t0 = time.perf_counter()
    gen = 0
    steps = 0
    while steps < 30:
        step_toks = eng.spec_step_lanes(lanes)
        gen += sum(len(x) for x in step_toks)
        steps += 1
    dt = time.perf_counter() - t0
    for a in areas:
        eng.seq_drop(a)
    b32 = {"lanes": 4, "steps": steps, "gen_toks": gen, "ms_per_step": round(dt * 1000.0 / steps, 3), "agg_tok_s": round(gen / max(1e-6, dt), 2)}
    res["batched"]["c4_ctx32k"] = b32
    print(f"[{args.tag}] batch c4_ctx32k : agg={b32['agg_tok_s']:7.1f} tok/s | {b32['ms_per_step']:6.3f} ms/step", flush=True)

    eng.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
