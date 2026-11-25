#!/usr/bin/env python3
"""Direct in-process Q27_PROFILE=1 engine micro-profiler on Huihui NVFP4 + FP8 KV + DSpine."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("Q27_PROFILE", "1")
os.environ.setdefault("Q27_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "libq27.so"))
os.environ.setdefault("MAX_SEQS", "8")
os.environ.setdefault("VRAM_BUDGET_GIB", "25.2")
os.environ.setdefault("STATE_HOST_CACHE", "8")
os.environ.setdefault("MIN_KV_TOKENS", "131072")
os.environ.setdefault("VISION", "fp8")

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "engine", "python"))
sys.path.insert(0, os.path.join(_REPO, "frontend"))


def main() -> None:
    import q27lib
    import tokenizer as tok_mod

    model_dir = os.environ["WEIGHTS"]  # the target checkpoint folder (NVFP4)
    draft_dir = os.environ["DRAFT_DIR"]  # the DSpine draft snapshot folder

    t0 = time.perf_counter()
    eng = q27lib.Engine(model_dir, max_tokens=131072, kv_dtype="fp8", draft=draft_dir)
    print(f"[ENG] Load time: {time.perf_counter()-t0:.2f} s | max_tokens={eng.max_tokens} max_lanes={eng.max_lanes} spec_block={eng.spec_block}", flush=True)
    # Pre-allocate a few host cache slots so checkpoint timing isn't first-alloc
    eng.cache_grow(4)

    tok = tok_mod.Tokenizer(model_dir)
    ids_short = tok.prompt_ids(
        [{"role": "user", "content": "Write a complete C++20 lock-free SPSC ring buffer with comments and tests."}],
        enable_thinking=False,
    )

    # 1. Single-stream DSpine verify (short context)
    print("\n=== PROFILE 1: Single-stream short prompt + DSpine verify ===", flush=True)
    eng.reset()
    t0 = time.perf_counter()
    _ = eng.spec_begin(ids_short, temperature=0.0, seed=42)
    t_pf = (time.perf_counter() - t0) * 1000.0
    for _ in range(3):
        eng.spec_step()
    eng.profile_dump()  # reset counters before timed run
    t0 = time.perf_counter()
    gen = 0
    steps = 0
    eos_ids = getattr(tok, "eos_ids", {151645, 151643})
    while steps < 40:
        toks = eng.spec_step()
        gen += len(toks)
        steps += 1
        if any(t in eos_ids for t in toks):
            break
    dt = time.perf_counter() - t0
    print(f"[P1] short DSpine: prefill={t_pf:.2f}ms | {gen} toks / {steps} steps ({gen/max(1,steps):.2f} tok/step) | {gen/dt:.1f} tok/s | {dt*1000/max(1,steps):.2f} ms/step", flush=True)
    print("--- P1 Q27_PROFILE ---", flush=True)
    eng.profile_dump()

    # 2. 20K prefill + prefix checkpoint + immediate hit restore
    print("\n=== PROFILE 2: 20K prefill + prefix checkpoint + immediate hit ===", flush=True)
    lines20 = [f"[r:{i:05d}] Record 0x{(i*2654435761)&0xFFFF:04x}: latency={(i%97)+1}.3ms lane={i%8}." for i in range(520)]
    ids_20k = tok.prompt_ids([{"role": "user", "content": "\n".join(lines20)}], enable_thinking=False)
    eng.prefix_begin(ids_20k, spec=True)
    eng.profile_dump()
    t0 = time.perf_counter()
    _ = eng.spec_begin(ids_20k, temperature=0.0, seed=42)
    t_pf20 = (time.perf_counter() - t0) * 1000.0
    print(f"[P2] cold 20K ({len(ids_20k)} toks): {t_pf20:.2f} ms ({len(ids_20k)/(t_pf20/1000):.0f} tok/s)", flush=True)
    print("--- P2 Cold 20K Q27_PROFILE ---", flush=True)
    eng.profile_dump()

    t0 = time.perf_counter()
    eng.checkpoint()
    t_ckpt = (time.perf_counter() - t0) * 1000.0
    print(f"[P2] checkpoint(): {t_ckpt:.2f} ms", flush=True)

    # Simulate warm turn hitting checkpoint immediately
    ids_20k_plus = ids_20k + tok.encode("\nTurn 2: what is record 00005? Reply briefly.")
    for k in range(3):
        t0 = time.perf_counter()
        reuse = eng.prefix_begin(ids_20k_plus, spec=True)
        t_pb = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        _ = eng.spec_begin(ids_20k_plus[reuse:], temperature=0.0, seed=42)
        t_sb = (time.perf_counter() - t1) * 1000.0
        t2 = time.perf_counter()
        eng.checkpoint()
        t_ck2 = (time.perf_counter() - t2) * 1000.0
        print(f"[P2] warm hit #{k+1}: reuse={reuse}/{len(ids_20k_plus)} (+{len(ids_20k_plus)-reuse}) | prefix_begin={t_pb:.2f}ms | spec_begin={t_sb:.2f}ms | checkpoint={t_ck2:.2f}ms", flush=True)
        ids_20k_plus = ids_20k_plus + tok.encode(f"\nTurn {k+3}: check record 0000{k+1}.")

    # 3. 32K prefill + DSpine decode
    print("\n=== PROFILE 3: 32K prompt prefill + DSpine decode ===", flush=True)
    lines32 = [f"[r:{i:05d}] Record 0x{(i*2654435761)&0xFFFF:04x}: latency={(i%97)+1}.3ms lane={i%8}." for i in range(850)]
    ids_32k = tok.prompt_ids([{"role": "user", "content": "\n".join(lines32) + "\nWrite a C++20 SPSC ring buffer."}], enable_thinking=False)
    eng.reset()
    t0 = time.perf_counter()
    _ = eng.spec_begin(ids_32k, temperature=0.0, seed=42)
    t_pf32 = (time.perf_counter() - t0) * 1000.0
    for _ in range(3):
        eng.spec_step()
    eng.profile_dump()
    t0 = time.perf_counter()
    gen = 0
    steps = 0
    while steps < 30:
        toks = eng.spec_step()
        gen += len(toks)
        steps += 1
        if any(t in eos_ids for t in toks):
            break
    dt = time.perf_counter() - t0
    print(f"[P3] 32K ({len(ids_32k)} toks): prefill={t_pf32:.2f}ms ({len(ids_32k)/(t_pf32/1000):.0f} tok/s) | decode={gen/dt:.1f} tok/s ({dt*1000/max(1,steps):.2f} ms/step)", flush=True)
    print("--- P3 32K Decode Q27_PROFILE ---", flush=True)
    eng.profile_dump()

    # 4. Batched C=4 short context
    print("\n=== PROFILE 4: Batched C=4 short context ===", flush=True)
    areas = []
    for l in range(4):
        eng.reset()
        _ = eng.spec_begin(
            tok.prompt_ids([{"role": "user", "content": f"[Lane {l}] Write a complete C++20 lock-free SPSC ring buffer."}], enable_thinking=False),
            temperature=0.0, seed=42,
        )
        if l < 3:
            areas.append(eng.seq_park())
    lanes = [None] + areas
    for _ in range(3):
        eng.spec_step_lanes(lanes)
    eng.profile_dump()
    t0 = time.perf_counter()
    gen = 0
    steps = 0
    while steps < 30:
        res = eng.spec_step_lanes(lanes)
        gen += sum(len(x) for x in res)
        steps += 1
    dt = time.perf_counter() - t0
    print(f"[P4] C=4 short: {gen} toks / {steps} steps | agg={gen/dt:.1f} tok/s | {dt*1000/steps:.2f} ms/step | lane_blocks={[eng.last_lane_block(z) for z in range(4)]}", flush=True)
    print("--- P4 C=4 Short Q27_PROFILE ---", flush=True)
    eng.profile_dump()

    # Test Q27_SPEC_ADAPT_BATCH=0 vs default on C=4
    eng.configure(Q27_SPEC_ADAPT_BATCH="0")
    for _ in range(3):
        eng.spec_step_lanes(lanes)
    t0 = time.perf_counter()
    gen = 0
    steps = 0
    while steps < 30:
        res = eng.spec_step_lanes(lanes)
        gen += sum(len(x) for x in res)
        steps += 1
    dt = time.perf_counter() - t0
    print(f"[P4b] C=4 short (ADAPT_BATCH=0): {gen} toks / {steps} steps | agg={gen/dt:.1f} tok/s | {dt*1000/steps:.2f} ms/step | lane_blocks={[eng.last_lane_block(z) for z in range(4)]}", flush=True)
    for a in areas:
        eng.seq_drop(a)

    # 5. Batched C=4 32K long-context decode
    print("\n=== PROFILE 5: Batched C=4 32K long-context decode ===", flush=True)
    eng.configure(Q27_SPEC_ADAPT_BATCH=None)
    areas = []
    for l in range(4):
        eng.reset()
        lines_l = [f"[l{l}:{i:05d}] Record 0x{(i*2654435761 + l*7919)&0xFFFF:04x}: latency={(i%97)+1}.3ms lane={i%8}." for i in range(850)]
        ids_l = tok.prompt_ids([{"role": "user", "content": "\n".join(lines_l) + "\nWrite a C++20 SPSC ring buffer."}], enable_thinking=False)
        _ = eng.spec_begin(ids_l, temperature=0.0, seed=42)
        if l < 3:
            areas.append(eng.seq_park())
    lanes = [None] + areas
    for _ in range(3):
        eng.spec_step_lanes(lanes)
    eng.profile_dump()
    t0 = time.perf_counter()
    gen = 0
    steps = 0
    while steps < 25:
        res = eng.spec_step_lanes(lanes)
        gen += sum(len(x) for x in res)
        steps += 1
    dt = time.perf_counter() - t0
    print(f"[P5] C=4 32K: {gen} toks / {steps} steps | agg={gen/dt:.1f} tok/s | {dt*1000/steps:.2f} ms/step | lane_blocks={[eng.last_lane_block(z) for z in range(4)]}", flush=True)
    print("--- P5 C=4 32K Q27_PROFILE ---", flush=True)
    eng.profile_dump()
    for a in areas:
        eng.seq_drop(a)

    eng.close()


if __name__ == "__main__":
    main()
