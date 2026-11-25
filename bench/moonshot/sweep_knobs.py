#!/usr/bin/env python3
"""Sweep runtime SM120 tuning knobs on Huihui all-NVFP4 + FP8 KV + DSpine in one GPU session."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("Q27_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "libq27.so"))
os.environ.setdefault("MAX_SEQS", "8")
os.environ.setdefault("VRAM_BUDGET_GIB", "25.2")
os.environ.setdefault("STATE_HOST_CACHE", "8")
os.environ.setdefault("MIN_KV_TOKENS", "131072")
os.environ.setdefault("VISION", "fp8")

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "engine", "python"))
sys.path.insert(0, os.path.join(_REPO, "frontend"))

import q27lib
import tokenizer as tok_mod
from bench_suite import CODE_PROMPT, make_filler


def measure_single(eng, ids, steps=35):
    eng.reset()
    _ = eng.spec_begin(ids, temperature=0.0, seed=42)
    for _ in range(3):
        eng.spec_step()
    t0 = time.perf_counter()
    gen = 0
    for _ in range(steps):
        gen += len(eng.spec_step())
    dt = time.perf_counter() - t0
    return dt * 1000.0 / steps, gen / dt


def measure_c4(eng, ids_list, steps=25):
    areas = []
    for i, ids in enumerate(ids_list):
        eng.reset()
        _ = eng.spec_begin(ids, temperature=0.0, seed=42)
        if i < len(ids_list) - 1:
            areas.append(eng.seq_park())
    lanes = [None] + areas
    for _ in range(3):
        eng.spec_step_lanes(lanes)
    t0 = time.perf_counter()
    gen = 0
    for _ in range(steps):
        gen += sum(len(x) for x in eng.spec_step_lanes(lanes))
    dt = time.perf_counter() - t0
    for a in areas:
        eng.seq_drop(a)
    return dt * 1000.0 / steps, gen / dt


def main():
    model_dir = os.environ["WEIGHTS"]  # the target checkpoint folder (NVFP4)
    draft_dir = os.environ["DRAFT_DIR"]  # the DSpine draft snapshot folder
    eng = q27lib.Engine(model_dir, max_tokens=131072, kv_dtype="fp8", draft=draft_dir)
    tok = tok_mod.Tokenizer(model_dir)
    ids_code = tok.prompt_ids([{"role": "user", "content": CODE_PROMPT}], enable_thinking=False)
    ids_c4_32k = [
        tok.prompt_ids([{"role": "user", "content": f"Ctx {l}:\n{make_filler(32000, seed=3000+l)}\n\n" + CODE_PROMPT}], enable_thinking=False)
        for l in range(4)
    ]

    print("=== SWEEP 1: Single-stream short (skinny knobs) ===", flush=True)
    for label, cfg in [
        ("default (w=2,fit=1,l2=8,ext=1)", dict(Q27_SKINNY_WAVES=None, Q27_SKINNY_FIT=None, Q27_SKINNY_L2_CHUNKS=None, Q27_SKINNY_EXT_REDUCE_M=None)),
        ("waves=1", dict(Q27_SKINNY_WAVES="1", Q27_SKINNY_FIT=None, Q27_SKINNY_L2_CHUNKS=None, Q27_SKINNY_EXT_REDUCE_M=None)),
        ("waves=3", dict(Q27_SKINNY_WAVES="3", Q27_SKINNY_FIT=None, Q27_SKINNY_L2_CHUNKS=None, Q27_SKINNY_EXT_REDUCE_M=None)),
        ("waves=4", dict(Q27_SKINNY_WAVES="4", Q27_SKINNY_FIT=None, Q27_SKINNY_L2_CHUNKS=None, Q27_SKINNY_EXT_REDUCE_M=None)),
        ("fit=0 (ceil)", dict(Q27_SKINNY_WAVES="2", Q27_SKINNY_FIT="0", Q27_SKINNY_L2_CHUNKS=None, Q27_SKINNY_EXT_REDUCE_M=None)),
        ("ext_reduce_m=16 (in-kernel)", dict(Q27_SKINNY_WAVES=None, Q27_SKINNY_FIT=None, Q27_SKINNY_L2_CHUNKS=None, Q27_SKINNY_EXT_REDUCE_M="16")),
        ("l2=16", dict(Q27_SKINNY_WAVES=None, Q27_SKINNY_FIT=None, Q27_SKINNY_L2_CHUNKS="16", Q27_SKINNY_EXT_REDUCE_M=None)),
        ("l2=4", dict(Q27_SKINNY_WAVES=None, Q27_SKINNY_FIT=None, Q27_SKINNY_L2_CHUNKS="4", Q27_SKINNY_EXT_REDUCE_M=None)),
    ]:
        eng.configure(**cfg)
        ms, tps = measure_single(eng, ids_code)
        print(f"  {label:32s}: {ms:6.3f} ms/step | {tps:6.1f} tok/s", flush=True)

    print("=== SWEEP 2: C=4 32K (attention tc_splits + skinny waves) ===", flush=True)
    eng.configure(Q27_SKINNY_WAVES=None, Q27_SKINNY_FIT=None, Q27_SKINNY_L2_CHUNKS=None, Q27_SKINNY_EXT_REDUCE_M=None, Q27_SPEC_ADAPT_BATCH="0")
    for sp in (None, "24", "16", "12", "8"):
        eng.configure(Q27_ATTN_TC_SPLITS=sp)
        ms, tps = measure_c4(eng, ids_c4_32k, steps=20)
        print(f"  tc_splits={str(sp):6s}: {ms:6.3f} ms/step | agg={tps:7.1f} tok/s", flush=True)

    eng.close()


if __name__ == "__main__":
    main()
