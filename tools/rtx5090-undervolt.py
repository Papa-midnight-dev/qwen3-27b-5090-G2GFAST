#!/usr/bin/env python3
"""True NVML Undervolt & Longevity Controller for NVIDIA RTX 5090 (Blackwell) on Linux.

How true undervolting works on Linux without MSI Afterburner:
  1. Shift the P0 Graphics V/F curve UPWARD (+core_offset MHz via nvmlDeviceSetClockOffsets).
     Every frequency point now requires LESS voltage (typically -70 to -120 mV).
  2. Cap the top of the boost curve (--max-clock, e.g. 2750 MHz via nvmlDeviceSetGpuLockedClocks
     with min=210 MHz so idle P8 power stays low). This flattens the top of the V/F curve so the
     GPU never climbs into the inefficient high-voltage tail (>1.0 V).
  3. Set a longevity-first power cap (--watts 450, STRICTLY capped at <= 550 W).
  4. Apply a conservative, ECC-safe GDDR7 memory clock offset (+1000 MHz default) because LLM
     decode on RTX 5090 is overwhelmingly memory-bandwidth bound.

Usage:
  rtx5090-undervolt.py --status                      # read-only (no root required)
  sudo rtx5090-undervolt.py --apply                  # 450W, +180 MHz V/F, 2750 MHz cap, +1000 MHz mem
  sudo rtx5090-undervolt.py --apply --watts 450 --core-offset 200 --max-clock 2750 --mem-offset 1000
  sudo rtx5090-undervolt.py --reset                  # restore factory defaults
"""
from __future__ import annotations

import argparse
import ctypes
import sys

HARD_MAX_WATTS = 550  # Non-negotiable card longevity guardrail.
DEFAULT_WATTS = 450
DEFAULT_CORE_OFFSET_MHZ = 180
DEFAULT_MAX_CLOCK_MHZ = 2750
DEFAULT_MIN_CLOCK_MHZ = 210
DEFAULT_MEM_OFFSET_MHZ = 1000

NVML_SUCCESS = 0
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_MEM = 2
NVML_PSTATE_0 = 0
NVML_FEATURE_ENABLED = 1
NVML_TEMPERATURE_GPU = 0


class NvmlClockOffsetV1(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("type", ctypes.c_uint),
        ("pstate", ctypes.c_uint),
        ("clockOffsetMHz", ctypes.c_int),
        ("minClockOffsetMHz", ctypes.c_int),
        ("maxClockOffsetMHz", ctypes.c_int),
    ]


NVML_CLOCK_OFFSET_V1 = ctypes.sizeof(NvmlClockOffsetV1) | (1 << 24)


def load_nvml() -> ctypes.CDLL:
    for name in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("Could not load libnvidia-ml.so.1")
    lib.nvmlErrorString.restype = ctypes.c_char_p
    lib.nvmlErrorString.argtypes = [ctypes.c_uint]
    return lib


def check(lib: ctypes.CDLL, rc: int, what: str) -> None:
    if rc != NVML_SUCCESS:
        msg = lib.nvmlErrorString(rc)
        raise RuntimeError(f"{what} failed: {msg.decode() if msg else rc} (rc={rc})")


def get_clock_offset(lib: ctypes.CDLL, dev: ctypes.c_void_p, clock_type: int, pstate: int = NVML_PSTATE_0) -> NvmlClockOffsetV1:
    info = NvmlClockOffsetV1(version=NVML_CLOCK_OFFSET_V1, type=clock_type, pstate=pstate)
    check(lib, lib.nvmlDeviceGetClockOffsets(dev, ctypes.byref(info)), f"nvmlDeviceGetClockOffsets(type={clock_type})")
    return info


def set_clock_offset(lib: ctypes.CDLL, dev: ctypes.c_void_p, clock_type: int, offset_mhz: int, pstate: int = NVML_PSTATE_0) -> int:
    cur = get_clock_offset(lib, dev, clock_type, pstate)
    clamped = max(cur.minClockOffsetMHz, min(cur.maxClockOffsetMHz, offset_mhz))
    if clamped != offset_mhz:
        print(f"Note: clamped clock_type={clock_type} offset {offset_mhz} -> {clamped} MHz "
              f"(driver range [{cur.minClockOffsetMHz}, {cur.maxClockOffsetMHz}])", file=sys.stderr)
    req = NvmlClockOffsetV1(
        version=NVML_CLOCK_OFFSET_V1,
        type=clock_type,
        pstate=pstate,
        clockOffsetMHz=clamped,
    )
    check(lib, lib.nvmlDeviceSetClockOffsets(dev, ctypes.byref(req)), f"nvmlDeviceSetClockOffsets(type={clock_type}, {clamped})")
    return clamped


def print_status(lib: ctypes.CDLL, dev: ctypes.c_void_p) -> None:
    name = ctypes.create_string_buffer(96)
    check(lib, lib.nvmlDeviceGetName(dev, name, 96), "nvmlDeviceGetName")
    plim, pdef, pmin, pmax, pdraw = (ctypes.c_uint() for _ in range(5))
    check(lib, lib.nvmlDeviceGetPowerManagementLimit(dev, ctypes.byref(plim)), "GetPowerLimit")
    check(lib, lib.nvmlDeviceGetPowerManagementDefaultLimit(dev, ctypes.byref(pdef)), "GetDefaultPowerLimit")
    check(lib, lib.nvmlDeviceGetPowerManagementLimitConstraints(dev, ctypes.byref(pmin), ctypes.byref(pmax)), "GetPowerConstraints")
    check(lib, lib.nvmlDeviceGetPowerUsage(dev, ctypes.byref(pdraw)), "GetPowerUsage")
    gr_clk, mem_clk, temp, pstate, pm = (ctypes.c_uint() for _ in range(5))
    check(lib, lib.nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, ctypes.byref(gr_clk)), "GetGrClock")
    check(lib, lib.nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, ctypes.byref(mem_clk)), "GetMemClock")
    check(lib, lib.nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, ctypes.byref(temp)), "GetTemp")
    check(lib, lib.nvmlDeviceGetPerformanceState(dev, ctypes.byref(pstate)), "GetPState")
    check(lib, lib.nvmlDeviceGetPersistenceMode(dev, ctypes.byref(pm)), "GetPersistenceMode")

    gr_off = get_clock_offset(lib, dev, NVML_CLOCK_GRAPHICS, NVML_PSTATE_0)
    mem_off = get_clock_offset(lib, dev, NVML_CLOCK_MEM, NVML_PSTATE_0)

    print(f"GPU:              {name.value.decode()}")
    print(f"Persistence Mode: {'Enabled' if pm.value else 'Disabled'}")
    print(f"Power:            {pdraw.value / 1000:.1f} W / {plim.value / 1000:.0f} W "
          f"(default {pdef.value / 1000:.0f} W, hw range {pmin.value / 1000:.0f}-{pmax.value / 1000:.0f} W, guardrail <= {HARD_MAX_WATTS} W)")
    print(f"Clocks:           Graphics {gr_clk.value} MHz | Memory {mem_clk.value} MHz | P{pstate.value} | {temp.value} C")
    print(f"P0 Core Offset:   {gr_off.clockOffsetMHz:+d} MHz (allowed [{gr_off.minClockOffsetMHz}, {gr_off.maxClockOffsetMHz}])")
    print(f"P0 Mem Offset:    {mem_off.clockOffsetMHz:+d} MHz (allowed [{mem_off.minClockOffsetMHz}, {mem_off.maxClockOffsetMHz}])")


def main() -> int:
    ap = argparse.ArgumentParser(description="RTX 5090 true NVML undervolt & longevity controller")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    ap.add_argument("--status", action="store_true", help="Show current GPU power/clock/offset status")
    ap.add_argument("--apply", action="store_true", help="Apply true undervolt + longevity power cap")
    ap.add_argument("--reset", action="store_true", help="Restore factory defaults (0 offsets, unlocked clocks, default power)")
    ap.add_argument("--watts", type=int, default=DEFAULT_WATTS, help=f"Power limit in Watts (default: {DEFAULT_WATTS}, HARD MAX: {HARD_MAX_WATTS})")
    ap.add_argument("--core-offset", type=int, default=DEFAULT_CORE_OFFSET_MHZ, help=f"P0 graphics V/F offset in MHz (default: +{DEFAULT_CORE_OFFSET_MHZ})")
    ap.add_argument("--max-clock", type=int, default=DEFAULT_MAX_CLOCK_MHZ, help=f"Boost ceiling in MHz for curve flattening (default: {DEFAULT_MAX_CLOCK_MHZ}, 0=unlocked)")
    ap.add_argument("--min-clock", type=int, default=DEFAULT_MIN_CLOCK_MHZ, help=f"Min locked clock in MHz (default: {DEFAULT_MIN_CLOCK_MHZ} for low idle power)")
    ap.add_argument("--mem-offset", type=int, default=DEFAULT_MEM_OFFSET_MHZ, help=f"P0 GDDR7 memory offset in MHz (default: +{DEFAULT_MEM_OFFSET_MHZ})")
    args = ap.parse_args()

    if not (args.status or args.apply or args.reset):
        args.status = True

    if args.apply and args.watts > HARD_MAX_WATTS:
        print(f"REFUSED: --watts {args.watts} exceeds hard longevity guardrail ({HARD_MAX_WATTS} W).", file=sys.stderr)
        return 2

    lib = load_nvml()
    check(lib, lib.nvmlInit_v2(), "nvmlInit_v2")
    try:
        dev = ctypes.c_void_p()
        check(lib, lib.nvmlDeviceGetHandleByIndex_v2(args.gpu, ctypes.byref(dev)), "nvmlDeviceGetHandleByIndex_v2")

        if args.reset:
            set_clock_offset(lib, dev, NVML_CLOCK_GRAPHICS, 0, NVML_PSTATE_0)
            set_clock_offset(lib, dev, NVML_CLOCK_MEM, 0, NVML_PSTATE_0)
            check(lib, lib.nvmlDeviceResetGpuLockedClocks(dev), "nvmlDeviceResetGpuLockedClocks")
            pdef = ctypes.c_uint()
            check(lib, lib.nvmlDeviceGetPowerManagementDefaultLimit(dev, ctypes.byref(pdef)), "GetDefaultPowerLimit")
            check(lib, lib.nvmlDeviceSetPowerManagementLimit(dev, pdef.value), "SetPowerManagementLimit(default)")
            print(f"Reset GPU {args.gpu} to factory defaults (0 MHz offsets, unlocked clocks, {pdef.value / 1000:.0f} W).")
            print_status(lib, dev)
            return 0

        if args.apply:
            pmin, pmax = ctypes.c_uint(), ctypes.c_uint()
            check(lib, lib.nvmlDeviceGetPowerManagementLimitConstraints(dev, ctypes.byref(pmin), ctypes.byref(pmax)), "GetPowerConstraints")
            target_mw = args.watts * 1000
            if target_mw < pmin.value or target_mw > min(pmax.value, HARD_MAX_WATTS * 1000):
                raise RuntimeError(
                    f"Requested {args.watts} W is outside safe bounds [{pmin.value / 1000:.0f}, {min(pmax.value / 1000, HARD_MAX_WATTS):.0f}] W"
                )
            check(lib, lib.nvmlDeviceSetPersistenceMode(dev, NVML_FEATURE_ENABLED), "nvmlDeviceSetPersistenceMode(1)")
            check(lib, lib.nvmlDeviceSetPowerManagementLimit(dev, target_mw), f"nvmlDeviceSetPowerManagementLimit({args.watts}W)")
            gr_applied = set_clock_offset(lib, dev, NVML_CLOCK_GRAPHICS, args.core_offset, NVML_PSTATE_0)
            mem_applied = set_clock_offset(lib, dev, NVML_CLOCK_MEM, args.mem_offset, NVML_PSTATE_0)
            if args.max_clock > 0:
                check(lib, lib.nvmlDeviceSetGpuLockedClocks(dev, args.min_clock, args.max_clock),
                      f"nvmlDeviceSetGpuLockedClocks({args.min_clock}, {args.max_clock})")
            else:
                check(lib, lib.nvmlDeviceResetGpuLockedClocks(dev), "nvmlDeviceResetGpuLockedClocks")
            print(f"Applied RTX 5090 True Undervolt: {args.watts} W cap | Core {gr_applied:+d} MHz (cap {args.max_clock} MHz) | Mem {mem_applied:+d} MHz")
            print_status(lib, dev)
            return 0

        print_status(lib, dev)
        return 0
    finally:
        lib.nvmlShutdown()


if __name__ == "__main__":
    raise SystemExit(main())
