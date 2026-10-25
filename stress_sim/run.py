#!/usr/bin/env python3
"""Reproducible differential C++/Python verification; exits nonzero on mismatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import numpy as np
from reference import (Network, convolution, dense, golden_arrays, lif, pool,
                       readout, SHAPES)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = Path(os.environ.get("SPIKEFACE_SRC", ROOT / "Spikeface_HLS"))
HARNESS = Path(os.environ.get("SPIKEFACE_HARNESS", HERE / "harness"))
SOURCES = ["conv1_layer_axi", "conv1_pool", "conv1_pool_axi", "conv2_layer",
           "conv2_layer_axi", "conv2_pool", "conv2_pool_axi", "fc1_layer",
           "fc1_layer_axi", "fc2_layer_axi"]


def build():
    include = Path(os.environ.get("XILINX_HLS_INCLUDE", "/home/pranavsai/WORK/Vitis_HLS/2022.2/include"))
    if not (include / "ap_fixed.h").is_file():
        raise RuntimeError("Set XILINX_HLS_INCLUDE to the real Vitis HLS include directory")
    cmd = ["g++", "-std=c++14", "-O2", "-Wno-unknown-pragmas", f"-I{include}", f"-I{SRC}",
           str(HERE / "harness.cpp"), *[str(SRC / (n + ".cpp")) for n in SOURCES], "-o", str(HARNESS)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=240)
    return {"compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
            "headers": str(include), "command": cmd}


class Checks:
    def __init__(self):
        self.rows = []

    def equal(self, name, got, expected):
        a, b = np.asarray(got).ravel(), np.asarray(expected).ravel()
        same_shape = a.shape == b.shape
        bad = np.flatnonzero(a != b) if same_shape else np.arange(max(a.size, b.size))
        row = {"name": name, "values": int(b.size), "mismatches": int(bad.size), "pass": same_shape and bad.size == 0}
        if bad.size and same_shape:
            row["first"] = {"index": int(bad[0]), "actual": int(a[bad[0]]), "expected": int(b[bad[0]])}
        self.rows.append(row)
        if not row["pass"]:
            print("FAIL", row, flush=True)


def execute(mode, data):
    completed = subprocess.run([str(HARNESS), mode],
                               input=np.asarray(data, dtype="<i4").tobytes(),
                               capture_output=True, timeout=180)
    if completed.returncode:
        raise RuntimeError(f"{mode} C++ failed: {completed.stderr.decode(errors='replace')}")
    # Xilinx's simulation library appends a stream-depth diagnostic at exit.
    payload = re.sub(rb"INFO \[HLS SIM\]: The maximum depth reached by any hls::stream\(\) instance in the design is \d+\r?\n$", b"", completed.stdout)
    return np.frombuffer(payload, dtype="<i4")


def verify_case(checks, label, mode, frames, weights=None, beta=30, threshold=32, clears=None, resets=None):
    frames = np.asarray(frames, dtype=np.int64)
    weights = weights or {}
    n = len(frames)
    modes = list(SHAPES) if mode == "chain" else [mode]
    packet = [n, beta, threshold]
    for name in modes:
        if name in SHAPES:
            packet.extend(np.asarray(weights[name]).ravel())
    for t, frame in enumerate(frames):
        packet.append(int(resets[t]) if resets is not None else 0)
        packet.extend(frame.ravel())
        if mode in ("fc2", "chain"):
            packet.append(int(clears[t] if clears is not None else t == 0))
    actual = execute(mode, packet)
    pos = 0
    def take(name, expected, mask=None):
        nonlocal pos
        expected = np.asarray(expected).ravel()
        if mask is not None:
            expected = expected & mask
        got = actual[pos:pos + expected.size]
        pos += expected.size
        checks.equal(label + "/" + name, got, expected)
    mem = {"conv1": np.zeros((8, 32, 32), dtype=np.int64),
           "conv2": np.zeros((16, 16, 16), dtype=np.int64), "fc1": np.zeros(64, dtype=np.int64)}
    accum = np.zeros(40, dtype=np.int64)
    network = Network(weights, beta, threshold) if mode == "chain" else None
    trace = []
    for t, frame in enumerate(frames):
        if resets is not None and resets[t]:
            for state in mem.values():
                state.fill(0)
            if network:
                for state in network.mem.values():
                    state.fill(0)
        if network:
            result = network.step(frame)
            if clears is not None and clears[t]:
                # network.step already added this timestep; recompute readout reset.
                result["fc2"], network.accum = readout(result["fc1"], weights["fc2"], network.accum, True)
            elif t == 0:
                pass
            current_modes = ["conv1", "pool1", "conv2", "pool2", "fc1", "fc2"]
        else:
            result = {}
            current_modes = [mode]
            if mode in ("conv1", "conv2"):
                result[mode], mem[mode], taps = convolution(frame, weights[mode], mem[mode], beta, threshold)
            elif mode == "fc1":
                result[mode], mem[mode], taps = dense(frame, weights[mode], mem[mode], beta, threshold)
            elif mode.startswith("pool"):
                result[mode] = pool(frame)
            else:
                result[mode], accum = readout(frame, weights[mode], accum, bool(clears[t]) if clears is not None else t == 0)
        for name in current_modes:
            prefix = f"t{t}/{name}"
            if name == "fc2":
                p = np.empty((40, 4), dtype=np.int64)
                p[:, 0] = result[name]
                p[:, 1], p[:, 2], p[:, 3] = 0, 15, 15
                p[-1, 1] = 1
                got = actual[pos:pos+160].reshape(40, 4)
                pos += 160
                for k, field in enumerate(["score_bits", "last", "keep", "strb"]):
                    checks.equal(label + "/" + prefix + "/" + field, got[:, k], p[:, k])
                trace.append(result[name].copy())
            else:
                take(prefix + "/plain", result[name])
                take(prefix + "/axi", result[name])
                if name in mem:
                    take(prefix + "/membrane", network.mem[name] if network else mem[name], 255)
                    count = result["taps"][["conv1", "conv2", "fc1"].index(name)] if network else taps
                    take(prefix + "/active_taps", [count, count])
    checks.equal(label + "/protocol_length", [pos], [actual.size])
    print(f"Checked {label}: {n} timestep(s)", flush=True)
    return np.asarray(trace)


def golden(checks):
    names = ["golden_conv1_array.h", "golden_pool.h", "golden_conv2.h", "golden_pool2.h", "golden_fc1.h", "golden_fc2.h"]
    g = {name: golden_arrays(SRC / name) for name in names}
    c1, p1, c2, p2, f1, f2 = (g[name] for name in names)
    weights = {"conv1": np.rint(c1["GOLDEN_ARR_KERNELS"]*128).astype(int),
               "conv2": np.rint(c2["GOLDEN_CONV2_WEIGHTS"]*128).astype(int),
               "fc1": np.rint(f1["GOLDEN_FC1_WEIGHTS"]*128).astype(int),
               "fc2": np.rint(f2["GOLDEN_FC2_WEIGHTS"]*64).astype(int)}
    net = Network(weights)
    for t, frame in enumerate(c1["GOLDEN_ARR_INPUT"]):
        result = net.step(frame.astype(int))
        for name, expected in [("conv1", c1["GOLDEN_ARR_SPIKES"][t]), ("pool1", p1["GOLDEN_POOL_OUTPUT"][t]*4),
                               ("conv2", c2["GOLDEN_CONV2_SPIKES"][t]), ("pool2", p2["GOLDEN_POOL2_OUTPUT"][t]*4),
                               ("fc1", f1["GOLDEN_FC1_SPIKES"][t]), ("fc2", f2["GOLDEN_FC2_SCORES"][t]*(1<<24))]:
            checks.equal(f"python/golden/t{t}/{name}", result[name], expected)
        checks.equal(f"python/golden/t{t}/taps", result["taps"], [c1["GOLDEN_ARR_ACTIVE_TAPS"][t], c2["GOLDEN_CONV2_ACTIVE_TAPS"][t], f1["GOLDEN_FC1_ACTIVE_TAPS"][t]])
    for name, frames in [("conv1", c1["GOLDEN_ARR_INPUT"]), ("pool1", p1["GOLDEN_POOL_INPUT"]),
                         ("conv2", c2["GOLDEN_CONV2_INPUT"]*4), ("pool2", p2["GOLDEN_POOL2_INPUT"]),
                         ("fc1", f1["GOLDEN_FC1_INPUT"]*4), ("fc2", f2["GOLDEN_FC2_INPUT"])]:
        verify_case(checks, "golden/"+name, name, frames, weights)
    trace = verify_case(checks, "golden/chain", "chain", c1["GOLDEN_ARR_INPUT"], weights)
    checks.equal("hls/golden/final_trace", trace, f2["GOLDEN_FC2_SCORES"]*(1<<24))
    return weights


def stress(checks, rng, cases):
    # Exhaust every membrane code with each signed beta code, at positive and
    # negative thresholds; include rounding ties and saturation currents.
    m, b, t = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128), [-128, 0, 1, 32, 127], indexing="ij")
    m, b, t = m.ravel(), b.ravel(), t.ravel()
    cur = rng.choice([-524288, -524287, -192, -64, 0, 64, 192, 524286, 524287], m.size)
    packet = np.column_stack([cur, m, b, t])
    got = execute("lif", np.r_[m.size, packet.ravel()]).reshape(-1, 2)
    spikes, membrane = lif(cur, m, b, t)
    checks.equal("stress/lif/spikes", got[:, 0], spikes)
    checks.equal("stress/lif/membranes", got[:, 1], membrane & 255)
    print(f"Checked {m.size} LIF combinations", flush=True)
    cur = np.arange(-(1 << 19), 1 << 19, dtype=np.int64)
    packet = np.column_stack([cur, np.zeros_like(cur), np.zeros_like(cur), np.full_like(cur, 32)])
    got = execute("lif", np.r_[cur.size, packet.ravel()]).reshape(-1, 2)
    spikes, membrane = lif(cur, np.zeros_like(cur), 0, 32)
    checks.equal("exhaustive/lif/all_current_codes/spikes", got[:, 0], spikes)
    checks.equal("exhaustive/lif/all_current_codes/membranes", got[:, 1], membrane & 255)
    for mode, channels, side in [("pool1", 8, 32), ("pool2", 16, 16)]:
        frames = np.zeros((16, channels, side, side), dtype=np.int64)
        for pattern in range(16):
            for bit, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                frames[pattern, :, dy::2, dx::2] = (pattern >> bit) & 1
        verify_case(checks, "exhaustive/"+mode+"/patterns", mode, frames)
    for k in range(cases):
        weights = {name: rng.integers(-128, 128, shape, dtype=np.int64) for name, shape in SHAPES.items()}
        beta, threshold = [(30, 32), (0, 32), (32, 127), (127, -128), (-128, 0)][k % 5]
        for name, shape, levels in [("conv1", (32, 32), 2), ("pool1", (8, 32, 32), 2),
                                    ("conv2", (8, 16, 16), 5), ("pool2", (16, 16, 16), 2),
                                    ("fc1", (1024,), 5), ("fc2", (64,), 2)]:
            frames = rng.integers(0, levels, (4, *shape))
            frames[0] = 0
            frames[1] = levels-1
            frames[2] *= rng.random(shape) < .05
            verify_case(checks, f"stress/{k}/{name}", name, frames, weights, beta, threshold,
                        clears=[1, 0, 1, 0] if name == "fc2" else None)
        frames = rng.integers(0, 2, (5, 32, 32))
        frames[0], frames[1] = 0, 1
        verify_case(checks, f"stress/{k}/chain", "chain", frames, weights, beta, threshold, clears=[1, 0, 0, 1, 0])
    # Sign-changing adversarial prefix overflows distinguish saturating stepwise
    # accumulation from a final clipped NumPy/torch matmul.
    for sign in [-1, 1]:
        w = np.full((64, 1024), 127 if sign == 1 else -128, dtype=np.int64)
        w[:, 600:] = -128 if sign == 1 else 127
        verify_case(checks, f"stress/fc1/prefix_overflow/{sign}", "fc1", np.full((3, 1024), 4), {"fc1": w})
    for sign in [-1, 1]:
        w = np.full((40, 64), 127 if sign == 1 else -128, dtype=np.int64)
        verify_case(checks, f"stress/fc2/temporal_saturation/{sign}", "fc2", np.ones((160, 64)), {"fc2": w}, clears=[1]+[0]*158+[1])
    weights = {name: rng.integers(-32, 33, shape) for name, shape in SHAPES.items()}
    frames = rng.integers(0, 2, (12, 32, 32))
    frames[8:] = frames[:4]
    trace = verify_case(checks, "stress/reset/chain", "chain", frames, weights,
                        clears=[1]+[0]*7+[1]+[0]*3, resets=[1]+[0]*7+[1]+[0]*3)
    checks.equal("stress/reset/repeat_image_scores", trace[:4], trace[8:])
    for name, shape, levels in [("conv1", (32, 32), 2), ("conv2", (8, 16, 16), 5), ("fc1", (1024,), 5)]:
        frames = rng.integers(0, levels, (8, *shape))
        frames[6:] = frames[:2]
        verify_case(checks, "stress/reset/"+name, name, frames, weights, resets=[1, 0, 0, 0, 0, 0, 1, 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    started = time.time()
    checks = Checks()
    metadata = {"headers": os.environ.get("XILINX_HLS_INCLUDE", "/home/pranavsai/WORK/Vitis_HLS/2022.2/include"),
                "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]} if args.no_build else build()
    error = None
    try:
        golden(checks)
        stress(checks, np.random.default_rng(args.seed), args.cases)
    except Exception as exc:
        error = repr(exc)
        print("ERROR", error, flush=True)
    report = {"seed": args.seed, "cases": args.cases, "elapsed_seconds": round(time.time()-started, 3),
              "python": platform.python_version(), "numpy": np.__version__, "build": metadata,
              "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(SRC.glob("*")) if p.suffix in (".h", ".cpp")},
              "checks": checks.rows, "error": error,
              "pass": error is None and bool(checks.rows) and all(row["pass"] for row in checks.rows)}
    (HERE / args.output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"pass": report["pass"], "checks": len(checks.rows), "values": sum(x["values"] for x in checks.rows),
                      "failed_checks": sum(not x["pass"] for x in checks.rows), "seconds": report["elapsed_seconds"]}), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
