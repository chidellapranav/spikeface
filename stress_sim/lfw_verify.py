#!/usr/bin/env python3
"""Verify new LFW HLS against scalar float32 and the supplied PyTorch model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.lfw import load_reference
from stress_sim.lfw_reference import Network, conv, dense, export_params, lif

HERE = ROOT / "stress_sim"
EXE = HERE / "build/lfw_harness"
SOURCE = Path(os.environ.get("SPIKEFACE_LFW_SRC", ROOT / "Spikeface_HLS"))
EXE = Path(os.environ.get("SPIKEFACE_LFW_HARNESS", EXE))


def build():
    include = os.environ.get("XILINX_HLS_INCLUDE", "/home/pranavsai/WORK/Vitis_HLS/2022.2/include")
    EXE.parent.mkdir(exist_ok=True)
    cmd = ["g++", "-std=c++14", "-O2", "-ffp-contract=off", "-Wno-unknown-pragmas", f"-I{include}",
           f"-I{SOURCE}", str(HERE/"lfw_harness.cpp"), str(SOURCE/"lfw_model.cpp"),
           str(SOURCE/"lfw_axi.cpp"), "-o", str(EXE)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)


def run_cpp(mode, frames, params, resets, config=(.9375, .5, .9375, .5, .9375)):
    packet = [struct.pack("<I", len(frames)), np.asarray(config, dtype="<f4").tobytes()]
    for name in (["conv1", "conv2", "fc1"] if mode == "chain" else [{"readout": "fc1"}.get(mode, mode)]):
        packet.append(np.asarray(params[name], dtype="<f4").tobytes())
    for frame, reset in zip(frames, resets):
        packet += [struct.pack("<I", reset), np.asarray(frame, dtype="<f4" if mode in ("chain", "conv1") else "<u4").tobytes()]
    p = subprocess.run([str(EXE), mode], input=b"".join(packet), capture_output=True, timeout=240)
    if p.returncode:
        raise RuntimeError(p.stderr.decode())
    payload = re.sub(rb"INFO \[HLS SIM\]: The maximum depth reached by any hls::stream\(\) instance in the design is \d+\r?\n$", b"", p.stdout)
    raw = np.frombuffer(payload, dtype="<u4")
    pos, trace = 0, []
    def take(n, floating=False):
        nonlocal pos
        result = raw[pos:pos+n]
        pos += n
        if len(result) != n:
            raise RuntimeError("Short harness output")
        return result.view("<f4") if floating else result
    for _ in frames:
        row = {}
        if mode in ("conv1", "chain"):
            row.update(spikes1=take(8192), axi_spikes1=take(8192), mem1=take(8192, True))
        if mode in ("conv2", "chain"):
            row.update(spikes2=take(4096), axi_spikes2=take(4096), mem2=take(4096, True))
        if mode in ("readout", "chain"):
            row.update(scores=take(53, True), mem3=take(53, True), evidence=take(53, True))
            packets = take(53*4).reshape(53, 4)
            row.update(axi_scores=packets[:, 0].copy().view("<f4"), last=packets[:, 1], keep=packets[:, 2], strb=packets[:, 3])
        trace.append(row)
    if pos != raw.size:
        raise RuntimeError("Excess harness output")
    return trace


def torch_trace(model, image, timesteps):
    import torch
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).reshape(1, 1, 64, 64))
    result, evidence = [], torch.zeros((1, 53))
    with torch.inference_mode():
        m1, m2, m3 = model.init_leaky()
        for t in range(timesteps):
            s1, m1 = model.lif1(model.pool1(model.bn1(model.conv1(x))), m1)
            s2, m2 = model.lif2(model.pool2(model.bn2(model.conv2(s1))), m2)
            _, m3 = model.lif3(model.bn3(model.fc1(s2.flatten(1))), m3)
            evidence = evidence + m3
            result.append({"spikes1": s1.numpy().ravel().copy(), "spikes2": s2.numpy().ravel().copy(),
                           "mem1": m1.numpy().ravel().copy(), "mem2": m2.numpy().ravel().copy(),
                           "mem3": m3.numpy().ravel().copy(), "scores": (evidence/(t+1)).numpy().ravel().copy()})
    return result


class Checks:
    def __init__(self):
        self.rows = []

    def compare(self, name, a, b, exact=True):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        if a.shape != b.shape:
            raise RuntimeError(f"shape mismatch: {name}")
        good = a == b if exact else np.isclose(a, b, atol=1e-4, rtol=1e-4)
        row = {"name": name, "values": int(a.size), "mismatches": int(np.count_nonzero(~good)),
               "max_absolute_error": float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0,
               "comparison": "exact" if exact else "atol=rtol=1e-4", "pass": bool(np.all(good))}
        self.rows.append(row)
        if not row["pass"]:
            print("FAIL", row, flush=True)


def verify(checks, name, mode, frames, params, resets, config=(.9375, .5, .9375, .5, .9375)):
    actual = run_cpp(mode, frames, params, resets, config)
    reference = Network(params, config)
    for t, (frame, reset, row) in enumerate(zip(frames, resets, actual)):
        if reset:
            reference.reset()
        if mode == "chain":
            expected = reference.step(frame)
        elif mode == "conv1":
            s, reference.m1 = lif(conv(frame, params[mode], 8, 64), reference.m1, config[0], config[1])
            expected = {"spikes1": s, "mem1": reference.m1}
        elif mode == "conv2":
            s, reference.m2 = lif(conv(frame, params[mode], 16, 32), reference.m2, config[2], config[3])
            expected = {"spikes2": s, "mem2": reference.m2}
        else:
            current = dense(frame, params["fc1"])
            reference.m3 = np.float32(config[4])*reference.m3+current
            reference.evidence = reference.evidence+reference.m3
            reference.steps += 1
            expected = {"mem3": reference.m3, "evidence": reference.evidence,
                        "scores": reference.evidence/np.float32(reference.steps)}
        for key, value in expected.items():
            if key.startswith("spikes"):
                checks.compare(f"{name}/t{t}/{key}", row[key], value)
                checks.compare(f"{name}/t{t}/axi_{key}", row["axi_"+key], value)
            else:
                # Bit patterns include signed zeros and all mantissa bits.
                checks.compare(f"{name}/t{t}/{key}/bits", row[key].view("<u4"), np.asarray(value, dtype="<f4").ravel().view("<u4"))
        if "scores" in expected:
            checks.compare(f"{name}/t{t}/axi_scores/bits", row["axi_scores"].view("<u4"), np.asarray(expected["scores"], dtype="<f4").view("<u4"))
            checks.compare(f"{name}/t{t}/last", row["last"], [0]*52+[1])
            for field in ("keep", "strb"):
                checks.compare(f"{name}/t{t}/{field}", row[field], [15]*53)
    print(f"Checked LFW {name}: {len(frames)} timesteps", flush=True)
    return actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--cases", type=int, default=4)
    p.add_argument("--random-cases", type=int, default=8)
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--output", default=str(HERE/"lfw_results.json"))
    args = p.parse_args()
    if not args.no_build:
        build()
    model, state, labels, metadata = load_reference(args.repo)
    params = export_params(state)
    checks, rng, start = Checks(), np.random.default_rng(20261024), time.perf_counter()
    images = np.load(Path(args.repo)/"data/spikeface_lfw/X_test.npy", allow_pickle=False)
    indices = list(range(min(args.cases, len(images))))
    if args.cases > 0:
        # Keep full-dataset counterexamples in the regression suite. Do not
        # hide their PyTorch trace failures behind easy first-image fixtures.
        indices = sorted(set(indices + [4, 267]))
    for i in indices:
        frames = np.repeat(images[i:i+1], 25, axis=0)
        hls = verify(checks, f"checkpoint/image{i}/chain", "chain", frames, params, [1]+[0]*24)
        pytorch = torch_trace(model, images[i], 25)
        for t, (got, expected) in enumerate(zip(hls, pytorch)):
            for key, value in expected.items():
                checks.compare(f"pytorch/image{i}/t{t}/{key}", got[key], value, key.startswith("spikes"))
        checks.compare(f"pytorch/image{i}/prediction", [np.argmax(hls[-1]["scores"])], [np.argmax(pytorch[-1]["scores"])])
    # Independent stages with real checkpoint params and pathological activation.
    for mode, shape in [("conv1", (64, 64)), ("conv2", (8, 32, 32)), ("readout", (16, 16, 16))]:
        frames = rng.integers(0, 2, (8, *shape)).astype(np.float32)
        frames[0], frames[1], frames[6:] = 0, 1, frames[:2].copy()
        verify(checks, "stress/"+mode, mode, frames, params, [1, 0, 0, 0, 0, 0, 1, 0])
    # Exactly representable parameters isolate logic from backend reduction noise.
    simple = {"conv1": np.zeros(224, dtype=np.float32), "conv2": np.zeros(3248, dtype=np.float32),
              "fc1": np.zeros(217247, dtype=np.float32)}
    for name, offset, channels in [("conv1", 200, 8), ("conv2", 3200, 16), ("fc1", 217088, 53)]:
        simple[name][:offset] = rng.integers(-4, 5, offset).astype(np.float32)/32
        simple[name][offset:offset+channels] = rng.integers(-4, 5, channels).astype(np.float32)/32
        simple[name][offset+channels:offset+2*channels] = rng.choice([-1, .5, 1], channels)
        simple[name][offset+2*channels:] = rng.integers(-4, 5, channels).astype(np.float32)/32
    frames = rng.integers(0, 2, (8, 64, 64)).astype(np.float32)
    frames[6:] = frames[:2]
    trace = verify(checks, "stress/exact_chain", "chain", frames, simple, [1, 0, 0, 0, 0, 0, 1, 0])
    for t in range(2):
        checks.compare("stress/repeat_image/"+str(t), trace[t]["scores"], trace[t+6]["scores"])
    for mode, count, offset, channels, shape in [("conv1", 224, 200, 8, (64, 64)), ("conv2", 3248, 3200, 16, (8, 32, 32))]:
        special = dict(simple)
        special[mode] = np.zeros(count, dtype=np.float32)
        special[mode][offset+channels:offset+channels*2] = 1
        special[mode][offset+channels*2:] = .5  # exact threshold; firing must be strict >
        verify(checks, "boundary/threshold/"+mode, mode, np.zeros((8, *shape), dtype=np.float32), special, [1, 0, 0, 0, 1, 0, 0, 0])
    for case in range(args.random_cases):
        random_params = {}
        for name, count, offset, channels in [("conv1", 224, 200, 8), ("conv2", 3248, 3200, 16), ("fc1", 217247, 217088, 53)]:
            a = rng.integers(-16, 17, count).astype(np.float32)/32
            a[offset+channels:offset+2*channels] = rng.choice([-2, -1, 0, .5, 1, 2], channels)
            random_params[name] = a
        config = [(0., .5, 0., .5, 0.), (.9375, .5, .9375, .5, .9375), (1., .5, 1., .5, 1.)][case % 3]
        for mode, shape in [("conv1", (64, 64)), ("conv2", (8, 32, 32)), ("readout", (16, 16, 16))]:
            frames = rng.integers(0, 2, (4, *shape)).astype(np.float32)
            frames[0], frames[1] = 0, 1
            verify(checks, f"random/{case}/{mode}", mode, frames, random_params, [1, 0, 0, 1], config)
        verify(checks, f"random/{case}/chain", "chain", rng.integers(0, 2, (4, 64, 64)).astype(np.float32), random_params, [1, 0, 0, 1], config)
    scalar = [r for r in checks.rows if not r["name"].startswith("pytorch/")]
    pytorch = [r for r in checks.rows if r["name"].startswith("pytorch/")]
    result = {"reference": metadata, "seed": 20261024, "random_cases": args.random_cases, "seconds": time.perf_counter()-start,
              "checks": checks.rows, "pass": all(r["pass"] for r in checks.rows),
              "scalar_reference_bit_exact": all(r["pass"] for r in scalar),
              "pytorch_trace_equivalent": all(r["pass"] for r in pytorch),
              "checkpoint_image_indices": indices,
              "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"Spikeface_HLS").glob("lfw*")}}
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"pass": result["pass"], "checks": len(checks.rows), "values": sum(r["values"] for r in checks.rows),
                      "failed": sum(not r["pass"] for r in checks.rows), "seconds": result["seconds"]}), flush=True)
    return int(not result["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
