#!/usr/bin/env python3
"""Compile and run every original repository testbench with real Xilinx headers."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Spikeface_HLS"
MAPPING = {
    "tb_spikeface_neuron": [], "tb_conv1_array": [],
    "tb_conv1_array_axi": ["conv1_layer_axi"],
    "tb_conv1_pool": ["conv1_pool_axi"], "tb_conv1_pool_axi": ["conv1_pool_axi"],
    "tb_conv2_array": ["conv2_layer_axi"], "tb_conv2_array_axi": ["conv2_layer_axi"],
    "tb_conv2_pool": ["conv2_pool"], "tb_conv2_pool_axi": ["conv2_pool_axi"],
    "tb_fc1_layer": ["fc1_layer"], "tb_fc1_layer_axi": ["fc1_layer_axi"],
    "tb_fc2_layer_axi": ["fc2_layer_axi"],
}


def main():
    include = os.environ.get("XILINX_HLS_INCLUDE", "/home/pranavsai/WORK/Vitis_HLS/2022.2/include")
    build = HERE / "build"
    build.mkdir(exist_ok=True)
    rows = []
    for test, sources in MAPPING.items():
        start = time.monotonic()
        cmd = ["g++", "-std=c++14", "-O2", "-Wno-unknown-pragmas", f"-I{include}", f"-I{SRC}",
               str(SRC / (test + ".cpp")), *[str(SRC / (s + ".cpp")) for s in sources], "-o", str(build / test)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        log = p.stdout + p.stderr
        if not p.returncode:
            p = subprocess.run([str(build / test)], capture_output=True, text=True, timeout=180)
            log += p.stdout + p.stderr
        (build / (test + ".log")).write_text(log)
        row = {"test": test, "pass": p.returncode == 0 and "PASS:" in log,
               "returncode": p.returncode, "seconds": round(time.monotonic()-start, 3)}
        rows.append(row)
        print(row, flush=True)
    (HERE / "original_results.json").write_text(json.dumps(rows, indent=2)+"\n")
    return int(not all(row["pass"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
