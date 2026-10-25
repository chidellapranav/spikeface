#!/usr/bin/env python3
"""Demonstrate test sensitivity using isolated deliberate source defects."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Spikeface_HLS"


def main():
    variants = {
        "rounding_half_up": [("spikeface_types.h", "AP_RND_CONV", "AP_RND")],
        "invalid_strobes": [("fc2_layer_axi.cpp", "pkt.strb = -1;", "pkt.strb = 0;")],
        "ignored_image_reset": [(name+"_layer_axi.cpp", "clear_state ? mem_t(0) :", "") for name in ["conv1", "conv2", "fc1"]],
        "pool_scale_error": [("conv1_pool_axi.cpp", "(0.25)", "(0.5)")],
    }
    rows = []
    for name, changes in variants.items():
        folder = HERE / "build" / ("mutation_"+name)
        folder.mkdir(parents=True, exist_ok=True)
        for p in SRC.iterdir():
            if p.suffix in (".cpp", ".h"):
                shutil.copy2(p, folder / p.name)
        for file, before, after in changes:
            p = folder / file
            text = p.read_text()
            if before not in text:
                raise RuntimeError(f"Mutation not applied: {name}: {file}")
            p.write_text(text.replace(before, after))
        env = dict(os.environ, SPIKEFACE_SRC=str(folder), SPIKEFACE_HARNESS=str(folder / "harness"))
        result = folder / "results.json"
        with (folder / "run.log").open("w") as log:
            p = subprocess.run([sys.executable, str(HERE / "run.py"), "--cases", "0", "--output", str(result)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        data = json.loads(result.read_text()) if result.exists() else {}
        failed = [r for r in data.get("checks", []) if not r["pass"]]
        row = {"mutation": name, "detected": p.returncode == 1 and data.get("error") is None and bool(failed),
               "failed_checks": len(failed), "first": failed[0] if failed else None}
        rows.append(row)
        print(row, flush=True)
    (HERE / "mutation_results.json").write_text(json.dumps(rows, indent=2)+"\n")
    return int(not all(r["detected"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
