#!/usr/bin/env python3
"""Reject deliberate LFW reset, threshold, readout and packet defects."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    args = p.parse_args()
    variants = {
        "immediate_instead_of_delayed_reset": ("lfw_model.cpp", "(previous>threshold)", "(integrated>threshold)"),
        "inclusive_threshold": ("lfw_model.cpp", "spikes[idx]=updated>threshold", "spikes[idx]=updated>=threshold"),
        "sum_instead_of_mean": ("lfw_model.cpp", "evidence[oc]/float(steps)", "evidence[oc]"),
        "bad_tlast": ("lfw_axi.cpp", "packet.last=i==52", "packet.last=i==51"),
    }
    rows = []
    for name, (file, before, after) in variants.items():
        folder = ROOT/"stress_sim/build"/("lfw_mutation_"+name)
        folder.mkdir(parents=True, exist_ok=True)
        for source in (ROOT/"Spikeface_HLS").glob("lfw*"):
            shutil.copy2(source, folder/source.name)
        path = folder/file
        text = path.read_text()
        if before not in text:
            raise RuntimeError("Mutation not applied")
        path.write_text(text.replace(before, after))
        env = dict(os.environ, SPIKEFACE_LFW_SRC=str(folder), SPIKEFACE_LFW_HARNESS=str(folder/"harness"))
        result = folder/"results.json"
        with (folder/"run.log").open("w") as log:
            completed = subprocess.run([sys.executable, str(ROOT/"stress_sim/lfw_verify.py"), "--repo", args.repo,
                                        "--cases", "0", "--random-cases", "1", "--output", str(result)], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        data = json.loads(result.read_text()) if result.exists() else {}
        failed = [row for row in data.get("checks", []) if not row["pass"]]
        row = {"mutation": name, "detected": completed.returncode == 1 and bool(failed),
               "failed_checks": len(failed), "first": failed[0] if failed else None}
        rows.append(row)
        print(row, flush=True)
    (ROOT/"stress_sim/lfw_mutation_results.json").write_text(json.dumps(rows, indent=2)+"\n")
    return int(not all(row["detected"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
