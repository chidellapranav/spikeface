#!/usr/bin/env python3
"""Run native Vitis RTL campaigns in isolated build directories and save evidence.

No behavioral replacement of floating-point IP, forced RTL signals, warning
filters, or relaxed score tolerances are used.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TOPS = ["lfw_conv1_axi", "lfw_conv2_axi", "lfw_readout_axi", "lfw_chain_verify"]


def execute(top, campaign, vitis, stalls):
    work = campaign/top
    work.mkdir(parents=True, exist_ok=True)
    log_path = work/"vitis.log"
    build_root = work/"projects"
    env = dict(os.environ, SPIKEFACE_HLS_STAGE="cosim", SPIKEFACE_HLS_BLOCK=top,
               SPIKEFACE_HLS_BUILD_DIR=str(build_root), SPIKEFACE_FP_ADD_IMPL="fabric",
               SPIKEFACE_COSIM_TEST="stress", SPIKEFACE_COSIM_STALL=str(int(stalls)))
    start = time.time()
    with log_path.open("w") as log:
        process = subprocess.Popen([vitis, "-f", str(ROOT/"build_lfw_hls.tcl")], cwd=work,
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
    solution = build_root/top/"sol"
    reports = sorted((solution/"sim/report").glob("*cosim.rpt"))
    text = log_path.read_text(errors="replace")
    report_text = "\n".join(p.read_text(errors="replace") for p in reports)
    result = {"top": top, "returncode": status, "seconds": round(time.time()-start, 3),
              "random_stalls": stalls, "transactions": 4, "images": 2, "timesteps_per_image": 2,
              "fp_add_impl": "fabric", "clock_period_ns": 20,
              "opmode_warning_lines": text.count("OPMODE Input Warning"),
              "errors": [line for line in text.splitlines() if "ERROR:" in line],
              "log": str(log_path.relative_to(ROOT)), "cosim_report": report_text,
              "pass": status == 0 and bool(re.search(r"Verilog\s*\|\s*Pass", report_text, re.I))
                      and "C/RTL co-simulation finished: PASS" in text}
    p = solution/"syn/report"/(top+"_csynth.xml")
    if p.exists():
        xml = ET.parse(p).getroot()
        result["synthesis"] = {"estimated_clock_ns": xml.findtext(".//EstimatedClockPeriod"),
                               "resources": {n.tag: n.text for n in xml.findall(".//AreaEstimates/Resources/*")}}
    (work/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ["top", "pass", "seconds", "opmode_warning_lines", "returncode"]}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--top", choices=TOPS, action="append")
    p.add_argument("--no-stalls", action="store_true")
    p.add_argument("--name", default="rtl_stress")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name) or args.jobs < 1:
        p.error("Use a plain campaign name and jobs >= 1")
    fixture = ROOT/"stress_sim/build/lfw_fixture.bin"
    if not fixture.is_file() or int.from_bytes(fixture.read_bytes()[:4], "little") != 2:
        p.error("Export a two-timestep fixture first using export_lfw_fixture.py --timesteps 2")
    vitis = os.environ.get("VITIS_HLS") or shutil.which("vitis_hls") or "/home/pranavsai/WORK/Vitis_HLS/2022.2/bin/vitis_hls"
    campaign = ROOT/"stress_sim/build"/args.name
    if campaign.exists():
        p.error("Campaign directory exists; select a new --name to preserve prior evidence")
    campaign.mkdir(parents=True)
    selected = args.top or TOPS[:3]
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs, len(selected))) as pool:
        futures = [pool.submit(execute, top, campaign, vitis, not args.no_stalls) for top in selected]
        results = [f.result() for f in futures]
    sources = [ROOT/"build_lfw_hls.tcl", ROOT/"stress_sim/tb_lfw_rtl_stress.cpp",
               ROOT/"stress_sim/rtl_chain.cpp", ROOT/"stress_sim/tb_lfw_rtl_chain.cpp", fixture,
               *sorted((ROOT/"Spikeface_HLS").glob("lfw*"))]
    summary = {"campaign": args.name, "seconds": round(time.time()-start, 3), "results": results,
               "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               "pass": all(r["pass"] for r in results)}
    (ROOT/"stress_sim"/(args.name+"_results.json")).write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"pass": summary["pass"], "seconds": summary["seconds"]}), flush=True)
    return int(not summary["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
