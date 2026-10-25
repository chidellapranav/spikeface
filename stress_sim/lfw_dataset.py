#!/usr/bin/env python3
"""Evaluate actual LFW HLS C++/AXI inference on every supplied test image."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.lfw import load_reference
from stress_sim.lfw_reference import export_params
from stress_sim.lfw_verify import run_cpp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    args = p.parse_args()
    _, state, classes, metadata = load_reference(args.repo)
    params = export_params(state)
    x = np.load(Path(args.repo)/"data/spikeface_lfw/X_test.npy", allow_pickle=False)
    labels = np.load(Path(args.repo)/"data/spikeface_lfw/y_test.npy", allow_pickle=False)
    baseline = json.loads((ROOT/"stress_sim/lfw_evaluation.json").read_text())
    if metadata["checkpoint_sha256"] != baseline["metadata"]["checkpoint_sha256"] or len(x) != baseline["images"]:
        raise ValueError("First run app.lfw for this exact checkpoint and complete dataset")
    for name in ("X_test.npy", "y_test.npy"):
        digest = hashlib.sha256((Path(args.repo)/"data/spikeface_lfw"/name).read_bytes()).hexdigest()
        if baseline["dataset_sha256"].get(name) != digest:
            raise ValueError("Dataset bytes differ from the paired PyTorch baseline")
    if labels.tolist() != baseline["labels"] or metadata["classes"] != baseline["metadata"]["classes"]:
        raise ValueError("Label order differs from the paired PyTorch baseline")
    predictions, logits = [], []
    max_error, close, axi_exact, packets_ok = 0., True, True, True
    start = time.perf_counter()
    for i, image in enumerate(x):
        trace = run_cpp("chain", np.repeat(image[None], 25, axis=0), params, [1]+[0]*24)
        last = trace[-1]
        predictions.append(int(np.argmax(last["axi_scores"])))
        logits.append(last["axi_scores"].astype(float).tolist())
        expected = np.asarray(baseline["logits"][i], dtype=np.float32)
        max_error = max(max_error, float(np.max(np.abs(last["axi_scores"]-expected))))
        close &= bool(np.allclose(last["axi_scores"], expected, atol=1e-4, rtol=1e-4))
        for row in trace:
            axi_exact &= bool(np.array_equal(row["axi_scores"].view("<u4"), row["scores"].view("<u4")))
            axi_exact &= bool(np.array_equal(row["spikes1"], row["axi_spikes1"])) and bool(np.array_equal(row["spikes2"], row["axi_spikes2"]))
            packets_ok &= bool(np.array_equal(row["last"], [0]*52+[1])) and bool(np.all(row["keep"] == 15)) and bool(np.all(row["strb"] == 15))
        if (i+1) % 25 == 0 or i+1 == len(x):
            print(f"HLS LFW inference {i+1}/{len(x)}; correct={np.count_nonzero(np.asarray(predictions)==labels[:i+1])}; max error={max_error:.8g}", flush=True)
    correct = int(np.count_nonzero(np.asarray(predictions) == labels))
    agreement = int(np.count_nonzero(np.asarray(predictions) == baseline["predictions"]))
    confusion = np.zeros((53, 53), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    report = {"model": "new LFW HLS float32 C++ and AXI chain", "metadata": metadata,
              "images": len(x), "correct": correct, "accuracy": correct/len(x), "target_exceeded": correct/len(x) > .96,
              "pytorch_prediction_agreement": agreement, "max_logit_absolute_error": max_error,
              "all_logits_close_at_1e_4": bool(close), "plain_axi_bit_exact": bool(axi_exact), "packets_valid": bool(packets_ok),
              "seconds": time.perf_counter()-start, "predictions": predictions, "labels": labels.tolist(), "logits": logits,
              "confusion_matrix": confusion.tolist(),
              "pass": agreement == len(x) and close and axi_exact and packets_ok,
              "scope": "C++ simulation, not measured board inference or RTL co-simulation"}
    (ROOT/"stress_sim/lfw_hls_evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ["pass", "images", "correct", "accuracy", "pytorch_prediction_agreement", "max_logit_absolute_error", "seconds"]}), flush=True)
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
