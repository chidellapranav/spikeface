#!/usr/bin/env python3
"""Create deterministic binary verification data for native HLS C/RTL tools."""
import argparse
from pathlib import Path
import struct
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.lfw import load_reference
from stress_sim.lfw_reference import Network, export_params


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--timesteps", type=int, default=25)
    args = p.parse_args()
    if not 1 <= args.timesteps <= 25:
        p.error("--timesteps must be in 1..25")
    _, state, _, _ = load_reference(args.repo)
    params = export_params(state)
    image = np.load(Path(args.repo)/"data/spikeface_lfw/X_test.npy", allow_pickle=False)[0]
    network = Network(params)
    output = ROOT/"stress_sim/build/lfw_fixture.bin"
    output.parent.mkdir(exist_ok=True)
    with output.open("wb") as f:
        f.write(struct.pack("<I", args.timesteps))
        for name in ("conv1", "conv2", "fc1"):
            f.write(params[name].astype("<f4").tobytes())
        f.write(image.astype("<f4").tobytes())
        for t in range(args.timesteps):
            row = network.step(image)
            for name in ("spikes1", "spikes2"):
                f.write(row[name].astype("<u4").tobytes())
            f.write(row["scores"].astype("<f4").tobytes())
    print(output)


if __name__ == "__main__":
    main()
