"""Independent integer model of the committed HLS assignment semantics.

All internal currents are integer codes with 12 fractional bits. This models
every saturating accumulator assignment, not merely a final clipped dot product.
No Xilinx headers or HLS implementation code are used here.
"""
from pathlib import Path
import ast
import re
import numpy as np

ACC_MIN, ACC_MAX = -(1 << 19), (1 << 19) - 1
SHAPES = {"conv1": (8, 5, 5), "conv2": (16, 8, 5, 5),
          "fc1": (64, 1024), "fc2": (40, 64)}


def sat(x):
    return np.clip(x, ACC_MIN, ACC_MAX)


def round_even_div(x, divisor):
    """Exact signed nearest-even division, including negative halfway cases."""
    q, r = np.divmod(np.asarray(x, dtype=np.int64), divisor)
    return q + ((2 * r > divisor) | ((2 * r == divisor) & ((q & 1) != 0)))


def lif(current, membrane, beta=30, threshold=32):
    # beta and membrane both Q3.5: product has 10 fractional bits, hence *4.
    updated = sat(sat(np.asarray(membrane, dtype=np.int64) * beta * 4) + current)
    quantized = np.clip(round_even_div(updated, 128), -128, 127)
    spikes = (quantized >= threshold).astype(np.int64)
    return spikes, np.where(spikes, 0, quantized)


def convolution(x, weights, membrane, beta=30, threshold=32):
    x = np.asarray(x, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    binary = x.ndim == 2
    if binary:
        x, w = x[None], w[:, None]
    padded = np.pad(x, ((0, 0), (2, 2), (2, 2)))
    current = np.zeros((w.shape[0], *x.shape[1:]), dtype=np.int64)
    taps = 0
    for ic in range(x.shape[0]):
        partial = np.zeros_like(current)
        for ky in range(5):
            for kx in range(5):
                patch = padded[ic, ky:ky+x.shape[1], kx:kx+x.shape[2]]
                partial = sat(partial + w[:, ic, ky, kx, None, None] * patch * (32 if binary else 8))
                taps += int(np.count_nonzero(patch)) * w.shape[0]
        current = sat(current + partial)
    spikes, mem = lif(current, membrane, beta, threshold)
    return spikes, mem, taps


def pool(x):
    # Output codes have two fractional bits: a sum of four spikes is its code.
    x = np.asarray(x, dtype=np.int64)
    return x[:, 0::2, 0::2] + x[:, 0::2, 1::2] + x[:, 1::2, 0::2] + x[:, 1::2, 1::2]


def dense(x, weights, membrane, beta=30, threshold=32):
    x, w = np.asarray(x, dtype=np.int64).ravel(), np.asarray(weights, dtype=np.int64)
    current = np.zeros(w.shape[0], dtype=np.int64)
    for i in range(x.size):
        current = sat(current + w[:, i] * x[i] * 8)
    spikes, mem = lif(current, membrane, beta, threshold)
    return spikes, mem, int(np.count_nonzero(x)) * w.shape[0]


def readout(x, weights, accum, clear=False):
    current = np.zeros(40, dtype=np.int64)
    for i, value in enumerate(np.asarray(x).ravel()):
        current = sat(current + np.asarray(weights)[:, i] * int(value) * 64)
    accum = current if clear else sat(accum + current)
    # Widen Q8.12 to Q8.24 by appending 12 zero bits.
    return accum * 4096, accum


class Network:
    def __init__(self, weights, beta=30, threshold=32):
        for name, shape in SHAPES.items():
            a = np.asarray(weights[name])
            if a.shape != shape or a.dtype.kind not in "iu" or np.any((a < -128) | (a > 127)):
                raise ValueError(f"{name}: expected signed 8-bit integer codes with shape {shape}")
        self.weights, self.beta, self.threshold = weights, beta, threshold
        self.reset()

    def reset(self):
        self.mem = {"conv1": np.zeros((8, 32, 32), dtype=np.int64),
                    "conv2": np.zeros((16, 16, 16), dtype=np.int64),
                    "fc1": np.zeros(64, dtype=np.int64)}
        self.accum = np.zeros(40, dtype=np.int64)

    def step(self, frame, clear=False):
        if clear:
            self.reset()
        c1, self.mem["conv1"], t1 = convolution(frame, self.weights["conv1"], self.mem["conv1"], self.beta, self.threshold)
        p1 = pool(c1)
        c2, self.mem["conv2"], t2 = convolution(p1, self.weights["conv2"], self.mem["conv2"], self.beta, self.threshold)
        p2 = pool(c2)
        f1, self.mem["fc1"], t3 = dense(p2, self.weights["fc1"], self.mem["fc1"], self.beta, self.threshold)
        scores, self.accum = readout(f1, self.weights["fc2"], self.accum, clear)
        return {"conv1": c1, "pool1": p1, "conv2": c2, "pool2": p2,
                "fc1": f1, "fc2": scores, "taps": np.array([t1, t2, t3])}


def golden_arrays(path):
    """Read numeric C initializers without executing source; validate element count."""
    text = Path(path).read_text()
    out = {}
    pattern = r"static const (?:double|int|long)\s+(\w+)((?:\[\d+\])+)\s*=\s*(\{.*?\});"
    for m in re.finditer(pattern, text, re.S):
        shape = tuple(map(int, re.findall(r"\d+", m[2])))
        literal = re.sub(r"(?<=\d)L\b", "", m[3]).replace("{", "[").replace("}", "]")
        data = ast.literal_eval(literal)
        def flatten(v):
            if isinstance(v, list):
                return [z for item in v for z in flatten(item)]
            return [v]
        out[m[1]] = np.asarray(flatten(data)).reshape(shape)
    return out
