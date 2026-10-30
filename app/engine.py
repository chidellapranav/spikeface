"""Validated software inference and exact-spike evaluation for SpikeFace."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps
from Spike_old.stress_sim.reference import Network, SHAPES, golden_arrays

ROOT = Path(__file__).resolve().parents[1]


def valid_frames(frames):
    a = np.asarray(frames)
    if a.ndim not in (2, 3) or a.shape[1:] not in ((1024,), (32, 32)) or not 1 <= a.shape[0] <= 500:
        raise ValueError("Spikes must have shape (T,1024) or (T,32,32), with 1 ≤ T ≤ 500")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError("Spikes must contain only exact 0 and 1 values")
    return a.astype(np.int64).reshape(-1, 32, 32)


class Model:
    def __init__(self, weights, metadata=None, beta=30, threshold=32):
        self.metadata = metadata or {}
        if not isinstance(self.metadata, dict):
            raise ValueError("Model metadata must be a JSON object")
        self.beta, self.threshold = int(beta), int(threshold)
        if self.beta != beta or self.threshold != threshold or not 0 <= self.beta <= 32 or not 1 <= self.threshold <= 127:
            raise ValueError("Deployment requires integer beta code 0..32 and threshold code 1..127")
        Network(weights, self.beta, self.threshold)  # validates shapes and code ranges
        self.weights = {k: np.array(v, dtype=np.int64, copy=True) for k, v in weights.items()}
        labels = self.metadata.get("labels", [f"Identity {i}" for i in range(40)])
        if not isinstance(labels, list) or len(labels) != 40 or not all(isinstance(s, str) for s in labels):
            raise ValueError("labels must contain exactly 40 strings")
        self.labels = labels

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            weights = {name: f[name] for name in SHAPES}
            meta = json.loads(str(f["metadata"].item())) if "metadata" in f else {}
            return cls(weights, meta, f["beta_code"].item() if "beta_code" in f else 30,
                       f["threshold_code"].item() if "threshold_code" in f else 32)

    @classmethod
    def golden(cls):
        src = ROOT / "Spike_old/Spikeface_HLS"
        spec = [("conv1", "golden_conv1_array.h", "GOLDEN_ARR_KERNELS", 128),
                ("conv2", "golden_conv2.h", "GOLDEN_CONV2_WEIGHTS", 128),
                ("fc1", "golden_fc1.h", "GOLDEN_FC1_WEIGHTS", 128),
                ("fc2", "golden_fc2.h", "GOLDEN_FC2_WEIGHTS", 64)]
        weights = {name: np.rint(golden_arrays(src / file)[key]*scale).astype(np.int64)
                   for name, file, key, scale in spec}
        return cls(weights, {"name": "Committed golden-vector fixture", "verification_fixture": True,
                             "accuracy": "Not measured; this fixture is not an accuracy benchmark"})

    def save(self, path):
        np.savez_compressed(path, **self.weights, metadata=json.dumps(self.metadata),
                            beta_code=self.beta, threshold_code=self.threshold)

    def infer(self, frames):
        frames = valid_frames(frames)
        network = Network(self.weights, self.beta, self.threshold)
        trace = []
        taps = np.zeros(3, dtype=np.int64)
        start = time.perf_counter()
        for frame in frames:
            r = network.step(frame)
            trace.append(r["fc2"])
            taps += r["taps"]
        seconds = time.perf_counter()-start
        scores = np.asarray(trace[-1], dtype=np.float64) / (1 << 24)
        top = np.argsort(-scores, kind="stable")[:5]
        return {"prediction": int(top[0]), "identity": self.labels[top[0]],
                "scores": scores.tolist(), "top5": [{"class": int(i), "label": self.labels[i], "score": float(scores[i])} for i in top],
                "timesteps": len(frames), "seconds": seconds, "backend": "integer HLS reference",
                "trace_codes": np.asarray(trace).tolist(), "metadata": self.metadata,
                "sparsity": dict(zip(["conv1", "conv2", "fc1"], (1-taps/(np.array([204800, 819200, 65536])*len(frames))).tolist())),
                "score_at_limit": bool(np.any((network.accum == -(1 << 19)) | (network.accum == (1 << 19)-1)))}


def image_frames(source, timesteps=50, seed=0):
    if isinstance(timesteps, bool) or not isinstance(timesteps, int) or not 1 <= timesteps <= 500:
        raise ValueError("Timesteps must be an integer in 1..500")
    with Image.open(source) as im:
        if im.width * im.height > 20_000_000:
            raise ValueError("Image is too large")
        # Explicit deployment default; original training preprocessing is unavailable.
        resampling = getattr(Image, "Resampling", Image)
        gray = ImageOps.exif_transpose(im).convert("L").resize((32, 32), resampling.BILINEAR)
        image = np.asarray(gray, dtype=np.float64)/255.0
    return (np.random.default_rng(seed).random((timesteps, 32, 32)) < image).astype(np.int64)


def fixture_frames():
    return golden_arrays(ROOT / "Spike_old/Spikeface_HLS/golden_conv1_array.h")["GOLDEN_ARR_INPUT"].astype(np.int64)


def evaluate(model, dataset):
    """Paired fixed-spike evaluation; dataset must be an untouched held-out split."""
    with np.load(dataset, allow_pickle=False) as f:
        frames, labels = f["frames"], f["labels"]
        expected = f["score_codes"] if "score_codes" in f else None
    if frames.ndim not in (3, 4) or len(frames) == 0 or labels.shape != (len(frames),):
        raise ValueError("Dataset needs frames (N,T,1024)/(N,T,32,32) and labels (N,)")
    if labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= 40)):
        raise ValueError("Labels must be integer class indices 0..39")
    if expected is not None and expected.shape != (len(frames), frames.shape[1], 40):
        raise ValueError("score_codes must have shape (N,T,40)")
    predictions, wrong, mismatches = [], [], 0
    for i, (spikes, label) in enumerate(zip(frames, labels)):
        result = model.infer(spikes)  # fresh state for every image
        predictions.append(result["prediction"])
        if predictions[-1] != label:
            wrong.append(i)
        if expected is not None:
            mismatches += int(np.count_nonzero(np.asarray(result["trace_codes"]) != expected[i]))
    correct, n = len(frames)-len(wrong), len(frames)
    # Wilson 95% interval makes small-test-set uncertainty explicit.
    p, z = correct/n, 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"images": n, "correct": correct, "accuracy": p, "target_exceeded": p > .96,
            "wilson_95": [center-half, center+half], "wrong_indices": wrong,
            "predictions": predictions, "score_code_mismatches": mismatches if expected is not None else None}
