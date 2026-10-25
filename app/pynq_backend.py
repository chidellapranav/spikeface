"""Optional board backend. Import only on a PYNQ board with matching hardware."""
import time
import numpy as np
from .engine import valid_frames


class PynqModel:
    def __init__(self, model, bitstream):
        from PS.spikeface_driver import SpikeFaceOverlay, WEIGHT_FRAC_BITS
        self.metadata = dict(model.metadata, hardware_bitstream=bitstream)
        self.labels = model.labels
        self.sf = SpikeFaceOverlay(bitstream)
        weights = {k: v.astype(np.float64)/(1 << WEIGHT_FRAC_BITS[k]) for k, v in model.weights.items()}
        try:
            self.sf.configure(weights, model.beta/32, model.threshold/32)
        except Exception:
            self.close()
            raise

    def infer(self, frames):
        frames = valid_frames(frames)
        start = time.perf_counter()
        pred, scores, extras = self.sf.run_inference(spike_frames=frames, T=len(frames),
                                                    fresh_state=True, return_trace=True, collect_taps=True)
        top = np.argsort(-scores, kind="stable")[:5]
        codes = np.rint(extras["trace"]*(1 << 24)).astype(np.int64)
        return {"prediction": int(pred), "identity": self.labels[pred], "scores": scores.tolist(),
                "top5": [{"class": int(i), "label": self.labels[i], "score": float(scores[i])} for i in top],
                "trace_codes": codes.tolist(), "timesteps": len(frames), "seconds": time.perf_counter()-start,
                "backend": "PYNQ programmable logic", "metadata": self.metadata,
                "sparsity": extras["sparsity"],
                "score_at_limit": bool(np.any((codes[-1] == -(1 << 31)) | (codes[-1] == ((1 << 19)-1)*4096)))}

    def close(self):
        if hasattr(self, "sf"):
            for name in ("spike_in_buf", "score_out_buf"):
                buffer = getattr(self.sf, name, None)
                if buffer is not None:
                    buffer.freebuffer()
