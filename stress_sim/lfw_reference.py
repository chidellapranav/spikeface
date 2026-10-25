"""Float32 scalar-reduction specification of the authoritative LFW architecture."""
import numpy as np


def export_params(state):
    result = {}
    for name, bn in [("conv1", "bn1"), ("conv2", "bn2"), ("fc1", "bn3")]:
        def array(key):
            return state[key].detach().cpu().numpy().astype(np.float32)
        scale = array(bn+".weight")/np.sqrt(array(bn+".running_var")+np.float32(1e-5))
        shift = array(bn+".bias")-array(bn+".running_mean")*scale
        result[name] = np.concatenate([array(name+".weight").ravel(), array(name+".bias"), scale, shift])
    return result


def conv(x, params, channels, side):
    """Explicit sequential float32 MACs; no BLAS reduction or kernel folding."""
    inputs = 1 if channels == 8 else 8
    wsize = channels*inputs*25
    weights = params[:wsize].reshape(channels, inputs, 5, 5)
    bias, scale, shift = params[wsize:].reshape(3, channels)
    a = np.asarray(x, dtype=np.float32).reshape(inputs, side, side)
    pad = np.pad(a, ((0, 0), (2, 2), (2, 2)))
    summed = np.zeros((channels, side, side), dtype=np.float32)
    for ic in range(inputs):
        for ky in range(5):
            for kx in range(5):
                term = pad[ic, ky:ky+side, kx:kx+side]*weights[:, ic, ky, kx, None, None]
                summed = summed+term
    bn = (summed+bias[:, None, None])*scale[:, None, None]+shift[:, None, None]
    pooled = np.maximum(np.maximum(bn[:, 0::2, 0::2], bn[:, 0::2, 1::2]),
                        np.maximum(bn[:, 1::2, 0::2], bn[:, 1::2, 1::2]))
    return pooled


def lif(current, membrane, beta, threshold):
    integrated = np.float32(beta)*membrane+current
    updated = integrated-np.where(membrane > threshold, np.float32(threshold), np.float32(0))
    return (updated > threshold).astype(np.uint32), updated


def dense(x, params):
    x = np.asarray(x, dtype=np.float32)
    weights = params[:217088].reshape(53, 4096)
    bias, scale, shift = params[217088:].reshape(3, 53)
    value = np.zeros(53, dtype=np.float32)
    for i in range(4096):
        value = value+weights[:, i]*x.ravel()[i]
    return (value+bias)*scale+shift


class Network:
    def __init__(self, params, config=(.9375, .5, .9375, .5, .9375)):
        self.params, self.config = params, config
        self.reset()

    def reset(self):
        self.m1, self.m2 = np.zeros((8, 32, 32), dtype=np.float32), np.zeros((16, 16, 16), dtype=np.float32)
        self.m3, self.evidence = np.zeros(53, dtype=np.float32), np.zeros(53, dtype=np.float32)
        self.steps = 0

    def step(self, image, clear=False):
        if clear:
            self.reset()
        b1, t1, b2, t2, b3 = self.config
        c1 = conv(image, self.params["conv1"], 8, 64)
        s1, self.m1 = lif(c1, self.m1, b1, t1)
        c2 = conv(s1, self.params["conv2"], 16, 32)
        s2, self.m2 = lif(c2, self.m2, b2, t2)
        current = dense(s2, self.params["fc1"])
        self.m3 = np.float32(b3)*self.m3+current
        self.evidence = self.evidence+self.m3
        self.steps += 1
        scores = self.evidence/np.float32(self.steps)
        return {"spikes1": s1, "spikes2": s2, "mem1": self.m1.copy(), "mem2": self.m2.copy(),
                "mem3": self.m3.copy(), "evidence": self.evidence.copy(), "scores": scores}
