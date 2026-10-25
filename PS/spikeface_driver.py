"""
SpikeFace PS driver for the PYNQ-Z2 (Python / PYNQ).

Runs ON THE BOARD, after the bitstream (.bit) and hardware handoff (.hwh) have
been copied over. It configures the six PL blocks, streams spike frames in via
DMA, sequences the blocks timestep by timestep, and reads back the 40 class
scores.

STATUS: written against the committed design (spikeface/system_150926.xsa and
rebuild_vivado.tcl) but NOT YET RUN ON HARDWARE. Expect a short bring-up round.
See PS/README.md and PS/DEPLOYMENT_AND_ACCURACY.md.

Register names, offsets and DMA widths below were read from the committed
system_top.hwh and the Vitis-generated driver headers inside the XSA.
"""

import time
import numpy as np
from pynq import Overlay, allocate, Clocks

# ---------------------------------------------------------------------------
# Design constants (verified against the committed XSA / hwh)
# ---------------------------------------------------------------------------
BITSTREAM_PATH = "/home/xilinx/spikeface/spikeface.bit"   # spikeface.hwh must sit beside it

IP_NAMES = {
    "conv1": "conv1_layer_axi_0",
    "pool1": "conv1_pool_axi_0",     # free-running: ap_start tied high in hardware
    "conv2": "conv2_layer_axi_0",
    "pool2": "conv2_pool_axi_0",     # free-running: ap_start tied high in hardware
    "fc1":   "fc1_layer_axi_0",
    "fc2":   "fc2_layer_axi_0",
    "dma_in":  "axi_dma_0",          # MM2S only, 8-bit stream  -> conv1
    "dma_out": "axi_dma_1",          # S2MM only, 32-bit stream <- fc2
}
COMPUTE_BLOCKS = ("conv1", "conv2", "fc1", "fc2")   # need an explicit ap_start
SPIKING_BLOCKS = ("conv1", "conv2", "fc1")          # have beta / threshold

# Fixed-point formats (fractional bits), from spikeface_types.h / fc2_layer_axi.h
WEIGHT_FRAC_BITS = {"conv1": 7, "conv2": 7, "fc1": 7, "fc2": 6}  # ap_fixed<8,1> ; fc2 ap_fixed<8,2>
MEM_FRAC_BITS = 5            # mem_t = ap_fixed<8,3> (beta, threshold)
SCORE_FRAC_BITS = 24         # axis_score_t = ap_fixed<32,8>

# Weight array shapes, C-order flatten == HLS array index order
WEIGHT_SHAPES = {"conv1": (8, 5, 5), "conv2": (16, 8, 5, 5), "fc1": (64, 1024), "fc2": (40, 64)}

# Byte offset of each weights array inside the block's AXI-Lite space, from the
# generated xxxx_hw.h headers (..._ADDR_WEIGHTS_BASE). All weights are 8-bit.
WEIGHT_OFFSET = {"conv1": 0x100, "conv2": 0x1000, "fc1": 0x10000, "fc2": 0x1000}

IN_ELEMENTS = 32 * 32        # one input spike frame
OUT_CLASSES = 40
T_TIMESTEPS = 50

# Possible MAC taps per timestep, for sparsity reporting
POSSIBLE_TAPS = {"conv1": 8 * 32 * 32 * 25, "conv2": 16 * 16 * 16 * 200, "fc1": 64 * 1024}


# ---------------------------------------------------------------------------
# Pure helpers (no hardware needed; unit-testable)
# ---------------------------------------------------------------------------
def float_to_code(values, frac_bits, name="value", total_bits=8):
    """
    Convert already-quantized fixed-point values to raw two's-complement codes.

    `values` must lie on the fixed-point grid (multiples of 2**-frac_bits) --
    i.e. the w_*_q arrays from the Python fixed-point reference, NOT raw
    float weights. Off-grid input is rejected rather than silently rounded.
    """
    v = np.asarray(values, dtype=np.float64)
    if total_bits != 8 or not 0 <= frac_bits <= 7:
        raise ValueError("This driver supports 8-bit codes with 0..7 fractional bits")
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError(f"{name}: values must be nonempty and finite")
    scaled = v * (1 << frac_bits)
    codes = np.rint(scaled)
    if np.max(np.abs(scaled - codes)) > 1e-6:
        raise ValueError(f"{name}: values are not on the Q{frac_bits} grid. "
                         "Pass the quantized weights (w_*_q), not raw float weights.")
    lo, hi = -(1 << (total_bits - 1)), (1 << (total_bits - 1)) - 1
    if codes.min() < lo or codes.max() > hi:
        raise ValueError(f"{name}: value outside the representable {total_bits}-bit range "
                         f"[{lo / (1 << frac_bits)}, {hi / (1 << frac_bits)}].")
    return codes.astype(np.int8).view(np.uint8)


def pack_words(codes_u8):
    """Pack a flat uint8 array into little-endian uint32 words (element 0 = low byte)."""
    flat = np.asarray(codes_u8, dtype=np.uint8).ravel()
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return flat.view("<u4")


def unpack_scores(raw_u32):
    """Reinterpret fc2's raw 32-bit patterns (ap_fixed<32,8>) as floats."""
    raw = np.asarray(raw_u32, dtype=np.uint32).copy()
    return raw.view(np.int32).astype(np.float64) / float(1 << SCORE_FRAC_BITS)


def encode_frame(image01, rng):
    """Bernoulli rate coding: a pixel of intensity p spikes with probability p."""
    return (rng.random(image01.shape) < image01).astype(np.uint8)


# ---------------------------------------------------------------------------
class SpikeFaceOverlay:
    def __init__(self, bitstream_path=BITSTREAM_PATH, seed=0, timeout_s=5.0):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.bitstream_path = bitstream_path
        self.timeout_s = timeout_s
        self.rng = np.random.default_rng(seed)
        self._params = None
        self.overlay = Overlay(bitstream_path)
        self._bind()
        # The fabric clock the design actually runs at (latency numbers depend on it).
        self.fclk_mhz = Clocks.fclk0_mhz
        print(f"[SpikeFace] fabric clock FCLK0 = {self.fclk_mhz} MHz")
        self.spike_in_buf = allocate(shape=(IN_ELEMENTS,), dtype=np.uint8)
        self.score_out_buf = allocate(shape=(OUT_CLASSES,), dtype=np.uint32)

    def _bind(self):
        self.blocks = {k: getattr(self.overlay, IP_NAMES[k]) for k in
                       ("conv1", "pool1", "conv2", "pool2", "fc1", "fc2")}
        self.dma_in = getattr(self.overlay, IP_NAMES["dma_in"])
        self.dma_out = getattr(self.overlay, IP_NAMES["dma_out"])
        support = [hasattr(self.blocks[k].register_map, "clear_state") for k in SPIKING_BLOCKS]
        if any(support) and not all(support):
            raise RuntimeError("Mixed HLS reset interfaces: rebuild all three spiking IPs together")
        self.supports_clear_state = all(support)

    # -------------------------------------------------------------- setup
    def configure(self, weights, beta=0.9375, threshold=1.0):
        """
        weights: dict conv1/conv2/fc1/fc2 -> array of QUANTIZED weights with the
                 shapes in WEIGHT_SHAPES (conv1 is (8,5,5), not (8,1,5,5)).
        beta, threshold: LIF parameters as real values (0.9375 = 0.95 quantized).
        Call once after boot / after every PL reload.
        """
        if not np.isscalar(beta) or not np.isscalar(threshold) or not 0 <= beta <= 1 or threshold <= 0:
            raise ValueError("Require scalar beta in [0,1] and threshold > 0")
        beta_code = int(float_to_code(beta, MEM_FRAC_BITS, "beta")[()])
        thr_code = int(float_to_code(threshold, MEM_FRAC_BITS, "threshold")[()])
        prepared = {}
        for name in WEIGHT_SHAPES:
            w = np.asarray(weights[name])
            if w.shape != WEIGHT_SHAPES[name]:
                raise ValueError(f"{name}: expected weight shape {WEIGHT_SHAPES[name]}")
            float_to_code(w, WEIGHT_FRAC_BITS[name], f"{name} weights")
            if WEIGHT_OFFSET[name] + w.size > len(self.blocks[name].mmio.array) * 4:
                raise RuntimeError(f"{name}: AXI-Lite address window cannot reach weights; rebuild the PL address map")
            prepared[name] = w.copy()
        self._params = None  # partial MMIO failure must not leave inference enabled
        for name in SPIKING_BLOCKS:
            self.blocks[name].register_map.beta = beta_code
            self.blocks[name].register_map.threshold = thr_code
        for name in ("conv1", "conv2", "fc1", "fc2"):
            self._write_weights(name, prepared[name])
        self._params = dict(weights=prepared, beta=beta, threshold=threshold)

    def _write_weights(self, name, w):
        w = np.asarray(w)
        if w.shape != WEIGHT_SHAPES[name]:
            raise ValueError(f"{name}: weights shape {w.shape}, expected {WEIGHT_SHAPES[name]}")
        words = pack_words(float_to_code(w, WEIGHT_FRAC_BITS[name], f"{name} weights"))
        mm = self.blocks[name].mmio
        first = WEIGHT_OFFSET[name] // 4
        window_words = len(mm.array)
        if first + words.size > window_words:
            raise RuntimeError(
                f"{name}: weights need AXI-Lite bytes 0x{WEIGHT_OFFSET[name]:X}.."
                f"0x{WEIGHT_OFFSET[name] + w.size - 1:X}, but this block's address window is only "
                f"0x{window_words * 4:X} bytes in the loaded bitstream. The weights cannot be "
                "reached from the PS. Fix the address range in Vivado's Address Editor "
                "(see PS/DEPLOYMENT_AND_ACCURACY.md, 'fc1 address window').")
        mm.array[first:first + words.size] = words
        if not np.array_equal(np.asarray(mm.array[first:first + words.size]), words):
            raise RuntimeError(f"{name}: weight read-back mismatch -- write did not land.")

    # -------------------------------------------------------------- state reset
    def reset_pl_state(self):
        """
        Legacy hardware has no per-image reset. Reprogramming the PL
        re-initialises those memories to zero; weights/parameters are then
        rewritten. Slow (~seconds) but needs no HLS change. Verify it with the
        determinism check in PS/DEPLOYMENT_AND_ACCURACY.md before trusting it.
        """
        if self._params is None:
            raise RuntimeError("configure() has not been called yet.")
        self.overlay = Overlay(self.bitstream_path)   # downloads the bitstream again
        self._bind()
        self.configure(**self._params)

    # -------------------------------------------------------------- scheduler
    def _diagnose(self):
        parts = []
        for k in COMPUTE_BLOCKS:
            c = self.blocks[k].register_map.CTRL
            parts.append(f"{k}(idle={int(c.AP_IDLE)})")
        parts.append(f"dma_in.idle={self.dma_in.sendchannel.idle}")
        parts.append(f"dma_out.idle={self.dma_out.recvchannel.idle}")
        return " ".join(parts)

    def _wait(self, cond, what):
        t0 = time.perf_counter()
        while not cond():
            if time.perf_counter() - t0 > self.timeout_s:
                raise TimeoutError(f"timeout waiting for {what}. State: {self._diagnose()}")

    def _wait_dma(self, channel, what, expected_bytes):
        def complete():
            if channel.error:
                raise RuntimeError(f"{what}: DMA error. State: {self._diagnose()}")
            if not channel.running:
                raise RuntimeError(f"{what}: DMA halted. State: {self._diagnose()}")
            return channel.idle
        self._wait(complete, what)
        # PYNQ wait() performs receive-buffer cache invalidation and records
        # transferred bytes. Polling .idle alone skipped this housekeeping.
        channel.wait()
        if channel.transferred != expected_bytes:
            raise RuntimeError(f"{what}: transferred {channel.transferred} bytes, expected {expected_bytes}")

    def _run_one_timestep(self, frame_flat, clear_accum, clear_state=False):
        """
        Start order is DOWNSTREAM-FIRST (fc2, fc1, conv2, conv1). Each block only
        drains its input stream once started; starting upstream-first lets a
        producer fill a not-yet-started consumer's FIFO and stall (deadlock).
        The two pooling blocks are free-running and need no start.
        """
        for k in COMPUTE_BLOCKS:
            self._wait(lambda k=k: self.blocks[k].register_map.CTRL.AP_IDLE, f"{k} idle")
        self.blocks["fc2"].register_map.clear_accum = 1 if clear_accum else 0
        if self.supports_clear_state:
            for k in SPIKING_BLOCKS:
                self.blocks[k].register_map.clear_state = int(clear_state)
        for k in ("fc2", "fc1", "conv2", "conv1"):
            self.blocks[k].register_map.CTRL.AP_START = 1
        self.spike_in_buf[:] = frame_flat
        self.dma_out.recvchannel.transfer(self.score_out_buf)   # arm the receiver first
        self.dma_in.sendchannel.transfer(self.spike_in_buf)
        self._wait_dma(self.dma_in.sendchannel, "input DMA", IN_ELEMENTS)
        self._wait_dma(self.dma_out.recvchannel, "output DMA (fc2 TLAST)", OUT_CLASSES * 4)
        for k in COMPUTE_BLOCKS:
            self._wait(lambda k=k: self.blocks[k].register_map.CTRL.AP_IDLE, f"{k} complete")

    def read_active_taps(self):
        """MAC taps actually computed in the last call, per spiking block."""
        out = {}
        for k in SPIKING_BLOCKS:
            rm = self.blocks[k].register_map
            out[k] = (int(rm.total_active_taps_2) << 32) | int(rm.total_active_taps_1)
        return out

    def run_inference(self, image01=None, T=T_TIMESTEPS, spike_frames=None,
                      fresh_state=True, return_trace=False, collect_taps=False):
        """
        image01: 32x32 floats in [0,1], encoded on the PS each timestep -- OR --
        spike_frames: (T,32,32) or (T,1024) array of {0,1} to replay exact spikes.
        fresh_state: isolate images (new clear_state flags, or legacy PL reload).
        Returns (predicted_class, final_scores[, extras]).
        """
        if self._params is None:
            raise RuntimeError("Call configure() first.")
        if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)) or not 1 <= T <= 500:
            raise ValueError("T must be an integer in 1..500")
        if image01 is not None and spike_frames is not None:
            raise ValueError("Provide image01 or spike_frames, not both")
        if spike_frames is not None:
            frames = np.asarray(spike_frames)
            if frames.ndim not in (2, 3) or frames.shape[1:] not in ((32, 32), (IN_ELEMENTS,)):
                raise ValueError("spike_frames must have shape (T,32,32) or (T,1024)")
            if frames.shape[0] < T:
                raise ValueError(f"spike_frames has {frames.shape[0]} timesteps, need {T}")
            if not np.all((frames == 0) | (frames == 1)):
                raise ValueError("spike_frames must contain only 0/1")
            frames = frames.astype(np.uint8).reshape(-1, IN_ELEMENTS)
        elif image01 is None:
            raise ValueError("Provide image01 or spike_frames.")
        else:
            image01 = np.asarray(image01, dtype=np.float64)
            if image01.shape != (32, 32) or not np.all(np.isfinite(image01)) or np.any((image01 < 0) | (image01 > 1)):
                raise ValueError("image01 must be a finite 32x32 array in [0,1]")
        if fresh_state and not self.supports_clear_state:
            self.reset_pl_state()
        trace, taps = [], {k: 0 for k in SPIKING_BLOCKS}
        for t in range(T):
            f = frames[t] if spike_frames is not None else \
                encode_frame(np.asarray(image01, dtype=np.float64), self.rng).ravel()
            self._run_one_timestep(f, clear_accum=(t == 0), clear_state=(fresh_state and t == 0))
            if return_trace:
                trace.append(unpack_scores(self.score_out_buf).copy())
            if collect_taps:
                for k, v in self.read_active_taps().items():
                    taps[k] += v
        scores = unpack_scores(self.score_out_buf)
        pred = int(np.argmax(scores))
        extras = {}
        if return_trace:
            extras["trace"] = np.array(trace)
        if collect_taps:
            extras["sparsity"] = {k: 1.0 - taps[k] / (POSSIBLE_TAPS[k] * T) for k in taps}
        return (pred, scores, extras) if extras else (pred, scores)


if __name__ == "__main__":
    # Minimal smoke test. Weight files: quantized arrays exported from the Python
    # reference (see README). Shapes: conv1 (8,5,5) conv2 (16,8,5,5) fc1 (64,1024) fc2 (40,64).
    sf = SpikeFaceOverlay()
    w = {k: np.load(f"w_{k}_q.npy") for k in ("conv1", "conv2", "fc1", "fc2")}
    sf.configure(w)
    img = np.load("test_image_32x32.npy")            # float in [0,1]
    pred, scores = sf.run_inference(img, T=50)
    print("predicted class:", pred, " score:", round(float(scores[pred]), 4))
