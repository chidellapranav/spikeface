"""PYNQ driver for the NEW 53-class LFW design; not compatible with legacy XSA.

Offsets below were checked against native Vitis 2022.2 synthesis RTL. Recheck
them against exported headers/HWH after any interface change. Board unverified.
"""
import time
import numpy as np
from pynq import Overlay, allocate, Clocks

IP_NAMES = {"conv1": "lfw_conv1_axi_0", "conv2": "lfw_conv2_axi_0", "fc1": "lfw_readout_axi_0"}
PARAM_COUNTS = {"conv1": 224, "conv2": 3248, "fc1": 217247}
RESET_OFFSETS = {"conv1": 0x2C, "conv2": 0x2C, "fc1": 0x24}


def float_bits(value):
    value = np.float32(value)
    if not np.isfinite(value):
        raise ValueError("Nonfinite float parameter")
    return int(np.asarray(value).view(np.uint32))


class LFWOverlay:
    def __init__(self, bitstream, params, timeout_s=60., config=(.9375, .5, .9375, .5, .9375)):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        config = np.asarray(config, dtype=np.float32)
        if config.shape != (5,) or not np.all(np.isfinite(config)) or np.any((config[[0, 2, 4]] < 0) | (config[[0, 2, 4]] > 1)) or np.any(config[[1, 3]] <= 0):
            raise ValueError("Invalid LFW beta/threshold configuration")
        validated = {}
        for name, count in PARAM_COUNTS.items():
            a = np.asarray(params[name], dtype=np.float32)
            if a.shape != (count,) or not np.all(np.isfinite(a)):
                raise ValueError(f"{name}: expected {count} finite float parameters")
            validated[name] = a
        self.buffers = {}
        self.timeout_s = timeout_s
        self.overlay = Overlay(bitstream)
        self.blocks = {k: getattr(self.overlay, v) for k, v in IP_NAMES.items()}
        self.dma_in, self.dma_out = self.overlay.axi_dma_0, self.overlay.axi_dma_1
        self.fclk_mhz = Clocks.fclk0_mhz
        try:
            for name, a in validated.items():
                buffer = allocate(shape=a.shape, dtype=np.float32)
                self.buffers[name] = buffer
                buffer[:] = a
                buffer.flush()
                address = int(buffer.physical_address)
                self.blocks[name].write(0x10, address & 0xFFFFFFFF)
                self.blocks[name].write(0x14, address >> 32)
                index = {"conv1": 0, "conv2": 2, "fc1": 4}[name]
                self.blocks[name].write(0x1C, float_bits(config[index]))
                if name != "fc1":
                    self.blocks[name].write(0x24, float_bits(config[index+1]))
            self.buffers["input"] = allocate(shape=(4096,), dtype=np.float32)
            self.buffers["output"] = allocate(shape=(53,), dtype=np.float32)
        except Exception:
            self.close()
            raise

    def _wait(self, predicate, name):
        start = time.perf_counter()
        while not predicate():
            if time.perf_counter()-start > self.timeout_s:
                raise TimeoutError(f"LFW hardware timeout: {name}; reload overlay before retrying")

    def _dma_wait(self, channel, name, byte_count):
        def done():
            if channel.error or not channel.running:
                raise RuntimeError(f"{name}: DMA error or halted engine; reload overlay before retrying")
            return channel.idle
        self._wait(done, name)
        channel.wait()  # receive cache invalidation and transferred count
        if channel.transferred != byte_count:
            raise RuntimeError(f"{name}: {channel.transferred} bytes received, expected {byte_count}")

    def infer(self, image, timesteps=25, return_trace=False):
        image = np.asarray(image, dtype=np.float32)
        if image.shape != (64, 64) or not np.all(np.isfinite(image)) or np.any((image < 0) | (image > 1)):
            raise ValueError("Image must be finite 64x64 grayscale in [0,1]")
        if isinstance(timesteps, bool) or not isinstance(timesteps, int) or not 1 <= timesteps <= 500:
            raise ValueError("timesteps must be an integer in 1..500")
        self.buffers["input"][:] = image.ravel()
        trace = []
        for t in range(timesteps):
            for name, ip in self.blocks.items():
                self._wait(lambda ip=ip: bool(ip.read(0)&4), name+" idle")
                ip.write(RESET_OFFSETS[name], int(t == 0))
            for name in ("fc1", "conv2", "conv1"):
                self.blocks[name].write(0x00, 1)
            self.dma_out.recvchannel.transfer(self.buffers["output"])
            self.dma_in.sendchannel.transfer(self.buffers["input"])
            self._dma_wait(self.dma_in.sendchannel, "input", 4096*4)
            self._dma_wait(self.dma_out.recvchannel, "scores", 53*4)
            for name, ip in self.blocks.items():
                self._wait(lambda ip=ip: bool(ip.read(0) & 4), name+" completion")
            if not np.all(np.isfinite(self.buffers["output"])):
                raise RuntimeError("Nonfinite LFW hardware scores; reload overlay before retrying")
            if return_trace:
                trace.append(np.asarray(self.buffers["output"]).copy())
        scores = np.asarray(self.buffers["output"]).copy()
        return (int(np.argmax(scores)), scores, np.asarray(trace)) if return_trace else (int(np.argmax(scores)), scores)

    def close(self):
        for buffer in self.buffers.values():
            buffer.freebuffer()
        self.buffers.clear()
