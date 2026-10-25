import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


class Memory(np.ndarray):
    counter = 0

    def __new__(cls, shape, dtype):
        a = np.zeros(shape, dtype=dtype).view(cls)
        cls.counter += 1
        a.physical_address = 0x10000000+cls.counter*0x100000
        a.flushed = a.freed = a.invalidated = False
        return a

    def flush(self):
        self.flushed = True

    def freebuffer(self):
        self.freed = True

    def invalidate(self):
        self.invalidated = True


class Channel:
    error, running, idle = False, True, True

    def transfer(self, buffer):
        self.buffer = buffer
        if buffer.size == 53:
            buffer[:] = np.arange(53, dtype=np.float32)

    def wait(self):
        self.transferred = self.buffer.nbytes
        if self.buffer.size == 53:
            self.buffer.invalidate()


class IP:
    def __init__(self):
        self.writes = []

    def read(self, offset):
        return 4

    def write(self, offset, value):
        self.writes.append((offset, value))


def board():
    overlay = SimpleNamespace(lfw_conv1_axi_0=IP(), lfw_conv2_axi_0=IP(), lfw_readout_axi_0=IP(),
                              axi_dma_0=SimpleNamespace(sendchannel=Channel()), axi_dma_1=SimpleNamespace(recvchannel=Channel()))
    spec = importlib.util.spec_from_file_location("lfw_driver_subject", Path(__file__).resolve().parents[1]/"PS/lfw_driver.py")
    driver = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"pynq": SimpleNamespace(Overlay=lambda path: overlay, allocate=Memory, Clocks=SimpleNamespace(fclk0_mhz=50))}):
        spec.loader.exec_module(driver)
    params = {k: np.zeros(n, dtype=np.float32) for k, n in driver.PARAM_COUNTS.items()}
    return driver, overlay, params


class LFWPSTests(unittest.TestCase):
    def test_parameter_pointers_and_float_codes(self):
        driver, overlay, params = board()
        sf = driver.LFWOverlay("test.bit", params)
        for name, ip in sf.blocks.items():
            self.assertTrue(sf.buffers[name].flushed)
            self.assertIn((0x10, sf.buffers[name].physical_address), ip.writes)
            self.assertIn((0x1C, 0x3F700000), ip.writes)
        sf.close()

    def test_reset_each_image_and_dma_cache(self):
        driver, _, params = board()
        sf = driver.LFWOverlay("test.bit", params)
        pred, scores, trace = sf.infer(np.zeros((64, 64)), 2, True)
        self.assertEqual(pred, 52)
        self.assertEqual(trace.shape, (2, 53))
        self.assertTrue(sf.buffers["output"].invalidated)
        for name, ip in sf.blocks.items():
            flags = [value for offset, value in ip.writes if offset == driver.RESET_OFFSETS[name]]
            self.assertEqual(flags, [1, 0])
        sf.infer(np.zeros((64, 64)), 1)
        for name, ip in sf.blocks.items():
            self.assertEqual([v for o, v in ip.writes if o == driver.RESET_OFFSETS[name]][-1], 1)
        memories = list(sf.buffers.values())
        sf.close()
        self.assertTrue(all(b.freed for b in memories))

    def test_reject_invalid_configuration_and_images(self):
        driver, _, params = board()
        for config in [(2, .5, .9, .5, .9), (.9, 0, .9, .5, .9)]:
            with self.assertRaises(ValueError):
                driver.LFWOverlay("test.bit", params, config=config)
        sf = driver.LFWOverlay("test.bit", params)
        for image in [np.zeros((32, 32)), np.full((64, 64), np.nan), np.full((64, 64), 2)]:
            with self.assertRaises(ValueError):
                sf.infer(image)
        sf.close()

    def test_dma_errors_and_short_packets(self):
        driver, _, params = board()
        sf = driver.LFWOverlay("test.bit", params)
        channel = sf.dma_out.recvchannel
        channel.error = True
        with self.assertRaises(RuntimeError):
            sf._dma_wait(channel, "rx", 212)
        channel.error = False
        channel.transfer(sf.buffers["output"])
        with self.assertRaises(RuntimeError):
            sf._dma_wait(channel, "rx", 211)
        sf.close()


if __name__ == "__main__":
    unittest.main()
