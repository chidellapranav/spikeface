"""Board-independent tests. Mocks model control/cache behavior, not FPGA logic."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


def load_driver():
    spec = importlib.util.spec_from_file_location("driver_test_subject", Path(__file__).resolve().parents[1]/"Spike_old/PS/spikeface_driver.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"pynq": SimpleNamespace(Overlay=None, allocate=None, Clocks=None)}):
        spec.loader.exec_module(module)
    return module


drv = load_driver()


class Buffer(np.ndarray):
    def __new__(cls, n, dtype):
        return np.zeros(n, dtype=dtype).view(cls)

    def invalidate(self):
        self.invalidated = True


class Channel:
    def __init__(self, name, log):
        self.name, self.log = name, log
        self.idle, self.running, self.error = True, True, False
        self.transferred = 0

    def transfer(self, buffer):
        self.log.append(self.name+" transfer")
        self.buffer = buffer
        if self.name == "rx":
            buffer[:] = np.arange(40, dtype=np.uint32) << 24

    def wait(self):
        self.log.append(self.name+" wait")
        self.transferred = self.buffer.nbytes
        if self.name == "rx":
            self.buffer.invalidate()


class Ctrl:
    AP_IDLE = 1

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __setattr__(self, key, value):
        if key == "AP_START":
            self.log.append(self.name+" start")
        object.__setattr__(self, key, value)


def fake_board():
    sf = drv.SpikeFaceOverlay.__new__(drv.SpikeFaceOverlay)
    sf.timeout_s, sf.rng, sf.supports_clear_state = .01, np.random.default_rng(0), True
    sf._params = None
    log = []
    sf.blocks = {}
    for name in drv.IP_NAMES:
        if name.startswith("dma"):
            continue
        rm = SimpleNamespace(CTRL=Ctrl(name, log), clear_state=0, total_active_taps_1=0, total_active_taps_2=0)
        sf.blocks[name] = SimpleNamespace(register_map=rm, mmio=SimpleNamespace(array=np.zeros(32768, dtype=np.uint32)))
    sf.dma_in = SimpleNamespace(sendchannel=Channel("tx", log))
    sf.dma_out = SimpleNamespace(recvchannel=Channel("rx", log))
    sf.spike_in_buf, sf.score_out_buf = Buffer(1024, np.uint8), Buffer(40, np.uint32)
    w = {name: np.zeros(shape) for name, shape in drv.WEIGHT_SHAPES.items()}
    sf.configure(w)
    return sf, log


class PSTests(unittest.TestCase):
    def test_code_packing_and_score_decoding(self):
        codes = drv.float_to_code(np.array([-1, -1/128, 0, 127/128]), 7)
        np.testing.assert_array_equal(codes, [128, 255, 0, 127])
        self.assertEqual(int(drv.pack_words(codes)[0]), 0x7F00FF80)
        self.assertEqual(int(drv.pack_words([1])[0]), 1)
        np.testing.assert_array_equal(drv.unpack_scores(np.array([0xFF000000, 0x01000000], dtype=np.uint32)), [-1, 1])

    def test_reject_invalid_fixed_point_values(self):
        for x in [[], [np.nan], [np.inf], [1.0], [.1]]:
            with self.assertRaises(ValueError):
                drv.float_to_code(x, 7)

    def test_scheduler_and_cache_completion(self):
        sf, log = fake_board()
        pred, scores = sf.run_inference(spike_frames=np.zeros((2, 1024)), T=2)
        self.assertEqual(pred, 39)
        self.assertTrue(sf.score_out_buf.invalidated)
        self.assertEqual(log[:6], ["fc2 start", "fc1 start", "conv2 start", "conv1 start", "rx transfer", "tx transfer"])
        self.assertIn("rx wait", log)
        self.assertTrue(all(sf.blocks[k].register_map.clear_state == 0 for k in drv.SPIKING_BLOCKS))
        sf.run_inference(spike_frames=np.zeros((1, 1024)), T=1)
        self.assertTrue(all(sf.blocks[k].register_map.clear_state == 1 for k in drv.SPIKING_BLOCKS))

    def test_invalid_input_never_starts_hardware(self):
        sf, log = fake_board()
        for value in [256, -.1, np.nan, 2]:
            with self.assertRaises(ValueError):
                sf.run_inference(spike_frames=np.full((1, 1024), value), T=1)
        for value in [0, -1, False, 1.5, 501]:
            with self.assertRaises(ValueError):
                sf.run_inference(spike_frames=np.zeros((1, 1024)), T=value)
        with self.assertRaises(ValueError):
            sf.run_inference(image01=np.zeros((16, 64)), T=1)
        self.assertEqual(log, [])

    def test_window_validation_before_mmio_writes(self):
        sf, _ = fake_board()
        sf.blocks["fc1"].mmio.array = np.zeros(16384, dtype=np.uint32)
        with self.assertRaisesRegex(RuntimeError, "address window"):
            sf.configure({k: np.ones(s)/128 for k, s in drv.WEIGHT_SHAPES.items()})
        self.assertEqual(np.count_nonzero(sf.blocks["conv1"].mmio.array), 0)

    def test_dma_error_and_short_transfer(self):
        sf, _ = fake_board()
        channel = sf.dma_out.recvchannel
        channel.error = True
        with self.assertRaisesRegex(RuntimeError, "DMA error"):
            sf._wait_dma(channel, "test", 160)
        channel.error = False
        channel.transfer(sf.score_out_buf)
        with self.assertRaisesRegex(RuntimeError, "expected 159"):
            sf._wait_dma(channel, "test", 159)

    def test_timeout(self):
        sf, _ = fake_board()
        sf.timeout_s = .0001
        with self.assertRaises(TimeoutError):
            sf._wait(lambda: False, "stuck")

    def test_legacy_reset_fallback(self):
        sf, _ = fake_board()
        sf.supports_clear_state = False
        with patch.object(sf, "reset_pl_state") as reset:
            sf.run_inference(spike_frames=np.zeros((1, 1024)), T=1)
            reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
