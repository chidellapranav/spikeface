import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from PS.run_lfw import load_bundle, smoke


class BundleTests(unittest.TestCase):
    def test_hash_detects_corrupt_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            payload = path / 'lfw_bundle.npz'
            np.savez(payload, images=np.zeros((2, 64, 64), dtype=np.float32))
            (path / 'manifest.json').write_text(json.dumps({
                'format_version': 1,
                'payload_sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}))
            _, arrays = load_bundle(folder)
            self.assertEqual(arrays['images'].shape, (2, 64, 64))
            with payload.open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaises(ValueError):
                load_bundle(folder)

    def test_smoke_detects_state_leak_on_repeated_image(self):
        class Board:
            def __init__(self):
                self.calls = []

            def infer(self, image, timesteps, return_trace):
                self.calls.append(int(image[0, 0]))
                trace = np.zeros((timesteps, 53), dtype=np.float32)
                if len(self.calls) == 3:
                    trace[0, 0] = 1
                return 0, trace[-1], trace

        board = Board()
        arrays = {'images': np.stack([np.zeros((64, 64)), np.ones((64, 64))]),
                  'smoke_indices': np.array([0, 1]),
                  'expected_traces': np.zeros((2, 2, 53), dtype=np.float32)}
        records = smoke(board, arrays, 2)
        self.assertEqual(board.calls, [0, 1, 0])
        self.assertEqual([r['passed'] for r in records], [True, True, False])


if __name__ == '__main__':
    unittest.main()
