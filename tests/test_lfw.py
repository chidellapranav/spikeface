import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import numpy as np
from PIL import Image

REFERENCE = Path(__file__).resolve().parents[2]/"trmp_spike"
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("snntorch") is not None and REFERENCE.is_dir()


@unittest.skipUnless(AVAILABLE, "Requires optional torch/snnTorch and the reference checkout")
class LFWAppTests(unittest.TestCase):
    def test_reference_model_and_metadata(self):
        from app.lfw import load_reference
        model, state, labels, meta = load_reference(REFERENCE)
        self.assertEqual(len(labels), 53)
        self.assertEqual(tuple(state["fc1.weight"].shape), (53, 4096))
        self.assertFalse(model.training)
        self.assertTrue(meta["hardware_compatible"])
        self.assertEqual(meta["hls_config"], [.9375, .5, .9375, .5, .9375])

    def test_labeled_image_database(self):
        from app.lfw import evaluate_lfw, load_reference
        _, _, labels, _ = load_reference(REFERENCE)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"images"
            folder = root/labels[0]
            folder.mkdir(parents=True)
            Image.new("L", (64, 64), 127).save(folder/"test.png")
            with redirect_stdout(io.StringIO()):
                report = evaluate_lfw(REFERENCE, images_root=root, output=Path(d)/"accuracy.json")
            self.assertEqual(report["images"], 1)
            self.assertEqual(len(report["confusion_matrix"]), 53)
            self.assertEqual(len(report["image_files"]), 1)
            Image.new("L", (32, 32), 127).save(folder/"small.png")
            with self.assertRaisesRegex(ValueError, "64x64"):
                evaluate_lfw(REFERENCE, images_root=root, output=Path(d)/"bad.json")

    def test_label_map_validation(self):
        from app.lfw import evaluate_lfw
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d)/"class_names.npy", np.array(["wrong"]))
            with self.assertRaisesRegex(ValueError, "class_names"):
                evaluate_lfw(REFERENCE, dataset_dir=d, output=Path(d)/"out.json")

    def test_pynq_requires_bitstream(self):
        from app.lfw import evaluate_lfw
        with self.assertRaisesRegex(ValueError, "bitstream"):
            evaluate_lfw(REFERENCE, backend="pynq")


if __name__ == "__main__":
    unittest.main()
