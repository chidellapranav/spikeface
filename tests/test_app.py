import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from app.engine import Model, evaluate, fixture_frames, image_frames, valid_frames
from app.server import infer_request
from Spike_old.stress_sim.reference import golden_arrays


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model.golden()
        cls.frames = fixture_frames()

    def test_fixture_trace_and_repeated_images(self):
        expected = golden_arrays(Path(__file__).resolve().parents[1] / "Spike_old/Spikeface_HLS/golden_fc2.h")["GOLDEN_FC2_SCORES"]
        first = self.model.infer(self.frames)
        self.model.infer(1-self.frames[:3])
        second = self.model.infer(self.frames)
        np.testing.assert_array_equal(first["trace_codes"], expected*(1 << 24))
        self.assertEqual(first["trace_codes"], second["trace_codes"])
        self.assertEqual(first["prediction"], 12)

    def test_model_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"fixture.npz"
            self.model.save(path)
            loaded = Model.load(path)
            self.assertEqual(loaded.metadata, self.model.metadata)
            for k in self.model.weights:
                np.testing.assert_array_equal(loaded.weights[k], self.model.weights[k])

    def test_invalid_weights_and_labels(self):
        for invalid in [np.zeros((64, 1024), dtype=float), np.full((64, 1024), 128), np.zeros(3, dtype=int)]:
            w = dict(self.model.weights, fc1=invalid)
            with self.assertRaises(ValueError):
                Model(w)
        with self.assertRaises(ValueError):
            Model(self.model.weights, {"labels": ["only one"]})

    def test_input_validation_before_cast(self):
        for invalid in [np.full((1, 1024), 256), np.full((1, 1024), -.5), np.full((1, 1024), np.nan),
                        np.zeros((0, 1024)), np.zeros((2, 16, 64)), np.zeros((501, 1024))]:
            with self.assertRaises(ValueError):
                valid_frames(invalid)

    def test_encoding_is_seeded_and_brightness_extremes(self):
        for value in [0, 255, 128]:
            data = io.BytesIO()
            Image.new("L", (40, 50), value).save(data, format="PNG")
            content = data.getvalue()
            a = image_frames(io.BytesIO(content), 4, 9)
            b = image_frames(io.BytesIO(content), 4, 9)
            np.testing.assert_array_equal(a, b)
            if value in (0, 255):
                self.assertTrue(np.all(a == (value == 255)))
        for steps in [False, 0, 501, 1.5]:
            with self.assertRaises(ValueError):
                image_frames(io.BytesIO(content), steps)

    def test_evaluation_paired_scores(self):
        trace = self.model.infer(self.frames)["trace_codes"]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"heldout_format_fixture.npz"
            np.savez(path, frames=self.frames[None], labels=np.array([12]), score_codes=np.asarray(trace)[None])
            result = evaluate(self.model, path)
            self.assertEqual(result["correct"], 1)
            self.assertEqual(result["score_code_mismatches"], 0)
            self.assertLess(result["wilson_95"][0], .5)
            np.savez(path, frames=self.frames[None], labels=np.array([40]))
            with self.assertRaises(ValueError):
                evaluate(self.model, path)

    def test_request_validation(self):
        for request in [[], {}, {"image": "!bad!"}, {"image": "AA==", "seed": -1}, {"image": "AA==", "seed": True}]:
            with self.assertRaises((ValueError, OSError)):
                infer_request(self.model, request)
        data = io.BytesIO()
        Image.new("L", (32, 32), 0).save(data, format="PNG")
        response = infer_request(self.model, {"image": base64.b64encode(data.getvalue()).decode(), "timesteps": 1, "seed": 0})
        self.assertEqual(response["timesteps"], 1)
        self.assertNotIn("trace_codes", response)
        self.assertEqual(len(response["scores"]), 40)


if __name__ == "__main__":
    unittest.main()
