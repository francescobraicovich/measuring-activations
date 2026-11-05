import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.calibrate import fit_thresholds, load_calibration, save_calibration
from src.data import Window, windows_from_rows
from src.model_hooks import collect_activations, outgoing_norms


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


class ConvProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[3., 4.], [0., 2.], [1., 0.]]))

    def forward(self, x):
        return x @ self.weight


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = ConvProjection()
        self.transformer = SimpleNamespace(h=[SimpleNamespace(mlp=SimpleNamespace(c_proj=self.output))])
        self.config = SimpleNamespace(n_inner=3, n_embd=2)

    def forward(self, input_ids, use_cache=False):
        values = torch.stack([input_ids.float(), input_ids.float() + 1, input_ids.float() + 2], -1)
        return self.output(values)


class DataCalibrationTest(unittest.TestCase):
    def test_windows_keep_article_boundaries_and_are_repeatable(self):
        rows = [{"text": " = First = "}, {"text": "one two three four five six"},
                {"text": " = Second = "}, {"text": "seven eight nine ten eleven twelve"}]
        windows = windows_from_rows(rows, Tokenizer(), "train", 4, 4)
        self.assertEqual(windows, windows_from_rows(rows, Tokenizer(), "train", 4, 4))
        self.assertEqual([w.passage_id for w in windows],
                         ["train-article-00000"] * 2 + ["train-article-00001"] * 2)
        with self.assertRaises(ValueError):
            windows_from_rows(rows, Tokenizer(), "train", 100, 4)

    def test_projection_norms_and_capture(self):
        model = Model()
        torch.testing.assert_close(outgoing_norms(model, "gpt2", 0), torch.tensor([5., 2., 1.]))
        # A single channel's output delta has the corresponding weight norm.
        delta = model.output(torch.tensor([1., 0., 0.])) - model.output(torch.zeros(3))
        self.assertEqual(torch.linalg.vector_norm(delta).item(), 5.)
        windows = [Window([1, 2], "one", "article", "train")]
        with tempfile.TemporaryDirectory() as directory:
            data = collect_activations(model, "gpt2", 0, windows, Path(directory) / "values.npy")
            np.testing.assert_array_equal(data, [[1, 2, 3], [2, 3, 4]])
            self.assertFalse(model.output._forward_pre_hooks)

    def test_thresholds_require_train_and_preserve_provenance(self):
        data = np.array([[1., 10.], [3., 20.]], dtype=np.float32)
        thresholds = fit_thresholds(data, [.5])
        torch.testing.assert_close(thresholds[.5]["N"], torch.tensor([2., 15.]))
        self.assertEqual(thresholds[.5]["G"].item(), 6.5)
        with self.assertRaises(ValueError):
            fit_thresholds(data, [.5], split="test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.pt"
            save_calibration(path, thresholds, {"split": "train"})
            self.assertEqual(load_calibration(path)["metadata"]["split"], "train")


if __name__ == "__main__":
    unittest.main()
