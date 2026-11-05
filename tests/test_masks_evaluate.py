import unittest
from types import SimpleNamespace

import torch

from src.masks import build_mask, random_mask, condition_seed
from src.evaluate import prediction_metrics, evaluate_batch


class MasksTest(unittest.TestCase):
    def test_stable_ties_and_exact_counts(self):
        values = torch.tensor([[[1., 1., 0., 1.], [4., 3., 2., 1.]]])
        expected = torch.tensor([[[True, False, True, False], [False, False, True, True]]])
        self.assertTrue(torch.equal(build_mask(values, "L", .5), expected))
        self.assertTrue(torch.equal(build_mask(values, "W", .5, outgoing_norms=torch.ones(4)), expected))
        for method in "NLGW":
            self.assertFalse(build_mask(values, method, 0).any())
            self.assertTrue(build_mask(values, method, 1).all())

    def test_thresholds_are_strict(self):
        values = torch.tensor([[0., 1., 2.]])
        self.assertTrue(torch.equal(build_mask(values, "N", .5, {"N": torch.tensor([0., 2., 2.])}), torch.tensor([[False, True, False]])))
        self.assertTrue(torch.equal(build_mask(values, "G", .5, {"G": torch.tensor(1.)}), torch.tensor([[True, False, False]])))

    def test_random_counts_and_reproducibility(self):
        counts = torch.tensor([[0, 2, 4], [3, 1, 2]])
        seed = condition_seed("model", 2, "window", "N", .75, 0)
        first = random_mask(counts, 4, seed)
        self.assertTrue(torch.equal(first.sum(-1), counts))
        self.assertTrue(torch.equal(first, random_mask(counts, 4, seed)))
        self.assertNotEqual(seed, condition_seed("model", 2, "window", "N", .75, 1))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): build_mask(torch.ones(2, 3), "L", 1.1)
        with self.assertRaises(ValueError): build_mask(torch.ones(2, 3), "N", .5, {"N": torch.ones(2)})
        with self.assertRaises(ValueError): random_mask(torch.tensor([5]), 4, 1)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 4)
        self.projection = torch.nn.Linear(4, 5, bias=False)
    def forward(self, ids, use_cache=False):
        return SimpleNamespace(logits=self.projection(self.embedding(ids)))


class EvaluationTest(unittest.TestCase):
    def test_metrics_direction_and_identity(self):
        ids = torch.tensor([[0, 1, 0]])
        base = torch.tensor([[[2., 0.], [0., 2.], [0., 0.]]])
        metrics = prediction_metrics(base, base, ids, 0, 2)
        self.assertEqual(metrics["kl"].item(), 0.)
        self.assertEqual(metrics["nll_delta"].item(), 0.)
        changed = torch.zeros_like(base)
        self.assertGreater(prediction_metrics(base, changed, ids, 0, 2)["kl"].item(), 0.)

    def test_hook_cleanup_identity_and_random(self):
        torch.manual_seed(1)
        model = ToyModel().eval()
        ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        windows = [SimpleNamespace(window_id="a"), SimpleNamespace(window_id="b")]
        rows = evaluate_batch(model, model.projection, ids, windows, "toy", 0, "L", 0, None, torch.ones(4), start=0, stop=3)
        self.assertEqual(rows[0]["kl"], 0.)
        self.assertEqual(len(model.projection._forward_pre_hooks), 0)
        rows = evaluate_batch(model, model.projection, ids, windows, "toy", 0, "L", .5, None, torch.ones(4), random_control=True, start=0, stop=3)
        self.assertEqual(rows[0]["observed_fraction"], .5)
        self.assertEqual(len(model.projection._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
