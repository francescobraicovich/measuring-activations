import unittest
from types import SimpleNamespace

import torch

from scripts.ablate import paired_summary
from src.evaluate import evaluate_batch, prediction_metrics


class StatisticsTest(unittest.TestCase):
    def test_paired_summary_weights_windows_and_averages_random_repetitions(self):
        rows = []
        for window, cluster, effect in [("a", "first", 1.), ("b", "first", 2.), ("c", "second", 6.)]:
            common = dict(model="gpt2", layer=6, fraction=.75, window_id=window, cluster_id=cluster)
            rows.append(dict(common, method="L", repetition=0, kl=effect + 2))
            for repetition, random_kl in enumerate([1., 2., 3.]):
                rows.append(dict(common, method="L_random", repetition=repetition, kl=random_kl))
        summary = paired_summary(rows, replicates=100, seed=7).iloc[0]
        self.assertEqual(summary.paired_kl_difference, 3.)
        self.assertEqual(summary.clusters, 2)
        self.assertGreaterEqual(summary.lower, 1.5)
        self.assertLessEqual(summary.upper, 6.)
        self.assertTrue(paired_summary(rows, replicates=100, seed=7).equals(paired_summary(rows, replicates=100, seed=7)))
        with self.assertRaises(ValueError):
            paired_summary(
                [row for row in rows if not (row["method"] == "L_random" and row["window_id"] == "c")], replicates=2)

    def test_default_scoring_positions_include_16_and_126_only(self):
        ids = torch.zeros((1, 128), dtype=torch.long)
        base = torch.zeros((1, 128, 2))
        changed = base.clone()
        changed[:, 15, 0] = 10
        changed[:, 127, 0] = 10
        self.assertEqual(prediction_metrics(base, changed, ids)["kl"].item(), 0.)
        changed[:, 16, 0] = 10
        changed[:, 126, 0] = 10
        result = prediction_metrics(base, changed, ids)["kl"].item()
        single = prediction_metrics(base, changed, ids, 16, 17)["kl"].item()
        self.assertAlmostEqual(result, 2 * single / 111, places=6)

    def test_baseline_is_reused(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 3)
                self.calls = 0
            def forward(self, ids, use_cache=False):
                self.calls += 1
                return SimpleNamespace(logits=self.projection(torch.nn.functional.one_hot(ids, 3).float()))
        model = Model().eval()
        ids = torch.tensor([[0, 1, 2]])
        baseline = model(ids).logits
        evaluate_batch(model, model.projection, ids, [SimpleNamespace(window_id="a")],
                       "model", 0, "L", .5, None, torch.ones(3), start=0, stop=2,
                       baseline_logits=baseline)
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(model.projection._forward_pre_hooks), 0)

    def test_random_control_is_independent_of_batch_order(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(8, 3, bias=False)
            def forward(self, ids, use_cache=False):
                values = torch.arange(8, dtype=torch.float32).expand(*ids.shape, 8) + ids.unsqueeze(-1)
                return SimpleNamespace(logits=self.projection(values))
        torch.manual_seed(4)
        model = Model().eval()
        ids = torch.tensor([[0, 1, 2], [1, 2, 0]])
        windows = [SimpleNamespace(window_id="a"), SimpleNamespace(window_id="b")]
        def run(tokens, batch):
            return evaluate_batch(model, model.projection, tokens, batch, "model", 0, "L", .5,
                                  None, torch.ones(8), random_control=True, start=0, stop=2)
        forward = run(ids, windows)
        reverse = run(ids.flip(0), windows[::-1])
        for metric in forward[0]:
            self.assertAlmostEqual(forward[0][metric], reverse[1][metric], places=6)


if __name__ == "__main__":
    unittest.main()
