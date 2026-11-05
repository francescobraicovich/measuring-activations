import unittest
import numpy as np
import pandas as pd
import torch
from analysis.followup.localization import restricted_mask, geometry, clustered_difference


class LocalizationTest(unittest.TestCase):
    def test_position_masks_are_disjoint_and_complete(self):
        mask = torch.ones((2, 128, 4), dtype=torch.bool)
        prefix = restricted_mask(mask, 0, 16)
        suffix = restricted_mask(mask, 16, 128)
        self.assertTrue(torch.equal(prefix | suffix, mask))
        self.assertFalse((prefix & suffix).any())
        self.assertEqual(prefix.sum().item(), 2 * 16 * 4)
        with self.assertRaises(ValueError):
            restricted_mask(mask, 16, 129)

    def test_geometric_norm_preserves_cancellation(self):
        activations = torch.tensor([[[1., 1.], [1., 0.]]])
        weight = torch.tensor([[1., 0.], [-1., 0.]])
        norms = torch.linalg.vector_norm(weight, dim=1)
        mask = torch.ones_like(activations, dtype=torch.bool)
        net, individual = geometry(activations, mask, weight, norms, 0, 2)
        self.assertAlmostEqual(net.item(), .5)
        self.assertAlmostEqual(individual.item(), 1.5)
        with self.assertRaises(ValueError):
            geometry(activations, mask[..., :1], weight, norms, 0, 2)

    def test_article_bootstrap_paired_and_weighted(self):
        rows = []
        for window, cluster, d in [('a', 'one', 1.), ('b', 'one', 3.), ('c', 'two', 5.)]:
            rows.extend([{'window_id': window, 'cluster_id': cluster, 'method': 'W', 'kl': d},
                         {'window_id': window, 'cluster_id': cluster, 'method': 'L', 'kl': 0.}])
        frame = pd.DataFrame(rows)
        result = clustered_difference(frame, 'kl', 'W', 'L')
        self.assertEqual(result['articles'], 2)
        self.assertAlmostEqual(result['difference'], 3.)
        self.assertTrue(np.isfinite([result['low'], result['high']]).all())
        with self.assertRaises(ValueError):
            clustered_difference(frame.iloc[:-1], 'kl', 'W', 'L')


if __name__ == '__main__':
    unittest.main()
