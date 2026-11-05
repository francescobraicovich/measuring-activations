import unittest

import numpy as np
import torch

from analysis.expanded.methods import FRACTIONS, build_mask, fit_calibration, random_mask
from src.masks import build_mask as original_mask


class ExpandedMethodsTest(unittest.TestCase):
    def test_calibration_uses_training_means_and_reports_ties(self):
        train = np.array([[1., 10.], [3., 10.], [5., 14.]], dtype=np.float32)
        calibration = fit_calibration(train, [0.5])
        torch.testing.assert_close(calibration["means"], torch.tensor([3., 34. / 3]))
        centered = np.abs(train - calibration["means"].numpy())
        np.testing.assert_allclose(calibration["thresholds"][.5]["N_centered"], np.quantile(centered, .5, axis=0))
        threshold = calibration["thresholds"][.5]["G_centered"].item()
        self.assertEqual(calibration["tie_fractions"][.5]["G_centered"], np.mean(centered == threshold))
        self.assertEqual(FRACTIONS[-3:], (.95, .99, 1.))

    def test_centered_masks_and_zero_intervention_scoring(self):
        values = torch.tensor([[2., 11., 4., 3.]])
        means = torch.tensor([2., 10., 2., 0.])
        thresholds = {"N_centered": torch.tensor([.5, 2., 2., 4.]), "G_centered": torch.tensor(2.)}
        self.assertTrue(torch.equal(build_mask(values, "N_centered", .5, thresholds, means=means), torch.tensor([[True, True, False, True]])))
        self.assertTrue(torch.equal(build_mask(values, "G_centered", .5, thresholds, means=means), torch.tensor([[True, True, False, False]])))
        self.assertTrue(torch.equal(build_mask(values, "L_centered", .5, means=means), torch.tensor([[True, True, False, False]])))
        self.assertTrue(torch.equal(build_mask(values, "W_centered", .5, outgoing_norms=torch.tensor([1., 3., 1., 1.]), means=means), torch.tensor([[True, False, True, False]])))

    def test_stable_ties_endpoints_and_original_equivalence(self):
        values = torch.tensor([[1., 1., 0., 1.]])
        means = torch.zeros(4)
        for method in ("L", "W"):
            args = {"outgoing_norms": torch.ones(4)}
            expected = torch.tensor([[True, False, True, False]])
            self.assertTrue(torch.equal(build_mask(values, method + "_centered", .5, means=means, **args), expected))
            self.assertTrue(torch.equal(build_mask(values, method, .5, **args), original_mask(values, method, .5, **args)))
        for method in ("N_centered", "L_centered", "G_centered", "W_centered"):
            self.assertFalse(build_mask(values, method, 0, means=means).any())
            self.assertTrue(build_mask(values, method, 1, means=means).all())

    def test_random_control_matches_counts(self):
        values = torch.tensor([[0., 1., 2.], [2., 1., 0.]])
        mask = build_mask(values, "L_centered", .5, means=torch.zeros(3))
        control = random_mask(mask.sum(-1), 3, 42)
        self.assertTrue(torch.equal(control.sum(-1), mask.sum(-1)))
        self.assertTrue(torch.equal(control, random_mask(mask.sum(-1), 3, 42)))

    def test_invalid_calibration_and_masks(self):
        for data in (np.empty((0, 2)), np.array([[float('nan')]]), np.ones(2)):
            with self.assertRaises(ValueError): fit_calibration(data)
        with self.assertRaises(ValueError): fit_calibration(np.ones((2, 2)), split="test")
        with self.assertRaises(ValueError): fit_calibration(np.ones((2, 2)), [1.1])
        with self.assertRaises(ValueError): build_mask(torch.ones(2, 3), "L_centered", .5, means=torch.ones(2))
        with self.assertRaises(ValueError): build_mask(torch.ones(2, 3), "N_centered", .5, means=torch.ones(3))
        with self.assertRaises(ValueError): build_mask(torch.ones(2, 3), "unknown", .5)


if __name__ == "__main__":
    unittest.main()
