import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from analysis.depth_extension import run
from analysis.expanded.methods import METHODS, build_mask, fit_calibration
from src.data import Window
from src.masks import random_mask


class DepthExtensionTest(unittest.TestCase):
    def test_masks_and_random_controls_match_token_counts(self):
        values = np.array([[0, 1, 2, 3], [2, 0, 3, 1], [1, 2, 0, 3]], dtype=np.float32)
        calibration = fit_calibration(values, [.75])
        activations = torch.tensor([[0., 2., 2., 4.], [3., 1., 0., 2.]])
        for method in METHODS:
            mask = build_mask(activations, method, .75, calibration['thresholds'][.75], torch.ones(4), calibration['means'])
            counts = mask.sum(-1)
            if method in ('L', 'W', 'L_centered', 'W_centered'):
                self.assertEqual(counts.tolist(), [3, 3])
            self.assertTrue(torch.equal(random_mask(counts, 4, 9).sum(-1), counts))

    def test_gate_rejects_changed_artifact_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / 'artifact'
            artifact.write_text('original')
            gate = root / 'gate.json'
            with patch.object(run, 'source_hashes', return_value={'source': 'first'}):
                run.write_json(gate, {'sources': {'source': 'first'}, 'entries': [{'artifacts': {str(artifact): run.file_digest(artifact)}}]})
                run.verify_gate(gate)
                artifact.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                    run.verify_gate(gate)
            with patch.object(run, 'source_hashes', return_value={'source': 'second'}):
                with self.assertRaisesRegex(ValueError, 'Source files changed'):
                    run.verify_gate(gate)

    def test_test_stage_requires_both_gates_before_model_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {'source': 'digest'}
            entries = [{'model': m, 'layer': l, 'artifacts': {}} for m, layers in run.LAYERS.items() for l in layers]
            run.write_json(root / 'calibration_complete.json', {'sources': sources, 'entries': entries})
            with patch.object(run, 'source_hashes', return_value=sources), patch.object(run, 'run_layer') as execute, \
                 patch.object(sys, 'argv', ['run', 'test', '--output', str(root)]):
                with self.assertRaises(FileNotFoundError):
                    run.main()
                execute.assert_not_called()
                run.write_json(root / 'validation_complete.json', {'sources': sources, 'entries': [{'model': 'gpt2', 'layer': 4, 'artifacts': {}}]})
                with self.assertRaisesRegex(ValueError, 'Incomplete validation'):
                    run.main()
                execute.assert_not_called()

    def test_checkpoint_reuse_and_signature_mismatch(self):
        class Model:
            def __call__(self, ids, use_cache=False):
                return type('Output', (), {'logits': torch.zeros((*ids.shape, 2))})()
        windows = [Window([0] * 128, 'window', 'article', 'validation')]
        metrics = {'kl': 0., 'baseline_nll': 1., 'nll_delta': 0., 'flip_rate': 0.,
                   'observed_fraction': .75, 'removed_abs_fraction': .1, 'removed_squared_fraction': .1,
                   'removed_weighted_fraction': .1, 'removed_centered_abs_fraction': .1,
                   'removed_centered_weighted_fraction': .1}
        def fake_evaluate(*args):
            fraction = args[7]
            return [{**metrics, 'observed_fraction': fraction}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / 'calibration.pt'
            torch.save({'metadata': {}}, calibration_path)
            directory = root / 'run'
            with patch.object(run, 'model_and_windows', return_value=(Model(), windows)), \
                 patch.object(run, 'projection', return_value=object()), \
                 patch.object(run, 'outgoing_norms', return_value=torch.ones(4)), \
                 patch.object(run, 'validate_hook'), patch.object(run, 'source_hashes', return_value={'source': 'digest'}), \
                 patch.object(run, 'evaluate', side_effect=fake_evaluate) as evaluate:
                run.run_layer('gpt2', 0, 'validation', 1, {}, calibration_path, directory, 1, 'cpu')
                self.assertEqual(evaluate.call_count, 18)
                evaluate.reset_mock()
                run.run_layer('gpt2', 0, 'validation', 1, {}, calibration_path, directory, 1, 'cpu')
                evaluate.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'signature changed'):
                    run.run_layer('gpt2', 0, 'validation', 1, {}, calibration_path, directory, 2, 'cpu')
                checkpoint = directory / 'batch_0000.json'
                rows = json.loads(checkpoint.read_text())
                rows.pop()
                checkpoint.write_text(json.dumps(rows))
                with self.assertRaisesRegex(ValueError, 'Invalid checkpoint'):
                    run.run_layer('gpt2', 0, 'validation', 1, {}, calibration_path, directory, 1, 'cpu')


if __name__ == '__main__':
    unittest.main()
