import unittest

import pandas as pd

from analysis.expanded.paired_analysis import analyze, paired


def fixture():
    rows = []
    for window, cluster, value in [('a', 'one', 1.), ('b', 'one', 2.), ('c', 'two', 4.)]:
        for base in ('N', 'L', 'G', 'W'):
            for suffix, delta in (('', 0.), ('_centered', 1.), ('_random', 2.), ('_centered_random', 3.)):
                for repetition in range(3 if suffix.endswith('_random') else 1):
                    rows.append({'model': 'gpt2', 'layer': 6, 'window_id': window, 'cluster_id': cluster,
                                 'method': base + suffix, 'fraction': .25, 'repetition': repetition,
                                 'kl': value + delta + (repetition if suffix.endswith('_random') else 0),
                                 'observed_fraction': .75, 'removed_abs_fraction': .2,
                                 'removed_centered_abs_fraction': .3})
    return pd.DataFrame(rows)


class PairedAnalysisTest(unittest.TestCase):
    def test_pairs_and_random_repetition_mean(self):
        frame = fixture()
        rows = analyze(frame)
        self.assertEqual(len(rows), 14)
        centered = rows[(rows['first'] == 'N_centered') & (rows['second'] == 'N')].iloc[0]
        self.assertEqual(centered.fraction, .25)
        self.assertAlmostEqual(centered.paired_kl_difference, 1.)
        self.assertAlmostEqual(centered.lower, 1.)
        self.assertAlmostEqual(centered.upper, 1.)
        random = rows[(rows['first'] == 'N') & (rows['second'] == 'N_random')].iloc[0]
        self.assertAlmostEqual(random.paired_kl_difference, -3.)
        self.assertEqual(random.articles, 2)
        self.assertEqual(random.windows, 3)

    def test_missing_or_inconsistent_pairs_fail(self):
        frame = fixture()
        missing = frame[~((frame.method == 'L_centered') & (frame.window_id == 'c'))]
        with self.assertRaisesRegex(ValueError, 'Unmatched windows'):
            paired(missing, 'L_centered', 'L', .25)
        changed = frame.copy()
        changed.loc[(changed.method == 'L_centered') & (changed.window_id == 'c'), 'cluster_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            paired(changed, 'L_centered', 'L', .25)
        changed = frame.copy()
        changed.loc[changed.index[0], 'kl'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            analyze(changed)


if __name__ == '__main__':
    unittest.main()
