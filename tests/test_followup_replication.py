import unittest
import pandas as pd
from analysis.followup.replication import paired, select_windows

class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': list(range(140 if 'enough' in text else 20))}

class ReplicationTest(unittest.TestCase):
    def test_article_boundaries_and_skip_short(self):
        rows = []
        for i in range(38):
            rows.extend([{'text': f'= Article {i} ='}, {'text': 'short' if i == 5 else 'enough'}])
        windows = select_windows(rows, FakeTokenizer())
        self.assertEqual(len(windows), 32)
        self.assertEqual(windows[0].passage_id, 'test-article-00004')
        self.assertEqual(windows[1].passage_id, 'test-article-00006')
        self.assertTrue(all(len(w.input_ids) == 128 and w.window_id.endswith('0000000') for w in windows))
        self.assertEqual(len({w.passage_id for w in windows}), 32)

    def test_paired_rejects_missing_and_reports_direction(self):
        rows = []
        for i in range(32):
            for method, kl in [('W', 2.), ('L', 1.)]:
                rows.append({'method': method, 'window_id': str(i), 'cluster_id': str(i), 'kl': kl})
        frame = pd.DataFrame(rows)
        result = paired(frame, 'W', 'L')
        self.assertEqual((result['difference'], result['lower'], result['upper'], result['positive_articles']), (1., 1., 1., 32))
        with self.assertRaises(ValueError):
            paired(frame.iloc[:-1], 'W', 'L')

if __name__ == '__main__':
    unittest.main()
