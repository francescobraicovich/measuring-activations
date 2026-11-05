import unittest

from analysis.large_study.data import sample_windows


class Tokenizer:
    def __init__(self, extra=0):
        self.extra = extra

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': list(range(len(text.split()) + self.extra))}

    def get_vocab(self):
        return {'extra': self.extra}


class SampleTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{'text': ' = First = '}, {'text': 'one two three four five six seven eight nine'},
                     {'text': ' = Second = '}, {'text': 'one two three four five'},
                     {'text': ' = Third = '}]
        self.tokenizers = {'a': Tokenizer(), 'b': Tokenizer(2)}

    def test_balanced_disjoint_reproducible_selection(self):
        windows, manifest = sample_windows(self.rows, self.tokenizers, 4, 4)
        self.assertEqual((windows, manifest), sample_windows(self.rows, self.tokenizers, 4, 4))
        expected = ['test-article-00000', 'test-article-00001',
                    'test-article-00000', 'test-article-00001']
        for result in windows.values():
            self.assertEqual([window.passage_id for window in result], expected)
            self.assertEqual(len({window.window_id for window in result}), 4)
            self.assertTrue(all(len(window.input_ids) == 4 for window in result))
        self.assertEqual(manifest['article_count'], 2)
        self.assertEqual(manifest['common_available_windows'], 5)
        self.assertEqual(manifest['models']['a']['windows'][2]['token_offset'], 4)

    def test_invalid_requests(self):
        for count, size in [(0, 4), (6, 4), (1, 1)]:
            with self.assertRaises(ValueError):
                sample_windows(self.rows, self.tokenizers, count, size)
        with self.assertRaises(ValueError):
            sample_windows(self.rows, {}, 1, 4)


if __name__ == '__main__':
    unittest.main()
