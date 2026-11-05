"""Deterministic, article-balanced held-out WikiText windows."""
import hashlib
import json
import re

from src.data import Window

DATASET_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sample_windows(rows, tokenizers, count=512, window_size=128):
    """Select matching article rounds, using disjoint token windows per model.

    Each round visits articles in source order. Only articles with a complete
    window in every tokenizer participate, and each round requires that window
    to exist for every tokenizer. This gives identical article multiplicities.
    """
    if not tokenizers or count < 1 or window_size < 2:
        raise ValueError('tokenizers and positive count are required; window_size must be at least 2')
    articles, current, texts = [], [], []
    for row in rows:
        text = row['text']
        texts.append(text)
        if re.match(r'^\s*=\s*[^=\n]+?\s*=\s*$', text):
            if current:
                articles.append('\n'.join(current))
            current = [text]
        elif text.strip():
            current.append(text)
    if current:
        articles.append('\n'.join(current))
    encoded = {name: [list(tokenizer(article, add_special_tokens=False)['input_ids'])
                      for article in articles] for name, tokenizer in tokenizers.items()}
    capacities = [min(len(encoded[name][i]) // window_size for name in tokenizers)
                  for i in range(len(articles))]
    available = sum(capacities)
    if available < count:
        raise ValueError(f'Only {available} common complete windows available; requested {count}')
    selected = []
    for round_index in range(max(capacities)):
        for i, capacity in enumerate(capacities):
            if capacity > round_index:
                selected.append((i, round_index * window_size))
                if len(selected) == count:
                    break
        if len(selected) == count:
            break
    windows = {}
    model_manifest = {}
    for name, tokenizer in tokenizers.items():
        result, records = [], []
        for i, offset in selected:
            article_id = f'test-article-{i:05d}'
            window_id = f'{article_id}-token-{offset:07d}'
            ids = encoded[name][i][offset:offset + window_size]
            result.append(Window(ids, window_id, article_id, 'test'))
            records.append({'window_id': window_id, 'article_id': article_id, 'token_offset': offset,
                            'input_ids_sha256': _digest(json.dumps(ids, separators=(',', ':')))})
        windows[name] = result
        backend = getattr(tokenizer, 'backend_tokenizer', None)
        representation = backend.to_str() if backend is not None else json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(',', ':'))
        model_manifest[name] = {'tokenizer_sha256': _digest(representation), 'windows': records,
                                'available_windows': sum(len(ids) // window_size for ids in encoded[name]),
                                'eligible_articles': sum(len(ids) >= window_size for ids in encoded[name])}
    manifest = {'dataset': 'Salesforce/wikitext', 'dataset_config': 'wikitext-2-raw-v1',
                'dataset_revision': DATASET_REVISION, 'split': 'test',
                'dataset_text_sha256': _digest(json.dumps(texts, ensure_ascii=False, separators=(',', ':'))),
                'window_size': window_size, 'window_count': count, 'article_count': len({i for i, _ in selected}),
                'total_articles': len(articles), 'common_eligible_articles': sum(c > 0 for c in capacities),
                'common_available_windows': available, 'selection': 'source-order article round-robin, common tokenizer capacity',
                'models': model_manifest}
    return windows, manifest


def load_sample(tokenizers, count=512, window_size=128, revision=DATASET_REVISION):
    if revision != DATASET_REVISION:
        raise ValueError('Unexpected dataset revision')
    from datasets import load_dataset
    rows = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test', revision=revision)
    return sample_windows(rows, tokenizers, count, window_size)
