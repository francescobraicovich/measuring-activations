#!/usr/bin/env python3
"""Targeted exploratory replication on 32 additional WikiText test articles."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.expanded.methods import build_mask
from analysis.expanded.run import evaluate
from scripts.ablate import file_digest, validate_hook
from src.data import Window
from src.model_hooks import outgoing_norms, projection

OUTPUT = ROOT / 'results/followup/replication'
CALIBRATION = ROOT / 'results/depth_extension/calibration/gpt2_0/calibration.pt'
MODEL_ID = 'openai-community/gpt2'
METHODS = ('L', 'W', 'W_centered')


def select_windows(rows, tokenizer, count=32):
    """First 128-token window in the first 32 eligible articles after article 3.

    Article boundaries exactly mirror src.data.windows_from_rows.
    """
    articles, current = [], []
    for row in rows:
        text = row['text']
        if re.match(r'^\s*=\s*[^=\n]+?\s*=\s*$', text):
            if current:
                articles.append('\n'.join(current))
            current = [text]
        elif text.strip():
            current.append(text)
    if current:
        articles.append('\n'.join(current))
    windows = []
    for index, article in enumerate(articles):
        if index < 4:
            continue
        ids = tokenizer(article, add_special_tokens=False)['input_ids']
        if len(ids) < 128:
            continue
        article_id = f'test-article-{index:05d}'
        windows.append(Window(list(ids[:128]), article_id + '-token-0000000', article_id, 'test'))
        if len(windows) == count:
            break
    if len(windows) != count:
        raise ValueError(f'Only {len(windows)} eligible new articles; {count} required')
    return windows


def paired(frame, first, second, metric='kl'):
    left = frame[frame.method.eq(first)][['window_id', 'cluster_id', metric]]
    right = frame[frame.method.eq(second)][['window_id', 'cluster_id', metric]]
    if left.window_id.duplicated().any() or right.window_id.duplicated().any():
        raise ValueError('Duplicate window in paired comparison')
    values = left.merge(right, on=['window_id', 'cluster_id'], suffixes=('_first', '_second'), validate='one_to_one')
    if len(values) != 32 or len(values) != len(left) or len(values) != len(right):
        raise ValueError('The paired comparison must contain all 32 articles')
    difference = values[metric + '_first'].to_numpy() - values[metric + '_second'].to_numpy()
    rng = np.random.default_rng(1729)
    draws = difference[rng.integers(0, len(difference), (1000, len(difference)))].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return {'first': first, 'second': second, 'metric': metric, 'difference': float(difference.mean()),
            'lower': float(low), 'upper': float(high), 'positive_articles': int((difference > 0).sum()),
            'negative_articles': int((difference < 0).sum()), 'articles': len(values), 'replicates': 1000}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calibration = torch.load(CALIBRATION, map_location='cpu', weights_only=False)
    metadata = calibration['metadata']
    assert metadata['split'] == 'train' and metadata['layer'] == 0 and metadata['model'] == MODEL_ID
    token = AutoTokenizer.from_pretrained(MODEL_ID, revision=metadata['model_commit'])
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test', revision=metadata['dataset_revision'])
    windows = select_windows(dataset, token)
    original = pd.read_parquet(ROOT / 'results/depth_extension/test/gpt2_0/metrics.parquet')
    if set(w.passage_id for w in windows) & set(original.cluster_id):
        raise ValueError('New article selection overlaps the original test articles')
    selection = {'scope': 'targeted exploratory replication selected after original results',
        'selection': 'first complete 128-token window from first 32 eligible test articles after article 3',
        'scoring_positions': '16–126 inclusive', 'mask_positions': '0–127 inclusive',
        'calibration_sha256': file_digest(CALIBRATION), 'model_commit': metadata['model_commit'],
        'dataset_revision': metadata['dataset_revision'], 'fraction': .75,
        'conditions': ['L', 'W', 'W_centered', 'L_random', 'W_random', 'W_centered_random', 'identity', 'all_zero'],
        'bootstrap_seed': 1729, 'bootstrap_replicates': 1000,
        'windows': [{'window_id': w.window_id, 'article_id': w.passage_id,
                     'input_ids_sha256': hashlib.sha256(np.asarray(w.input_ids, dtype=np.int64).tobytes()).hexdigest()} for w in windows]}
    frozen = OUTPUT / 'selection.json'
    if frozen.exists() and json.loads(frozen.read_text()) != selection:
        raise ValueError('Frozen selection changed')
    if not frozen.exists():
        frozen.write_text(json.dumps(selection, indent=2) + '\n')
    print('Selection frozen:', len(windows), 'new articles', flush=True)
    device = 'mps'
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=metadata['model_commit'], torch_dtype=torch.float32).to(device).eval()
    if getattr(model.config, '_commit_hash', None) != metadata['model_commit']:
        raise ValueError('Model revision mismatch')
    module = projection(model, 'gpt2', 0)
    norms = outgoing_norms(model, 'gpt2', 0).to(device)
    validate_hook(model, module, torch.tensor([windows[0].input_ids], device=device), len(norms))
    rows = []
    for offset in range(0, len(windows), 2):
        batch = windows[offset:offset+2]
        ids = torch.tensor([w.input_ids for w in batch], device=device)
        def check(_module, inputs):
            a = inputs[0]
            for method in METHODS:
                mask = build_mask(a, method, .75, calibration['thresholds'][.75], norms, calibration['means'])
                if not bool((mask.sum(-1) == int(.75 * a.shape[-1])).all()):
                    raise RuntimeError(f'Quota mismatch for {method}')
            return inputs
        handle = module.register_forward_pre_hook(check)
        try:
            with torch.inference_mode():
                baseline = model(ids, use_cache=False).logits
        finally:
            handle.remove()
        for method in (*METHODS, *(m + '_random' for m in METHODS), 'identity', 'all_zero'):
            is_random = method.endswith('_random')
            underlying = method.removesuffix('_random') if is_random else ('L' if method in ('identity', 'all_zero') else method)
            fraction = 0. if method == 'identity' else 1. if method == 'all_zero' else .75
            result = evaluate(model, module, ids, batch, 'gpt2', 0, underlying, fraction, calibration, norms, is_random, 0, baseline)
            for window, metrics in zip(batch, result):
                rows.append({'model': 'gpt2', 'layer': 0, 'window_id': window.window_id, 'cluster_id': window.passage_id,
                             'method': method, 'fraction': fraction, 'repetition': 0, **metrics})
        print(f'Evaluated {min(offset+2, len(windows))}/{len(windows)} articles', flush=True)
    table = pd.DataFrame(rows)
    if not np.isfinite(table.select_dtypes('number')).all().all():
        raise ValueError('Non-finite metric')
    if not table[table.method.eq('identity')].kl.abs().le(1e-6).all():
        raise ValueError('Identity differs')
    for method in METHODS:
        a = table[table.method.eq(method)].sort_values('window_id')
        b = table[table.method.eq(method + '_random')].sort_values('window_id')
        if not np.array_equal(a.observed_fraction.to_numpy(), b.observed_fraction.to_numpy()):
            raise ValueError('Random control differs in counts')
    table.to_csv(OUTPUT / 'metrics.csv', index=False)
    table.to_parquet(OUTPUT / 'metrics.parquet', index=False)
    comparisons = [paired(table, a, b, metric) for metric in ('kl', 'nll_delta', 'flip_rate')
                   for a, b in (('W', 'L'), ('W_centered', 'W'), ('W_centered', 'L'))]
    pd.DataFrame(comparisons).to_csv(OUTPUT / 'paired_comparisons.csv', index=False)
    (OUTPUT / 'run.json').write_text(json.dumps({'status': 'passed', 'scope': 'exploratory',
        'selection_sha256': file_digest(frozen), 'calibration_sha256': file_digest(CALIBRATION),
        'source_sha256': file_digest(Path(__file__)), 'window_count': len(windows),
        'device': device, 'dtype': 'float32'}, indent=2) + '\n')
    print('Saved replication results', flush=True)


if __name__ == '__main__':
    main()
