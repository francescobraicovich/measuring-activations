"""Check the article's data explorer and reported comparisons."""
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_explorer_precedes_results_and_retains_controls():
    class Identifiers(HTMLParser):
        def __init__(self):
            super().__init__()
            self.values = []

        def handle_starttag(self, tag, attrs):
            self.values.extend(value for name, value in attrs if name == 'id')

    html = (ROOT / 'blog.html').read_text()
    parser = Identifiers()
    parser.feed(html)
    for identifier in ('data-explorer', 'chart', 'depth-chart', 'model', 'layer', 'metric', 'fraction', 'ranking', 'scale', 'horizontal-scale', 'rows', 'pairedrows'):
        assert parser.values.count(identifier) == 1
    assert html.index('id="definitions"') < html.index('id="chart"') < html.index('id="results"') < html.index('id="depth-chart"')


def test_reported_75_percent_rankings():
    data = pd.read_csv(ROOT / 'results/large_study/data/summary.csv')
    selected = data[(data.fraction == .75) & data.method.isin(['N', 'L', 'G', 'W']) & data.metric.isin(['kl', 'flip_rate'])]
    for model, winner in [('smollm2', 'G'), ('gpt2', 'N')]:
        for metric in ('kl', 'flip_rate'):
            results = selected[(selected.model == model) & (selected.metric == metric)]
            assert len(results) == 20
            assert all(group.loc[group['mean'].idxmin(), 'method'] == winner for _, group in results.groupby('layer'))


def test_horizontal_scale_switches_without_removing_table_rows():
    import subprocess

    program = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync(process.argv[1], 'utf8');
const script = html.match(/<\/main><script>([\s\S]*?)<\/script>/)[1];
const elements = {};
for (const id of ['model','layer','metric','fraction','ranking','scale','horizontal-scale','download','chart','depth-chart','crosshead','crossrows','pairedrows','rows']) {
  elements[id] = {id, _value: '', get value() {return this._value}, set value(value) {this._value=String(value)}, innerHTML: '', appendChild(child) {if (!this.value) this.value=child.value}};
}
elements.ranking.value = 'original';
elements.scale.value = 'linear';
elements['horizontal-scale'].value = 'linear';
const plots = {};
const context = {document: {getElementById: id => elements[id], createElement: () => ({})},
  Plotly: {react: (id, traces, layout) => {plots[id] = {traces, layout}}}};
vm.runInNewContext(script, context);
assert.strictEqual(plots.chart.layout.xaxis.type, 'linear');
assert(plots.chart.traces.some(trace => trace.x.includes(0)));
assert(elements.rows.innerHTML.includes('Identity'));
elements['horizontal-scale'].value = 'log';
elements['horizontal-scale'].onchange();
assert.strictEqual(plots.chart.layout.xaxis.type, 'log');
assert.strictEqual(plots.chart.layout.xaxis.autorange, true);
assert(plots.chart.traces.length > 0);
assert(plots.chart.traces.every(trace => trace.x.every(value => value === null || value > 0)));
assert(elements.rows.innerHTML.includes('Identity'));
assert.notStrictEqual(plots['depth-chart'].layout.xaxis.type, 'log');
elements['horizontal-scale'].value = 'linear';
elements['horizontal-scale'].onchange();
assert.strictEqual(plots.chart.layout.xaxis.type, 'linear');
assert(plots.chart.traces.some(trace => trace.x.includes(0)));
assert.strictEqual(plots.chart.layout.xaxis.range[0], 0);
assert.strictEqual(plots.chart.layout.xaxis.range[1], 1);
'''
    subprocess.run(['node', '-e', program, str(ROOT / 'blog.html')], check=True)
