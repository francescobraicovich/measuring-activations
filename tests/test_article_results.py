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

    html = (ROOT / 'article/index.html').read_text()
    parser = Identifiers()
    parser.feed(html)
    for identifier in ('data-explorer', 'chart', 'depth-chart', 'model', 'layer', 'metric', 'fraction', 'ranking', 'scale', 'rows', 'pairedrows'):
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


def test_model_switch_preserves_available_selections():
    import subprocess

    program = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync(process.argv[1], 'utf8');
const script = html.match(/<\/main><script>([\s\S]*?)<\/script>/)[1];
const elements = {};
for (const id of ['model','layer','metric','fraction','ranking','scale','chart','depth-chart','depth-title','crosshead','crossrows','pairedrows','rows']) {
  elements[id] = {id, _value: '', _html: '', options: [],
    get value() {return this._value}, set value(value) {this._value=String(value)},
    get selectedOptions() {return [{textContent: this.value}]},
    get innerHTML() {return this._html}, set innerHTML(value) {this._html=value; this.options=[]; this._value=''},
    appendChild(child) {this.options.push(String(child.value)); if (this.options.length === 1) this.value=child.value}};
}
elements.ranking.value = 'original';
elements.scale.value = 'log';
const plots = {};
vm.runInNewContext(script, {document: {getElementById: id => elements[id], createElement: () => ({})},
  Plotly: {react: (id, traces, layout, config) => {plots[id] = {traces, layout, config}}}});
assert.strictEqual(plots.chart.layout.xaxis.type, 'linear');
assert(html.includes('<option value="log">Logarithmic</option><option value="linear">Linear</option>'));
for (const id of ['chart', 'depth-chart']) {
  assert.strictEqual(plots[id].layout.hovermode, 'x unified');
  assert.strictEqual(plots[id].layout.legend.x, .5);
  assert.strictEqual(plots[id].layout.legend.xanchor, 'center');
  assert.strictEqual(plots[id].layout.xaxis.spikemode, 'across');
  assert.strictEqual(plots[id].layout.yaxis.type, 'log');
  assert.strictEqual(plots[id].layout.xaxis.fixedrange, true);
  assert.strictEqual(plots[id].layout.yaxis.fixedrange, true);
  assert.strictEqual(plots[id].config.displayModeBar, false);
  assert.strictEqual(plots[id].config.scrollZoom, false);
  assert.strictEqual(plots[id].config.doubleClick, false);
  assert.strictEqual(plots[id].traces.find(t => t.name && t.name.startsWith('N ·')).line.color, '#d72000');
  assert.strictEqual(plots[id].traces.find(t => t.name && t.name.startsWith('G ·')).line.color, '#253367');
}
const comparison = plots.chart.traces.find(t => t.name && t.name.startsWith('N ·') && t.showlegend === false);
assert(comparison.x.length > 0);
assert(comparison.hovertemplate.includes('Observed removal'));
assert(comparison.hovertemplate.includes('color:#d72000'));
assert.strictEqual(comparison.hoverlabel.font.color, '#d72000');
const measured = plots.chart.traces.find(t => t.name && t.name.startsWith('N ·') && t.showlegend !== false);
assert(comparison.y.every(value => measured.y.includes(value)));
assert(comparison.customdata.every(([observed]) => measured.x.includes(observed)));
assert.strictEqual(plots.chart.layout.xaxis.hoverformat, '.1%');
assert(!plots.chart.traces.some(trace => trace.name === 'Identity' || trace.name === 'All zero'));
assert(!plots['depth-chart'].traces.some(trace => trace.name === 'Identity' || trace.name === 'All zero'));
assert(elements.rows.innerHTML.includes('Identity'));
assert(!html.includes('id="download"'));
assert(!elements.metric.options.includes('nll_delta'));
assert(!elements.metric.options.includes('baseline_nll'));
assert.strictEqual(plots.chart.traces.find(t => t.name === 'Random').line.color, '#000000');
assert(!html.includes('id="horizontal-scale"'));
elements.layer.value = '-1'; elements.layer.onchange();
assert.strictEqual(plots.chart.traces.find(trace => trace.name && trace.name.startsWith('N ·')).x.length, 16);
elements.fraction.value = '0.9'; elements.fraction.onchange();
elements.metric.value = 'flip_rate'; elements.metric.onchange();
assert.strictEqual(plots.chart.layout.yaxis.tickformat, '.0%');
assert(plots.chart.traces.find(t => t.showlegend === false && t.name && t.name.startsWith('N ·')).hovertemplate.includes('.2%'));
elements.ranking.value = 'both'; elements.ranking.onchange();
elements.scale.value = 'log'; elements.scale.onchange();
elements.model.value = 'gpt2'; elements.model.onchange();
assert.strictEqual(elements.layer.value, '-1');
assert.strictEqual(plots.chart.traces.find(trace => trace.name && trace.name.startsWith('N ·')).x.length, 16);
assert.strictEqual(Number(elements.fraction.value), .9);
assert(elements['depth-title'].textContent.includes('Token flip rate across layers at 90% nominal removal'));
assert.strictEqual(elements.metric.value, 'flip_rate');
assert.strictEqual(elements.ranking.value, 'both');
assert.strictEqual(elements.scale.value, 'log');
elements.layer.value = '6'; elements.layer.onchange();
elements.model.value = 'smollm2'; elements.model.onchange();
assert(elements.layer.options.includes(elements.layer.value));
elements.layer.value = '16'; elements.layer.onchange();
elements.model.value = 'gpt2'; elements.model.onchange();
assert.strictEqual(elements.layer.value, '6');
assert.strictEqual(Number(elements.fraction.value), .9);
assert(elements['depth-title'].textContent.includes('Token flip rate across layers at 90% nominal removal'));
'''
    subprocess.run(['node', '-e', program, str(ROOT / 'article/index.html')], check=True)


def test_sidenotes_use_blog_footnotes_and_interactions():
    import subprocess

    html = (ROOT / 'article/index.html').read_text()
    for number in (1, 2):
        assert html.count(f'id="fn{number}"') == 1
        assert html.count(f'id="fnref{number}"') == 1
        assert f'href="#fn{number}"' in html
    assert '<article class="article prose">' in html
    assert '<mark>as well as sequentially across all layers</mark><sup class="footnote-ref">' in html
    assert '<mark>not sufficient to establish a general trend across model generations</mark><sup class="footnote-ref">' in html
    assert 'notes-enhanced .footnotes' in html
    program = r'''
const assert = require('assert'), events = {}, created = [];
function element() { return {handlers:{},style:{},hidden:false,childNodes:[],classList:{add(){}},
  addEventListener(name, fn){this.handlers[name]=fn}, setAttribute(name,v){this[name]=v},
  append(...items){this.childNodes.push(...items)}, remove(){}, focus(){}, contains(){return false},
  getBoundingClientRect(){return {left:100,right:200,top:100,bottom:120,height:100,width:235}},
  closest(){return {getBoundingClientRect(){return {left:100,right:800}}}}}; }
const reference = element(), anchor = {hash:'#fn1',textContent:'[1]',id:'fnref1'}, highlighted = {nodeName:'MARK'};
reference.previousSibling = highlighted;
reference.querySelector=()=>anchor;reference.before=trigger=>{reference.trigger=trigger};
const target = {cloneNode(){return {childNodes:[],querySelectorAll(){return []}}}};
globalThis.document={querySelectorAll(selector){return selector==='.footnote-ref'?[reference]:[]},
 getElementById(){return target},createElement(){const e=element();created.push(e);return e},
 body:{append(){},classList:{add(){}}},addEventListener(name,fn){events[name]=fn},querySelector(){return null}};
globalThis.addEventListener=()=>{};globalThis.innerWidth=1200;globalThis.innerHeight=800;
require(process.argv[1]);
const trigger=reference.trigger, panel=created[0];
assert.strictEqual(trigger.childNodes[0], highlighted);
trigger.handlers.pointerenter({pointerType:'mouse'});
assert.strictEqual(panel.hidden,false);assert.strictEqual(trigger['aria-expanded'],'true');
assert.strictEqual(panel.style.left,'818px');
events.keydown({key:'Escape'});assert.strictEqual(panel.hidden,true);
trigger.handlers.click();assert.strictEqual(panel.hidden,false);
events.pointerdown({target:{}});assert.strictEqual(panel.hidden,true);
'''
    subprocess.run(['node', '-e', program, str(ROOT / 'article/source/footnotes.js')], check=True)
