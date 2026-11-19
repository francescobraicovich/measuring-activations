"""Check the essay's definition matrices against saved study results."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'article/source'


def test_matrix_metrics_and_figure_order():
    source = (ROOT / 'article/index.html').read_text()
    assert re.findall(r'<(?:figcaption|p class="note figure-caption")>Figure (\d+)\.', source) == list('123456')
    assert 'projection-geometry-title' not in source
    assert 'I would start with G' not in source
    data = re.search(r'<script type="application/json" id="comparison-data">(.*?)</script>', source).group(1)
    assert set(json.loads(data)['models']) == {'gpt2', 'smollm2'}
    program = r'''
const assert = require('node:assert/strict'), fs = require('node:fs');
const script = process.argv[1];
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const api = require(script);
assert.equal(api.interpolate([[.5,.5,2], [1,1,4]], .75), 3);
assert.equal(api.interpolate([[.5,.5,2]], .75), null);
assert.equal(api.interpolate([[.5,.5,2]], -.1), null);
assert(Math.abs(api.ratio(data.models,'gpt2',6,'N','G',.75).value - .579) < .002);
assert.deepEqual(api.color(1, data.colors.N, data.colors.G), [255, 255, 255]);
assert.deepEqual(api.color(1 / 8, data.colors.N, data.colors.G), [215, 32, 0]);
assert.deepEqual(api.color(8, data.colors.N, data.colors.G), [37, 51, 103]);
for (const [model, layers] of Object.entries(data.models))
  for (const layer of Object.keys(layers))
    for (const first of Object.keys(data.definitions))
      for (const second of Object.keys(data.definitions))
        for (const fraction of data.fractions)
          for (const column of [2,3]) {
            const result = api.ratio(data.models,model,layer,first,second,fraction,column);
            assert(result && Number.isFinite(result.value) && result.value > 0);
          }
class Element {
  constructor() { this.children=[]; this.style={}; this.listeners={}; this.value=''; this.textContent=''; }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children=[]; }
  addEventListener(name, fn) { this.listeners[name]=fn; }
}
const elements = Object.fromEntries(
  ['comparison-first','comparison-second','comparison-metric','winner-metric','comparison-direction','comparison-gradient','comparison-better-first','comparison-better-second','comparison-matrices','winner-matrices']
    .map(id => [id,new Element()])
);
elements['comparison-data']={textContent:JSON.stringify(data)};
global.document={getElementById:id=>elements[id],createElement:()=>new Element()};
delete require.cache[require.resolve(script)]; require(script);
const table = id => elements[id].children[0].children[1];
const cell = id => table(id).children[2].children[0].children[2];
assert.equal(elements['comparison-first'].value,'N');
assert.equal(elements['comparison-second'].value,'G');
assert.equal(elements['comparison-metric'].value,'kl');
assert.equal(cell('winner-matrices').textContent,'N');
assert.equal(cell('comparison-matrices').textContent,'0.93×');
for (const id of ['winner-matrices', 'comparison-matrices']) {
  assert.equal(elements[id].children.length, 2);
  for (const panel of elements[id].children) {
    assert.equal(panel.children.length, 2);
    const [, head, body] = panel.children[1].children;
    assert.equal(head.children[0].children.length, 13);
    assert.equal(body.children[5].children.length, 13);
    assert.equal(body.children[5].children[0].textContent, 'All layers');
  }
}
elements['comparison-metric'].value='flip_rate';
elements['comparison-metric'].listeners.change();
assert.equal(cell('winner-matrices').textContent,'N');
assert.equal(cell('comparison-matrices').textContent,'1.02×');
assert.match(cell('comparison-matrices').title,/Token flip rate ratio 1\.025/);
assert.equal(elements['comparison-direction'].textContent,'Token flip rate ratio (N ÷ G)');
assert.equal(elements['comparison-better-first'].textContent,'N better');
assert.equal(elements['comparison-better-second'].textContent,'G better');
elements['comparison-second'].value='W';
elements['comparison-second'].listeners.change();
assert.equal(elements['comparison-better-second'].textContent,'W better');
assert.equal(elements['comparison-gradient'].style.background, 'linear-gradient(to right, #d72000, #fff, #1bb6af)');
const previousRatio = cell('comparison-matrices');
elements['winner-metric'].value='flip_rate';
elements['winner-metric'].listeners.change();
assert.equal(cell('winner-matrices').textContent,'G');
assert.equal(cell('comparison-matrices'),previousRatio);
'''
    subprocess.run(['node', '-e', program, str(HERE / 'definition_comparison.js')],
                   input=data, text=True, check=True)
