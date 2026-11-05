"""Check the article's executable definitions against the experimental masks."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from analysis.large_study.write_report import inline_svg, measurement, substitute
from src.calibrate import fit_thresholds
from src.masks import build_mask

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'analysis/large_study/strength_definitions.js'


def javascript_results(cases):
    program = '''
const fs = require('fs');
require(process.argv[1]);
const methods = globalThis.ActivationStrengthMethods;
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(c => methods.calculate(c.activations, c.norms, c.fraction, c.calibration))));
'''
    process = subprocess.run(['node', '-e', program, str(SCRIPT)], input=json.dumps(cases),
                             text=True, capture_output=True, check=True)
    return json.loads(process.stdout)


def test_interactive_masks_match_experiment_including_ties_and_endpoints():
    generator = np.random.default_rng(17)
    fixtures = [
        # Equal scores and exact zeros exercise stable channel-index ties and
        # strict threshold comparisons, rather than just random easy cases.
        (np.array([0., -1., 1., 2.]), np.array([1., 2., 2., .5]),
         np.array([[0., 0., 0., 0.], [0., 1., 1., 1.], [1., 2., 2., 2.], [2., 3., 3., 3.]])),
        (generator.normal(size=8), generator.uniform(.1, 4, size=8), generator.uniform(0, 5, size=(12, 8))),
    ]
    cases = []
    for activations, norms, calibration in fixtures:
        for fraction in (0., .1, .25, .5, .75, .9, 1.):
            cases.append({'activations': activations.tolist(), 'norms': norms.tolist(),
                          'fraction': fraction, 'calibration': calibration.T.tolist()})
    results = javascript_results(cases)
    for case, result in zip(cases, results):
        values = torch.tensor(case['activations'], dtype=torch.float32)
        calibration = np.array(case['calibration']).T
        thresholds = fit_thresholds(calibration, [case['fraction']])[case['fraction']]
        for method in ('N', 'L', 'G', 'W'):
            actual = build_mask(values, method, case['fraction'], thresholds, case['norms'])
            assert actual.tolist() == result['masks'][method]


def test_weight_changes_only_contribution_norm_selection():
    case = {'activations': [.2, -.4, -.8, 1.2, -1.6, 2.4, .12, -.65],
            'norms': [4, .5, 4, .4, .8, .25, 5, 1.5], 'fraction': .5,
            'calibration': [[scale * (.1 + 1.8 * i / 11) for i in range(12)]
                            for scale in [.1, .25, .5, 1, 2, 3, .15, .8]]}
    changed = case | {'norms': [4, .5, .1, .4, .8, .25, 5, 1.5]}
    original, modified = javascript_results([case, changed])
    for method in ('N', 'L', 'G'):
        assert original['masks'][method] == modified['masks'][method]
    assert original['masks']['W'] != modified['masks']['W']
    assert {method: [i for i, value in enumerate(mask) if value]
            for method, mask in original['masks'].items()} == {
                'N': [4, 5, 6, 7], 'L': [0, 1, 6, 7], 'G': [0, 1, 6], 'W': [1, 3, 5, 6]}


def test_invalid_interactive_input_is_rejected():
    process = subprocess.run(['node', '-e', '''
require(process.argv[1]);
const {calculate, quantile} = globalThis.ActivationStrengthMethods;
for (const operation of [() => calculate([1], [1], -.1), () => calculate([1], [], .5),
  () => calculate([NaN], [1], .5), () => calculate([1], [-1], .5),
  () => quantile([], .5), () => quantile([1], NaN)]) {
  let rejected = false;
  try { operation(); } catch (error) { rejected = error instanceof RangeError; }
  if (!rejected) process.exit(1);
}
''', str(SCRIPT)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


def test_article_claims_reject_missing_or_duplicate_measurements():
    import pandas as pd
    table = pd.DataFrame([{'model': 'gpt2', 'mean': .1}])
    assert measurement(table, model='gpt2')['mean'] == .1
    with pytest.raises(ValueError):
        measurement(table, model='smollm2')
    with pytest.raises(ValueError):
        measurement(pd.concat([table, table]), model='gpt2')
    with pytest.raises(ValueError):
        substitute('An unchecked value: [[unknown]]', {})


def test_interactive_controls_update_the_display_without_a_browser():
    """Exercise actual control handlers in a minimal, isolated DOM interface."""
    import re
    markup = (SCRIPT.parent / 'strength_definitions.html').read_text()
    identifiers = re.findall(r'id="([^"]+)"', markup)
    program = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const identifiers = JSON.parse(process.argv[2]);
const elements = Object.fromEntries(identifiers.map(id => [id, {
  value: '', innerHTML: '', textContent: '', listeners: {},
  addEventListener(event, handler) { this.listeners[event] = handler; },
  setAttribute(name, value) { this[name] = value; }
}]));
elements['strength-definitions'].querySelector = selector => {
  const element = elements[selector.slice(1)];
  assert(element, `Missing element ${selector}`);
  return element;
};
for (const [id, value] of Object.entries({'strength-fraction':'50', 'strength-channel':'2',
  'strength-activation':'-0.8', 'strength-weight':'4'})) elements[id].value = value;
globalThis.document = {getElementById: id => elements[id]};
globalThis.katex = require(process.argv[3]);
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));
const before = elements['strength-mask-rows'].innerHTML;
assert(before.includes('selected-count'));
assert(before.includes('class="katex"'));
elements['strength-weight'].value = '.1';
elements['strength-weight'].listeners.input({target: elements['strength-weight']});
assert.notStrictEqual(before, elements['strength-mask-rows'].innerHTML);
assert(elements['strength-detail'].textContent.includes('W removes'));
elements['strength-fraction'].value = '100';
elements['strength-fraction'].listeners.input();
assert.strictEqual((elements['strength-mask-rows'].innerHTML.match(/8 \/ 8/g) || []).length, 4);
elements['strength-fraction'].value = '0';
elements['strength-fraction'].listeners.input();
assert.strictEqual((elements['strength-mask-rows'].innerHTML.match(/0 \/ 8/g) || []).length, 4);
elements['strength-channel'].value = '0';
elements['strength-channel'].listeners.change();
assert.strictEqual(elements['strength-activation'].value, .2);
elements['strength-activation'].value = '3';
elements['strength-activation'].listeners.input({target: elements['strength-activation']});
assert.strictEqual(elements['strength-activation-value'].value, '3.00');
assert(elements['strength-bars']['aria-label'].includes('magnitude 3.00'));
'''
    process = subprocess.run(['node', '-e', program, str(SCRIPT), json.dumps(identifiers),
                              str(ROOT / 'assets/vendor/katex/katex.min.js')],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


def test_inline_figures_do_not_share_clip_paths(tmp_path):
    from html.parser import HTMLParser
    class Identifiers(HTMLParser):
        def __init__(self):
            super().__init__()
            self.values = []

        def handle_starttag(self, tag, attrs):
            self.values.extend(value for name, value in attrs if name == 'id')

    markup = '<svg id="figure_1"><defs><clipPath id="clip_1"></clipPath></defs><g clip-path="url(#clip_1)"><use href="#figure_1"/></g></svg>'
    figures = []
    for name in ('first', 'second'):
        path = tmp_path / f'{name}.svg'
        path.write_text(markup)
        result = inline_svg(path, 'Title', 'Description')
        assert f'url(#{name}-clip_1)' in result
        assert f'href="#{name}-figure_1"' in result
        figures.append(result)
    parser = Identifiers()
    parser.feed(''.join(figures))
    assert len(parser.values) == len(set(parser.values))

