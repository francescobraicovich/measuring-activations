/* Exact synthetic demonstration of the four experimental selection rules. */
(function (global) {
  'use strict';
  const scales = [0.1, 0.25, 0.5, 1, 2, 3, 0.15, 0.8];
  const histories = scales.map(scale => Array.from({length: 12}, (_, i) => scale * (0.1 + 1.8 * i / 11)));
  const initialActivations = [0.2, -0.4, -0.8, 1.2, -1.6, 2.4, 0.12, -0.65];
  const initialNorms = [4, 0.5, 4, 0.4, 0.8, 0.25, 5, 1.5];

  function quantile(values, fraction) {
    if (!values.length || !values.every(Number.isFinite) || !Number.isFinite(fraction) || fraction < 0 || fraction > 1) {
      throw new RangeError('Quantile requires finite observations and a fraction in [0, 1]');
    }
    const sorted = [...values].sort((a, b) => a - b);
    const position = fraction * (sorted.length - 1);
    const lower = Math.floor(position), upper = Math.ceil(position);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
  }

  function calculate(activations, norms, fraction, calibration = histories) {
    if (!Array.isArray(activations) || !activations.length || !activations.every(Number.isFinite) ||
        !Array.isArray(norms) || norms.length !== activations.length || !norms.every(x => Number.isFinite(x) && x >= 0) ||
        !Array.isArray(calibration) || calibration.length !== activations.length ||
        !Number.isFinite(fraction) || fraction < 0 || fraction > 1) {
      throw new RangeError('Expected finite channel values, nonnegative matching norms, and a fraction in [0, 1]');
    }
    const magnitude = activations.map(Math.abs);
    const weighted = magnitude.map((value, i) => value * norms[i]);
    const channelThresholds = calibration.map(values => quantile(values, fraction));
    const pooledThreshold = quantile(calibration.flat(), fraction);
    const count = Math.floor(fraction * activations.length);
    function ranks(scores) {
      const order = scores.map((_, i) => i).sort((i, j) => scores[i] - scores[j] || i - j);
      const result = Array(order.length);
      order.forEach((channel, index) => { result[channel] = index + 1; });
      return result;
    }
    const magnitudeRanks = ranks(magnitude), weightedRanks = ranks(weighted);
    function thresholdMask(thresholds) {
      return magnitude.map((value, i) => fraction === 1 || (fraction > 0 && value < thresholds[i]));
    }
    return {
      magnitude, weighted, channelThresholds, pooledThreshold, magnitudeRanks, weightedRanks,
      masks: {
        N: thresholdMask(channelThresholds),
        L: magnitudeRanks.map(rank => rank <= count),
        G: thresholdMask(magnitude.map(() => pooledThreshold)),
        W: weightedRanks.map(rank => rank <= count)
      }
    };
  }
  global.ActivationStrengthMethods = {quantile, calculate, histories, initialActivations, initialNorms};
  if (typeof document === 'undefined') return;
  const root = document.getElementById('strength-definitions');
  if (!root) return;
  const get = id => root.querySelector('#' + id);
  const activations = [...initialActivations], norms = [...initialNorms];
  const names = 'ABCDEFGH'.split('');
  const format = value => value.toFixed(2).replace('-', '−');
  const math = expression => global.katex.renderToString(expression, {
    throwOnError: true, strict: 'error', trust: false, output: 'htmlAndMathml'
  });
  const magnitudeLabel = math(String.raw`\lvert a\rvert`);
  const normLabel = math(String.raw`\lVert\mathbf{w}\rVert_2`);
  const definitions = {
    N: ['Neuron percentile', math(String.raw`\lvert a\rvert\,/\,q_j(p)`)],
    L: ['Layer percentile at token', math(String.raw`\lvert a\rvert\quad(\mathrm{rank})`)],
    G: ['Layer percentile, pooled', math(String.raw`\lvert a\rvert\,/\,q_{\mathrm{pool}}(p)`)],
    W: ['Layer percentile, weight adjusted', math(String.raw`\lvert a\rvert\,\lVert\mathbf{w}\rVert_2\quad(\mathrm{rank})`)]
  };

  function render() {
    const fraction = Number(get('strength-fraction').value) / 100;
    const selected = Number(get('strength-channel').value);
    const result = calculate(activations, norms, fraction);
    get('strength-fraction-value').value = Math.round(100 * fraction) + '%';
    get('strength-activation-value').value = format(activations[selected]);
    get('strength-weight-value').value = format(norms[selected]);
    get('strength-mask-rows').innerHTML = Object.keys(definitions).map(method => {
      const cells = names.map((name, i) => {
        const removed = result.masks[method][i];
        const value = method === 'W' ? result.weighted[i] : result.magnitude[i];
        const reference = method === 'N' ? '/ ' + format(result.channelThresholds[i]) :
          method === 'G' ? '/ ' + format(result.pooledThreshold) :
          'rank ' + (method === 'L' ? result.magnitudeRanks[i] : result.weightedRanks[i]);
        return `<td class="${removed ? 'removed' : 'retained'} ${i === selected ? 'edited-channel' : ''}" aria-label="${method}, neuron ${name}: ${removed ? 'selected for zeroing' : 'retained'}, score ${format(value)}, ${reference}"><span class="selection-symbol">${removed ? '×' : '○'}</span><span>${format(value)}</span><small>${reference}</small></td>`;
      }).join('');
      const count = result.masks[method].filter(Boolean).length;
      return `<tr><th scope="row"><strong>${method}</strong> ${definitions[method][0]}<small>${definitions[method][1]}</small></th>${cells}<td class="selected-count">${count} / ${names.length}</td></tr>`;
    }).join('');

    const bars = names.map((name, i) => `<div class="activation-column ${i === selected ? 'is-edited' : ''}"><span class="activation-value">${format(result.magnitude[i])}</span><div class="activation-track"><div class="activation-bar" style="height:${result.magnitude[i] / 5 * 100}%"></div></div><strong>${name}</strong><span class="weight-value">${normLabel} ${format(norms[i])}</span></div>`).join('');
    get('strength-bars').innerHTML = '<div class="bar-axis-label">' + magnitudeLabel + '<br><span>0–5</span></div>' + bars;
    get('strength-bars').setAttribute('aria-label', names.map((name, i) => `Neuron ${name}: magnitude ${format(result.magnitude[i])}, output norm ${format(norms[i])}`).join('; '));
    get('strength-detail').textContent = 'Neuron ' + names[selected] + ': ' + Object.keys(definitions).map(method => method + ' ' + (result.masks[method][selected] ? 'removes' : 'retains')).join(' · ') + '. N/G counts may differ from the nominal quota.';
  }
  get('strength-fraction').addEventListener('input', render);
  get('strength-channel').addEventListener('change', () => {
    const selected = Number(get('strength-channel').value);
    get('strength-activation').value = activations[selected];
    get('strength-weight').value = norms[selected];
    render();
  });
  get('strength-activation').addEventListener('input', event => {
    activations[Number(get('strength-channel').value)] = Number(event.target.value);
    render();
  });
  get('strength-weight').addEventListener('input', event => {
    norms[Number(get('strength-channel').value)] = Number(event.target.value);
    render();
  });
  render();
})(globalThis);
