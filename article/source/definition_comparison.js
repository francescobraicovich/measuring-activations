/* Compare definitions at nominal or matched observed removal. */
(function () {
  'use strict';
  function interpolate(points, target, column = 2) {
    if (!Number.isFinite(target) || target < 0 || target > 1) return null;
    const sorted = [[0, 0], ...points.map(point => [point[1], point[column]])].sort((a, b) => a[0] - b[0]);
    for (let i = 0; i < sorted.length; i++) {
      if (Math.abs(target - sorted[i][0]) < 1e-12) return sorted[i][1];
      if (i > 0 && target < sorted[i][0]) {
        const [x0, y0] = sorted[i - 1], [x1, y1] = sorted[i];
        return y0 + (target - x0) / (x1 - x0) * (y1 - y0);
      }
    }
    return null;
  }
  function ratio(data, model, layer, first, second, target, column = 2) {
    const group = data[model]?.[String(layer)];
    if (!group || !group[first] || !group[second]) return null;
    const a = interpolate(group[first], target, column), b = interpolate(group[second], target, column);
    if (a === null || b === null || a < 0 || b < 0) return null;
    return {value: a === b ? 1 : a / b, first: a, second: b};
  }
  function winners(group, target, column) {
    const points = Object.entries(group).map(([method, series]) => [method, series.find(point => point[0] === target)]);
    const minimum = Math.min(...points.map(([, point]) => point[column]));
    return points.filter(([, point]) => point[column] === minimum);
  }
  function color(value, firstColor, secondColor) {
    const level = Math.max(-1, Math.min(1, Math.log2(value) / 3));
    const terminal = level < 0 ? firstColor : secondColor;
    return [1, 3, 5].map(index => {
      const channel = parseInt(terminal.slice(index, index + 2), 16);
      return Math.round(255 + (channel - 255) * Math.abs(level));
    });
  }
  function foreground(rgb) {
    const linear = rgb.map(channel => {
      const normalized = channel / 255;
      return normalized <= .04045 ? normalized / 12.92 : ((normalized + .055) / 1.055) ** 2.4;
    });
    const luminance = .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
    return luminance < .179 ? '#fff' : '#000';
  }
  if (typeof module !== 'undefined') module.exports = {interpolate, ratio, winners, color};
  if (typeof document === 'undefined') return;
  const {models, layers, fractions, metrics, definitions, colors} = JSON.parse(document.getElementById('comparison-data').textContent);
  const winnerMetric = document.getElementById('winner-metric');
  const metric = document.getElementById('comparison-metric');
  const first = document.getElementById('comparison-first'), second = document.getElementById('comparison-second');
  const direction = document.getElementById('comparison-direction');
  first.value = 'N'; second.value = 'G'; winnerMetric.value = metric.value = metrics[0];
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function header(text, scope) {
    const th = element('th', text);
    th.scope = scope;
    return th;
  }
  function matrix(container, title, removal, fill) {
    container.replaceChildren();
    for (const [model, name] of [['gpt2', 'GPT-2'], ['smollm2', 'SmolLM2']]) {
      const panel = element('div', undefined, 'comparison-table-wrap');
      panel.appendChild(element('h3', name));
      const table = element('table');
      table.appendChild(element('caption', `${name}: ${title}, by layer and ${removal} removal`, 'visually-hidden'));
      const head = element('thead'), headRow = element('tr');
      headRow.appendChild(header('Layer', 'col'));
      for (const fraction of fractions) headRow.appendChild(header(`${Math.round(fraction * 100)}%`, 'col'));
      head.appendChild(headRow); table.appendChild(head);
      const body = element('tbody');
      for (const layer of layers[model]) {
        const row = element('tr');
        row.appendChild(header(layer === -1 ? 'All layers' : String(layer), 'row'));
        for (const fraction of fractions) {
          const cell = element('td');
          fill(cell, model, layer, fraction);
          row.appendChild(cell);
        }
        body.appendChild(row);
      }
      table.appendChild(body); panel.appendChild(table); container.appendChild(panel);
    }
  }
  function outcome(selected) {
    const column = metrics.indexOf(selected) + 2;
    const flip = selected === 'flip_rate';
    const label = flip ? 'Token flip rate' : 'KL divergence';
    const format = value => flip ? `${(value * 100).toPrecision(3)}%` : `${value.toPrecision(4)} nats/token`;
    return {column, label, format};
  }
  function renderWinners() {
    const {column, label, format} = outcome(winnerMetric.value);
    matrix(document.getElementById('winner-matrices'), `lowest ${label}`, 'nominal', (cell, model, layer, fraction) => {
      const best = winners(models[model][String(layer)], fraction, column);
      cell.textContent = best.map(([method]) => method).join('/');
      cell.className = best.length === 1 ? `winner-${best[0][0]}` : 'winner-tie';
      cell.title = best.map(([method, point]) => `${definitions[method]}: ${format(point[column])}; observed removal ${(point[1] * 100).toFixed(1)}%`).join(' / ');
    });
  }
  function renderComparison() {
    const {column, label, format} = outcome(metric.value);
    direction.textContent = `${label} ratio (${first.value} ÷ ${second.value})`;
    const same = first.value === second.value;
    const firstColor = colors[first.value], secondColor = colors[second.value];
    document.getElementById('comparison-gradient').style.background = same
      ? '#fff'
      : `linear-gradient(to right, ${firstColor}, #fff, ${secondColor})`;
    document.getElementById('comparison-better-first').textContent = same ? '' : `${first.value} better`;
    document.getElementById('comparison-better-second').textContent = same ? '' : `${second.value} better`;
    matrix(document.getElementById('comparison-matrices'), `${label} ratio (${first.value} ÷ ${second.value})`, 'mean observed', (cell, model, layer, fraction) => {
      const result = ratio(models, model, layer, first.value, second.value, fraction, column);
      if (result === null) {cell.textContent = '—'; cell.title = 'Not available'; return;}
      const {value} = result;
      cell.textContent = value === Infinity ? '∞' : `${Number(value.toPrecision(value > .95 && value < 1.05 ? 3 : 2))}×`;
      const rgb = color(value, firstColor, secondColor);
      cell.style.backgroundColor = `rgb(${rgb.join(',')})`;
      cell.style.color = foreground(rgb);
      cell.title = `${first.value}: ${format(result.first)}; ${second.value}: ${format(result.second)}; ${label} ratio ${value.toPrecision(4)}`;
    });
  }
  winnerMetric.addEventListener('change', renderWinners);
  for (const select of [metric, first, second]) select.addEventListener('change', renderComparison);
  renderWinners();
  renderComparison();
})();
