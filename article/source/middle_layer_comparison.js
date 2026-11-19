(() => {
  if (typeof Plotly === 'undefined') return;
  const panels = document.getElementById('middle-layer-panels');
  const controls = document.getElementById('middle-layer-controls');
  const button = controls.querySelector('button');
  const methods = ['random', 'G', 'N'];

  for (const method of methods) {
    const key = document.createElement('span');
    key.textContent = label(method);
    key.className = 'curve-key';
    key.style.setProperty('--curve-color', colors[method]);
    if (method === 'random') key.style.setProperty('--curve-style', 'dashed');
    controls.append(key);
  }

  function drawPanel(model, layer, title) {
    const panel = document.createElement('div');
    panels.append(panel);
    const rows = data.filter(row => row.model === model && row.layer === layer);
    const traces = [];
    for (const method of methods) {
      const measurements = rows.filter(row => row.method === method);
      const observed = new Map(measurements.filter(row => row.metric === 'observed_fraction')
        .map(row => [row.fraction, row.mean]));
      const rates = measurements.filter(row => row.metric === 'flip_rate').sort((a, b) => a.fraction - b.fraction);
      const x = rates.map(row => observed.get(row.fraction));
      traces.push({
        type: 'scatter', mode: 'lines', fill: 'toself', fillcolor: colors[method], opacity: .12,
        line: {width: 0}, hoverinfo: 'skip',
        x: x.concat(x.slice().reverse()),
        y: rates.map(row => row.upper).concat(rates.map(row => row.lower).reverse()),
      }, {
        type: 'scatter', mode: 'lines', name: label(method),
        x: [0, ...x], y: [0, ...rates.map(row => row.mean)],
        line: {color: colors[method], dash: method === 'random' ? 'dash' : 'solid', width: 2},
        hovertemplate: '%{x:.1%} removed<br>%{y:.2%} flipped<extra>%{fullData.name}</extra>',
      });
    }
    return Plotly.newPlot(panel, traces, {
      title: {text: title, x: 0, xanchor: 'left', font: {size: 14}},
      font: {family: 'system-ui, sans-serif', size: 12, color: '#000'},
      height: 265, margin: {l: 53, r: 16, t: 36, b: 49}, showlegend: false,
      xaxis: {title: {text: 'Mean observed removal'}, range: [0, 1], dtick: .25, tickformat: '.0%', fixedrange: true},
      yaxis: {title: {text: 'Token flip rate'}, range: [0, .26], dtick: .05, tickformat: '.0%', fixedrange: true},
    }, {responsive: true, displayModeBar: false});
  }

  Promise.all([drawPanel('gpt2', 6, 'GPT-2 · layer 6'), drawPanel('smollm2', 16, 'SmolLM2 · layer 16')]).then(() => {
    panels.nextElementSibling.style.display = 'none';
    controls.hidden = false;
    button.addEventListener('click', () => {
      const visible = button.getAttribute('aria-pressed') === 'false';
      button.setAttribute('aria-pressed', visible);
      for (const panel of panels.children) {
        const bands = panel.data.flatMap((trace, index) => trace.fill === 'toself' ? [index] : []);
        Plotly.restyle(panel, {visible}, bands);
      }
    });
  });
})();
