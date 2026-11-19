(() => {
  if (typeof Plotly === 'undefined') return;
  const {models, metrics, definitions, colors} = JSON.parse(document.getElementById('window-distribution-data').textContent);
  const controls = document.getElementById('window-distribution-controls');
  const chart = document.getElementById('window-distribution-chart');
  const model = document.getElementById('window-model');
  const layer = document.getElementById('window-layer');
  const metric = document.getElementById('window-metric');
  const removal = document.getElementById('window-removal');
  const sample = document.getElementById('window-distribution-sample');
  const definition = document.getElementById('window-definition');

  function setOptions(select, choices, preferred) {
    select.replaceChildren(...choices.map(([value, label]) => new Option(label, value)));
    if (choices.some(([value]) => value === preferred)) select.value = preferred;
  }

  function updateLayers(preferred) {
    const layers = Object.keys(models[model.value]).sort((a, b) => (a === '-1') - (b === '-1') || a - b);
    setOptions(layer, layers.map(value => [value, value === '-1' ? 'All layers' : value]), preferred);
  }

  function updateRemoval(preferred) {
    const fractions = Object.keys(models[model.value][layer.value]).sort((a, b) => a - b);
    setOptions(removal, fractions.map(value => [value, `${Number((100 * value).toPrecision(4))}%`]), preferred);
  }

  function updateDefinitions(preferred) {
    const available = models[model.value][layer.value][removal.value];
    setOptions(definition, Object.entries(definitions).filter(([value]) => value in available), preferred);
  }

  function render() {
    const outcomes = models[model.value][layer.value][removal.value][definition.value];
    const [edges, counts, markers] = outcomes[metric.value];
    const color = colors[definition.value];
    const isPercent = metric.value === 'flip_rate';
    const format = value => isPercent ? `${value.toFixed(1)}%` : Number(value.toPrecision(4)).toString();
    const labels = ['Median', '90th percentile', 'Maximum'];
    const dashes = ['solid', 'dash', 'dot'];
    const traces = [{
      type: 'bar', name: 'Windows', showlegend: false,
      x: counts.map((_, index) => (edges[index] + edges[index + 1]) / 2),
      y: counts, width: counts.map((_, index) => edges[index + 1] - edges[index]),
      customdata: counts.map((_, index) => [format(edges[index]), format(edges[index + 1])]),
      hovertemplate: '%{customdata[0]} to %{customdata[1]}<br>%{y} windows<extra></extra>',
      marker: {color: color + '38', line: {color: color + '59', width: 1}},
    }];
    const height = Math.max(...counts) * 1.07;
    markers.forEach((value, index) => traces.push({
      type: 'scatter', mode: 'lines', name: `${labels[index]} · ${format(value)}`,
      x: [value, value], y: [0, height], line: {color, dash: dashes[index], width: 2},
      hoverinfo: 'skip',
    }));
    return Plotly.react(chart, traces, {
      height: 360, margin: {l: 55, r: 14, t: 15, b: 48},
      font: {family: 'system-ui, sans-serif', color: '#000'},
      xaxis: {title: {text: metrics[metric.value]}, automargin: true, zeroline: false},
      yaxis: {title: {text: 'Number of windows'}, range: [0, height], automargin: true},
      legend: {orientation: 'h', y: 1, yanchor: 'bottom'}, bargap: 0,
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    }, {responsive: true, displayModeBar: false}).then(() => {
      sample.textContent = `Current selection: ${counts.reduce((total, count) => total + count, 0)} windows.`;
    });
  }

  setOptions(metric, Object.entries(metrics), 'flip_rate');
  updateLayers('16');
  updateRemoval('0.75');
  updateDefinitions('G');
  render().then(() => {
    chart.nextElementSibling.style.display = 'none';
    controls.hidden = false;
  });
  model.addEventListener('change', () => {
    updateLayers(layer.value);
    updateRemoval(removal.value);
    updateDefinitions(definition.value);
    render();
  });
  layer.addEventListener('change', () => {
    updateRemoval(removal.value);
    updateDefinitions(definition.value);
    render();
  });
  metric.addEventListener('change', render);
  definition.addEventListener('change', render);
  removal.addEventListener('change', () => {
    updateDefinitions(definition.value);
    render();
  });
})();
