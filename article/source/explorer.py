#!/usr/bin/env python3
"""Build a self-contained interactive view of the large-study summaries."""
import argparse
import json
from pathlib import Path

import pandas as pd
from plotly.offline.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {'model', 'layer', 'method', 'fraction', 'metric', 'mean', 'lower', 'upper', 'clusters', 'replicates'}


def build(input_path, output_path):
    paths = sorted(input_path.glob('*/summary.csv'))
    if (input_path / 'summary.csv').exists():
        paths.append(input_path / 'summary.csv')
    paths += sorted((input_path.parent / 'all_layers_combined').glob('*/summary.csv'))
    if not paths:
        raise ValueError(f'No summary.csv files found under {input_path}')
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f'Missing summary columns: {sorted(missing)}')
    keys = ['model', 'layer', 'method', 'fraction', 'metric']
    if data[keys].isna().any().any() or data.duplicated(keys).any():
        raise ValueError('Summary contains missing or duplicate identifiers')
    numeric = ['fraction', 'mean', 'lower', 'upper', 'clusters', 'replicates']
    if data[numeric].isna().any().any():
        raise ValueError('Summary contains missing values')
    paired_paths = sorted(input_path.glob('*/paired_comparisons.csv'))
    paired_paths += sorted((input_path.parent / 'all_layers_combined').glob('*/paired_comparisons.csv'))
    if (input_path / 'paired_comparisons.csv').exists():
        paired_paths.append(input_path / 'paired_comparisons.csv')
    paired = pd.concat([pd.read_csv(path) for path in paired_paths], ignore_index=True) if paired_paths else pd.DataFrame()
    if not paired.empty:
        required_paired = REQUIRED | {'reference'}
        if required_paired - set(paired.columns):
            raise ValueError(f'Missing paired comparison columns: {sorted(required_paired - set(paired.columns))}')
        paired_keys = keys + ['reference']
        if paired[paired_keys].isna().any().any() or paired.duplicated(paired_keys).any() or paired[numeric].isna().any().any():
            raise ValueError('Paired comparisons contain missing or duplicate values')
    payload = data[sorted(REQUIRED)].to_json(orient='records', double_precision=15)
    paired_payload = paired.to_json(orient='records', double_precision=15)
    html = (TEMPLATE.replace('__PLOTLY__', get_plotlyjs())
            .replace('__DATA__', payload.replace('</', '<\\/'))
            .replace('__PAIRED__', paired_payload.replace('</', '<\\/')))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    print(f'Wrote {output_path} from {len(data)} summary rows')


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Activation ablation study</title><script>__PLOTLY__</script>
<style>
body{font:15px system-ui,sans-serif;margin:0;color:#172131;background:#f5f7fa}main{max-width:1250px;margin:auto;padding:28px}h1{margin-bottom:4px}h2{margin-top:28px}p{line-height:1.5;color:#445166}.controls{display:flex;gap:18px;flex-wrap:wrap;background:white;padding:16px;border:1px solid #dce3eb;border-radius:8px}label{display:grid;gap:5px;font-weight:600}select{padding:7px;min-width:145px;border:1px solid #aebacc;border-radius:4px;background:white}#chart,#depth-chart{height:540px;background:white;margin-top:18px}.scroll{overflow:auto;background:white;border:1px solid #dce3eb}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #e6ebf0}th{background:#eff3f8;position:sticky;top:0}th:first-child,td:first-child{text-align:left}.note{font-size:13px}.empty{padding:20px}.result-table-title{padding:10px 8px;text-align:left;font:600 14px/1.4 system-ui,sans-serif;color:#000;background:#fff;white-space:normal}
</style></head><body><main>
<h1>Activation ablation study</h1><p>Explore held-out measurements by model, layer, method and nominal removal fraction.</p>
<div class="plot-frame">
<h3>Model output changes as more activations are removed</h3>
<div class="controls"><label>Model<select id="model"></select></label><label>Layer<select id="layer"></select></label><label>Outcome<select id="metric"></select></label><label>Ranking<select id="ranking"><option value="original">Original</option><option value="centered">Mean-centered</option><option value="both">Both</option></select></label><label>Vertical scale<select id="scale"><option value="log">Logarithmic</option><option value="linear">Linear</option></select></label></div>
<div id="chart" aria-label="Interactive ablation outcome plot"></div>
</div>
<p class="note figure-caption">Figure 2. For an all-layer run, observed removal and removed-mass values are averages across layers; sample sizes and fraction grids can differ from individual-layer runs. Points use mean observed removal, which can differ from the nominal fraction. Error bars are 95% article-cluster bootstrap intervals for the selected outcome. Hover to compare the nearest measured values. The actual observed removal fraction is shown for each method; use the legend to hide or show methods. The logarithmic scale omits zero and negative values; switch to linear to see them. Mean-centering subtracts each neuron's average activation on the training text before taking the magnitude used for ranking. The model keeps the original values of unselected activations and sets the selected ones to zero. Random is one shared comparator per nominal fraction. Its per-token count matches L and W exactly. Calibrated N and G thresholds can remove a different observed fraction, so their counts need not match random exactly.</p>
<!-- Additional results -->
<div class="plot-frame">
<h3 id="depth-title">Model output changes across layers</h3>
<div class="controls" id="depth-controls"><label>Nominal fraction for layer and paired comparisons<select id="fraction"></select></label></div><div id="depth-chart" aria-label="Ablation outcomes by layer at a fixed nominal removal fraction"></div>
</div>
<p class="note">Uses the selected model, outcome, ranking and nominal fraction. Hover for observed removal; calibrated methods can have different realized budgets. The lines connect only the five sampled individual layers; the all-layer experiment is excluded. A logarithmic vertical scale can reveal small differences, but does not display zero or negative values.</p>
<p class="note"><strong>Methods:</strong> N selects activations below a per-neuron training threshold; G uses one global training threshold. L selects the smallest magnitudes within each token; W selects the smallest magnitude multiplied by the outgoing weight norm. Random selects neurons uniformly. Identity removes none; all zero removes all.</p><div class="scroll"><table aria-describedby="layer-summary-note"><caption class="result-table-title">KL divergence and observed removal by layer</caption><thead id="crosshead"></thead><tbody id="crossrows"></tbody></table></div><p class="note" id="layer-summary-note">Each cell shows mean KL ± approximate 95% article-bootstrap uncertainty and observed removal for the selected nominal fraction. Identity and all-zero controls appear only at their measured fractions.</p><div class="scroll"><table aria-describedby="paired-summary-note"><caption class="result-table-title">Paired differences in KL divergence and observed removal</caption><thead><tr><th>Method</th><th>Reference</th><th>KL difference ± 95% uncertainty</th><th>Observed removal difference ± 95% uncertainty</th><th>Articles</th><th>Bootstrap replicates</th></tr></thead><tbody id="pairedrows"></tbody></table></div><p class="note" id="paired-summary-note">For the selected model, layer and nominal fraction, each difference is method minus reference, computed within the same windows. Bootstrap resampling preserves article clusters and pairing. Observed removal differences show whether the actual removal rates match.</p><div class="scroll"><table aria-describedby="outcome-summary-note"><caption class="result-table-title">Ablation outcomes by method and removal fraction</caption><thead><tr><th>Method</th><th>Nominal removal</th><th>Observed removal</th><th>KL ± 95% uncertainty</th><th>NLL change ± 95% uncertainty</th><th>Token flip rate ± 95% uncertainty</th><th>Articles</th><th>Bootstrap replicates</th></tr></thead><tbody id="rows"></tbody></table></div><p class="note" id="outcome-summary-note">Numerical ± values are the mean and half the width of the 95% article-bootstrap percentile interval; they approximate the potentially asymmetric bounds shown by the plot whiskers. Random repetitions, if present, are averaged within each window before summarization. Values reflect the available study output.</p>
</main><script>
const data=__DATA__;
const paired=__PAIRED__;
const $=id=>document.getElementById(id);
const metricNames={kl:'KL divergence (nats per scored token)',nll_delta:'NLL change (nats per scored token)',flip_rate:'Token flip rate',observed_fraction:'Observed fraction removed',removed_abs_fraction:'Removed absolute activation mass',removed_squared_fraction:'Removed squared activation mass',removed_weighted_fraction:'Removed weighted activation mass',removed_centered_abs_fraction:'Removed centered absolute activation mass',removed_centered_weighted_fraction:'Removed centered weighted activation mass',baseline_nll:'Baseline NLL'};
const colors={N:'#d72000',L:'#ee6100',G:'#253367',W:'#1bb6af',N_centered:'#9093a2',L_centered:'#ffad0a',G_centered:'#253367',W_centered:'#1bb6af',random:'#000000',identity:'#9093a2',all_zero:'#9093a2'};
const symbols={N:'diamond',L:'circle',G:'square',W:'cross',random:'triangle-up',identity:'circle-open',all_zero:'circle-open'};
function label(m){const names={N:'N · neuron percentile',L:'L · layer percentile at token',G:'G · pooled layer percentile',W:'W · weight-adjusted layer percentile',random:'Random',identity:'Identity',all_zero:'All zero'};return m.endsWith('_centered')?names[m.replace('_centered','')]+' (centered)':names[m]||m}
function options(element, values, preferred){element.innerHTML='';values.forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=element.id==='metric'?(metricNames[v]||v):element.id==='model'?(v==='smollm2'?'SmolLM2-135M':v==='gpt2'?'GPT-2 Small':v):element.id==='layer'&&v===-1?'All layers sequentially':element.id==='fraction'?(100*v).toFixed(v===.999?1:v===.995?1:0)+'%':v;element.appendChild(o)});if(values.map(String).includes(String(preferred))) element.value=preferred}
const selectedLayers={};
function chooseLayers(){let model=$('model').value, current=$('layer').value;let values=[...new Set(data.filter(r=>r.model===model).map(r=>r.layer))].sort((a,b)=>a-b);let available=values.map(String);let preferred=available.includes(current)?current:available.includes(selectedLayers[model])?selectedLayers[model]:values.includes(16)?16:values.includes(6)?6:values[0];options($('layer'),values,preferred);selectedLayers[model]=$('layer').value;chooseFractions()}
function chooseFractions(){let current=$('fraction').value;let values=[...new Set(data.filter(r=>r.model===$('model').value).map(r=>r.fraction))].sort((a,b)=>a-b);options($('fraction'),values,values.map(String).includes(current)?current:values.includes(.75)?.75:values[0]);render()}
function num(v,digits=3){return v==null?'—':Number(v).toLocaleString('en-US',{maximumSignificantDigits:digits})}
function pct(v){return v==null?'—':num(100*v)+'%'}
function comparisonPoints(points, anchors, logarithmic){
 const usable=points.filter(p=>Number.isFinite(p.r.mean)&&(!logarithmic||p.r.mean>0));
 return anchors.map(x=>{
  const nearest=usable.reduce((best,p)=>!best||Math.abs(p.obs.mean-x)<Math.abs(best.obs.mean-x)?p:best,null);
  return nearest?{x,point:nearest}:null;
 }).filter(Boolean);
}
function uncertainty(r,scale=1,suffix=''){return r?`${num(scale*r.mean)} ± ${num(scale*(r.upper-r.lower)/2,2)}${suffix}`:'—'}
function render(){
 const subset=data.filter(r=>r.model===$('model').value && String(r.layer)===$('layer').value); const metric=$('metric').value; const mode=$('ranking').value;
 const methods=['N','L','G','W','N_centered','L_centered','G_centered','W_centered','random','identity','all_zero'].filter(m=>mode==='both'||!m.includes('_centered')||mode==='centered').filter(m=>mode!=='centered'||m.includes('_centered')||['random','identity','all_zero'].includes(m));
 const index=new Map(subset.map(r=>[[r.method,r.fraction,r.metric].join('|'),r])); const lookup=(m,f,k)=>index.get([m,f,k].join('|')); const anchors=[...new Set(subset.filter(r=>methods.includes(r.method)&&!['identity','all_zero'].includes(r.method)&&r.metric===metric&&Number.isFinite(r.mean)&&($('scale').value!=='log'||r.mean>0)).map(r=>lookup(r.method,r.fraction,'observed_fraction')?.mean).filter(Number.isFinite))].sort((a,b)=>a-b); const valueFormat=metric==='kl'?'.3g':'.2%'; let traces=[];let rows=[];
 for(const method of methods){let points=subset.filter(r=>r.method===method&&r.metric===metric).sort((a,b)=>a.fraction-b.fraction).map(r=>({r,obs:lookup(method,r.fraction,'observed_fraction')})).filter(p=>p.obs&&!['identity','all_zero'].includes(method)); if(points.length){let centered=method.endsWith('_centered');traces.push({type:'scatter',mode:'lines+markers',name:label(method),legendgroup:method,marker:{symbol:symbols[method.replace('_centered','')]||'circle',size:7},x:points.map(p=>p.obs.mean),y:points.map(p=>p.r.mean),customdata:points.map(p=>p.r.fraction),line:{color:colors[method]||'#9093a2',dash:centered?'dash':method==='random'?'dot':'solid'},hoverinfo:'skip'});const comparison=comparisonPoints(points,anchors,$('scale').value==='log');traces.push({type:'scatter',mode:'lines',name:label(method),legendgroup:method,showlegend:false,x:comparison.map(p=>p.x),y:comparison.map(p=>p.point.r.mean),customdata:comparison.map(p=>[p.point.obs.mean,p.point.r.fraction]),line:{color:'rgba(0,0,0,0)',width:1},hoverlabel:{bgcolor:'#ffffff',bordercolor:colors[method],font:{color:colors[method]}},hovertemplate:`<span style="color:${colors[method]}">● ${label(method)}</span>: %{y:${valueFormat}}<br>Observed removal: %{customdata[0]:.1%}<br>Nominal removal: %{customdata[1]:.1%}<extra></extra>`});traces.push({type:'scatter',mode:'lines',showlegend:false,legendgroup:method,hoverinfo:'skip',x:points.flatMap(p=>[p.obs.mean,p.obs.mean,null]),y:points.flatMap(p=>[p.r.lower,p.r.upper,null]),line:{color:colors[method]||'#9093a2',width:1},connectgaps:false})}
 let fractions=[...new Set(subset.filter(r=>r.method===method).map(r=>r.fraction))].sort((a,b)=>a-b);for(const f of fractions){let kl=lookup(method,f,'kl'),obs=lookup(method,f,'observed_fraction'),nll=lookup(method,f,'nll_delta'),flip=lookup(method,f,'flip_rate');let meta=kl||obs||nll||flip;if(!meta)continue; rows.push(`<tr><td>${label(method)}</td><td>${pct(f)}</td><td>${pct(obs?.mean)}</td><td>${uncertainty(kl)}</td><td>${uncertainty(nll)}</td><td>${uncertainty(flip,100,'%')}</td><td>${meta.clusters}</td><td>${meta.replicates}</td></tr>`)} }
 const fraction=Number($('fraction').value); const all=data.filter(r=>r.model===$('model').value && r.layer!==-1); const layers=[...new Set(all.map(r=>r.layer))].sort((a,b)=>a-b); $('crosshead').innerHTML='<tr><th>Layer</th>'+methods.map(m=>`<th>${label(m)}</th>`).join('')+'</tr>'; $('crossrows').innerHTML=layers.map(layer=>'<tr><td>'+layer+'</td>'+methods.map(m=>{let kl=all.find(r=>r.layer===layer&&r.method===m&&r.fraction===fraction&&r.metric==='kl');let obs=all.find(r=>r.layer===layer&&r.method===m&&r.fraction===fraction&&r.metric==='observed_fraction');return '<td>'+(kl?`${uncertainty(kl)}<br>Removed ${pct(obs?.mean)}`:'—')+'</td>'}).join('')+'</tr>').join('');
 const depthTraces=[]; for(const method of methods){const points=layers.map(layer=>({layer,result:all.find(r=>r.layer===layer&&r.method===method&&r.fraction===fraction&&r.metric===metric),obs:all.find(r=>r.layer===layer&&r.method===method&&r.fraction===fraction&&r.metric==='observed_fraction')})).filter(p=>p.result&&p.obs&&!['identity','all_zero'].includes(method));if(!points.length)continue;let centered=method.endsWith('_centered');depthTraces.push({type:'scatter',mode:'lines+markers',name:label(method),marker:{symbol:symbols[method.replace('_centered','')]||'circle',size:7},x:points.map(p=>p.layer),y:points.map(p=>p.result.mean),customdata:points.map(p=>p.obs.mean),error_y:{type:'data',symmetric:false,array:points.map(p=>Math.max(0,p.result.upper-p.result.mean)),arrayminus:points.map(p=>Math.max(0,p.result.mean-p.result.lower)),visible:true},line:{color:colors[method]||'#9093a2',dash:centered?'dash':method==='random'?'dot':'solid'},hovertemplate:`Observed removal: %{customdata:.1%}<br>Mean: %{y:${valueFormat}}<extra>%{fullData.name}</extra>`})}
 const comparisons=paired.filter(r=>r.model===$('model').value&&String(r.layer)===$('layer').value&&r.fraction===fraction&&r.metric==='kl'&&methods.includes(r.method)); $('pairedrows').innerHTML=comparisons.map(r=>{let obs=paired.find(v=>v.model===r.model&&v.layer===r.layer&&v.method===r.method&&v.reference===r.reference&&v.fraction===r.fraction&&v.metric==='observed_fraction');return `<tr><td>${label(r.method)}</td><td>${label(r.reference)}</td><td>${uncertainty(r)}</td><td>${uncertainty(obs,100,'%')}</td><td>${r.clusters}</td><td>${r.replicates}</td></tr>`}).join('')||'<tr><td colspan="6" class="empty">No paired comparisons available for this selection.</td></tr>';
 $('rows').innerHTML=rows.join('')||'<tr><td colspan="8" class="empty">No measurements available for this selection.</td></tr>';
 Plotly.react('chart',traces,{xaxis:{title:{text:'Mean observed fraction removed'},type:'linear',tickformat:'.0%',hoverformat:'.1%',autorange:false,range:[0,1],fixedrange:true,showspikes:true,spikemode:'across',spikesnap:'hovered data',spikecolor:'#9093a2',spikethickness:1},yaxis:{title:{text:metricNames[metric]||metric},type:$('scale').value,tickformat:metric==='kl'?undefined:'.0%',fixedrange:true},margin:{t:20,r:24,b:125,l:88},font:{family:'system-ui, sans-serif',size:12,color:'#000000'},legend:{orientation:'h',x:.5,xanchor:'center',y:-.22,font:{size:11}},hovermode:'x unified',hoverdistance:-1,spikedistance:-1,paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff'},{responsive:true,displayModeBar:false,scrollZoom:false,doubleClick:false});
 $('depth-title').textContent=`${$('model').selectedOptions[0].textContent} · ${metricNames[metric]||metric} across layers at ${pct(fraction)} nominal removal`;
 Plotly.react('depth-chart',depthTraces,{xaxis:{title:{text:'Layer index'},tickmode:'array',tickvals:layers,fixedrange:true,showspikes:true,spikemode:'across',spikesnap:'hovered data',spikecolor:'#9093a2',spikethickness:1},yaxis:{title:{text:metricNames[metric]||metric},type:$('scale').value,tickformat:metric==='kl'?undefined:'.0%',fixedrange:true},margin:{t:20,r:24,b:125,l:88},font:{family:'system-ui, sans-serif',size:12,color:'#000000'},legend:{orientation:'h',x:.5,xanchor:'center',y:-.22,font:{size:11}},hovermode:'x unified',hoverdistance:-1,spikedistance:-1,paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff'},{responsive:true,displayModeBar:false,scrollZoom:false,doubleClick:false});
}
let models=[...new Set(data.map(r=>r.model))].sort();let preferred=models.find(m=>m.toLowerCase().includes('smollm2'))||models[0];options($('model'),models,preferred);let metrics=[...new Set(data.map(r=>r.metric))].filter(m=>!['observed_fraction','nll_delta','baseline_nll'].includes(m));options($('metric'),metrics,metrics.includes('kl')?'kl':metrics[0]);$('model').onchange=chooseLayers;$('layer').onchange=()=>{selectedLayers[$('model').value]=$('layer').value;render()};$('metric').onchange=render;$('ranking').onchange=render;$('fraction').onchange=render;$('scale').onchange=render;chooseLayers();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/large_study/data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/large_study/interactive.html')
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == '__main__':
    main()
