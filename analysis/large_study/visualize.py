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
    payload = data[list(REQUIRED)].to_json(orient='records', double_precision=15)
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
body{font:15px system-ui,sans-serif;margin:0;color:#172131;background:#f5f7fa}main{max-width:1250px;margin:auto;padding:28px}h1{margin-bottom:4px}h2{margin-top:28px}p{line-height:1.5;color:#445166}.controls{display:flex;gap:18px;flex-wrap:wrap;background:white;padding:16px;border:1px solid #dce3eb;border-radius:8px}label{display:grid;gap:5px;font-weight:600}select{padding:7px;min-width:145px;border:1px solid #aebacc;border-radius:4px;background:white}#chart,#depth-chart{height:540px;background:white;margin-top:18px}.scroll{overflow:auto;background:white;border:1px solid #dce3eb}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #e6ebf0}th{background:#eff3f8;position:sticky;top:0}th:first-child,td:first-child{text-align:left}.note{font-size:13px}.empty{padding:20px}
</style></head><body><main>
<h1>Activation ablation study</h1><p>Explore held-out measurements by model, layer, method and nominal removal fraction.</p>
<div class="controls"><label>Model<select id="model"></select></label><label>Layer<select id="layer"></select></label><label>Outcome<select id="metric"></select></label><label>Cross-layer nominal fraction<select id="fraction"></select></label><label>Ranking<select id="ranking"><option value="original">Original</option><option value="centered">Mean-centered</option><option value="both">Both</option></select></label><label>Horizontal scale<select id="horizontal-scale"><option value="linear">Linear</option><option value="log">Logarithmic</option></select></label><label>Vertical scale<select id="scale"><option value="linear">Linear</option><option value="log">Logarithmic</option></select></label><button id="download">Download all summary data (CSV)</button></div>
<div id="chart" aria-label="Interactive ablation outcome plot"></div>
<p class="note">Figure 2. Points use mean observed removal, which can differ from the nominal fraction. A logarithmic horizontal scale omits points at zero removal. Error bars are 95% article-cluster bootstrap intervals for the selected outcome. Hover for nominal fractions; use the legend to hide or show methods. Mean-centering changes the ranking score; selected activations are set to zero. Random is one shared comparator per nominal fraction. Its per-token count matches L and W exactly. Calibrated N and G thresholds can remove a different observed fraction, so their counts need not match random exactly.</p>
<h2>How the selected outcome varies across layers</h2><p class="note">Uses the selected model, outcome, ranking and nominal fraction. Hover for observed removal; calibrated methods can have different realized budgets. The lines connect only the five sampled layers. A logarithmic vertical scale can reveal small differences, but does not display zero or negative values.</p><div id="depth-chart" aria-label="Interactive comparison across sampled layers"></div>
<p class="note"><strong>Methods:</strong> N selects activations below a per-neuron training threshold; G uses one global training threshold. L selects the smallest magnitudes within each token; W selects the smallest magnitude multiplied by the outgoing weight norm. Random selects neurons uniformly. Identity removes none; all zero removes all.</p><h2>Cross-layer comparison</h2><p class="note">Each cell shows mean KL ± approximate 95% article-bootstrap uncertainty and observed removal for the selected nominal fraction. Identity and all-zero controls appear only at their measured fractions.</p><div class="scroll"><table><thead id="crosshead"></thead><tbody id="crossrows"></tbody></table></div><h2>Paired comparisons</h2><p class="note">For the selected model, layer and nominal fraction, each difference is method minus reference, computed within the same windows. Bootstrap resampling preserves article clusters and pairing. Observed removal differences show whether the actual removal rates match.</p><div class="scroll"><table><thead><tr><th>Method</th><th>Reference</th><th>KL difference ± 95% uncertainty</th><th>Observed removal difference ± 95% uncertainty</th><th>Articles</th><th>Bootstrap replicates</th></tr></thead><tbody id="pairedrows"></tbody></table></div><h2>Measurements by fraction and method</h2><p class="note">Numerical ± values are the mean and half the width of the 95% article-bootstrap percentile interval; they approximate the potentially asymmetric bounds shown by the plot whiskers. Random repetitions, if present, are averaged within each window before summarization. Values reflect the available study output.</p><div class="scroll"><table><thead><tr><th>Method</th><th>Nominal removal</th><th>Observed removal</th><th>KL ± 95% uncertainty</th><th>NLL change ± 95% uncertainty</th><th>Token flip rate ± 95% uncertainty</th><th>Articles</th><th>Bootstrap replicates</th></tr></thead><tbody id="rows"></tbody></table></div>
</main><script>
const data=__DATA__;
const paired=__PAIRED__;
const $=id=>document.getElementById(id);
const metricNames={kl:'KL divergence (nats per scored token)',nll_delta:'NLL change (nats per scored token)',flip_rate:'Token flip rate',observed_fraction:'Observed fraction removed',removed_abs_fraction:'Removed absolute activation mass',removed_squared_fraction:'Removed squared activation mass',removed_weighted_fraction:'Removed weighted activation mass',removed_centered_abs_fraction:'Removed centered absolute activation mass',removed_centered_weighted_fraction:'Removed centered weighted activation mass',baseline_nll:'Baseline NLL'};
const colors={N:'#4f7398',L:'#26736d',G:'#7b8050',W:'#ad6540',random:'#7b8089',identity:'#243735',all_zero:'#243735'};
function label(m){const names={N:'N · neuron percentile',L:'L · layer percentile at token',G:'G · pooled layer percentile',W:'W · weight-adjusted layer percentile',random:'Random',identity:'Identity',all_zero:'All zero'};return m.endsWith('_centered')?names[m.replace('_centered','')]+' (centered)':names[m]||m}
function options(element, values, preferred){element.innerHTML='';values.forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=element.id==='metric'?(metricNames[v]||v):element.id==='model'?(v==='smollm2'?'SmolLM2-135M':v==='gpt2'?'GPT-2 Small':v):element.id==='fraction'?(100*v).toFixed(v===.999?1:v===.995?1:0)+'%':v;element.appendChild(o)});if(values.map(String).includes(String(preferred))) element.value=preferred}
function chooseLayers(){let values=[...new Set(data.filter(r=>r.model===$('model').value).map(r=>r.layer))].sort((a,b)=>a-b); options($('layer'),values,values.includes(16)?16:values.includes(6)?6:values[0]);chooseFractions()}
function chooseFractions(){let values=[...new Set(data.filter(r=>r.model===$('model').value).map(r=>r.fraction))].sort((a,b)=>a-b);options($('fraction'),values,values.includes(.75)?.75:values[0]);render()}
function num(v,digits=3){return v==null?'—':Number(v).toLocaleString('en-US',{maximumSignificantDigits:digits})}
function pct(v){return v==null?'—':num(100*v)+'%'}
function uncertainty(r,scale=1,suffix=''){return r?`${num(scale*r.mean)} ± ${num(scale*(r.upper-r.lower)/2,2)}${suffix}`:'—'}
function render(){
 const subset=data.filter(r=>r.model===$('model').value && String(r.layer)===$('layer').value); const metric=$('metric').value; const mode=$('ranking').value; const horizontalScale=$('horizontal-scale').value;
 const methods=['N','L','G','W','N_centered','L_centered','G_centered','W_centered','random','identity','all_zero'].filter(m=>mode==='both'||!m.includes('_centered')||mode==='centered').filter(m=>mode!=='centered'||m.includes('_centered')||['random','identity','all_zero'].includes(m));
 const index=new Map(subset.map(r=>[[r.method,r.fraction,r.metric].join('|'),r])); const lookup=(m,f,k)=>index.get([m,f,k].join('|')); let traces=[];let rows=[];
 for(const method of methods){let points=subset.filter(r=>r.method===method&&r.metric===metric).sort((a,b)=>a.fraction-b.fraction).map(r=>({r,obs:lookup(method,r.fraction,'observed_fraction')})).filter(p=>p.obs&&(horizontalScale!=='log'||p.obs.mean>0)); if(points.length){let centered=method.endsWith('_centered');traces.push({type:'scatter',mode:'lines+markers',name:label(method),legendgroup:method,x:points.map(p=>p.obs.mean),y:points.map(p=>p.r.mean),customdata:points.map(p=>p.r.fraction),line:{color:colors[method[0]]||colors[method]||'#606b78',dash:centered?'dash':method==='random'?'dot':'solid'},hovertemplate:'Observed removal: %{x:.1%}<br>Nominal removal: %{customdata:.1%}<br>Mean: %{y:.3g}<extra>%{fullData.name}</extra>'});traces.push({type:'scatter',mode:'lines',showlegend:false,legendgroup:method,hoverinfo:'skip',x:points.flatMap(p=>[p.obs.mean,p.obs.mean,null]),y:points.flatMap(p=>[p.r.lower,p.r.upper,null]),line:{color:colors[method[0]]||colors[method]||'#606b78',width:1},connectgaps:false})}
 let fractions=[...new Set(subset.filter(r=>r.method===method).map(r=>r.fraction))].sort((a,b)=>a-b);for(const f of fractions){let kl=lookup(method,f,'kl'),obs=lookup(method,f,'observed_fraction'),nll=lookup(method,f,'nll_delta'),flip=lookup(method,f,'flip_rate');let meta=kl||obs||nll||flip;if(!meta)continue; rows.push(`<tr><td>${label(method)}</td><td>${pct(f)}</td><td>${pct(obs?.mean)}</td><td>${uncertainty(kl)}</td><td>${uncertainty(nll)}</td><td>${uncertainty(flip,100,'%')}</td><td>${meta.clusters}</td><td>${meta.replicates}</td></tr>`)} }
 const fraction=Number($('fraction').value); const all=data.filter(r=>r.model===$('model').value); const layers=[...new Set(all.map(r=>r.layer))].sort((a,b)=>a-b); $('crosshead').innerHTML='<tr><th>Layer</th>'+methods.map(m=>`<th>${label(m)}</th>`).join('')+'</tr>'; $('crossrows').innerHTML=layers.map(layer=>'<tr><td>'+layer+'</td>'+methods.map(m=>{let kl=all.find(r=>r.layer===layer&&r.method===m&&r.fraction===fraction&&r.metric==='kl');let obs=all.find(r=>r.layer===layer&&r.method===m&&r.fraction===fraction&&r.metric==='observed_fraction');return '<td>'+(kl?`${uncertainty(kl)}<br>Removed ${pct(obs?.mean)}`:'—')+'</td>'}).join('')+'</tr>').join('');
 const depthTraces=[]; for(const method of methods){const points=layers.map(layer=>({layer,result:all.find(r=>r.layer===layer&&r.method===method&&r.fraction===fraction&&r.metric===metric),obs:all.find(r=>r.layer===layer&&r.method===method&&r.fraction===fraction&&r.metric==='observed_fraction')})).filter(p=>p.result&&p.obs);if(!points.length)continue;let centered=method.endsWith('_centered');depthTraces.push({type:'scatter',mode:'lines+markers',name:label(method),x:points.map(p=>p.layer),y:points.map(p=>p.result.mean),customdata:points.map(p=>p.obs.mean),error_y:{type:'data',symmetric:false,array:points.map(p=>Math.max(0,p.result.upper-p.result.mean)),arrayminus:points.map(p=>Math.max(0,p.result.mean-p.result.lower)),visible:true},line:{color:colors[method[0]]||colors[method]||'#606b78',dash:centered?'dash':method==='random'?'dot':'solid'},hovertemplate:'Layer %{x}<br>Observed removal: %{customdata:.1%}<br>Mean: %{y:.3g}<extra>%{fullData.name}</extra>'})}
 const comparisons=paired.filter(r=>r.model===$('model').value&&String(r.layer)===$('layer').value&&r.fraction===fraction&&r.metric==='kl'&&methods.includes(r.method)); $('pairedrows').innerHTML=comparisons.map(r=>{let obs=paired.find(v=>v.model===r.model&&v.layer===r.layer&&v.method===r.method&&v.reference===r.reference&&v.fraction===r.fraction&&v.metric==='observed_fraction');return `<tr><td>${label(r.method)}</td><td>${label(r.reference)}</td><td>${uncertainty(r)}</td><td>${uncertainty(obs,100,'%')}</td><td>${r.clusters}</td><td>${r.replicates}</td></tr>`}).join('')||'<tr><td colspan="6" class="empty">No paired comparisons available for this selection.</td></tr>';
 $('rows').innerHTML=rows.join('')||'<tr><td colspan="8" class="empty">No measurements available for this selection.</td></tr>';
 Plotly.react('chart',traces,{title:{text:metricNames[metric]||metric},xaxis:{title:{text:'Mean observed fraction removed'},type:horizontalScale,tickformat:horizontalScale==='log'?'.2~%':'.0%',...(horizontalScale==='log'?{autorange:true}:{autorange:false,range:[0,1]})},yaxis:{title:{text:metricNames[metric]||metric},type:$('scale').value},margin:{t:75,r:24,b:125,l:88},font:{family:'system-ui, sans-serif',size:12,color:'#243735'},legend:{orientation:'h',y:-.22,font:{size:11}},hovermode:'closest',paper_bgcolor:'#fcfbf8',plot_bgcolor:'#fcfbf8'},{responsive:true,displaylogo:false});
 Plotly.react('depth-chart',depthTraces,{title:{text:`${metricNames[metric]||metric} across layers at ${(100*fraction).toFixed(fraction===.999||fraction===.995?1:0)}% nominal removal`},xaxis:{title:{text:'Layer index'},tickmode:'array',tickvals:layers},yaxis:{title:{text:metricNames[metric]||metric},type:$('scale').value},margin:{t:75,r:24,b:125,l:88},font:{family:'system-ui, sans-serif',size:12,color:'#243735'},legend:{orientation:'h',y:-.22,font:{size:11}},hovermode:'closest',paper_bgcolor:'#fcfbf8',plot_bgcolor:'#fcfbf8'},{responsive:true,displaylogo:false});
}
$('download').onclick=()=>{let columns=['model','layer','method','fraction','metric','mean','lower','upper','clusters','replicates'];let csv=[columns.join(','),...data.map(r=>columns.map(c=>JSON.stringify(r[c]??'')).join(','))].join('\n');let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='large_study_summary.csv';a.click();URL.revokeObjectURL(a.href)};
let models=[...new Set(data.map(r=>r.model))].sort();let preferred=models.find(m=>m.toLowerCase().includes('smollm2'))||models[0];options($('model'),models,preferred);let metrics=[...new Set(data.map(r=>r.metric))].filter(m=>m!=='observed_fraction');options($('metric'),metrics,metrics.includes('kl')?'kl':metrics[0]);$('model').onchange=chooseLayers;$('layer').onchange=render;$('metric').onchange=render;$('ranking').onchange=render;$('fraction').onchange=render;$('scale').onchange=render;$('horizontal-scale').onchange=render;chooseLayers();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/large_study/test')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/large_study/interactive.html')
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == '__main__':
    main()
