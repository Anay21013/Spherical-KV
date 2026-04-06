from __future__ import annotations
 
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import torch
import numpy as np
 
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
 
import config as _cfg
import allocation as _alloc
from tiers import build_tiers
from codebook_loader import load_codebooks
from spherical_kv_pipeline import SphericalKVPipeline

TIER_NAMES  = {0: "Dropped", 1: "High (b1)", 2: "Mid (b2)", 3: "Low (b3)"}
TIER_COLORS = {0: "#6c757d", 1: "#2DC653", 2: "#4361EE", 3: "#F77F00"}
 
 
def load_model_and_tokenizer(model_name: str, device: torch.device):
    from transformers import AutoTokenizer, LlamaForCausalLM
    print(f"[load] Model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = LlamaForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map={"": device},
        attn_implementation="eager",
    )
    mdl.eval()
    return mdl, tok
def load_corpus(tokenizer, dataset: str, num_tokens: int) -> torch.Tensor:
    from evaluate import get_eval_tokens
    return get_eval_tokens(tokenizer, None, dataset, num_tokens)

 
def extract_prefill_distribution(
    pipeline: SphericalKVPipeline,
    total_slots: int,
) -> Dict[int, float]:
    counts = Counter()
    for tok in pipeline._retained_tokens:
        counts[tok.new_tier_id] += 1
 
    retained = sum(counts.values())
    counts[0] = max(0, total_slots - retained)
 
    return {tid: counts.get(tid, 0) / max(total_slots, 1) for tid in [0, 1, 2, 3]}
 
 
def extract_decode_distribution(
    pipeline: SphericalKVPipeline,
) -> Dict[int, float]:
    counts = Counter()
    for tok in pipeline._retained_tokens:
        counts[tok.new_tier_id] += 1
 
    stg_count = sum(r.shape[0] for r in pipeline.stg_r.values())
    counts[3] += stg_count
 
    total = sum(counts.values())
    if total == 0:
        return {tid: 0.0 for tid in [0, 1, 2, 3]}
 
    return {tid: counts.get(tid, 0) / total for tid in [0, 1, 2, 3]}
 
 
def run_at_budget(
    pipeline: SphericalKVPipeline,
    model,
    prefill_ids: torch.Tensor,
    bpt: float,
    total_slots: int,
    n_warm: int,
    n_meas: int,
    device: torch.device,
    n_trials,
) -> Tuple[Dict[int, float], Dict[int, float]]:

    _cfg.BITS_PER_TOKEN = bpt
    if pipeline._patched:
        pipeline.uninstall()
    pipeline.prefill(prefill_ids)
 
    prefill_dist = extract_prefill_distribution(pipeline, total_slots)
 
    n_total = n_warm + n_meas
    for trail in range(n_trials):
      current_ids = prefill_ids.clone()
      for step in range(n_total):
          with torch.no_grad():
              out = model(
                  input_ids=current_ids[:, -1:],
                  use_cache=False,
                  return_dict=True,
              )
          next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
          current_ids = torch.cat([current_ids, next_id], dim=-1)
      del out, current_ids
    if device.type == "cuda":
        torch.cuda.empty_cache()
 
    decode_dist = extract_decode_distribution(pipeline)
 
    pipeline.uninstall()
    if device.type == "cuda":
        torch.cuda.empty_cache()
 
    return prefill_dist, decode_dist
 
 
def sweep_budgets(
    pipeline, model, prefill_ids, total_slots,
    bpt_values, T, num_layers, num_kv_heads,
    n_warm, n_meas, device, n_trials
) -> Tuple[dict, dict]:
    """Run the full pipeline at each bpt and collect distributions."""
    prefill_data = {"bpt": [], "high": [], "mid": [], "low": [], "drop": []}
    decode_data  = {"bpt": [], "high": [], "mid": [], "low": [], "drop": []}
 
    n_points = len(bpt_values)
    for i, bpt in enumerate(bpt_values):
        budget_bits = int(bpt * T * num_layers * num_kv_heads)
        t0 = time.time()
 
        pf_dist, dc_dist = run_at_budget(
            pipeline, model, prefill_ids, bpt, total_slots,
            n_warm, n_meas, device, n_trials,
        )
        # pf_dist, dc_dist = run_at_budget(
        #     pipeline, model, prefill_ids, budget_bits, total_slots,
        #     n_warm, n_meas, device,
        # )
        dt = time.time() - t0
 
        for data, dist in [(prefill_data, pf_dist), (decode_data, dc_dist)]:
            data["bpt"].append(round(float(bpt), 1))
            data["high"].append(round(dist[1], 6))
            data["mid"].append(round(dist[2], 6))
            data["low"].append(round(dist[3], 6))
            data["drop"].append(round(dist[0], 6))
 
        print(f"  [{i+1:2d}/{n_points}] {bpt:5.1f} bpt  "
              f"PF: H={pf_dist[1]*100:5.1f}% M={pf_dist[2]*100:5.1f}% "
              f"L={pf_dist[3]*100:5.1f}% D={pf_dist[0]*100:5.1f}%  |  "
              f"DC: H={dc_dist[1]*100:5.1f}% M={dc_dist[2]*100:5.1f}% "
              f"L={dc_dist[3]*100:5.1f}% D={dc_dist[0]*100:5.1f}%  "
              f"({dt:.1f}s)")
 
    return prefill_data, decode_data
 
 
def generate_html(prefill_data, decode_data, tiers, meta, out_path):
    tier_info = {t.tier_id: t.token_bits() for t in tiers if t.tier_id != 0}
 
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SphericalKV Tier Allocation</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap');
  :root {{
    --bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
    --text-dim:#8b949e;--accent:#58a6ff;
    --high:#2DC653;--mid:#4361EE;--low:#F77F00;--drop:#6c757d;
  }}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{font-family:'DM Sans',system-ui,sans-serif;background:var(--bg);
        color:var(--text);min-height:100vh;padding:2rem;}}
  .header{{text-align:center;margin-bottom:2rem;}}
  .header h1{{font-size:1.8rem;font-weight:700;
    background:linear-gradient(135deg,var(--accent),var(--high));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:.3rem;}}
  .header .subtitle{{color:var(--text-dim);font-size:.95rem;}}
  .meta-bar{{display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap;
    margin-bottom:1.5rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;
    color:var(--text-dim);}}
  .meta-bar span{{background:var(--surface);border:1px solid var(--border);
    border-radius:6px;padding:.35rem .75rem;}}
  .meta-bar .val{{color:var(--accent);font-weight:600;}}
  .legend{{display:flex;justify-content:center;gap:1.5rem;margin-bottom:1.5rem;
    font-size:.85rem;}}
  .legend-item{{display:flex;align-items:center;gap:.4rem;}}
  .legend-dot{{width:12px;height:12px;border-radius:3px;}}
  .legend-bits{{font-family:'JetBrains Mono',monospace;color:var(--text-dim);
    font-size:.75rem;}}
  .slider-container{{background:var(--surface);border:1px solid var(--border);
    border-radius:12px;padding:1.5rem 2rem;max-width:700px;margin:0 auto 2rem;
    text-align:center;}}
  .slider-label{{font-size:.9rem;color:var(--text-dim);margin-bottom:.5rem;}}
  .slider-value{{font-family:'JetBrains Mono',monospace;font-size:2rem;
    font-weight:600;color:var(--accent);margin-bottom:.5rem;}}
  .slider-value .unit{{font-size:1rem;color:var(--text-dim);}}
  .slider-value .budget-bits{{font-size:.85rem;color:var(--text-dim);
    font-weight:400;display:block;margin-top:.2rem;}}
  input[type="range"]{{-webkit-appearance:none;width:100%;height:6px;
    border-radius:3px;background:var(--border);outline:none;}}
  input[type="range"]::-webkit-slider-thumb{{-webkit-appearance:none;width:22px;
    height:22px;border-radius:50%;background:var(--accent);cursor:pointer;
    box-shadow:0 0 8px rgba(88,166,255,.4);}}
  input[type="range"]::-moz-range-thumb{{width:22px;height:22px;border-radius:50%;
    background:var(--accent);cursor:pointer;border:none;}}
  .slider-endpoints{{display:flex;justify-content:space-between;
    font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--text-dim);
    margin-top:.3rem;}}
  .charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;
    max-width:1100px;margin:0 auto 2rem;}}
  .chart-card{{background:var(--surface);border:1px solid var(--border);
    border-radius:12px;padding:1rem;}}
  .chart-card h3{{text-align:center;font-size:1rem;font-weight:600;
    margin-bottom:.3rem;}}
  .chart-card .phase-tag{{display:block;text-align:center;font-size:.75rem;
    color:var(--text-dim);margin-bottom:.5rem;
    font-family:'JetBrains Mono',monospace;}}
  .diff-table{{max-width:700px;margin:0 auto 2rem;background:var(--surface);
    border:1px solid var(--border);border-radius:12px;overflow:hidden;}}
  .diff-table table{{width:100%;border-collapse:collapse;
    font-family:'JetBrains Mono',monospace;font-size:.82rem;}}
  .diff-table th{{background:rgba(88,166,255,.08);padding:.6rem 1rem;
    text-align:left;font-weight:600;color:var(--text-dim);
    border-bottom:1px solid var(--border);}}
  .diff-table td{{padding:.5rem 1rem;border-bottom:1px solid var(--border);}}
  .diff-table tr:last-child td{{border-bottom:none;}}
  .diff-pos{{color:var(--high);}} .diff-neg{{color:#f85149;}}
  .diff-zero{{color:var(--text-dim);}}
  .area-card{{background:var(--surface);border:1px solid var(--border);
    border-radius:12px;padding:1rem;max-width:1100px;margin:0 auto 2rem;}}
  .area-card h3{{text-align:center;font-size:1rem;font-weight:600;
    margin-bottom:.5rem;}}
  .stamp{{text-align:center;color:var(--text-dim);font-size:.75rem;
    font-family:'JetBrains Mono',monospace;margin-top:1rem;}}
  @media(max-width:768px){{.charts-row{{grid-template-columns:1fr;}}
    body{{padding:1rem;}}}}
</style>
</head>
<body>
 
<div class="header">
  <h1>SphericalKV Tier Allocation</h1>
  <p class="subtitle">Prefill and Decode time distributions at multiple budget points</p>
</div>
 
<div class="meta-bar">
  <span>Model: <span class="val">{meta['model']}</span></span>
  <span>Dataset: <span class="val">{meta['dataset']}</span></span>
  <span>T = <span class="val">{meta['T']}</span></span>
  <span>L = <span class="val">{meta['num_layers']}</span></span>
  <span>H_kv = <span class="val">{meta['num_kv_heads']}</span></span>
  <span>dh = <span class="val">{meta['head_dim']}</span></span>
  <span>Decode = <span class="val">{meta['n_decode']} steps</span></span>
  <span>Refresh = <span class="val">every {meta['refresh_cadence']}</span></span>
</div>
 
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:var(--high)"></div>
    High (b1) <span class="legend-bits">{tier_info.get(1,'?')}b</span></div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--mid)"></div>
    Mid (b2) <span class="legend-bits">{tier_info.get(2,'?')}b</span></div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--low)"></div>
    Low (b3) <span class="legend-bits">{tier_info.get(3,'?')}b</span></div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--drop)"></div>
    Dropped <span class="legend-bits">0b</span></div>
</div>
 
<div class="slider-container">
  <div class="slider-label">Bits Per Token (budget)</div>
  <div class="slider-value" id="bpt-display">--</div>
  <input type="range" id="bpt-slider" min="0" max="1" step="1" value="0"/>
  <div class="slider-endpoints"><span id="bpt-min">--</span><span id="bpt-max">--</span></div>
</div>
 
<div class="charts-row">
  <div class="chart-card">
    <h3>Prefill Allocation</h3>
    <span class="phase-tag"> Reuse & Stability proxy</span>
    <div id="pie-prefill"></div>
  </div>
  <div class="chart-card">
    <h3>Decode Allocation</h3>
    <span class="phase-tag">Compressed decode with online_refresh</span>
    <div id="pie-decode"></div>
  </div>
</div>
 
<div id="diff-section" class="diff-table"></div>
 
<div class="area-card">
  <h3>Tier Distribution vs Budget -- Prefill</h3>
  <div id="area-prefill"></div>
</div>
<div class="area-card">
  <h3>Tier Distribution vs Budget -- Decode</h3>
  <div id="area-decode"></div>
</div>
 
 
<script>
const P = {json.dumps(prefill_data)};
const D = {json.dumps(decode_data)};
const bpt = P.bpt, N = bpt.length;
const T = {meta['T']}, L = {meta['num_layers']}, H = {meta['num_kv_heads']};
const C = {{high:'#2DC653',mid:'#4361EE',low:'#F77F00',drop:'#6c757d'}};
const layout = {{paper_bgcolor:'transparent',plot_bgcolor:'transparent',
  font:{{family:'DM Sans,system-ui',color:'#e6edf3',size:12}},
  margin:{{t:10,b:10,l:10,r:10}},showlegend:false}};
 
const sl = document.getElementById('bpt-slider');
sl.min=0; sl.max=N-1; sl.value=Math.floor(N/2);
document.getElementById('bpt-min').textContent=bpt[0];
document.getElementById('bpt-max').textContent=bpt[N-1];
 
function bar(div,data,i){{
  const v=[data.high[i]*100,data.mid[i]*100,data.low[i]*100,data.drop[i]*100];
  const labels=['High (b1)','Mid (b2)','Low (b3)','Dropped'];
  const colors=[C.high,C.mid,C.low,C.drop];
  Plotly.react(div,[{{x:labels,y:v,type:'bar',
    marker:{{color:colors,line:{{color:'rgba(255,255,255,0.15)',width:1}}}},
    text:v.map(x=>x.toFixed(1)+'%'),textposition:'outside',
    textfont:{{family:'JetBrains Mono',size:12,color:'#e6edf3'}},
    hovertemplate:'%{{x}}: %{{y:.1f}}%<extra></extra>'}}],
    {{...layout,height:300,margin:{{t:15,b:40,l:45,r:15}},
    xaxis:{{color:'#8b949e',tickfont:{{size:10}}}},
    yaxis:{{title:'%',color:'#8b949e',gridcolor:'#21262d',zeroline:false,
      range:[0,Math.min(Math.max(...v)*1.25,105)]}}}},
    {{responsive:true,displayModeBar:false}});
}}
 
function diff(i){{
  const ts=['high','mid','low','drop'],ns=['High (b1)','Mid (b2)','Low (b3)','Dropped'],
    cs=[C.high,C.mid,C.low,C.drop];
  let rows='';
  for(let j=0;j<4;j++){{
    const pf=P[ts[j]][i]*100,dc=D[ts[j]][i]*100,d=dc-pf;
    const cls=d>.05?'diff-pos':d<-.05?'diff-neg':'diff-zero';
    rows+=`<tr><td><span style="color:${{cs[j]}}">&#x25CF;</span> ${{ns[j]}}</td>
      <td>${{pf.toFixed(1)}}%</td><td>${{dc.toFixed(1)}}%</td>
      <td class="${{cls}}">${{d>0?'+':''}}${{d.toFixed(1)}}pp</td></tr>`;
  }}
  document.getElementById('diff-section').innerHTML=`<table><thead><tr>
    <th>Tier</th><th>Prefill</th><th>Decode</th><th>Delta</th></tr></thead>
    <tbody>${{rows}}</tbody></table>`;
}}
 
function area(div,data){{
  const tr=k=>{{return{{x:bpt,y:data[k].map(v=>v*100),name:k,fill:'tonexty',
    fillcolor:C[k]+'55',line:{{color:C[k],width:1.5}},stackgroup:'s'}}}};
  Plotly.react(div,[tr('high'),tr('mid'),tr('low'),tr('drop')],{{...layout,height:280,
    margin:{{t:20,b:50,l:60,r:30}},
    xaxis:{{title:'Bits per Token',color:'#8b949e',gridcolor:'#21262d',zeroline:false}},
    yaxis:{{title:'%',color:'#8b949e',gridcolor:'#21262d',zeroline:false,range:[0,100]}},
    showlegend:true,legend:{{orientation:'h',y:1.15,x:.5,xanchor:'center',
      font:{{size:11,color:'#8b949e'}}}},
    shapes:[{{type:'line',x0:bpt[0],x1:bpt[0],y0:0,y1:100,
      line:{{color:'#58a6ff',width:2,dash:'dot'}}}}]
  }},{{responsive:true,displayModeBar:false}});
}}
 
function update(){{
  const i=parseInt(sl.value),b=bpt[i];
  const bits=Math.round(b*T*L*H);
  document.getElementById('bpt-display').innerHTML=
    b+' <span class="unit">bpt</span>'+
    '<span class="budget-bits">= '+bits.toLocaleString()+' bits global budget</span>';
  bar('pie-prefill',P,i); bar('pie-decode',D,i); diff(i);
  Plotly.relayout('area-prefill',{{'shapes[0].x0':b,'shapes[0].x1':b}});
  Plotly.relayout('area-decode',{{'shapes[0].x0':b,'shapes[0].x1':b}});
}}
 
area('area-prefill',P); area('area-decode',D);
sl.addEventListener('input',update); update();
</script>
</body></html>"""
 
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"\n[done] Dashboard -> {out_path}")
 
 
 
def main():
    ap = argparse.ArgumentParser(
        description="SphericalKV tier-allocation dashboard")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--codebook_dir",
                    default="codebooks/codebooks_llama_1b")
    ap.add_argument("--dataset", default="pg19")
    ap.add_argument("--context_length", type=int, default=512)
    ap.add_argument("--n_warm", type=int, default=8)
    ap.add_argument("--n_meas", type=int, default=64)
    ap.add_argument("--n_trials", type=int, default = 1)
    ap.add_argument("--bpt_min", type=float, default=None)
    ap.add_argument("--bpt_max", type=float, default=None)
    ap.add_argument("--bpt_steps", type=int, default=20,
                    help="Number of budget points (each runs full pipeline)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="tier_allocation_dashboard.html")
    ap.add_argument("--sink_tokens", type=int, default=4)
    args = ap.parse_args()
 
    device = torch.device(
        args.device if args.device
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[config] device={device}")
 
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    cfg = model.config
    num_layers   = cfg.num_hidden_layers
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim     = getattr(cfg, "head_dim",
                           cfg.hidden_size // cfg.num_attention_heads)
    group_size   = _cfg.GROUP_SIZE
    num_groups   = head_dim // group_size
 
    # Patch config globals to match this model
    _cfg.HEAD_DIM   = head_dim
    _cfg.NUM_GROUPS = num_groups
 
    tiers = build_tiers(head_dim)
    b3_bits = tiers[3].token_bits()
    b1_bits = tiers[1].token_bits()
    print(f"[tiers] b1={b1_bits}b  b2={tiers[2].token_bits()}b  "
          f"b3={b3_bits}b  (head_dim={head_dim})")
 
    codebooks = load_codebooks(args.codebook_dir, num_layers, num_kv_heads,
                               tiers)
 
    pipeline = SphericalKVPipeline(
        model=model, tokenizer=tokenizer,
        codebooks=codebooks, device=device,
        head_dim=head_dim, group_size=group_size,
        sink_tokens=args.sink_tokens,
        use_fused=(device.type == "cuda"),
    )
 
    bpt_lo = args.bpt_min if args.bpt_min is not None else max(b3_bits * 0.3, 5)
    bpt_hi = args.bpt_max if args.bpt_max is not None else b1_bits * 1.15
    bpt_values = np.linspace(bpt_lo, bpt_hi, args.bpt_steps)
    print(f"[sweep] {args.bpt_steps} budget points from {bpt_lo:.1f} to "
          f"{bpt_hi:.1f} bpt")
 
    T = args.context_length
    n_decode = args.n_warm + args.n_meas
    num_eval_tokens = T + n_decode + 16
    print(f"\n[data] Loading {num_eval_tokens} tokens from '{args.dataset}'")
    eval_ids = load_corpus(tokenizer, args.dataset, num_eval_tokens)
    prefill_ids = eval_ids[:T].unsqueeze(0).to(device)
 
    total_slots = T * num_layers * num_kv_heads
 
    print(f"\n{'='*72}")
    print(f"  BUDGET SWEEP  --  {args.bpt_steps} points x "
          f"(prefill + {n_decode} decode steps) each")
    print(f"{'='*72}")
    t_start = time.time()
 
    prefill_data, decode_data = sweep_budgets(
        pipeline, model, prefill_ids, total_slots,
        bpt_values, T, num_layers, num_kv_heads,
        args.n_warm, args.n_meas, device, args.n_trials
    )
 
    t_total = time.time() - t_start
    print(f"\n[sweep] Total time: {t_total:.0f}s  "
          f"({t_total/args.bpt_steps:.1f}s per budget point)")
 
    from config import REFRESH_CADENCE
    meta = {
        "model": args.model.split("/")[-1],
        "dataset": args.dataset,
        "T": T,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "total_slots": total_slots,
        "n_decode": n_decode,
        "refresh_cadence": REFRESH_CADENCE,
    }
    generate_html(prefill_data, decode_data, tiers, meta, args.output)
    print(f"\nOpen {args.output} in your browser to explore the dashboard.")
 
 
if __name__ == "__main__":
    main()