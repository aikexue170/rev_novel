"""Assemble the final comparison report from merged load results, the multiq run, and training evals.

usage: final_report.py <out_dir> <merged_load_dir> <multiq_dir> <train_dir>
"""
import sys,json,pathlib,math,subprocess,tempfile
out,load,multiq,train=[pathlib.Path(a) for a in sys.argv[1:5]];out.mkdir(exist_ok=True)
L=json.loads((load/'summary.json').read_text());M=json.loads((multiq/'summary.json').read_text()) if (multiq/'summary.json').exists() else None
LABELS={'jev':'Hosted Jev (OpenRouter)','small2h':'Qwen3.5-2B + head · H100','small4h':'Qwen3.5-4B + head · H100','small4b':'Qwen3.5-4B + head · B200','small9h':'Qwen3.5-9B + head · H100','small9b':'Qwen3.5-9B + head · B200','dense27h':'Qwen3.8-27B + head · H100','dense27b':'Qwen3.8-27B + head · B200','small4bal':'Qwen3.5-4B balanced + head · B200','small9bal':'Qwen3.5-9B balanced + head · B200'}
ORDER=['jev','small2h','small4h','small4b','small4bal','small9h','small9b','small9bal','dense27h','dense27b']
rows=L['rows'];arms=[a for a in ORDER if any(r['arm']==a for r in rows)]
def wilson(k,n,z=1.96):
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
def fmt(v,f):return '' if v is None else f.format(v)
jev1=next(r for r in rows if r['arm']=='jev' and r['concurrency']==1);jev_cost=jev1['cost_per_1k']
lines=['# Beating Jev with open weights: final comparison','',
'Same 975-question public holdout for every system (ContractNLI, RACE, CosmosQA, MultiRC, Social IQa; never trained on). One question per request. Cloud client in us-west. Hosted Jev = `typesafe/jev-1.13` via OpenRouter, billed cost from the API. Self-hosted = Qwen backbone + rank-16 LoRA + pointer head, merged BF16 weights, served with cross-request dynamic batching on one Modal GPU; cost = GPU + provisioned CPU/RAM list price ÷ measured answers per second at that concurrency (no idle time, no startup).','',
'## Headline: best operating point per system','',
'| System | Accuracy (975) | 95% CI | p50 latency, 1 in flight | Best throughput | Cost / 1k answers at best throughput | vs Jev cost |','|---|---:|---:|---:|---:|---:|---:|']
best={}
for a in arms:
 rs=[r for r in rows if r['arm']==a];r1=next((r for r in rs if r['concurrency']==1),rs[0])
 if a=='jev':b=max(rs,key=lambda r:r['answers_per_second']);cost=jev_cost
 else:b=min(rs,key=lambda r:r['cost_per_1k']);cost=b['cost_per_1k']
 k=round(r1['accuracy']*975);lo,hi=wilson(k,975);best[a]=dict(acc=r1['accuracy'],p50=r1['p50_ms'],tp=b['answers_per_second'],conc=b['concurrency'],cost=cost)
 lines.append(f"| {LABELS[a]} | **{r1['accuracy']*100:.1f}%** | {lo*100:.1f}–{hi*100:.1f} | {r1['p50_ms']:.0f} ms | {b['answers_per_second']:.0f}/s @ c={b['concurrency']} | ${cost:.4f} | {f'{jev_cost/cost:.1f}x cheaper' if cost<=jev_cost else f'{cost/jev_cost:.1f}x more expensive'} |" if a!='jev' else f"| {LABELS[a]} | {r1['accuracy']*100:.1f}% | {lo*100:.1f}–{hi*100:.1f} | {r1['p50_ms']:.0f} ms | {b['answers_per_second']:.0f}/s @ c={b['concurrency']} (client-limited) | ${cost:.4f} | — |")
lines+=['','Accuracy is from the concurrency-1 pass; every other level re-answered all 975 questions and matched to within a few near-tied answers (see per-level table). Jev throughput is limited by how many requests the client kept in flight, not by Jev.','','![headline](throughput_cost.png)','','## Every concurrency level','','| Arm | GPU | Conc. | Accuracy | p50 ms | p95 ms | Answers/s | GPU busy | Cost / 1k |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for a in arms:
 for r in sorted((r for r in rows if r['arm']==a),key=lambda r:r['concurrency']):
  lines.append('| {} | {} | {} | {:.2f}% | {:.0f} | {:.0f} | {:.1f} | {} | ${:.4f} |'.format(a,r['gpu'],r['concurrency'],r['accuracy']*100,r['p50_ms'],r['p95_ms'],r['answers_per_second'],fmt(r['gpu_busy'],'{:.0%}'),r['cost_per_1k']))
lines+=['','![latency](latency.png)','','![accuracy](accuracy.png)','']
if M:
 lines+=['## Shared context, 1 / 5 / 20 questions per request (one request in flight)','','Four invoice contexts × 20 yes/no questions (synthetic, out-of-distribution for our training data; Jev was stronger here in earlier sessions). e2e p50 from the cloud client; GPU ms is the server-side forward for all question rows of the request.','','| System | Questions/request | Accuracy | e2e p50 | e2e p95 | GPU ms | Cost / 1k answers |','|---|---:|---:|---:|---:|---:|---:|']
 for r in M['rows']:
  lines.append('| {} | {} | {} | {:.0f} ms | {:.0f} ms | {} | {} |'.format(LABELS.get(r['arm'],r['arm']),r['size'],fmt(r['accuracy'],'{:.1%}'),r['p50_ms'],r['p95_ms'],fmt(r['gpu_ms'],'{:.0f}'),fmt(r['cost_per_1k'],'${:.4f}')))
 lines+=['','![multiq](multiq.png)','']
# training summary
lines+=['## Training','','One epoch of LoRA (rank 16, attention/DeltaNet projections) + 256-d pointer head on the 19,792-decision public mix (rows ≤4096 tokens kept), paired option orders with a KL consistency term, one B200 each. Holdout accuracy during training (never used for selection; final checkpoint used):','','| Model | 4,096 ex | 8,192 ex | 12,288 ex | 16,384 ex | Final | Final, options shuffled | Train time |','|---|---:|---:|---:|---:|---:|---:|---:|']
jobs=json.loads((train/'jobs.json').read_text())
PY='/tmp/jev-modal-latency-env/bin/python'
for key,job in jobs.items():
 tmp=pathlib.Path(tempfile.mkdtemp())/'p'
 if subprocess.run([PY,'-m','modal','volume','get','jev-decision-training',job['volume_run']+'/progress.jsonl',str(tmp),'--force'],capture_output=True).returncode!=0:continue
 ev={};last=None
 for l in tmp.read_text().splitlines():
  d=json.loads(l)
  if 'eval' in d:ev[d['eval']]=d['accuracy']
  if 'examples' in d:last=d
 g=lambda k:f"{ev[k]*100:.1f}%" if k in ev else ''
 lines.append(f"| {job['model']} | {g('development_at_4096')} | {g('development_at_8192')} | {g('development_at_12288')} | {g('development_at_16384')} | **{g('trained_development')}** | {g('trained_development_permuted')} | {last['elapsed_seconds']/60:.0f} min |" if last else f"| {job['model']} | | | | | | | |")
lines+=['','## Serving notes','','- Right padding on a causal model needs no attention mask for the positions read; masked vs unmasked outputs were identical on 40 mixed-length rows (`/parity`).','- Triton autotunes fla\'s DeltaNet kernels per (batch, length) key, ~8 s each; the server reuses the first tuned config per kernel for new keys and pre-warms a fixed set of batch sizes (`server.py`).','- Length-sorted, marginal-cost sub-batching (16-token rounding). Padding + per-forward overhead still cost ~20–30% versus the 52k tokens/s (B200) / 30k tokens/s (H100) forward ceiling; sequence packing would recover most of that.','- Cost excludes startup (~40–120 s), idle time, and any provider margin; it is the operating cost of a saturated GPU at Modal list price, checked 2026-09-22.','']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'headline.json').write_text(json.dumps(best,indent=2));print('\n'.join(lines[:14]))
