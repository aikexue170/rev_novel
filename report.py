"""Turn a load-test run into tables: accuracy, latency, throughput, cost per 1,000 answers at measured throughput.

usage: report.py <load_dir> [<load_dir> ...]
Fetches summary.jsonl (and per-arm metadata) from the jev-benchmark-results volume, writes <load_dir>/REPORT.md and summary.json.
"""
import sys,json,subprocess,tempfile,pathlib,collections
PY='/tmp/jev-modal-latency-env/bin/python'
# Modal list prices, USD per second, checked 2026-09-21 (modal.com/pricing). CPU/RAM are the provisioned 8 cores + 96 GiB.
GPU_RATES={'H100':3.95/3600,'B200':6.25/3600,'L40S':1.95/3600,'A100-80GB':2.50/3600,'H200':4.54/3600}
CPU_RAM=8*(0.047/3600)+96*(0.008/3600)
def fetch(volume,remote,dest):
 r=subprocess.run([PY,'-m','modal','volume','get',volume,remote,str(dest),'--force'],capture_output=True,text=True);return r.returncode==0
def gpu_key(name):
 n=(name or '').upper()
 for k in GPU_RATES:
  if k.split('-')[0] in n:return k
 return None
def load(d):
 d=pathlib.Path(d);tmp=pathlib.Path(tempfile.mkdtemp())
 assert fetch('jev-benchmark-results',d.name+'/summary.jsonl',tmp/'summary.jsonl'),'no summary for '+d.name
 rows=[json.loads(l) for l in (tmp/'summary.jsonl').read_text().splitlines()]
 meta={}
 for arm in {r['arm'] for r in rows}:
  if fetch('jev-benchmark-results',d.name+'/'+arm+'_metadata.jsonl',tmp/(arm+'.meta')):meta[arm]=json.loads((tmp/(arm+'.meta')).read_text().splitlines()[-1])
 return d,rows,meta
def main(dirs):
 for d,rows,meta in map(load,dirs):
  table=[];lines=['# Load test '+d.name,'','Exact 975-question public holdout, one question per request, cloud client in us-west. Self-hosted cost = (GPU + provisioned CPU/RAM list price per second) / measured answers per second, i.e. cost at the utilization actually achieved at that concurrency. Hosted Jev cost = billed usage returned by the API.','',
   '| Arm | GPU | Conc. | Accuracy | p50 ms | p95 ms | Answers/s | GPU busy | Cost / 1k answers |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
  for s in rows:
   if s['tag']!='sweep':continue
   arm=s['arm'];m=meta.get(arm,{});gpu=m.get('gpu','');st=s.get('server_stats') or {}
   if arm=='jev':cost=s['total_cost_usd']/s['requests']*1000 if s['total_cost_usd'] else None;gpu='hosted';util=None
   else:
    k=gpu_key(gpu);rate=GPU_RATES[k]+CPU_RAM if k else None;cost=rate/s['answers_per_second']*1000 if rate else None;util=st.get('gpu_seconds',0)/s['wall_seconds'] if st else None
   acc=s['correct']/s['requests']
   table.append(dict(arm=arm,gpu=gpu,concurrency=s['concurrency'],accuracy=acc,p50_ms=s['p50']*1000 if s['p50'] else None,p95_ms=s['p95']*1000 if s['p95'] else None,answers_per_second=s['answers_per_second'],gpu_busy=util,cost_per_1k=cost,errors=s['errors'],batches=st.get('batches'),batch_rows=st.get('batch_rows')))
   lines.append(f"| {arm} | {gpu} | {s['concurrency']} | {acc*100:.2f}% | {s['p50']*1000:.0f} | {s['p95']*1000:.0f} | {s['answers_per_second']:.1f} | {'' if util is None else f'{util*100:.0f}%'} | {'' if cost is None else f'${cost:.4f}'} |")
  lines+=['','Errors per level: '+', '.join(f"{t['arm']}@{t['concurrency']}={t['errors']}" for t in table if t['errors'])+('none' if not any(t['errors'] for t in table) else ''),'']
  for arm,m in meta.items():lines.append(f"- **{arm}**: {m.get('model')} · checkpoint `{m.get('checkpoint_run')}` ({m.get('training_examples')} training examples) · {m.get('gpu')} in {m.get('region')} · batching {m.get('batching')} · cold start {m.get('cold_seconds',0):.0f}s")
  (d/'REPORT.md').write_text('\n'.join(lines)+'\n');(d/'summary.json').write_text(json.dumps(dict(rows=table,metadata=meta),indent=2))
  print('\n'.join(lines))
if __name__=='__main__':main(sys.argv[1:])
