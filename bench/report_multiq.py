"""Summarize a multiq run: per arm and questions-per-request, median/p95 e2e, GPU time, accuracy, cost.

usage: bench/report_multiq.py <multiq_dir>   -> writes <dir>/summary.json and REPORT.md
"""
import sys,json,subprocess,tempfile,pathlib,statistics,collections
PY='/tmp/jev-modal-latency-env/bin/python'
ROOT=pathlib.Path(__file__).resolve().parents[1]
def rundir(a):   # a run directory: as given, or by name under <repo>/runs/
 p=pathlib.Path(a);return p if p.exists() or not (ROOT/'runs'/a).exists() else ROOT/'runs'/a
def fetch(remote,dest):
 return subprocess.run([PY,'-m','modal','volume','get','jev-benchmark-results',remote,str(dest),'--force'],capture_output=True,text=True).returncode==0
d=rundir(sys.argv[1]);tmp=pathlib.Path(tempfile.mkdtemp())
r=subprocess.run([PY,'-m','modal','volume','ls','jev-benchmark-results',d.name],capture_output=True,text=True);arms=sorted({l.split('/')[-1].split('_')[0].replace('.jsonl','') for l in r.stdout.split() if l.endswith('.jsonl') and not l.endswith('_warmup.jsonl') and not l.endswith('_metadata.jsonl')})
rows=[];meta={}
for arm in arms:
 if not fetch(d.name+'/'+arm+'.jsonl',tmp/(arm+'.jsonl')):continue
 if fetch(d.name+'/'+arm+'_metadata.jsonl',tmp/(arm+'.meta')):meta[arm]=json.loads((tmp/(arm+'.meta')).read_text().splitlines()[-1])
 recs=[json.loads(l) for l in (tmp/(arm+'.jsonl')).read_text().splitlines()]
 by=collections.defaultdict(list)
 for r in recs:by[r['size']].append(r)
 for size,rs in sorted(by.items()):
  ok=[r for r in rs if 'error' not in r];secs=sorted(r['seconds'] for r in ok);gpu=[r['gpu_seconds'] for r in ok if r.get('gpu_seconds')]
  cost=sum((r.get('usage') or {}).get('cost',0) or 0 for r in ok);sh=[r['shared'] for r in ok if r.get('shared')];srv=[r['server_seconds'] for r in ok if r.get('server_seconds')]
  med=lambda xs:statistics.median(xs) if xs else None
  rows.append(dict(arm=arm,size=size,requests=len(rs),errors=len(rs)-len(ok),accuracy=sum(r['correct'] for r in ok)/(size*len(ok)) if ok else None,p50_ms=statistics.median(secs)*1000 if secs else None,p95_ms=secs[int(len(secs)*.95)]*1000 if secs else None,gpu_ms=statistics.median(gpu)*1000 if gpu else None,cost_per_1k=cost/(size*len(ok))*1000 if cost else None,path=collections.Counter(r.get('path') for r in ok).most_common(1)[0][0] if ok else None,prefill_ms=med([x['prefill_ms'] for x in sh]),fork_ms=med([x['fork_ms'] for x in sh]),branch_ms=med([x['branch_ms'] for x in sh]),server_ms=med(srv)*1000 if srv else None,network_ms=(statistics.median(secs)-med(srv))*1000 if srv and secs else None))
lines=['# Multi-question latency '+d.name,'','Questions per request on one shared context, 5 repetitions each, one request in flight, cloud client in us-west. GPU = server-side forward (prefill + fork + branches on the shared path); network = e2e minus server time.','','| Arm | Q/req | Accuracy | e2e p50 | e2e p95 | GPU | prefill | fork | branches | network | path | Cost / 1k |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
def fmt(v,f):return '' if v is None else f.format(v)
for r in rows:lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(r['arm'],r['size'],fmt(r['accuracy'],'{:.1%}'),fmt(r['p50_ms'],'{:.0f} ms'),fmt(r['p95_ms'],'{:.0f} ms'),fmt(r['gpu_ms'],'{:.0f} ms'),fmt(r['prefill_ms'],'{:.0f}'),fmt(r['fork_ms'],'{:.0f}'),fmt(r['branch_ms'],'{:.0f}'),fmt(r['network_ms'],'{:.0f} ms'),r.get('path') or '',fmt(r['cost_per_1k'],'${:.4f}')))
for arm,m in meta.items():lines.append(f"\n- **{arm}**: {m.get('model')} · `{m.get('checkpoint_run')}` · {m.get('gpu')} in {m.get('region')}")
(d/'summary.json').write_text(json.dumps(dict(rows=rows,metadata=meta),indent=2));(d/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
