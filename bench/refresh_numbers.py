"""Build post/numbers.json (and post/holdout_latency_by_length.json) from one load-test run and one multi-question run,
so every number in the README charts comes from a single run with the same client and server version.

usage: bench/refresh_numbers.py <load_run_name> <multiq_summary.json> [arm=key ...]
  load_run_name        run directory on the jev-benchmark-results volume (e.g. final8_20260923-001846), summarized by bench/report.py
  multiq_summary.json  summary written by bench/report_multiq.py for the cache-busted 1/5/20 run
  arm=key              arm name -> chart key mapping (default: jev=jev small4b=4b small9b=9b dense27jb=27b)
Prints the README table rows as markdown.
"""
import sys,json,pathlib,subprocess,tempfile,statistics
PY=sys.executable;ROOT=pathlib.Path(__file__).resolve().parents[1]
run=sys.argv[1];mq=json.loads(pathlib.Path(sys.argv[2]).read_text())
keymap=dict(a.split('=') for a in sys.argv[3:]) or {'jev':'jev','small4b':'4b','small9b':'9b','dense27jb':'27b'}
tmp=pathlib.Path(tempfile.mkdtemp())
def fetch(name):
 p=tmp/name.replace('/','_');r=subprocess.run([PY,'-m','modal','volume','get','jev-benchmark-results',run+'/'+name,str(p),'--force'],capture_output=True,text=True);assert r.returncode==0,name+'\n'+r.stdout[-500:]+r.stderr[-500:];return p
rows=[json.loads(l) for l in fetch('summary.jsonl').read_text().splitlines() if '"sweep"' in l]
RATES={'NVIDIA B200':0.001736,'NVIDIA H100 80GB HBM3':0.001097};CPU=8*0.0000131;RAM=96*0.00000222
def cost(r):
 if r['arm']=='jev':return r['total_cost_usd']/r['requests']*1000
 return (RATES[r['gpu']]+CPU+RAM)/r['answers_per_second']*1000
gpus={}
for name in keymap:
 try:gpus[name]=json.loads(fetch(name+'_metadata.jsonl').read_text().splitlines()[0]).get('gpu')
 except AssertionError:gpus[name]=None
for r in rows:r['gpu']=gpus.get(r['arm'])
hero={};curve={}
for name,key in keymap.items():
 rs=sorted((r for r in rows if r['arm']==name),key=lambda r:r['concurrency']);assert rs,name
 c1=rs[0];assert c1['concurrency']==1
 best=min(rs,key=cost);hero[key]=dict(accuracy=100*c1['correct']/c1['requests'],p50_ms=c1['p50']*1000,p95_ms=c1['p95']*1000,cost_per_1k=cost(best),best_concurrency=best['concurrency'],answers_per_second=best['answers_per_second'])
 curve[key]=dict(levels=[r['concurrency'] for r in rs],cost_per_1k=[cost(r) for r in rs])
# latency by input length from the concurrency-1 pass; tokens per case from our own server's padded row length
sw={}
for name,key in keymap.items():
 sw[key]=[r for r in (json.loads(l) for l in fetch(name+'_sweep.jsonl').read_text().splitlines()) if r.get('concurrency')==1 and 'error' not in r]
tokkey=next(k for k in ['4b','9b','27b'] if k in sw);tok={r['case']:r['rows'][0]['batch_padded_tokens'] for r in sw[tokkey]}
edges=[0,256,512,1024,2048,8192];labels=['<256','256-512','512-1k','1k-2k','2k+'];bl={'edges':edges,'labels':labels,'run':run,'arms':{}}
for key,rs in sw.items():
 per=[]
 for lo,hi,lab in zip(edges,edges[1:],labels):
  sel=[r for r in rs if lo<=tok[r['case']]<hi];secs=sorted(r['seconds']*1000 for r in sel)
  per.append(dict(bucket=lab,n=len(sel),p50=secs[len(secs)//2],p95=secs[int(len(secs)*.95)],mean_tokens=statistics.mean(tok[r['case']] for r in sel)))
 bl['arms'][key]=per
mqn={}
for r in mq['rows']:
 key=keymap.get(r['arm'])
 if key:mqn.setdefault(key,{})[str(r['size'])]=dict(p50_ms=r['p50_ms'],accuracy=r['accuracy'],gpu_ms=r.get('gpu_ms'),path=r.get('path'))
out=dict(load_run=run,multiq_run=mq.get('metadata',{}).get('run') or sys.argv[2],hero=hero,cost_curve=curve,multiq=mqn)
(ROOT/'post/numbers.json').write_text(json.dumps(out,indent=1));(ROOT/'post/holdout_latency_by_length.json').write_text(json.dumps(bl,indent=1))
NAMES={'jev':'Hosted Jev','4b':'Qwen3.5-4B','9b':'Qwen3.5-9B','27b':'Qwen3.8-27B'}
print('\n| Model | Accuracy | Speed (end to end) | Cost per 1k |\n|---|---:|---:|---:|')
for k in ['jev','4b','9b','27b']:
 if k in hero:h=hero[k];print(f"| {NAMES[k]} | {h['accuracy']:.1f}% | {h['p50_ms']:.0f} ms | ${h['cost_per_1k']:.4f} |")
print('\n| Model | '+' | '.join(['under 256 tokens','256 to 512','512 to 1k','1k to 2k','over 2k'])+' |\n|---|---:|---:|---:|---:|---:|')
for k in ['jev','4b','9b','27b']:
 if k in bl['arms']:print(f"| {NAMES[k]} | "+' | '.join(f"{b['p50']:.0f} ms" for b in bl['arms'][k])+' |')
print('\n| Model | 1 question | 5 questions | 20 questions |\n|---|---:|---:|---:|')
for k in ['jev','4b','9b','27b']:
 if k in mqn:print(f"| {NAMES[k]} | "+' | '.join(f"{mqn[k][s]['p50_ms']:.0f} ms" for s in ['1','5','20'])+' |')
print('\nbest concurrency / answers per second:',{k:(h['best_concurrency'],round(h['answers_per_second'],1)) for k,h in hero.items()})
