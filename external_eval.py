"""Score the frozen external set (Jev's 102 official workflow questions + 160 public-task slice) on deployed servers,
and compare with Jev's saved answers on the same rows.

usage: external_eval.py arm=url [arm=url ...]
"""
import sys,json,time,datetime,collections,concurrent.futures,requests
from pathlib import Path
here=Path(__file__).resolve().parent
rows=[json.loads(l) for l in (here/'eval_sets/jev_official_262.jsonl').read_text().splitlines()]
jev={r['id']:r for r in (json.loads(l) for l in (here/'results/jev_official_answers.jsonl').read_text().splitlines())}
assert len(rows)==262 and all(r['id'] in jev for r in rows)
out=here/('external_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir()
OFFICIAL=lambda s:s.startswith('typesafe_')
def score(arm,url):
 s=requests.Session()
 for _ in range(200):
  try:
   r=s.post(url+'/ping',json={},timeout=1800)
   if r.status_code==200:break
  except Exception:pass
  time.sleep(10)
 meta=r.json();print('READY',arm,meta['gpu'],meta['training_examples'],flush=True)
 def one(x):
  for attempt in range(3):
   try:
    d=s.post(url+'/score',json={'state':x['state'],'questions':[{'id':'q','instructions':x['instructions'],'criteria':x['criteria']}]},timeout=600).json()
    a=d['answers']['q'];return dict(id=x['id'],source=x['source'],expected=x['expected'],choice=a['choice'],probabilities=a['probabilities'])
   except Exception as e:err=repr(e);time.sleep(2)
  return dict(id=x['id'],source=x['source'],expected=x['expected'],choice=None,error=err)
 with concurrent.futures.ThreadPoolExecutor(8) as pool:res=list(pool.map(one,rows))
 (out/(arm+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in res)+'\n');(out/(arm+'_metadata.json')).write_text(json.dumps(meta,indent=2))
 return res
arms={a.split('=',1)[0]:a.split('=',1)[1] for a in sys.argv[1:]}
results={arm:score(arm,url) for arm,url in arms.items()}
results['jev']=[dict(id=r['id'],source=r['source'],expected=r['expected'],choice=r.get('choice')) for r in (jev[x['id']] for x in rows)]
lines=['# External set: Jev official workflow questions + public-task slice','','| Arm | Official 102 | invoices 29 | agent traces 28 | customer service 24 | security 21 | Public slice 160 | errors |','|---|---:|---:|---:|---:|---:|---:|---:|']
summary={}
for arm,res in results.items():
 by=collections.defaultdict(lambda:[0,0])
 for r in res:
  by[r['source']][0]+=r['choice']==r['expected'];by[r['source']][1]+=1
 off=[sum(by[s][0] for s in by if OFFICIAL(s)),sum(by[s][1] for s in by if OFFICIAL(s))];pub=[sum(by[s][0] for s in by if not OFFICIAL(s)),sum(by[s][1] for s in by if not OFFICIAL(s))]
 g=lambda s:f"{by[s][0]}/{by[s][1]}"
 summary[arm]=dict(official=off,public=pub,by_source={k:v for k,v in by.items()})
 lines.append(f"| {arm} | **{off[0]}/{off[1]}** ({off[0]/off[1]*100:.1f}%) | {g('typesafe_invoice_processing')} | {g('typesafe_agent_trace_observability')} | {g('typesafe_customer_service')} | {g('typesafe_security_incidents')} | {pub[0]}/{pub[1]} ({pub[0]/pub[1]*100:.1f}%) | {sum(r.get('choice') is None for r in res)} |")
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2));print('\n'.join(lines));print(out)
