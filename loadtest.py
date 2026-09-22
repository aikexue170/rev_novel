"""Cloud-side load test: accuracy, latency and throughput of decision servers across concurrency levels.

Each arm answers the exact 975-question public holdout (one question per request) at every concurrency level,
so accuracy is verified under load and throughput is answers per wall-second. Hosted Jev is run at low
concurrency only (its throughput is the provider's to scale; we need its latency, accuracy and billed cost).
"""
import modal,json,datetime,hashlib,sys
from pathlib import Path
app=modal.App('jev-beat-loadtest');vol=modal.Volume.from_name('jev-benchmark-results',create_if_missing=True)
image=modal.Image.debian_slim(python_version='3.12').pip_install('requests')
@app.function(image=image,region='us-west',timeout=4*3600,volumes={'/results':vol},secrets=[modal.Secret.from_name('openrouter')])
def run(arms,cases,runid):
 import os,time,requests,concurrent.futures,random,threading,collections
 out=Path('/results')/runid;out.mkdir(exist_ok=True);lock=threading.Lock()
 def save(name,data):
  with lock:
   with (out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(data)+'\n')
 def post(s,url,body,headers=None,startup=False,timeout=120):
  start=time.perf_counter()
  for attempt in range(200 if startup else 3):
   try:r=s.post(url,json=body,headers=headers,timeout=1800 if startup else timeout)
   except requests.RequestException as e:
    if startup:time.sleep(5);continue
    raise
   if r.status_code in [429,502,503,504]:time.sleep(5 if startup else 2);continue
   r.raise_for_status();return r.json(),time.perf_counter()-start,attempt+1
  raise RuntimeError('Retries exhausted '+url)
 def bench(arm):
  name=arm['name'];hosted=arm.get('url')=='jev';s=requests.Session()
  adapter=requests.adapters.HTTPAdapter(pool_connections=128,pool_maxsize=128);s.mount('https://',adapter)
  if not hosted:
   url=arm['url'];meta,cold,_=post(s,url+'/ping',{},startup=True);save(name+'_metadata',dict(**meta,cold_seconds=cold));print('READY',name,meta.get('gpu'),meta.get('region'),flush=True)
  def request(c):
   q=c['question'];started=time.perf_counter()
   try:
    if hosted:
     data,elapsed,attempts=post(s,'https://openrouter.ai/api/alpha/decisions',dict(model='typesafe/jev-1.13',provider={'only':['TypeSafe'],'allow_fallbacks':False},state=c['state'],questions={'q':{'type':'choice','instructions':q['instructions'],'criteria':q['criteria']}}),{'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']},timeout=45)
    else:
     data,elapsed,attempts=post(s,url+'/score',dict(state=c['state'],questions=[{'id':'q','instructions':q['instructions'],'criteria':q['criteria']}]))
    a=data['answers']['q'];assert a['choice'] in q['criteria'];ps=a['probabilities'];assert set(ps)==set(q['criteria']) and abs(sum(ps.values())-1)<.03
    return dict(case=c['id'],source=q['source'],expected=q['expected'],choice=a['choice'],probabilities=ps,correct=a['choice']==q['expected'],seconds=elapsed,attempts=attempts,usage=data.get('usage'),rows=data.get('rows'),server_seconds=data.get('server_seconds'))
   except Exception as e:
    return dict(case=c['id'],source=q['source'],expected=q['expected'],error=repr(e),correct=False,seconds=time.perf_counter()-started)
  def sweep(level,tag,subset=None):
   order=list(cases if subset is None else subset);random.Random(1000+level).shuffle(order)
   if not hosted:post(s,url+'/reset_stats',{})
   t0=time.perf_counter();results=[]
   with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
    for r in pool.map(request,order):results.append(r)
   wall=time.perf_counter()-t0
   for r in results:save(name+'_'+tag,dict(**r,concurrency=level))
   stats=post(s,url+'/stats',{})[0] if not hosted else None
   secs=sorted(r['seconds'] for r in results if 'error' not in r)
   summary=dict(arm=name,tag=tag,concurrency=level,requests=len(results),errors=sum('error' in r for r in results),correct=sum(r['correct'] for r in results),wall_seconds=wall,answers_per_second=len(results)/wall,p50=secs[len(secs)//2] if secs else None,p95=secs[int(len(secs)*.95)] if secs else None,mean=sum(secs)/len(secs) if secs else None,server_stats=stats,total_cost_usd=sum((r.get('usage') or {}).get('cost',0) or 0 for r in results) if hosted else None)
   save('summary',summary);print('LEVEL',json.dumps({k:v for k,v in summary.items() if k!='server_stats'}),flush=True)
   with lock:vol.commit()
   return summary
  if hosted:
   sweep(2,'warmup',cases[:20])
   for level in arm.get('levels',[1,4]):sweep(level,'sweep')
  else:
   sweep(8,'warmup')
   for _ in range(15):_,e,_=post(s,url+'/noop',{});save(name+'_noop',{'seconds':e})
   for level in arm.get('levels',[1,2,4,8,16,32,64]):sweep(level,'sweep')
  print('DONE',name,flush=True)
 status={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as pool:
  fs={pool.submit(bench,a):a['name'] for a in arms}
  for f in concurrent.futures.as_completed(fs):
   try:f.result();status[fs[f]]='complete'
   except Exception as e:status[fs[f]]=repr(e);print('ERROR',fs[f],repr(e),flush=True)
   with lock:(out/'status.json').write_text(json.dumps(status));vol.commit()
 return {p.name:p.read_text() for p in out.iterdir() if p.is_file()}
if __name__=='__main__':
 here=Path(__file__).resolve().parent
 arms=json.loads(sys.argv[1]);label=sys.argv[2] if len(sys.argv)>2 else 'load'
 out=here/(label+'_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir()
 source=here/'eval_sets/public_holdout_975.jsonl';raw=source.read_bytes();rows=[json.loads(l) for l in raw.splitlines()];assert len(rows)==975
 cases=[{'id':r['id'],'state':r['state'],'question':{'instructions':r['instructions'],'criteria':r['criteria'],'expected':r['expected'],'source':r['source'],'group':r['group']}} for r in rows]
 (out/'manifest.json').write_text(json.dumps({'holdout_sha256':hashlib.sha256(raw).hexdigest(),'rows':975,'arms':arms},indent=2));(out/'loadtest.py').write_text(Path(__file__).read_text())
 with app.run(detach=True):
  call=run.spawn(arms,cases,out.name);job=dict(app_id=app.app_id,call_id=call.object_id,output=str(out));(out/'job.json').write_text(json.dumps(job,indent=2));print(json.dumps(job),flush=True)
