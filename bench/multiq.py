"""Shared-context multi-question latency: 1, 5 and 20 questions per request on the same state, one request in flight.

Default case set: cases/invoice_cases.json (4 invoices x 20 questions); cases/contract_cases_test.json for the 2k-token contracts.
Each arm answers every case at each size, `reps` times, in shuffled order. Reports e2e latency, GPU time, accuracy.
"""
import modal,json,datetime,sys
from pathlib import Path
app=modal.App('jev-beat-multiq');vol=modal.Volume.from_name('jev-benchmark-results',create_if_missing=True)
image=modal.Image.debian_slim(python_version='3.12').pip_install('requests')
@app.function(image=image,region='us-west',timeout=2*3600,volumes={'/results':vol},secrets=[modal.Secret.from_name('openrouter')])
def run(arms,cases,runid,sizes,reps,cachebust=False):
 import os,time,requests,random,threading,concurrent.futures,uuid
 out=Path('/results')/runid;out.mkdir(exist_ok=True);lock=threading.Lock()
 def save(name,data):
  with lock:
   with (out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(data)+'\n')
 def post(s,url,body,headers=None,startup=False):
  start=time.perf_counter()
  for attempt in range(200 if startup else 3):
   try:r=s.post(url,json=body,headers=headers,timeout=1800 if startup else 60)
   except requests.RequestException:
    if startup:time.sleep(5);continue
    raise
   if r.status_code in [429,502,503,504]:time.sleep(5 if startup else 2);continue
   r.raise_for_status();return r.json(),time.perf_counter()-start,attempt+1
  raise RuntimeError('Retries exhausted '+url)
 def bench(arm):
  name=arm['name'];hosted=arm.get('url')=='jev';s=requests.Session()
  if not hosted:
   url=arm['url'];meta,cold,_=post(s,url+'/ping',{},startup=True);save(name+'_metadata',dict(**meta,cold_seconds=cold));print('READY',name,meta.get('gpu'),flush=True)
  def request(c,size):
   qs=c['questions'][:size];started=time.perf_counter()
   state=c['state']
   if cachebust:
    ref=uuid.uuid4().hex   # keep the state's type: a dict gets an extra key, a string gets a suffix (both sides see the same object)
    state=dict(state,request_reference=ref) if isinstance(state,dict) else str(state)+'\n\nRequest reference: '+ref
   try:
    if hosted:data,elapsed,_=post(s,'https://openrouter.ai/api/alpha/decisions',dict(model='typesafe/jev-1.13',provider={'only':['TypeSafe'],'allow_fallbacks':False},state=state,questions={q['id']:{'type':'choice','instructions':q['instructions'],'criteria':q['criteria']} for q in qs}),{'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
    else:data,elapsed,_=post(s,url+('/score_shared' if arm.get('mode')=='shared' else '/score'),dict(state=state,questions=[{k:q[k] for k in ['id','instructions','criteria']} for q in qs],mode=arm.get('mode','auto')))
    answers=data['answers'];assert set(answers)=={q['id'] for q in qs}
    return dict(case=c['id'],size=size,seconds=elapsed,correct=sum(answers[q['id']]['choice']==q['expected'] for q in qs),answers={q['id']:answers[q['id']]['choice'] for q in qs},usage=data.get('usage'),server_seconds=data.get('server_seconds'),gpu_seconds=max((r['gpu_seconds'] for r in data.get('rows',[])),default=None),path=data.get('path'),shared=data.get('shared'))
   except Exception as e:return dict(case=c['id'],size=size,seconds=time.perf_counter()-started,error=repr(e),correct=0)
  # warm every (case,size) once
  for c in cases:
   for size in sizes:save(name+'_warmup',request(c,size))
  work=[(c,size) for c in cases for size in sizes for _ in range(reps)];random.Random(7).shuffle(work)
  for c,size in work:save(name,request(c,size))
  print('DONE',name,flush=True)
  with lock:vol.commit()
 status={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as pool:
  fs={pool.submit(bench,a):a['name'] for a in arms}
  for f in concurrent.futures.as_completed(fs):
   try:f.result();status[fs[f]]='complete'
   except Exception as e:status[fs[f]]=repr(e);print('ERROR',fs[f],repr(e),flush=True)
   with lock:(out/'status.json').write_text(json.dumps(status));vol.commit()
 return {p.name:p.read_text() for p in out.iterdir() if p.is_file()}
if __name__=='__main__':
 ROOT=Path(__file__).resolve().parents[1];arms=json.loads(sys.argv[1]);label=sys.argv[2] if len(sys.argv)>2 else 'multiq'
 cases_file=sys.argv[3] if len(sys.argv)>3 else str(ROOT/'cases/invoice_cases.json');sizes=json.loads(sys.argv[4]) if len(sys.argv)>4 else [1,5,20];cachebust=len(sys.argv)>5 and sys.argv[5]=='bust'
 cases=json.loads(Path(cases_file).read_text());assert len(cases)>=1
 out=ROOT/'runs'/(label+'_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir(parents=True);(out/'cases.json').write_text(json.dumps(cases));(out/'multiq.py').write_text(Path(__file__).read_text())
 with app.run(detach=True):
  call=run.spawn(arms,cases,out.name,sizes,5,cachebust);job=dict(app_id=app.app_id,call_id=call.object_id,output=str(out),cachebust=cachebust,sizes=sizes,cases_file=cases_file);(out/'job.json').write_text(json.dumps(job,indent=2));print(json.dumps(job),flush=True)
