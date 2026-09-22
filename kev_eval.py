"""Score Kev's released checkpoints (jaredpalmer/kev-4b, kev-9b) on our two frozen evaluation sets, through Kev's own
serving-format path (api.to_record -> model.encode -> model.probs), one question per request like our load tests.

usage: kev_eval.py [runs...]   default: jaredpalmer/kev-4b jaredpalmer/kev-9b
"""
import sys,json,datetime
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent;KEV=Path(os.environ.get('KEV_DIR',HERE.parent/'kev'))   # a checkout of github.com/jaredpalmer/kev
app=modal.App('jev-beat-kev-eval')
cache=modal.Volume.from_name('jev-model-cache');results=modal.Volume.from_name('jev-benchmark-results')
image=(modal.Image.debian_slim(python_version='3.12').env({'HF_HOME':'/cache/huggingface','KEV_DTYPE':'bf16','KEV_STRICT':'0'})
 .pip_install('torch==2.8.0','transformers==5.17.0','peft>=0.21','accelerate>=1.15','datasets>=3.0','numpy>=2.0','pydantic>=2.9','scikit-learn','flash-linear-attention==0.5.2','huggingface_hub')
 .add_local_dir(str(KEV/'kev'),remote_path='/root/kev'))
@app.function(image=image,gpu='H100',cpu=8,memory=65536,timeout=3*3600,volumes={'/cache':cache,'/results':results})
def run(run_name,rows,runid):
 import os,time,torch,traceback
 sys.path.insert(0,'/root')
 out=Path('/results')/runid;out.mkdir(exist_ok=True,parents=True);tag=run_name.split('/')[-1].replace('@','_')
 from kev.evaluate import load
 from kev.api import to_record,SystemOneRequest,to_answers
 t0=time.perf_counter();tok,model=load(run_name,'cuda');load_s=time.perf_counter()-t0
 print('LOADED',run_name,load_s,flush=True)
 preds=[];errors=0
 for i,x in enumerate(rows):
  state=x['state']
  if isinstance(state,str):
   try:parsed=json.loads(state);state=parsed if isinstance(parsed,(dict,list)) else state
   except Exception:pass
  crit=x['criteria'] if isinstance(x['criteria'],dict) else json.loads(x['criteria'])
  req={'state':state,'questions':{'q':{'type':'choice','instructions':x['instructions'],'criteria':crit}}}
  t=time.perf_counter()
  try:
   rec,meta=to_record(SystemOneRequest.model_validate(req))
   with torch.inference_mode():probs=model.probs(model.encode(tok,rec,strict=os.environ.get('KEV_STRICT','1')=='1'))
   ans=to_answers([p.tolist() if hasattr(p,'tolist') else p for p in probs],meta)['q']
   preds.append(dict(id=x['id'],source=x['source'],expected=x['expected'],choice=ans['choice'],probabilities=ans.get('probabilities'),seconds=time.perf_counter()-t))
  except Exception as e:
   errors+=1;preds.append(dict(id=x['id'],source=x['source'],expected=x['expected'],choice=None,error=repr(e)[:300],seconds=time.perf_counter()-t))
  if (i+1)%100==0:print('PROGRESS',tag,i+1,len(rows),flush=True)
 (out/(tag+'.jsonl')).write_text('\n'.join(json.dumps(p) for p in preds)+'\n');results.commit()
 acc=sum(p['choice']==p['expected'] for p in preds)/len(preds)
 return dict(run=run_name,rows=len(preds),errors=errors,accuracy=acc,load_seconds=load_s,gpu=torch.cuda.get_device_name())
if __name__=='__main__':
 runs=sys.argv[1:] or ['jaredpalmer/kev-4b','jaredpalmer/kev-9b']
 rows=[json.loads(l) for l in (HERE/'eval_sets/public_holdout_975.jsonl').read_text().splitlines()]+[json.loads(l) for l in (HERE/'eval_sets/jev_official_262.jsonl').read_text().splitlines()]
 out=HERE/('kev_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir();jobs={}
 with app.run(detach=True):
  for r in runs:
   call=run.spawn(r,rows,out.name);jobs[r]=dict(app_id=app.app_id,call_id=call.object_id)
  (out/'jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs),flush=True)
