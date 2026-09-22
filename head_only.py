"""Head-only ablation: frozen Qwen3.5-4B backbone (no LoRA), pointer heads trained on cached hidden states.

One forward pass per row (no backward through the backbone). Caches the hidden vectors at each option's last token
and at the decision token, at several depths, then trains a 256-d pointer head per depth (and on a concatenation of
depths) with cross-entropy over the offered options. Reports accuracy on the 975 holdout and Jev's official set.
"""
import sys,json,datetime
from pathlib import Path
import modal
from modal_image import blackwell_image as image
cache=modal.Volume.from_name('jev-model-cache');runs=modal.Volume.from_name('jev-decision-training')
app=modal.App('jev-beat-head-only')
@app.function(image=image,gpu='B200',cpu=8,memory=131072,timeout=4*3600,volumes={'/cache':cache,'/runs':runs})
def run(payload,runid):
 import os,time,random,math,collections,torch
 from transformers import AutoTokenizer,AutoModelForImageTextToText
 os.environ['FLA_USE_COMPILE']='0';torch.manual_seed(814);out=Path('/runs')/runid;out.mkdir(exist_ok=True);t0=time.perf_counter()
 name,revision=payload['model'];tok=AutoTokenizer.from_pretrained(name,revision=revision)
 full=AutoModelForImageTextToText.from_pretrained(name,revision=revision,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
 backbone=full.model.language_model;del full;torch.cuda.empty_cache()
 L=backbone.config.num_hidden_layers;depths=[d for d in payload['depths'] if d<=L];h=backbone.config.hidden_size
 def log(v):
  print(json.dumps(v),flush=True)
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
 def encode(r,seed=None):
  keys=list(r['criteria'])
  if seed is not None:random.Random(seed).shuffle(keys)
  ins=r['instructions'] if isinstance(r['instructions'],str) else json.dumps(r['instructions'],separators=(',',':'))
  state=r['state'] if isinstance(r['state'],str) else json.dumps(r['state'],separators=(',',':'))
  ids=tok.encode('State:\n'+state+'\nQuestion: '+ins+'\nOptions:\n',add_special_tokens=False);positions=[]
  for k in keys:
   desc=r['criteria'][k];ids+=tok.encode(str(k)+': '+str(desc or k)+'\n',add_special_tokens=False);positions.append(len(ids)-1)
  ids+=tok.encode('Decision:',add_special_tokens=False)
  return dict(ids=ids,positions=positions,keys=keys,label=keys.index(r['expected']),row=r)
 @torch.inference_mode()
 def extract(rows,label,max_tokens):
  """Returns list of (feats[dep][n_opt+1, h] as fp16 cpu, label, keys, source)."""
  rows=[x for x in rows if len(x['ids'])<=max_tokens];order=sorted(range(len(rows)),key=lambda i:len(rows[i]['ids']));feats=[None]*len(rows);clock=time.perf_counter();done=0
  i=0
  while i<len(order):
   longest=len(rows[order[i]]['ids']);j=i+1
   while j<len(order) and ((len(rows[order[j]]['ids'])+15)//16)*16*(j+1-i)<=payload['batch_tokens']:j+=1
   xs=[rows[k] for k in order[i:j]];length=((max(len(x['ids']) for x in xs)+15)//16)*16
   inp=torch.full((len(xs),length),tok.pad_token_id or 0,dtype=torch.long,device='cuda')
   for b,x in enumerate(xs):inp[b,:len(x['ids'])]=torch.tensor(x['ids'],device='cuda')
   hs=backbone(input_ids=inp,use_cache=False,output_hidden_states=True).hidden_states
   for b,x in enumerate(xs):
    pos=torch.tensor(x['positions']+[len(x['ids'])-1],device='cuda')
    feats[order[i+b]]={d:hs[d][b,pos].float().cpu().half() for d in depths}
   done+=len(xs);i=j
   if done%2000<len(xs):log({'extract':label,'done':done,'total':len(rows),'seconds':time.perf_counter()-clock})
  log({'extract':label,'done':done,'total':len(rows),'seconds':time.perf_counter()-clock})
  return [(f,x['label'],x['keys'],x['row']['source'],x['row']['expected']) for f,x in zip(feats,rows)]
 train=extract([encode(r,814+i) for i,r in enumerate(payload['train'])],'train',payload['max_tokens'])
 dev=extract([encode(r) for r in payload['development']],'holdout',16384);ext=extract([encode(r) for r in payload['external']],'external',16384)
 def tensors(items,ds):
  n=len(items);K=max(len(it[2]) for it in items);dim=h*len(ds)
  D=torch.zeros(n,dim,dtype=torch.float16);O=torch.zeros(n,K,dim,dtype=torch.float16);M=torch.zeros(n,K,dtype=torch.bool);Y=torch.tensor([it[1] for it in items])
  for i,it in enumerate(items):
   f=torch.cat([it[0][d] for d in ds],dim=-1);D[i]=f[-1];O[i,:f.shape[0]-1]=f[:-1];M[i,:f.shape[0]-1]=True
  return D,O,M,Y   # CPU fp16; batches are moved to the GPU as float32
 def train_head(cfg,epochs=8,lr=1e-3,bs=64):
  mix=isinstance(cfg,dict);ds=cfg['mix'] if mix else cfg;nl=len(ds)
  q=torch.nn.Linear(h,256,bias=False,device='cuda');k=torch.nn.Linear(h,256,bias=False,device='cuda')
  if mix:
   lns=torch.nn.ModuleList([torch.nn.LayerNorm(h,device='cuda') for _ in ds]);w=torch.nn.Parameter(torch.zeros(nl,device='cuda'));gamma=torch.nn.Parameter(torch.ones(1,device='cuda'))
   params=list(q.parameters())+list(k.parameters())+list(lns.parameters())+[w,gamma]
   def combine(X):   # X: [..., nl*h] -> softmax-weighted sum of per-layer normalized slices
    parts=X.view(*X.shape[:-1],nl,h);sm=torch.softmax(w,0)
    return gamma*sum(sm[i]*lns[i](parts[...,i,:]) for i in range(nl))
  else:
   ln=torch.nn.LayerNorm(h,device='cuda');params=list(q.parameters())+list(k.parameters())+list(ln.parameters());combine=ln
  opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
  D,O,M,Y=tensors(train,ds);n=D.shape[0];steps=0;total=epochs*math.ceil(n/bs)
  def logits(D,O,M):
   z=(k(combine(O))*q(combine(D))[:,None]).sum(-1)/16;return z.masked_fill(~M,-1e4)
  for ep in range(epochs):
   perm=torch.randperm(n)
   for s in range(0,n,bs):
    b=perm[s:s+bs];loss=torch.nn.functional.cross_entropy(logits(D[b].cuda().float(),O[b].cuda().float(),M[b].cuda()),Y[b].cuda());loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);steps+=1
    for g in opt.param_groups:g['lr']=lr*.5*(1+math.cos(math.pi*steps/total))
    opt.step();opt.zero_grad(set_to_none=True)
  def evaluate(items):
   Dv,Ov,Mv,Yv=tensors(items,ds);by=collections.defaultdict(lambda:[0,0]);answers=[]
   pred=[]
   with torch.no_grad():
    for s in range(0,Dv.shape[0],64):pred+=logits(Dv[s:s+64].cuda().float(),Ov[s:s+64].cuda().float(),Mv[s:s+64].cuda()).argmax(-1).cpu().tolist()
   for it,pr in zip(items,pred):
    ok=pr==it[1];by[it[3]][0]+=ok;by[it[3]][1]+=1;answers.append(dict(source=it[3],expected=it[4],choice=it[2][pr]))
   return dict(accuracy=sum(v[0] for v in by.values())/max(1,sum(v[1] for v in by.values())),by_source={s:v for s,v in by.items()},answers=answers)
  return evaluate(dev),evaluate(ext),(torch.softmax(w,0).detach().cpu().tolist() if mix else None)
 results={}
 for ds in payload['configs']:
  clock=time.perf_counter();d,e,weights=train_head(ds);off=[sum(v[0] for s,v in e['by_source'].items() if s.startswith('typesafe_')),sum(v[1] for s,v in e['by_source'].items() if s.startswith('typesafe_'))];pub=[sum(v[0] for s,v in e['by_source'].items() if not s.startswith('typesafe_')),sum(v[1] for s,v in e['by_source'].items() if not s.startswith('typesafe_'))]
  results[str(ds)]=dict(depths=ds,mix_weights=weights,holdout_accuracy=d['accuracy'],holdout_by_source=d['by_source'],official=off,public=pub,train_seconds=time.perf_counter()-clock,external_answers=e['answers'])
  log({'config':ds,'holdout':d['accuracy'],'official':off,'public':pub,'mix_weights':weights,'train_seconds':time.perf_counter()-clock})
  (out/'results.json').write_text(json.dumps(results,indent=1));runs.commit()
 (out/'status.json').write_text(json.dumps({'status':'complete','elapsed':time.perf_counter()-t0,'layers':L,'train_rows':len(train)}));runs.commit()
 return {k:{kk:vv for kk,vv in v.items() if kk!='external_answers'} for k,v in results.items()}
if __name__=='__main__':
 import gzip
 here=Path(__file__).resolve().parent
 payload=dict(model=('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),depths=list(range(12,33)),configs=[[18],[19],{'mix':[17,18,19,20,21]},{'mix':list(range(14,25))},{'mix':list(range(12,33))},{'mix':[18,20,24,28,32]}],max_tokens=4096,batch_tokens=32768,
  train=[json.loads(l) for l in gzip.decompress((here/'data/train.jsonl.gz').read_bytes()).decode().splitlines()],development=[json.loads(l) for l in (here/'eval_sets/public_holdout_975.jsonl').read_text().splitlines()],external=[json.loads(l) for l in (here/'eval_sets/jev_official_262.jsonl').read_text().splitlines()])
 out=here/('headmix_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir();(out/'head_only.py').write_text(Path(__file__).read_text())
 with app.run(detach=True):
  call=run.spawn(payload,out.name);job=dict(app_id=app.app_id,call_id=call.object_id,volume_run=out.name);(out/'job.json').write_text(json.dumps(job,indent=2));print(json.dumps(job),flush=True)
