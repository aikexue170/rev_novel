"""Full public-data LoRA + pointer-head training for a size ladder of Qwen3.5 dense models.

Recipe is the proven dense-pilot recipe (rank-16 LoRA on attention/DeltaNet projections, 256-d pointer head,
paired option orders with KL consistency, one epoch, cosine LR). Data is the full 19,792-decision public mix
(data_v3 train), filtered to <=4096 tokens inside the job; the frozen 975-question holdout is evaluation only.
"""
import sys,os,json,datetime
from pathlib import Path
import modal
from modal_image import blackwell_image as image
cache=modal.Volume.from_name('jev-model-cache')
app=modal.App('jev-beat-train')
runs=modal.Volume.from_name('jev-decision-training',create_if_missing=True)
MODELS={'2b':('Qwen/Qwen3.5-2B','15852e8c16360a2fea060d615a32b45270f8a8fc'),'4b':('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),'9b':('Qwen/Qwen3.5-9B','c202236235762e1c871ad0ccb60c8ee5ba337b9a'),'27b':('Qwen/Qwen3.8-27B','1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')}
@app.function(image=image,gpu='B200',cpu=8,memory=98304,timeout=6*3600,volumes={'/cache':cache,'/runs':runs})
def train(payload,runid,model_key):
 import os,time,random,math,collections,traceback,torch
 from transformers import AutoTokenizer,AutoModelForImageTextToText
 os.environ['CC']='gcc';os.environ['CXX']='g++';torch.set_num_threads(4);torch.manual_seed(814)
 out=Path('/runs')/runid;out.mkdir(exist_ok=True);start=time.perf_counter()
 def save(name,value):
  (out/name).write_text(json.dumps(value,indent=2));runs.commit()
 def log(value):
  print(json.dumps(value),flush=True)
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
 def results():return {p.name:p.read_text() for p in out.iterdir() if p.suffix in ['.json','.jsonl']}
 save('manifest.json',payload['manifest'])
 try:
  name,revision=payload['manifest']['models'][model_key];tok=AutoTokenizer.from_pretrained(name,revision=revision)
  full=AutoModelForImageTextToText.from_pretrained(name,revision=revision,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa')
  backbone=full.model.language_model;revision=full.config._commit_hash;del full;torch.cuda.empty_cache();backbone.requires_grad_(False)
  class LoRA(torch.nn.Module):
   def __init__(self,base):
    super().__init__();self.base=base
    self.a=torch.nn.Parameter(torch.empty(16,base.in_features,device='cuda',dtype=torch.float32));torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    self.b=torch.nn.Parameter(torch.zeros(base.out_features,16,device='cuda',dtype=torch.float32))
   def forward(self,x):
    update=torch.nn.functional.linear(torch.nn.functional.linear(x,self.a.to(x.dtype)),self.b.to(x.dtype))
    return self.base(x)+2*update
  adapters={};targets={'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'}
  for path,module in list(backbone.named_modules()):
   if isinstance(module,torch.nn.Linear) and path.split('.')[-1] in targets:
    parent,_,attr=path.rpartition('.');a=LoRA(module);setattr(backbone.get_submodule(parent),attr,a);adapters[path]=a
  h=backbone.config.hidden_size
  headq=torch.nn.Linear(h,256,bias=False,device='cuda');headk=torch.nn.Linear(h,256,bias=False,device='cuda')
  backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  backbone.config.use_cache=False
  ap=[p for a in adapters.values() for p in [a.a,a.b]];hp=list(headq.parameters())+list(headk.parameters());params=ap+hp
  opt=torch.optim.AdamW([{'params':ap,'lr':1e-4},{'params':hp,'lr':5e-4}],weight_decay=.01)
  metadata=dict(model=name,revision=revision,gpu=torch.cuda.get_device_name(),rank=16,alpha=32,trainable_parameters=sum(p.numel() for p in params),max_tokens=4096,accumulation=4,permutation_pairs=True,consistency_weight=.1,initial_checkpoint=None,optimizer_state='fresh Adam; zero-B LoRA and random pointer head',epochs=1,seed=814,adapter_lr=1e-4,head_lr=5e-4,training='uncompiled BF16 backbone; FP32 adapter and head parameters; fresh LoRA/head; frozen pretrained base',dataset='data_v3 train, <=4096 tokens',state_format='raw',layout='State:\\n{state}\\nQuestion: {instructions}\\nOptions:\\n{key}: {desc}\\n...Decision:',load_seconds=time.perf_counter()-start)
  save('metadata.json',metadata);log({'ready':metadata})
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
  alltrain=[encode(r,814+i) for i,r in enumerate(payload['train'])];devrows=[encode(r) for r in payload['development']]
  trainrows=[x for x in alltrain if len(x['ids'])<=4096];dropped=[x['row']['id'] for x in alltrain if len(x['ids'])>4096]
  assert max(len(x['ids']) for x in devrows)<=4096
  save('filtering.json',{'train':{'before':len(alltrain),'kept':len(trainrows),'dropped_ids':dropped},'development':{'before':len(devrows),'kept':len(devrows),'dropped_ids':[]}})
  def forward_batch(xs):
   length=((max(len(x['ids']) for x in xs)+127)//128)*128
   ids=torch.full((len(xs),length),tok.pad_token_id or 0,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids)
   for i,x in enumerate(xs):ids[i,:len(x['ids'])]=torch.tensor(x['ids'],device='cuda');mask[i,:len(x['ids'])]=1
   hidden=backbone(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
   result=[]
   for i,x in enumerate(xs):
    q=headq(hidden[i,len(x['ids'])-1].float());k=headk(hidden[i,x['positions']].float());result.append((k*q).sum(-1)/16)
   return result
  def evaluate(rows,label):
   backbone.eval();answers=[];loss=0;clock=time.perf_counter()
   order=sorted(range(len(rows)),key=lambda i:len(rows[i]['ids']))
   with torch.no_grad():
    for s in range(0,len(order),8):
     xs=[rows[i] for i in order[s:s+8]]
     if max(len(x['ids']) for x in xs)>2048:xs_groups=[[x] for x in xs]
     else:xs_groups=[xs]
     for g in xs_groups:
      zs=forward_batch(g)
      for x,z in zip(g,zs):
       p=z.softmax(-1);pred=x['keys'][int(z.argmax())]
       answers.append({'id':x['row']['id'],'source':x['row']['source'],'expected':x['row']['expected'],'choice':pred,'probabilities':dict(zip(x['keys'],p.cpu().tolist()))})
       loss+=float(torch.nn.functional.cross_entropy(z[None],torch.tensor([x['label']],device='cuda')))
   counts=collections.defaultdict(lambda:[0,0])
   for a in answers:counts[a['source']][0]+=a['choice']==a['expected'];counts[a['source']][1]+=1
   value={'accuracy':sum(a['choice']==a['expected'] for a in answers)/len(answers),'nll':loss/len(answers),'by_source':dict(counts),'seconds':time.perf_counter()-clock,'answers':answers};save(label+'.json',value);log({'eval':label,**{k:v for k,v in value.items() if k!='answers'}});return value
  import gc,contextlib
  backbone.train();opt.zero_grad(set_to_none=True);n=len(trainrows)
  parameter_storages={p.untyped_storage().data_ptr() for p in backbone.parameters()}
  def pack(t):
   if t.device.type=='cuda' and t.numel()*t.element_size()>=1048576 and t.untyped_storage().data_ptr() not in parameter_storages:
    return (t.device,t.detach().to('cpu'))
   return t
  def unpack(x):return x[1].to(x[0]) if isinstance(x,tuple) else x
  def activation_context(enabled):
   return torch.autograd.graph.saved_tensors_hooks(pack,unpack) if enabled else contextlib.nullcontext()
  def pair_loss(xs,indices):
   views=[]
   for x,i in zip(xs,indices):views.extend([x,encode(x['row'],900000+i)])
   zs=forward_batch(views);losses=[]
   for j in range(len(xs)):
    x,other=views[2*j:2*j+2];z,zother=zs[2*j:2*j+2]
    ce=(torch.nn.functional.cross_entropy(z[None],torch.tensor([x['label']],device='cuda'))+torch.nn.functional.cross_entropy(zother[None],torch.tensor([other['label']],device='cuda')))/2
    aligned=zother[[other['keys'].index(k) for k in x['keys']]];lp=z.log_softmax(-1);lq=aligned.log_softmax(-1)
    kl=((lp.exp()*(lp-lq)).sum()+(lq.exp()*(lq-lp)).sum())/2
    losses.append(ce+.1*kl)
   return torch.stack(losses).mean()
  indices=list(range(n));random.Random(319).shuffle(indices);groups=[]
  for start_index in range(0,n,256):
   window=sorted(indices[start_index:start_index+256],key=lambda i:len(trainrows[i]['ids']))
   groups.extend(window[j:j+4] for j in range(0,len(window),4))
  random.Random(320).shuffle(groups)
  save('training_order.json',[[trainrows[i]['row']['id'] for i in g] for g in groups])
  longest=max(range(n),key=lambda i:len(trainrows[i]['ids']))
  clock=time.perf_counter()
  with activation_context(False):
   probe=pair_loss([trainrows[longest]],[longest]);assert torch.isfinite(probe);probe.backward()
  torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
  log({'long_context_probe':{'tokens':len(trainrows[longest]['ids']),'seconds':time.perf_counter()-clock,'loss':float(probe.detach()),'peak_gb':torch.cuda.max_memory_allocated()/2**30}})
  del probe;opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
  evaluate(devrows,'baseline_development');backbone.train()
  losses=[];training_start=time.perf_counter();done=0
  def checkpoint(done,tag='checkpoint'):
   weights={'adapters':{k:{'a':a.a.detach().cpu(),'b':a.b.detach().cpu()} for k,a in adapters.items()},'headq':headq.state_dict(),'headk':headk.state_dict(),'metadata':metadata,'examples':done}
   torch.save(weights,out/(tag+'.pt'));torch.save(opt.state_dict(),out/'optimizer.pt');runs.commit()
  eval_points={4096,8192,12288,16384}
  for group in groups:
   length=max(len(trainrows[i]['ids']) for i in group)
   batch=4 if length<=1024 else 2 if length<=1536 else 1
   offload=False
   while True:
    try:
     group_losses=[]
     for offset in range(0,len(group),batch):
      ids=group[offset:offset+batch];xs=[trainrows[i] for i in ids]
      with activation_context(offload):
       loss=pair_loss(xs,ids);assert torch.isfinite(loss),'nonfinite loss'
       (loss*len(xs)/len(group)).backward()
      group_losses.extend([float(loss.detach())]*len(xs))
     norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
     break
    except torch.cuda.OutOfMemoryError:
     opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
     if batch>1:batch=max(1,batch//2)
     elif not offload:offload=True
     else:raise
     log({'oom_retry':{'done':done,'tokens':length,'microbatch':batch,'activation_offload':offload}})
   before=done;done+=len(group);losses.extend(group_losses)
   factor=.2+.8*.5*(1+math.cos(math.pi*done/n));opt.param_groups[0]['lr']=1e-4*factor;opt.param_groups[1]['lr']=5e-4*factor
   opt.step();opt.zero_grad(set_to_none=True)
   if done//128!=before//128 or done==n:
    elapsed=time.perf_counter()-training_start
    log({'examples':done,'total':n,'loss_last128':sum(losses[-128:])/len(losses[-128:]),'elapsed_seconds':elapsed,'examples_per_second':done/elapsed,'last_tokens':length,'last_microbatch':batch,'peak_gb':torch.cuda.max_memory_allocated()/2**30});runs.commit()
   if done//2048!=before//2048:checkpoint(done)
   crossed=[p for p in eval_points if before<p<=done]
   if crossed:
    evaluate(devrows,'development_at_'+str(crossed[0]));backbone.train()
  checkpoint(done)
  evaluate(devrows,'trained_development')
  permuted=[encode(x['row'],900+i) for i,x in enumerate(devrows)]
  evaluate(permuted,'trained_development_permuted')
  save('status.json',{'status':'complete','elapsed_seconds':time.perf_counter()-start,'checkpoint':str(out/'checkpoint.pt'),'training_examples':n});return results()
 except Exception:
  save('status.json',{'status':'failed','traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-start});print(traceback.format_exc(),flush=True);return results()
if __name__=='__main__':
 import shutil,hashlib
 keys=sys.argv[1:] or ['2b','4b','9b','27b']
 import gzip
 here=Path(__file__).resolve().parent;tp=here/os.environ.get('TRAIN_DATA','data/train.jsonl.gz')
 train_raw=gzip.decompress(tp.read_bytes()) if tp.suffix=='.gz' else tp.read_bytes();hold_raw=(here/'eval_sets/public_holdout_975.jsonl').read_bytes()
 payload={'train':[json.loads(l) for l in train_raw.splitlines()],'development':[json.loads(l) for l in hold_raw.splitlines()]}
 assert len(payload['development'])==975 and len(payload['train'])>1000
 tid={r['id'] for r in payload['train']};tg={r['group'] for r in payload['train']}
 assert not any(r['id'] in tid or r['group'] in tg for r in payload['development'])
 payload['manifest']={'models':{k:list(v) for k,v in MODELS.items()},'max_tokens':4096,'train_sha256':hashlib.sha256(train_raw).hexdigest(),'holdout_sha256':hashlib.sha256(hold_raw).hexdigest(),'train_rows':len(payload['train']),'holdout_rows':975}
 out=here/(os.environ.get('RUN_PREFIX','full_')+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir()
 (out/'manifest.json').write_text(json.dumps(payload['manifest'],indent=2));(out/'train.py').write_text(Path(__file__).read_text())
 jobs={}
 with app.run(detach=True):
  for key in keys:
   dest=out/key;dest.mkdir();runid=out.name+'_'+key
   call=train.spawn(payload,runid,key);job={'app_id':app.app_id,'call_id':call.object_id,'output':str(dest),'volume_run':runid,'model':MODELS[key][0]};(dest/'job.json').write_text(json.dumps(job,indent=2));jobs[key]=job
  (out/'jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(jobs),flush=True)
