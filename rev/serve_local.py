"""Local decision server: one Rev checkpoint (merged LoRA + pointer head) behind plain FastAPI + uvicorn. No Modal.

    python serve_local.py --repo robbalian/rev-qwen3.5-4b                 # Hugging Face repo id
    python serve_local.py --repo /path/to/dir-with-checkpoint.pt           # or a local directory
    python serve_local.py --repo robbalian/rev-qwen3.8-27b --selftest      # load, answer one hard-coded request, exit

Options: --base (default: the checkpoint's metadata['model']), --revision (default: metadata['revision'], else the
MODELS table below), --port 8000, --dtype bf16|fp16|fp32, --device cuda|cpu, --host 127.0.0.1.

Endpoints (same contract as serve/server.py):
  POST /score   {"state": <json or string>, "questions": [{"id": ..., "instructions": ..., "criteria": {key: desc}}]}
                -> {"answers": {id: {"choice": key, "probabilities": {key: p}}}, "server_seconds": float, ...}
  POST /ping    -> metadata (model, revision, checkpoint sha256, temperature, device, ...)
  GET  /health  -> {"ok": true}

All questions of one request go into one padded forward (right padding, no attention mask, as in serve/server.py's
run_rows). Requests are served one at a time; there is no cross-request batching here.

Requirements (the Modal image pins torch==2.14.0 and transformers==5.17.0; newer versions should also work):
    pip install torch transformers accelerate huggingface_hub fastapi uvicorn
Optional accelerations, used when importable: flash-linear-attention (fla, Triton kernels for the DeltaNet
layers), causal-conv1d (CUDA kernel for the DeltaNet conv; transformers falls back to torch conv1d without it).
A GPU with room for the base model in bf16 is needed for the 9B and 27B; the 4B runs on 12 GB.
"""
import argparse,hashlib,json,os,sys,threading,time
from pathlib import Path
# fla wraps its chunked gated-delta-rule op in torch.compile(fullgraph=True); with variable batch/length shapes that
# recompiles for seconds on every new shape (see serve/server.py). Must be set before fla is imported.
os.environ.setdefault('FLA_USE_COMPILE','0')

# Pinned base-model revisions, copied from serve/server.py. Used only when the checkpoint metadata carries no revision.
MODELS={'Qwen/Qwen3.8-27B':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','Qwen/Qwen3.5-9B':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','Qwen/Qwen3.5-4B':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','Qwen/Qwen3.5-2B':'15852e8c16360a2fea060d615a32b45270f8a8fc'}
DTYPES={'bf16':'bfloat16','fp16':'float16','fp32':'float32'}
LORA_TARGETS={'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'}
MAX_ROW_TOKENS=16384

SELFTEST_REQUEST={
 'state':{'invoice':{'number':'INV-2041','vendor':'Northwind Traders','total_usd':1240.00,'due':'2026-10-15','po_match':True,'approver_note':'looks fine'}},
 'questions':[
  {'id':'q1','instructions':'Is the invoice total greater than $500?','criteria':{'yes':'The total exceeds $500','no':'The total is $500 or less'}},
  {'id':'q2','instructions':'What should happen to this invoice next?','criteria':{'approve':'Approve for payment','reject':'Reject and return to vendor','escalate':'Escalate to a manager for review'}},
 ]}

def resolve_checkpoint(repo):
 """A local directory (or file) with checkpoint.pt, else a Hugging Face repo id: returns the local checkpoint path."""
 p=Path(repo)
 if p.is_file():return p
 if p.is_dir():
  c=p/'checkpoint.pt'
  if not c.exists():sys.exit('no checkpoint.pt in '+str(p))
  return c
 from huggingface_hub import hf_hub_download
 return Path(hf_hub_download(repo,'checkpoint.pt'))

def load(repo,base=None,revision=None,dtype='bf16',device='cuda'):
 import torch
 from transformers import AutoTokenizer,AutoModelForImageTextToText
 born=time.perf_counter()
 checkpoint_path=resolve_checkpoint(repo)
 checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
 for key in ('adapters','headq','headk','metadata'):assert key in checkpoint,'checkpoint lacks '+key
 metadata=dict(checkpoint['metadata'])
 name=base or metadata.get('model')
 assert name,'checkpoint metadata has no model name; pass --base'
 if base and metadata.get('model') and base!=metadata['model']:print('WARNING: --base %s differs from checkpoint model %s'%(base,metadata['model']),flush=True)
 revision=revision or metadata.get('revision') or MODELS.get(name)
 if revision is None:print('WARNING: no pinned revision for %s; loading the Hub default branch'%name,flush=True)
 torch_dtype=getattr(torch,DTYPES[dtype])
 if device.startswith('cuda'):
  assert torch.cuda.is_available(),'--device cuda but torch.cuda.is_available() is False'
  torch.backends.cuda.enable_cudnn_sdp(False)   # cuDNN SDPA autotunes per new (batch, length) shape; seconds each
 torch.manual_seed(814)
 print('loading %s @ %s (%s, %s)'%(name,revision,dtype,device),flush=True)
 tok=AutoTokenizer.from_pretrained(name,revision=revision)
 full=AutoModelForImageTextToText.from_pretrained(name,revision=revision,dtype=torch_dtype,device_map=device,attn_implementation='sdpa').eval()
 backbone=full.model.language_model;del full
 if device.startswith('cuda'):torch.cuda.empty_cache()
 # Merge the rank-16 LoRA into the frozen weights (alpha/rank = 32/16 = 2): serving runs the plain backbone.
 modules=dict(backbone.named_modules());merged=0
 for path,w in checkpoint['adapters'].items():
  module=modules[path];assert isinstance(module,torch.nn.Linear) and path.split('.')[-1] in LORA_TARGETS,path
  with torch.no_grad():module.weight.add_((2*(w['b'].float()@w['a'].float())).to(module.weight.dtype).to(module.weight.device))
  merged+=1
 assert merged==len(checkpoint['adapters']) and merged>0
 h=backbone.config.hidden_size
 headq=torch.nn.Linear(h,256,bias=False,device=device,dtype=torch.float32);headk=torch.nn.Linear(h,256,bias=False,device=device,dtype=torch.float32)
 headq.load_state_dict(checkpoint['headq']);headk.load_state_dict(checkpoint['headk'])
 # Optional CUDA kernel for the DeltaNet causal conv. transformers falls back to torch conv1d when it is absent.
 conv_status='torch conv1d fallback'
 if device.startswith('cuda'):
  try:
   import causal_conv1d
   model_module=sys.modules[type(backbone).__module__]
   if hasattr(model_module,'causal_conv1d_fn'):
    model_module.causal_conv1d_fn=lambda x,w,b=None,activation=None,**kw:causal_conv1d.causal_conv1d_fn(x,w,b,activation=activation)
    conv_status='causal_conv1d '+getattr(causal_conv1d,'__version__','?')
  except ImportError:pass
 try:
  import fla;fla_status='fla '+getattr(fla,'__version__','?')
 except Exception as e:fla_status='fla not importable ('+type(e).__name__+')'
 temperature=float(metadata.get('temperature',1.0));assert temperature>0
 info=dict(model=name,revision=revision,checkpoint=str(checkpoint_path),checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),training_examples=checkpoint.get('examples'),
  state_format=metadata.get('state_format','json'),temperature=temperature,device=device,dtype=dtype,precision='%s merged LoRA, FP32 pointer head'%dtype.upper(),causal_conv1d=conv_status,linear_attention_kernels=fla_status,
  merged_adapters=merged,server='serve_local.py',checkpoint_metadata=metadata)
 scorer=Scorer(tok,backbone,headq,headk,info)
 scorer.warm();info['load_seconds']=time.perf_counter()-born
 return scorer

class Scorer:
 def __init__(self,tok,backbone,headq,headk,info):
  self.tok,self.backbone,self.headq,self.headk,self.info=tok,backbone,headq,headk,info
  self.device=next(backbone.parameters()).device;self.pad_id=tok.pad_token_id or 0
  self.state_format=info['state_format'];self.temperature=info['temperature'];self.lock=threading.Lock()
 def encode(self,state,q):
  """Prompt layout from serve/server.py: State / Question / Options (one 'key: desc' line each) / Decision:."""
  s=state if (self.state_format=='raw' and isinstance(state,str)) else json.dumps(state,separators=(',',':'))
  ids=self.tok.encode('State:\n'+s+'\nQuestion: '+q['instructions']+'\nOptions:\n',add_special_tokens=False);positions=[];keys=[]
  for k,desc in q['criteria'].items():
   ids+=self.tok.encode(str(k)+': '+str(desc or k)+'\n',add_special_tokens=False);positions.append(len(ids)-1);keys.append(k)
  ids+=self.tok.encode('Decision:',add_special_tokens=False)
  return ids,positions,keys
 def run_rows(self,rows):
  """rows: list of (ids, option positions). One right-padded forward, no attention mask (causal model: real positions
  never attend to later pad tokens). Returns one probability list per row and the padded token count."""
  import torch
  with torch.inference_mode():
   length=((max(len(ids) for ids,_ in rows)+15)//16)*16
   inp=torch.full((len(rows),length),self.pad_id,dtype=torch.long,device=self.device)
   for i,(ids,_) in enumerate(rows):inp[i,:len(ids)]=torch.tensor(ids,device=self.device)
   hidden=self.backbone(input_ids=inp,attention_mask=None,use_cache=False).last_hidden_state
   out=[]
   for i,(ids,positions) in enumerate(rows):
    q=self.headq(hidden[i,len(ids)-1].float());k=self.headk(hidden[i,positions].float())
    out.append(((k*q).sum(-1)/16/self.temperature).softmax(-1).cpu().tolist())
   return out,inp.numel()
 def warm(self):
  ids,positions,_=self.encode(SELFTEST_REQUEST['state'],SELFTEST_REQUEST['questions'][0]);self.run_rows([(ids,positions)])
 def score(self,body):
  t0=time.perf_counter();questions=body['questions']
  enc=[self.encode(body['state'],q) for q in questions]
  for ids,_,_ in enc:
   if len(ids)>MAX_ROW_TOKENS:raise ValueError('row of %d tokens exceeds %d'%(len(ids),MAX_ROW_TOKENS))
  with self.lock:
   import torch
   t1=time.perf_counter();probs,padded=self.run_rows([(ids,positions) for ids,positions,_ in enc])
   if self.device.type=='cuda':torch.cuda.synchronize()
   gpu_seconds=time.perf_counter()-t1
  answers={}
  for q,(_,_,keys),p in zip(questions,enc,probs):
   answers[q['id']]={'choice':keys[max(range(len(keys)),key=lambda j:p[j])],'probabilities':dict(zip(keys,p))}
  rows=[dict(gpu_seconds=gpu_seconds,batch_rows=len(enc),batch_padded_tokens=padded,queued_seconds=0.)]*len(enc)
  return dict(answers=answers,server_seconds=time.perf_counter()-t0,path='rows',rows=rows,generated_tokens=0)

def build_app(scorer):
 import asyncio
 from fastapi import FastAPI,HTTPException
 api=FastAPI(title='rev decision server (local)')
 @api.get('/health')
 async def health():return {'ok':True}
 @api.post('/ping')
 async def ping(body:dict=None):return scorer.info
 @api.post('/score')
 async def score(body:dict):
  try:
   return await asyncio.to_thread(scorer.score,body)   # keep the event loop free while the GPU works
  except (KeyError,TypeError,ValueError,AttributeError) as e:
   raise HTTPException(status_code=400,detail='bad request: '+repr(e))
 return api

def main():
 ap=argparse.ArgumentParser(description=__doc__.split('\n\n')[0],formatter_class=argparse.RawDescriptionHelpFormatter,epilog=__doc__.split('\n',1)[1])
 ap.add_argument('--repo',required=True,help='Hugging Face repo id, or a local directory (or file) holding checkpoint.pt')
 ap.add_argument('--base',default=None,help='base model id (default: checkpoint metadata model, e.g. Qwen/Qwen3.5-4B)')
 ap.add_argument('--revision',default=None,help='base model commit (default: checkpoint metadata revision, else the MODELS table)')
 ap.add_argument('--port',type=int,default=8000);ap.add_argument('--host',default='127.0.0.1')
 ap.add_argument('--dtype',choices=sorted(DTYPES),default='bf16');ap.add_argument('--device',default='cuda',help='cuda, cuda:1, cpu, ...')
 ap.add_argument('--selftest',action='store_true',help='load the model, score one hard-coded request, print it and exit')
 a=ap.parse_args()
 scorer=load(a.repo,a.base,a.revision,a.dtype,a.device)
 print('READY',json.dumps({k:v for k,v in scorer.info.items() if k!='checkpoint_metadata'}),flush=True)
 if a.selftest:
  result=scorer.score(SELFTEST_REQUEST);print(json.dumps(result,indent=1))
  for qid,ans in result['answers'].items():print(qid,'->',ans['choice'])
  return
 import uvicorn
 uvicorn.run(build_app(scorer),host=a.host,port=a.port,log_level='info')

if __name__=='__main__':main()
