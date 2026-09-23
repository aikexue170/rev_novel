"""Throughput-oriented decision server: trained LoRA (merged) + pointer head, cross-request dynamic batching.

Every question is one independent causal row (context + question + options). Rows from concurrent requests are
gathered by a background worker into one padded forward pass (bounded by row count and padded-token budget),
so cost per answer is measured at realistic utilization instead of one serial request at a time.
"""
import sys,json,os
from pathlib import Path
import modal
# fla wraps its chunked gated-delta-rule op in torch.compile(fullgraph=True); with variable batch/length shapes that
# recompiles for seconds on every new shape. The Triton kernels themselves are unaffected by this flag.
os.environ['FLA_USE_COMPILE']='0'
from modal_image import hopper_image,blackwell_image
app=modal.App('jev-beat-serve')
runs=modal.Volume.from_name('jev-decision-training')
cache=modal.Volume.from_name('jev-model-cache')
REVISIONS={'Qwen/Qwen3.8-27B':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','Qwen/Qwen3.5-9B':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','Qwen/Qwen3.5-4B':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','Qwen/Qwen3.5-2B':'15852e8c16360a2fea060d615a32b45270f8a8fc'}

def serve(name,checkpoint_run,max_rows=128,max_padded_tokens=49152,window_ms=2.0,overhead_tokens=256):
 import os,time,math,threading,collections,asyncio,hashlib,torch,uvicorn
 from fastapi import FastAPI
 from transformers import AutoTokenizer,AutoModelForImageTextToText
 os.environ['CC']='gcc';os.environ['CXX']='g++'
 torch.manual_seed(814);torch.set_num_threads(4);born=time.perf_counter()
 # Slow-host guard. One-in-flight latency on an eager model is CPU launch-bound: hosts whose CPU runs this loop in
 # ~80 ms (GCP us-west today) give 2x the single-row forward time of hosts that run it in ~42 ms (Seattle). Hand such
 # a container back before loading weights so Modal places another; accept the host after 3 consecutive rejections.
 def _cpu_bench():
  t=time.perf_counter();x=0
  for i in range(2_000_000):x+=i
  return (time.perf_counter()-t)*1000
 _loop_ms=min(_cpu_bench() for _ in range(3));HOST_MAX_LOOP_MS=float(os.environ.get('HOST_MAX_LOOP_MS','60'));_accepted_slow=False
 _af=Path('/runs/.slow_host_attempts');_af.mkdir(exist_ok=True);_af=_af/(name.replace('/','_')+'_'+checkpoint_run)
 try:_att=int(_af.read_text()) if _af.exists() else 0
 except Exception:_att=0
 if _loop_ms>HOST_MAX_LOOP_MS and _att<3:
  _af.write_text(str(_att+1));runs.commit()
  raise RuntimeError('slow host: python loop %.0f ms > %.0f ms (region %s, cloud %s); returning container, attempt %d/3'%(_loop_ms,HOST_MAX_LOOP_MS,os.environ.get('MODAL_REGION'),os.environ.get('MODAL_CLOUD_PROVIDER'),_att+1))
 if _loop_ms>HOST_MAX_LOOP_MS:_accepted_slow=True
 elif _att:_af.write_text('0');runs.commit()
 print('HOST',dict(python_loop_ms=_loop_ms,region=os.environ.get('MODAL_REGION'),cloud=os.environ.get('MODAL_CLOUD_PROVIDER'),accepted_slow=_accepted_slow),flush=True)
 # cuDNN's SDPA backend builds/autotunes an execution plan per new (batch, length) shape, costing seconds each.
 torch.backends.cuda.enable_cudnn_sdp(False)
 # Triton autotune keys in fla's kernels vary with batch/length, and each new key re-benchmarks ~30 kernel variants
 # (about 8 s). Once a kernel has been tuned for any key, reuse its best-known config for new keys: Triton then skips
 # benchmarking (single-config path) and only compiles that config for the new specialization.
 from triton.runtime import autotuner as _autotuner
 _orig_prune=_autotuner.Autotuner.prune_configs
 def _reuse_prune(self,kwargs):
  if self.cache:
   return [collections.Counter(self.cache.values()).most_common(1)[0][0]]
  return _orig_prune(self,kwargs)
 _autotuner.Autotuner.prune_configs=_reuse_prune
 sdp_backends=dict(cudnn=torch.backends.cuda.cudnn_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled())
 revision=REVISIONS[name]
 tok=AutoTokenizer.from_pretrained(name,revision=revision)
 full=AutoModelForImageTextToText.from_pretrained(name,revision=revision,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
 backbone=full.model.language_model;del full;torch.cuda.empty_cache()
 checkpoint_path='/runs/'+checkpoint_run+'/checkpoint.pt'
 checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
 assert checkpoint['metadata']['model']==name,(checkpoint['metadata']['model'],name)
 checkpoint_sha256=hashlib.sha256(open(checkpoint_path,'rb').read()).hexdigest()
 # Merge LoRA into the frozen base weights: serving runs the plain backbone.
 targets={'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'};merged=0
 modules=dict(backbone.named_modules())
 for path,w in checkpoint['adapters'].items():
  module=modules[path];assert isinstance(module,torch.nn.Linear) and path.split('.')[-1] in targets
  with torch.no_grad():module.weight.add_((2*(w['b'].float()@w['a'].float())).to(module.weight.dtype).to('cuda'))
  merged+=1
 assert merged==len(checkpoint['adapters']) and merged>0
 h=backbone.config.hidden_size
 headq=torch.nn.Linear(h,256,bias=False,device='cuda',dtype=torch.float32);headk=torch.nn.Linear(h,256,bias=False,device='cuda',dtype=torch.float32)
 headq.load_state_dict(checkpoint['headq']);headk.load_state_dict(checkpoint['headk'])
 state_format=checkpoint['metadata'].get('state_format','json');temperature=float(checkpoint['metadata'].get('temperature',1.0))   # calibration: scores/T before the softmax (train/set_temperature.py)
 # Optimized causal_conv1d kernel (the image builds it for this GPU family); verify it runs.
 import causal_conv1d,importlib
 model_module=importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
 def reference_conv(x,w,b=None,activation=None,**kw):
  y=torch.nn.functional.conv1d(x.to(w.dtype),w.unsqueeze(1),b,padding=w.shape[-1]-1,groups=x.shape[1])[...,:x.shape[-1]]
  return (torch.nn.functional.silu(y) if activation in ('silu','swish') else y).to(x.dtype)
 with torch.inference_mode():
  x=torch.randn(2,256,101,device='cuda',dtype=torch.bfloat16);w=torch.randn(256,4,device='cuda',dtype=torch.bfloat16)
  conv_delta=float((causal_conv1d.causal_conv1d_fn(x,w,activation='silu').float()-reference_conv(x,w,activation='silu').float()).abs().max());assert conv_delta<.15,conv_delta
 model_module.causal_conv1d_fn=lambda x,w,b=None,activation=None,**kw:causal_conv1d.causal_conv1d_fn(x,w,b,activation=activation)
 try:
  import fla;from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk;fla_status='fla '+getattr(fla,'__version__','?')+' importable'
 except Exception as e:fla_status='fla import failed: '+repr(e)
 import triton;fla_status+='; triton '+triton.__version__
 pad_id=tok.pad_token_id or 0
 def encode(state,q):
  s=state if (state_format=='raw' and isinstance(state,str)) else json.dumps(state,separators=(',',':'))
  ids=tok.encode('State:\n'+s+'\nQuestion: '+q['instructions']+'\nOptions:\n',add_special_tokens=False);positions=[];keys=[]
  for k,desc in q['criteria'].items():
   ids+=tok.encode(str(k)+': '+str(desc or k)+'\n',add_special_tokens=False);positions.append(len(ids)-1);keys.append(k)
  ids+=tok.encode('Decision:',add_special_tokens=False)
  return ids,positions,keys
 @torch.inference_mode()
 def run_rows(rows,pad_to=None,use_mask=False):
  """rows: list of (ids,positions). Returns list of probability lists. Equal-length batches skip the attention mask."""
  lens=[len(r[0]) for r in rows];length=pad_to or ((max(lens)+15)//16)*16
  inp=torch.full((len(rows),length),pad_id,dtype=torch.long,device='cuda')
  for i,(ids,_) in enumerate(rows):inp[i,:len(ids)]=torch.tensor(ids,device='cuda')
  # Right padding on a causal model: real positions never attend to (later) pad tokens, and the recurrent/conv
  # layers are causal too, so no attention mask is needed for the positions we read. Skipping it keeps SDPA on
  # the flash/causal kernel instead of materializing a padded mask. `/parity` verifies this against the masked path.
  mask=None
  if use_mask:
   mask=torch.zeros_like(inp)
   for i,n in enumerate(lens):mask[i,:n]=1
  hidden=backbone(input_ids=inp,attention_mask=mask,use_cache=False).last_hidden_state
  out=[]
  for i,(ids,positions) in enumerate(rows):
   q=headq(hidden[i,len(ids)-1].float());k=headk(hidden[i,positions].float())
   out.append(((k*q).sum(-1)/16/temperature).softmax(-1))
  probs=torch.stack([torch.nn.functional.pad(p,(0,256-p.numel())) for p in out]).cpu()
  return [probs[i,:len(r[1])].tolist() for i,r in enumerate(rows)],inp.numel()
 # ---- dynamic batching worker ----
 batch_cfg={'overhead_tokens':overhead_tokens}   # replaced by the calibrated value (fixed forward cost in tokens) below
 queue=collections.deque();cond=threading.Condition();stats=dict(batches=0,rows=0,gpu_seconds=0.,padded_tokens=0,batch_rows=collections.Counter(),started=time.perf_counter())
 def worker():
  torch.cuda.synchronize()
  while True:
   with cond:
    while not queue:cond.wait()
   if window_ms>0:time.sleep(window_ms/1000)
   with cond:
    items=[queue.popleft() for _ in range(min(len(queue),max_rows))]
   items.sort(key=lambda it:len(it['ids']))
   # All rows from one request (one in flight): run them as a single forward. Splitting a 20-question request over
   # 16-token length buckets cost a second fixed overhead; cross-request batches below still use the marginal rule.
   single=len(items)<=32 and len({it.get('req') for it in items})==1 and ((len(items[-1]['ids'])+15)//16)*16*len(items)<=max_padded_tokens
   # Split into length-homogeneous sub-batches: forward cost is ~linear in padded tokens once a batch has
   # >=2k tokens, so never let a long row drag short rows up to its padded length. A sub-batch closes when
   # padding waste would exceed 25% (once it already holds >=2k padded tokens) or the token budget is hit.
   # Marginal-cost rule: rows are sorted by length, so appending row j only adds padding. Add it unless the extra
   # padded tokens it forces (for the rows already in the batch plus its own padding) cost more than one forward's
   # fixed overhead (~20 ms ~= 1k tokens at 50k tok/s), or the token budget would be exceeded. With 256 the rule split
   # a 20-row request (16-token buckets x 16 rows) into two forwards; 1024 keeps it in one.
   i=0
   while i<len(items):
    j=i+1;longest=((len(items[i]['ids'])+15)//16)*16
    while j<len(items):
     lnew=((len(items[j]['ids'])+15)//16)*16;extra=(lnew-longest)*(j-i)+(lnew-len(items[j]['ids']))
     if lnew*(j+1-i)>max_padded_tokens or (extra>batch_cfg['overhead_tokens'] and not single):break
     longest=lnew;j+=1
    # Triton autotunes fla's batch-keyed kernels for every new batch size (seconds each), so only run canonical
    # batch sizes: shrink to the largest canonical size that fits (leaving the rest queued) when the queue is deep,
    # otherwise pad with duplicate rows up to the next canonical size (results for duplicates are discarded).
    n=j-i;longest=((len(items[j-1]['ids'])+15)//16)*16
    if n not in CANON:
     up=min(c for c in CANON if c>n);down=max(c for c in CANON if c<n)
     if len(items)-i>=up or up*longest>max_padded_tokens:
      if len(items)-i>=up and up*longest<=max_padded_tokens:j=i+up
      else:j=i+down
     n=j-i
    sub=items[i:j];i=j;target=min(c for c in CANON if c>=n)
    rows=[(it['ids'],it['positions']) for it in sub]+[(sub[-1]['ids'],sub[-1]['positions'])]*(target-n)
    t0=time.perf_counter()
    try:
     probs,padded=run_rows(rows);torch.cuda.synchronize();dt=time.perf_counter()-t0;probs=probs[:n]
     stats['batches']+=1;stats['rows']+=len(sub);stats['gpu_seconds']+=dt;stats['padded_tokens']+=padded;stats['batch_rows'][len(sub)]+=1
     for it,p in zip(sub,probs):it['loop'].call_soon_threadsafe(it['future'].set_result,dict(probs=p,gpu_seconds=dt,batch_rows=len(sub),batch_padded_tokens=padded,queued_seconds=t0-it['t_enqueue']))
    except Exception as e:
     for it in sub:it['loop'].call_soon_threadsafe(it['future'].set_exception,RuntimeError(repr(e)))
 # Warm every canonical batch size before accepting traffic (Triton autotune is keyed on batch size; new lengths
 # for a seen batch size cost only a small specialization compile).
 CANON=list(range(1,17))+[20,24,28,32,40,48,56,64,80,96,112,128];warm_start=time.perf_counter();_base=tok.encode('The quick brown fox jumps over the lazy dog. '*400,add_special_tokens=False);assert len(_base)>=3072
 for _b,_l in [(1,256),(8,1024),(64,512),(4,3072)]+[(b,256) for b in CANON]:
  run_rows([(_base[:_l],[_l-3,_l-2])]*_b)
 torch.cuda.synchronize();warm_seconds=time.perf_counter()-warm_start
 threading.Thread(target=worker,daemon=True).start()
 metadata=dict(model=name,revision=revision,checkpoint_run=checkpoint_run,checkpoint_sha256=checkpoint_sha256,training_examples=checkpoint['examples'],state_format=state_format,temperature=temperature,gpu=torch.cuda.get_device_name(),region=os.environ.get('MODAL_REGION'),precision='BF16 merged LoRA, FP32 pointer head',execution='eager, sdpa, optimized causal_conv1d, FLA_USE_COMPILE=0, cudnn sdp off',sdp_backends=sdp_backends,batcher='v12: v9 + single-request rows run as one forward, cross-request marginal-cost overhead 256 padded tokens, atomic enqueue per request, GPU work off the event loop, calibrated cost-model routing between independent rows and the shared-context path',canonical_batch_sizes=CANON,warm_seconds=warm_seconds,batching=dict(max_rows=max_rows,max_padded_tokens=max_padded_tokens,window_ms=window_ms,overhead_tokens=overhead_tokens),load_seconds=time.perf_counter()-born,causal_conv1d_version=causal_conv1d.__version__,linear_attention_kernels=fla_status,conv_smoke_max_absolute_delta=conv_delta)
 api=FastAPI()
 @api.post('/ping')
 async def ping(body:dict):return metadata
 @api.post('/noop')
 async def noop(body:dict):return {'ok':True}
 @api.post('/stats')
 async def get_stats(body:dict):
  return dict(**{k:v for k,v in stats.items() if k!='batch_rows'},batch_rows={str(k):v for k,v in stats['batch_rows'].items()},uptime=time.perf_counter()-stats['started'])
 @api.post('/reset_stats')
 async def reset_stats(body:dict):
  stats.update(batches=0,rows=0,gpu_seconds=0.,padded_tokens=0,batch_rows=collections.Counter(),started=time.perf_counter());return {'ok':True}
 @api.post('/bench')
 async def bench(body:dict):
  """Time raw forwards for [batch,length] shapes with realistic-looking rows; returns median ms and tokens/s."""
  import statistics
  base=tok.encode('The quick brown fox jumps over the lazy dog. '*1200,add_special_tokens=False)
  results=[]
  for shape in body['shapes']:
   b,l=shape[:2];pad_to=shape[2] if len(shape)>2 else None
   ids=base[:l];assert len(ids)==l;positions=[l-3,l-2];rows=[(ids,positions)]*b
   with torch.inference_mode():
    torch.cuda.synchronize();t=time.perf_counter();run_rows(rows,pad_to);torch.cuda.synchronize();first=time.perf_counter()-t;times=[]
    for _ in range(body.get('reps',3)):
     t=time.perf_counter();run_rows(rows,pad_to);torch.cuda.synchronize();times.append(time.perf_counter()-t)
   m=statistics.median(times);results.append(dict(batch=b,length=l,pad_to=pad_to,first_call_ms=first*1000,median_ms=m*1000,tokens_per_second=b*(pad_to or l)/m,rows_per_second=b/m))
  return results
 @api.post('/profile')
 async def profile(body:dict):
  """cProfile the first forward on a (batch,length) shape this container has not seen; returns top cumulative entries."""
  import cProfile,pstats,io
  b,l=body['shape'];base=tok.encode('The quick brown fox jumps over the lazy dog. '*1200,add_special_tokens=False);rows=[(base[:l],[l-3,l-2])]*b
  pr=cProfile.Profile();torch.cuda.synchronize();t=time.perf_counter();pr.enable()
  with torch.inference_mode():run_rows(rows);torch.cuda.synchronize()
  pr.disable();elapsed=time.perf_counter()-t;buf=io.StringIO();pstats.Stats(pr,stream=buf).sort_stats('cumulative').print_stats(body.get('top',45))
  return dict(seconds=elapsed,stats=buf.getvalue())
 @api.post('/parity')
 async def parity(body:dict):
  """Masked vs unmasked outputs on real rows of mixed lengths (padded together). Returns max |dp| and choice agreement."""
  rows=[];keys=[]
  for c in body['cases']:
   for q in c['questions']:
    ids,positions,ks=encode(c['state'],q);rows.append((ids,positions));keys.append(ks)
  rows.sort(key=lambda r:len(r[0]))
  a,_=run_rows(rows,use_mask=True);b,_=run_rows(rows,use_mask=False)
  diffs=[max(abs(x-y) for x,y in zip(pa,pb)) for pa,pb in zip(a,b)]
  agree=sum(max(range(len(pa)),key=lambda j:pa[j])==max(range(len(pb)),key=lambda j:pb[j]) for pa,pb in zip(a,b))
  return dict(rows=len(rows),lengths=[len(r[0]) for r in rows],max_abs_prob_diff=max(diffs),mean_abs_prob_diff=sum(diffs)/len(diffs),choice_agreement=agree/len(rows))
 shared_lock=threading.Lock()
 def state_prefix_len(state):
  st=state if (state_format=='raw' and isinstance(state,str)) else json.dumps(state,separators=(',',':'))
  return len(tok.encode('State:\n'+st+'\n',add_special_tokens=False))
 SHARED_MIN_QUESTIONS=3;SHARED_MIN_PREFIX=64   # floors; the calibrated cost model below makes the actual choice
 @torch.inference_mode()
 def run_shared(seqs,positions_list,state_len,validate=False):
  """Shared-context execution: prefill the common prefix once, fork the cache (attention KV + recurrent/conv states)
  into one branch per question, run all branches in one batched forward. Right padding, no attention mask."""
  plen=min(min(len(x) for x in seqs)-1,state_len)
  while plen>0 and any(x[:plen]!=seqs[0][:plen] for x in seqs):plen-=1
  assert plen>0 and all(len(x)>plen for x in seqs)
  branches=[x[plen:] for x in seqs];n=len(branches);L=((max(map(len,branches))+15)//16)*16
  torch.cuda.synchronize();t0=time.perf_counter()
  pref=torch.tensor([seqs[0][:plen]],device='cuda');pc=backbone(input_ids=pref,use_cache=True).past_key_values
  torch.cuda.synchronize();t1=time.perf_counter()
  pc.reorder_cache(torch.zeros(n,dtype=torch.long,device='cuda'))   # independent copies for every branch
  torch.cuda.synchronize();t2=time.perf_counter()
  inp=torch.full((n,L),pad_id,dtype=torch.long,device='cuda')
  for i,b in enumerate(branches):inp[i,:len(b)]=torch.tensor(b,device='cuda')
  pos=torch.arange(plen,plen+L,device='cuda');fullmask=torch.ones((n,plen+L),dtype=torch.long,device='cuda')
  hidden=backbone(input_ids=inp,attention_mask=fullmask,position_ids=pos[None].expand(n,-1),cache_position=pos,past_key_values=pc,use_cache=True).last_hidden_state
  probs=[]
  for i,(b,positions) in enumerate(zip(branches,positions_list)):
   q=headq(hidden[i,len(b)-1].float());k=headk(hidden[i,[p-plen for p in positions]].float());probs.append(((k*q).sum(-1)/16/temperature).softmax(-1).cpu().tolist())
  torch.cuda.synchronize();t3=time.perf_counter()
  timing=dict(prefix_tokens=plen,branch_tokens=sum(map(len,branches)),computed_tokens=plen+sum(map(len,branches)),independent_tokens=sum(map(len,seqs)),prefill_ms=(t1-t0)*1000,fork_ms=(t2-t1)*1000,branch_ms=(t3-t2)*1000,gpu_ms=(t3-t0)*1000)
  validation=None
  if validate:
   t=time.perf_counter();ref,_=run_rows([(x,pp) for x,pp in zip(seqs,positions_list)]);torch.cuda.synchronize()
   validation=dict(independent_gpu_ms=(time.perf_counter()-t)*1000,max_abs_prob_diff=max(abs(a-b) for pa,pb in zip(probs,ref) for a,b in zip(pa,pb)),choice_agreement=sum(max(range(len(pa)),key=lambda j:pa[j])==max(range(len(pb)),key=lambda j:pb[j]) for pa,pb in zip(probs,ref))/n)
  return probs,timing,validation
 # ---- routing calibration: per-forward fixed cost and per-token cost of this model on this GPU, plus the fork cost ----
 def _median_ms(fn,reps=3):
  ts=[]
  for _ in range(reps):
   torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();ts.append(time.perf_counter()-t)
  return sorted(ts)[len(ts)//2]*1000
 _t1=_median_ms(lambda:run_rows([(_base[:256],[253,254])]));_t16=_median_ms(lambda:run_rows([(_base[:256],[253,254])]*16))
 _slope=max((_t16-_t1)/(15*256),1e-4);_fixed=max(_t1-256*_slope,1.0)
 _s256=[_base[:288] for _ in range(8)];_p256=[[285,286]]*8;_s1k=[_base[:1056] for _ in range(8)];_p1k=[[1053,1054]]*8
 run_shared(_s256,_p256,256);run_shared(_s1k,_p1k,1024)   # first calls compile/warm the shared path; measure the second ones
 _,_tm,_=run_shared(_s256,_p256,256);_,_tm2,_=run_shared(_s1k,_p1k,1024)
 _fork_tok=max((_tm2['fork_ms']-_tm['fork_ms'])/(8*768),0.);_fork_branch=max(_tm['fork_ms']/8-_fork_tok*256,0.)
 # Note: deriving this threshold from fixed_ms/ms_per_token (2,142 tokens on the 4B) cost 20% throughput under load
 # (26-37% more padded tokens per answer); the measured-best cross-request threshold stays at 256. See single-request rule in worker.
 calib=dict(fixed_ms=_fixed,ms_per_token=_slope,overhead_tokens=batch_cfg['overhead_tokens'],fork_ms_per_branch=_fork_branch,fork_ms_per_branch_token=_fork_tok,ms_1x256=_t1,ms_16x256=_t16,shared_8x256_ms=_tm['gpu_ms'],shared_8x1024_ms=_tm2['gpu_ms'])
 metadata['routing']=dict(rule='shared if 2*fixed + computed tokens*slope + fork(n, prefix) < fixed + independent padded tokens*slope; floors: >=%d questions, >=%d-token state'%(SHARED_MIN_QUESTIONS,SHARED_MIN_PREFIX),**calib)
 def choose_shared(n,prefix_tokens,seq_lens):
  """Estimated GPU ms of both paths from the calibration above; independent rows assumed to fit one forward."""
  if n<SHARED_MIN_QUESTIONS or prefix_tokens<SHARED_MIN_PREFIX:return False
  rows_ms=calib['fixed_ms']+sum(((l+15)//16)*16 for l in seq_lens)*calib['ms_per_token']
  shared_ms=2*calib['fixed_ms']+(prefix_tokens+sum(l-prefix_tokens for l in seq_lens))*calib['ms_per_token']+n*(calib['fork_ms_per_branch']+prefix_tokens*calib['fork_ms_per_branch_token'])
  return shared_ms<rows_ms
 # Host diagnostics: one-in-flight latency on an eager model is launch-bound, so a slow host CPU shows up directly.
 import subprocess as _sp
 try:_cpu_model=next((l.split(':',1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')),'?')
 except Exception:_cpu_model='?'
 try:_smi=_sp.run(['nvidia-smi','--query-gpu=name,clocks.sm,clocks.max.sm,power.limit,pstate,pcie.link.gen.current','--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout.strip()
 except Exception as e:_smi=repr(e)
 metadata['host']=dict(cpu=_cpu_model,python_loop_ms=_loop_ms,accepted_slow=_accepted_slow,host_max_loop_ms=HOST_MAX_LOOP_MS,nvidia_smi=_smi,region=os.environ.get('MODAL_REGION'),cloud=os.environ.get('MODAL_CLOUD_PROVIDER'),cpu_count=os.cpu_count())
 @api.post('/score_shared')
 async def score_shared(body:dict):
  t0=time.perf_counter();seqs=[];pl=[];keys=[]
  for q in body['questions']:
   ids,positions,ks=encode(body['state'],q);seqs.append(ids);pl.append(positions);keys.append(ks)
  def _go():
   with shared_lock:return run_shared(seqs,pl,state_prefix_len(body['state']),body.get('validate',False))
  probs,timing,validation=await asyncio.to_thread(_go)
  answers={q['id']:{'choice':ks[max(range(len(ks)),key=lambda j:p[j])],'probabilities':dict(zip(ks,p))} for q,ks,p in zip(body['questions'],keys,probs)}
  return dict(answers=answers,server_seconds=time.perf_counter()-t0,shared=timing,validation=validation,rows=[dict(gpu_seconds=timing['gpu_ms']/1000,batch_rows=len(seqs),batch_padded_tokens=timing['computed_tokens'],queued_seconds=0.)]*len(seqs),generated_tokens=0)
 @api.post('/score')
 async def score(body:dict):
  t0=time.perf_counter();loop=asyncio.get_running_loop();futures=[];meta=[]
  enc=[encode(body['state'],q) for q in body['questions']]
  for ids,_,_ in enc:assert len(ids)<=16384,len(ids)
  if body.get('mode','auto')=='auto' and len(enc)>=SHARED_MIN_QUESTIONS:
   sl=state_prefix_len(body['state'])
   if choose_shared(len(enc),sl,[len(e[0]) for e in enc]):
    seqs=[e[0] for e in enc];pl=[e[1] for e in enc];keys=[e[2] for e in enc]
    def _go():
     with shared_lock:return run_shared(seqs,pl,sl)
    probs,timing,_=await asyncio.to_thread(_go)   # keep the event loop free while the GPU works
    answers={q['id']:{'choice':ks[max(range(len(ks)),key=lambda j:p[j])],'probabilities':dict(zip(ks,p))} for q,ks,p in zip(body['questions'],keys,probs)}
    return dict(answers=answers,server_seconds=time.perf_counter()-t0,path='shared',shared=timing,rows=[dict(gpu_seconds=timing['gpu_ms']/1000,batch_rows=len(seqs),batch_padded_tokens=timing['computed_tokens'],queued_seconds=0.)]*len(seqs),generated_tokens=0)
  # Enqueue all rows under one lock acquisition so the batching window cannot close on a half-enqueued request
  # (with per-row appends, tokenizing 20 rows took longer than the 2 ms window and the request ran as two forwards).
  pend=[]
  for (ids,positions,keys),q in zip(enc,body['questions']):
   fut=loop.create_future();futures.append(fut);meta.append((q['id'],keys));pend.append(dict(ids=ids,positions=positions,loop=loop,future=fut,t_enqueue=time.perf_counter(),req=t0))
  with cond:queue.extend(pend);cond.notify()
  results=await asyncio.gather(*futures)
  answers={};detail=[]
  for (qid,keys),r in zip(meta,results):
   p=r['probs'];answers[qid]={'choice':keys[max(range(len(keys)),key=lambda j:p[j])],'probabilities':dict(zip(keys,p))}
   detail.append({k:r[k] for k in ['gpu_seconds','batch_rows','batch_padded_tokens','queued_seconds']})
  return dict(answers=answers,server_seconds=time.perf_counter()-t0,path='rows',rows=detail,generated_tokens=0)
 threading.Thread(target=uvicorn.run,args=(api,),kwargs=dict(host='0.0.0.0',port=8000,log_level='warning'),daemon=True).start();print('READY',json.dumps(metadata),flush=True)

COMMON=dict(port=8000,routing_region='us-west',cpu=8,memory=98304,max_containers=1,scaledown_window=300,startup_timeout=1800,unauthenticated=True,volumes={'/cache':cache,'/runs':runs})
# Checkpoint run directories on the jev-decision-training volume. Edit before `modal deploy`.
CHECKPOINTS={'4b_jb':'jb_20260923-002700_4b','27b_jb':'jb_20260922-211844_27b','9b':'full_20260922-091011_9b','27b':'full_20260922-091011_27b','4b':'full_20260922-091011_4b','2b':'full_20260922-091011_2b','4b_workflow':'synth_20260922-122120_4b','4b_balanced':'balanced_20260922-144201_4b','9b_balanced':'balanced_20260922-144201_9b'}

@app.server(image=blackwell_image,gpu='B200',compute_region='us-west',**COMMON)
class Small9b:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-9B',CHECKPOINTS['9b'])

@app.server(image=hopper_image,gpu='H100!',compute_region='us-west',**COMMON)
class Small9h:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-9B',CHECKPOINTS['9b'])

@app.server(image=blackwell_image,gpu='B200',compute_region='us-west',**COMMON)
class Dense27b:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.8-27B',CHECKPOINTS['27b'],max_padded_tokens=32768)
@app.server(image=blackwell_image,gpu='B200',compute_region='us-west',**{**COMMON,'max_containers':4,'min_containers':4})   # 4 warm GPUs for the full-suite run; set min_containers back to 0 afterwards
class Dense27jb:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.8-27B',CHECKPOINTS['27b_jb'],max_padded_tokens=32768)

@app.server(image=hopper_image,gpu='H100!',compute_region='us-west',**COMMON)
class Dense27h:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.8-27B',CHECKPOINTS['27b'],max_padded_tokens=32768)

@app.server(image=hopper_image,gpu='H100!',compute_region='us-west',**COMMON)
class Small4h:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-4B',CHECKPOINTS['4b'])

@app.server(image=blackwell_image,gpu='B200',compute_region='us-west',**COMMON)
class Small4b:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-4B',CHECKPOINTS['4b'])
@app.server(image=blackwell_image,gpu='B200',compute_region='us-west',**{**COMMON,'max_containers':4})   # up to 4 GPUs for full-suite benchmark runs
class Small4jb:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-4B',CHECKPOINTS['4b_jb'])

@app.server(image=hopper_image,gpu='H100!',compute_region='us-west',**COMMON)
class Small2h:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-2B',CHECKPOINTS['2b'])

@app.server(image=blackwell_image,gpu='B200',compute_region='us-west',**COMMON)
class Small4bwf:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-4B',CHECKPOINTS['4b_workflow'])

@app.server(image=blackwell_image,gpu='B200',compute_region='us-west',**COMMON)
class Small4bal:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-4B',CHECKPOINTS['4b_balanced'])

@app.server(image=blackwell_image,gpu='B200',compute_region='us-west',**COMMON)
class Small9bal:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-9B',CHECKPOINTS['9b_balanced'])
