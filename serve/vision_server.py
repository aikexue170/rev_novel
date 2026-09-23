"""Vision variant of the decision server: full Qwen VL model (vision tower kept), merged text-trained LoRA + head.

POST /score_image {state, image_png_b64, questions:[{id,instructions,criteria}]} -> same answer format as /score.
The image is placed before the text; option and decision positions are located in the text suffix.
"""
import sys,json,os
from pathlib import Path
import modal
os.environ['FLA_USE_COMPILE']='0'
from modal_image import blackwell_image
app=modal.App('jev-beat-vision');runs=modal.Volume.from_name('jev-decision-training');cache=modal.Volume.from_name('jev-model-cache')
REVISIONS={'Qwen/Qwen3.8-27B':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','Qwen/Qwen3.5-9B':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','Qwen/Qwen3.5-4B':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'}
CHECKPOINTS={'Qwen/Qwen3.8-27B':'full_20260922-091011_27b','Qwen/Qwen3.5-9B':'full_20260922-091011_9b','Qwen/Qwen3.5-4B':'full_20260922-091011_4b'}
def serve(name):
 import time,threading,collections,hashlib,base64,io,torch,uvicorn
 from fastapi import FastAPI
 from transformers import AutoTokenizer,AutoProcessor,AutoModelForImageTextToText
 from PIL import Image
 torch.backends.cuda.enable_cudnn_sdp(False)
 from triton.runtime import autotuner as _at
 _orig=_at.Autotuner.prune_configs
 def _reuse(self,kwargs):
  if self.cache:return [collections.Counter(self.cache.values()).most_common(1)[0][0]]
  return _orig(self,kwargs)
 _at.Autotuner.prune_configs=_reuse
 revision=REVISIONS[name];born=time.perf_counter()
 tok=AutoTokenizer.from_pretrained(name,revision=revision);proc=AutoProcessor.from_pretrained(name,revision=revision)
 full=AutoModelForImageTextToText.from_pretrained(name,revision=revision,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
 lm=full.model.language_model;ck=torch.load('/runs/'+CHECKPOINTS[name]+'/checkpoint.pt',map_location='cpu',weights_only=False);mods=dict(lm.named_modules())
 for path,w in ck['adapters'].items():
  with torch.no_grad():mods[path].weight.add_((2*(w['b'].float()@w['a'].float())).to(mods[path].weight.dtype).to('cuda'))
 h=lm.config.hidden_size;headq=torch.nn.Linear(h,256,bias=False,device='cuda');headk=torch.nn.Linear(h,256,bias=False,device='cuda');headq.load_state_dict(ck['headq']);headk.load_state_dict(ck['headk'])
 state_format=ck['metadata'].get('state_format','json');vs=tok.convert_tokens_to_ids('<|vision_start|>');ve=tok.convert_tokens_to_ids('<|vision_end|>');lock=threading.Lock()
 def suffix(state,q):
  s=state if (state_format=='raw' and isinstance(state,str)) else json.dumps(state,separators=(',',':'))
  ids=tok.encode('State:\n'+s+'\nQuestion: '+q['instructions']+'\nOptions:\n',add_special_tokens=False);positions=[];keys=[]
  for k,desc in q['criteria'].items():ids+=tok.encode(str(k)+': '+str(desc or k)+'\n',add_special_tokens=False);positions.append(len(ids)-1);keys.append(k)
  ids+=tok.encode('Decision:',add_special_tokens=False);return ids,positions,keys
 @torch.inference_mode()
 def score(image,state,questions):
  rows=[suffix(state,q) for q in questions];prefix='<|vision_start|><|image_pad|><|vision_end|>\n'
  # one processor pass for the image, text prefix only; then append each row's text ids
  enc=proc(text=[prefix],images=[image],return_tensors='pt');pre=enc['input_ids'][0].tolist()
  L=max(len(r[0]) for r in rows);n=len(rows);ids=torch.full((n,len(pre)+L),tok.pad_token_id or 0,dtype=torch.long)
  for i,(r,_,_) in enumerate(rows):ids[i,:len(pre)]=torch.tensor(pre);ids[i,len(pre):len(pre)+len(r)]=torch.tensor(r)
  pv=enc['pixel_values'].to('cuda',torch.bfloat16);thw=enc['image_grid_thw'].to('cuda')
  tt=torch.zeros_like(ids);tt[:,:len(pre)]=enc['mm_token_type_ids'][0]
  out=full.model(input_ids=ids.cuda(),mm_token_type_ids=tt.cuda(),pixel_values=pv.repeat(n,1) if n>1 else pv,image_grid_thw=thw.repeat(n,1),use_cache=False).last_hidden_state
  res=[]
  for i,(r,positions,keys) in enumerate(rows):
   off=len(pre);q=headq(out[i,off+len(r)-1].float());k=headk(out[i,[off+p for p in positions]].float());p=((k*q).sum(-1)/16).softmax(-1).cpu().tolist();res.append((keys,p))
  return res
 metadata=dict(model=name,checkpoint=CHECKPOINTS[name],gpu=torch.cuda.get_device_name(),mode='vision: image tokens before the text; text-trained LoRA + head',load_seconds=time.perf_counter()-born)
 api=FastAPI()
 @api.post('/ping')
 async def ping(body:dict):return metadata
 @api.post('/score_image')
 async def score_image(body:dict):
  t0=time.perf_counter();image=Image.open(io.BytesIO(base64.b64decode(body['image_png_b64']))).convert('RGB')
  with lock:res=score(image,body.get('state',''),body['questions'])
  torch.cuda.synchronize();answers={q['id']:{'choice':keys[max(range(len(keys)),key=lambda j:p[j])],'probabilities':dict(zip(keys,p))} for q,(keys,p) in zip(body['questions'],res)}
  return dict(answers=answers,server_seconds=time.perf_counter()-t0,image_tokens=int(metadata.get('image_tokens',0)))
 threading.Thread(target=uvicorn.run,args=(api,),kwargs=dict(host='0.0.0.0',port=8000,log_level='warning'),daemon=True).start();print('READY',json.dumps(metadata),flush=True)
COMMON=dict(port=8000,routing_region='us-west',compute_region='us-west',cpu=8,memory=98304,max_containers=1,scaledown_window=300,startup_timeout=1800,unauthenticated=True,volumes={'/cache':cache,'/runs':runs})
@app.server(image=blackwell_image.pip_install('pillow'),gpu='B200',**COMMON)
class Vis27b:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.8-27B')
@app.server(image=blackwell_image.pip_install('pillow'),gpu='B200',**COMMON)
class Vis9b:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-9B')
@app.server(image=blackwell_image.pip_install('pillow'),gpu='B200',**COMMON)
class Vis4b:
 @modal.enter()
 def start(self):serve('Qwen/Qwen3.5-4B')
