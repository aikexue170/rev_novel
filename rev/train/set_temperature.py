"""Write a calibration temperature into a training run's checkpoint metadata on the Modal volume.

The servers (serve/server.py, serve_local.py) divide the pointer-head scores by metadata['temperature'] before the softmax;
choices never change, only the spread of the probabilities. Fit the value with train/fit_temperature.py first.

usage: train/set_temperature.py <run_name> <temperature>
"""
import sys,subprocess,tempfile,pathlib,json
run,T=sys.argv[1],float(sys.argv[2]);assert 0.2<=T<=5,T
PY=sys.executable;tmp=pathlib.Path(tempfile.mkdtemp());ck=tmp/'checkpoint.pt'
def modal(*a):
 r=subprocess.run([PY,'-m','modal','volume',*a],capture_output=True,text=True);assert r.returncode==0,r.stdout[-800:]+r.stderr[-800:]
modal('get','jev-decision-training',run+'/checkpoint.pt',str(ck),'--force')
import torch
c=torch.load(ck,map_location='cpu',weights_only=False);old=c['metadata'].get('temperature');c['metadata']['temperature']=T;torch.save(c,ck)
modal('put','jev-decision-training',str(ck),run+'/checkpoint.pt','--force')
md=tmp/'metadata.json'
try:
 modal('get','jev-decision-training',run+'/metadata.json',str(md),'--force');m=json.loads(md.read_text());m['temperature']=T;md.write_text(json.dumps(m,indent=1));modal('put','jev-decision-training',str(md),run+'/metadata.json','--force')
except AssertionError:pass
print(f'{run}: temperature {old} -> {T} (checkpoint.pt and metadata.json updated on the volume; redeploy serve/server.py to pick it up)')
