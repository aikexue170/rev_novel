"""Quick status of training runs and load tests from the Modal volumes.

usage: bench/status.py train <full_dir>        # e.g. runs/full_20260922-091011
       bench/status.py load <load_dir>          # e.g. runs/pilot9_20260922-091346
"""
import sys,json,subprocess,tempfile,pathlib
PY='/tmp/jev-modal-latency-env/bin/python'
ROOT=pathlib.Path(__file__).resolve().parents[1]
def rundir(a):   # a run directory: as given, or by name under <repo>/runs/
 p=pathlib.Path(a);return p if p.exists() or not (ROOT/'runs'/a).exists() else ROOT/'runs'/a
def fetch(volume,remote):
 tmp=tempfile.mkdtemp();dest=pathlib.Path(tmp)/'f'
 r=subprocess.run([PY,'-m','modal','volume','get',volume,remote,str(dest),'--force'],capture_output=True,text=True)
 if r.returncode!=0:return None
 return dest.read_text()
mode,d=sys.argv[1],rundir(sys.argv[2])
if mode=='train':
 jobs=json.loads((d/'jobs.json').read_text())
 for key,job in jobs.items():
  txt=fetch('jev-decision-training',job['volume_run']+'/progress.jsonl')
  status=fetch('jev-decision-training',job['volume_run']+'/status.json')
  if not txt:print(key,'no progress yet');continue
  lines=[json.loads(l) for l in txt.splitlines()]
  prog=[l for l in lines if 'examples' in l];evals=[l for l in lines if 'eval' in l];oom=[l for l in lines if 'oom_retry' in l]
  last=prog[-1] if prog else {}
  print(f"{key}: {last.get('examples','?')}/{last.get('total','?')} ex  {last.get('examples_per_second',0):.2f} ex/s  loss {last.get('loss_last128',0):.3f}  elapsed {last.get('elapsed_seconds',0)/60:.1f} min  peak {last.get('peak_gb',0):.0f} GB  oom_retries {len(oom)}")
  for e in evals:print('   ',key,e['eval'],f"{e['accuracy']:.4f}",'nll',f"{e['nll']:.3f}")
  if status:print('    status:',json.loads(status).get('status'),json.loads(status).get('traceback','')[-600:])
elif mode=='load':
 txt=fetch('jev-benchmark-results',d.name+'/summary.jsonl')
 if not txt:print('no summary yet');sys.exit()
 for l in txt.splitlines():
  s=json.loads(l);st=s.get('server_stats') or {}
  util=st.get('gpu_seconds',0)/s['wall_seconds'] if st else None
  print(f"{s['arm']:8s} {s['tag']:7s} c={s['concurrency']:3d} acc={s['correct']}/{s['requests']} err={s['errors']} p50={s['p50']*1000 if s['p50'] else 0:6.1f}ms p95={s['p95']*1000 if s['p95'] else 0:6.1f}ms  {s['answers_per_second']:6.1f} ans/s  gpu_util={util if util is None else round(util,2)}  batches={st.get('batches')}  cost=${s['total_cost_usd'] or 0:.4f}")
 status=fetch('jev-benchmark-results',d.name+'/status.json');print('status:',status)
