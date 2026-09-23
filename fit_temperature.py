"""Fit a calibration temperature for a served checkpoint on a held-out decision set.

Sends every row (one question per request) to the server, then finds the scalar T that minimises negative
log-likelihood of the expected labels when the returned probabilities are re-tempered as p^(1/T) (equivalent to
dividing the head's scores by T, which is what the servers do with metadata['temperature']). Prints NLL and
expected calibration error at T=1 and at the best T. Choices never change.

usage: fit_temperature.py <server_url> <dev.jsonl> [concurrency=4]
Rows: {state, instructions, criteria, expected}. Use a set the checkpoint was not trained on (e.g. the synthetic
dev split written by synth_jevbench.py, or a slice of eval_sets/ that is not otherwise reported).
"""
import sys,json,math,time,requests,concurrent.futures
URL=sys.argv[1].rstrip('/');rows=[json.loads(l) for l in open(sys.argv[2])];conc=int(sys.argv[3]) if len(sys.argv)>3 else 4
s=requests.Session()
while True:
 try:
  if s.post(URL+'/ping',json={},timeout=900).status_code==200:break
 except requests.RequestException:pass
 time.sleep(5)
def ask(r):
 for attempt in range(5):
  try:
   d=s.post(URL+'/score',json=dict(state=r['state'],questions=[dict(id='q',instructions=r['instructions'],criteria=r['criteria'])]),timeout=300).json()
   return dict(expected=str(r['expected']),probs=d['answers']['q']['probabilities'])
  except Exception:time.sleep(3)
 raise RuntimeError('request failed 5x')
with concurrent.futures.ThreadPoolExecutor(conc) as ex:out=list(ex.map(ask,rows))
def temper(p,T):
 lp={k:math.log(max(v,1e-12))/T for k,v in p.items()};m=max(lp.values());z=sum(math.exp(v-m) for v in lp.values());return {k:math.exp(v-m)/z for k,v in lp.items()}
def nll(T):return -sum(math.log(max(temper(o['probs'],T).get(o['expected'],1e-12),1e-12)) for o in out)/len(out)
def ece(T,bins=10):
 conf=[];hit=[]
 for o in out:
  p=temper(o['probs'],T);k=max(p,key=p.get);conf.append(p[k]);hit.append(k==o['expected'])
 e=0
 for b in range(bins):
  idx=[i for i,c in enumerate(conf) if b/bins<c<=(b+1)/bins]
  if idx:e+=len(idx)/len(conf)*abs(sum(hit[i] for i in idx)/len(idx)-sum(conf[i] for i in idx)/len(idx))
 return e
acc=sum(max(o['probs'],key=o['probs'].get)==o['expected'] for o in out)/len(out)
best=min((nll(T/100),T/100) for T in range(40,301))
print(f'rows {len(out)}  accuracy {acc:.3f}')
print(f'T=1.00  nll {nll(1):.4f}  ece {ece(1):.4f}')
print(f'best T={best[1]:.2f}  nll {best[0]:.4f}  ece {ece(best[1]):.4f}')
for T in [0.8,1.0,1.2,1.5,2.0]:print(f'  T={T:.2f}  nll {nll(T):.4f}  ece {ece(T):.4f}')
