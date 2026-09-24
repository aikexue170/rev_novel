"""Recreate the JevBench v1.4.1 ranking chart (score vs cost) from the published results, and place Rev's 4B and 27B
on it as ESTIMATES under the v1.4 formula (jevbench/composite_v14.py), with every assumption stated.

usage: bench/jevbench_v14_chart.py <jevbench checkout> [out_dir=<repo>/results/benchmarkheaven/v1.4]

Inputs we can measure ourselves: public-item Intelligence (v1.3 formula, 3 of 4 tiers; the judge tier is not public),
public accuracy on the 231 public items, calibration on the public hard tier, serial latency and cost per 1,000.
Inputs only the maintainer can measure: sealed accuracy (308 unpublished items), the sealed-inclusive calibration term,
and speed from his own client. Sealed accuracy is therefore shown as a range, with a point at "same public-to-sealed
ratio as Jev".
"""
import sys,json,math,pathlib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
JB=pathlib.Path(sys.argv[1]);ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'results/benchmarkheaven/v1.4';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(JB))
from jevbench import composite_v14 as v14, composite_v13 as v13
board=json.loads((JB/'results/v1.4/jevbench-v1.4-results.json').read_text())
rows=[s for s in board['systems'] if s.get('ranked') and s.get('jevbench_score') is not None and (s.get('cost') or {}).get('usd_per_1000')]
jev=next(s for s in rows if s['key']=='jev-1.13.0')
ratio=jev['sealed_accuracy']/jev['public_accuracy']
# Our measured inputs (results/benchmarkheaven/summary_*.json, results/holdout/summary.json, jevbench_adapter/COST_BASIS.md)
OURS={'Rev Qwen3.5-4B':dict(i13=70.6,public=181/231,cal=75.4,p50=0.037,p95=0.064,usd=0.0220),
      'Rev Qwen3.8-27B':dict(i13=86.2,public=207/231,cal=80.6,p50=0.075,p95=0.202,usd=0.0819)}
def estimate(o,sealed):
    axes=dict(intelligence=v14.intelligence(o['i13'],sealed,o['public']),
              calibration=v14.calibration(o['cal'],o['cal']),          # assumes sealed calibration equals public calibration
              speed=v13.speed(o['p50'],o['p95'],'gpu'),                # his GPU: x2 + 0.15 s adjustment
              cost=v13.cost(o['usd']))
    return axes,v14.harmonic(axes)
est={}
for name,o in OURS.items():
    lo,hi=0.30,0.42;mid=ratio*o['public']
    est[name]=dict(inputs=o,sealed_mid=mid,mid=estimate(o,mid),lo=estimate(o,lo)[1],hi=estimate(o,hi)[1],jev_sealed=estimate(o,jev['sealed_accuracy'])[1])
# chart
BG='#fcfcfb';INK='#0b0b0b';INK2='#52514e';GRID='#e6e6e2';JEVC='#3b3b3b';C={'Rev Qwen3.5-4B':'#2a78d6','Rev Qwen3.8-27B':'#eb6834'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cfcfca','figure.facecolor':BG,'axes.facecolor':BG,'text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2})
fig,ax=plt.subplots(figsize=(11,6.4));fig.subplots_adjust(left=.08,right=.97,top=.84,bottom=.12)
fig.text(.03,.93,'JevBench v1.4.1 ranking, with Rev estimated',fontsize=16,weight='bold')
fig.text(.03,.885,'Grey: the %d ranked systems as published. Rev: estimates under the v1.4 formula. Bar: sealed accuracy 30%% to 42%%; dot: Jev\'s public-to-sealed ratio.'%len(rows),fontsize=9.5,color=INK2)
for s in rows:
    if s['key']=='jev-1.13.0':continue
    ax.scatter(s['cost']['usd_per_1000'],s['jevbench_score'],s=26,color='#b9b8b2',zorder=2)
ax.scatter(jev['cost']['usd_per_1000'],jev['jevbench_score'],s=90,color=JEVC,zorder=4);ax.annotate('Jev 1.13  %.1f'%jev['jevbench_score'],(jev['cost']['usd_per_1000'],jev['jevbench_score']),xytext=(8,-4),textcoords='offset points',fontsize=10.5,weight='bold')
for name,e in est.items():
    x=e['inputs']['usd'];ax.plot([x,x],[e['lo'],e['hi']],color=C[name],lw=3,alpha=.45,zorder=3,solid_capstyle='round')
    ax.scatter(x,e['mid'][1],s=90,color=C[name],zorder=5,edgecolor='white',linewidth=1.2)
    ax.annotate('%s  ~%.1f'%(name,e['mid'][1]),(x,e['mid'][1]),xytext=(9,4),textcoords='offset points',fontsize=10.5,weight='bold',color=C[name])
ax.set_xscale('log');ax.set_xlim(.008,.5);ax.set_ylim(0,75);ax.set_xlabel('Cost per 1,000 decisions (USD, log scale)');ax.set_ylabel('JevBench Score')
ax.set_xticks([.01,.02,.05,.1,.2],['$0.01','$0.02','$0.05','$0.10','$0.20']);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.grid(color=GRID,lw=.8);ax.set_axisbelow(True);ax.tick_params(length=0,pad=6)
fig.savefig(OUT/'jevbench_v14_estimate.png',dpi=170);fig.savefig(OUT/'jevbench_v14_estimate.svg')
out=dict(formula='jevbench/composite_v14.py @ '+board.get('revision',''),jev=dict(score=jev['jevbench_score'],axes=jev['axes'],public=jev['public_accuracy'],sealed=jev['sealed_accuracy']),
         jev_public_to_sealed_ratio=ratio,ours={n:dict(inputs=e['inputs'],sealed_assumed=e['sealed_mid'],axes=e['mid'][0],score=e['mid'][1],score_if_sealed_30=e['lo'],score_if_sealed_42=e['hi'],score_if_sealed_equals_jev=e['jev_sealed']) for n,e in est.items()},
         ranked_rows=len(rows),gap_over_25pp=sum(1 for s in rows if (s.get('public_minus_sealed_gap_pp') or 0)>25),intelligence_below_50=sum(1 for s in rows if s['axes']['intelligence']<50))
(OUT/'estimate.json').write_text(json.dumps(out,indent=1))
print(json.dumps(out,indent=1))
