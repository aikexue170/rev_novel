"""Charts for REV.md (the post). Numbers are the published ones from final/REPORT.md and the cache-busted multiq runs.

usage: post_charts.py [out_dir=post]   -> hero_holdout.png, latency_by_length.png, multiq.png, cost_concurrency.png, journey.png
"""
import sys,pathlib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'post');out.mkdir(exist_ok=True)
JEV='#3b3b3b';C={'4b':'#2a78d6','9b':'#1baf7a','27b':'#eb6834'};BG='#fcfcfb';INK='#0b0b0b';INK2='#52514e';GRID='#e6e6e2'
NAMES={'jev':'Hosted Jev','4b':'Qwen3.5-4B','9b':'Qwen3.5-9B','27b':'Qwen3.8-27B'};ORDER=['jev','4b','9b','27b']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cfcfca','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'figure.facecolor':BG,'axes.facecolor':BG})
def color(k):return JEV if k=='jev' else C[k]
def triptych(name,title,subtitle,panels):
 """panels: list of (panel title, unit, {arm: value}, fmt, lower_is_better). Missing panels are skipped."""
 panels=[p for p in panels if p[2]];n=len(panels)
 fig,axes=plt.subplots(1,n,figsize=(max(4.4*n+.6,8),4.6),squeeze=False);axes=axes[0];fig.subplots_adjust(left=.06,right=.98,top=.74,bottom=.2,wspace=.35)
 fig.text(.03,.92,title,fontsize=16,weight='bold');fig.text(.03,.855,subtitle,fontsize=10.5,color=INK2)
 for ax,(pt,unit,vals,fmt,lower) in zip(axes,panels):
  arms=[a for a in ORDER if a in vals];xs=range(len(arms));ys=[vals[a] for a in arms]
  ax.bar(xs,ys,color=[color(a) for a in arms],width=.62)
  for x,y in zip(xs,ys):ax.text(x,y,fmt(y),ha='center',va='bottom',fontsize=10.5,weight='bold')
  ax.axhline(vals['jev'],color=JEV,lw=1,ls='--',alpha=.6)
  ax.set_xticks(list(xs),[NAMES[a] for a in arms],fontsize=9.5);ax.set_title(pt+('  (lower is better)' if lower else '  (higher is better)'),loc='left',fontsize=12,weight='bold');ax.set_ylabel(unit)
  ax.grid(axis='y',color=GRID,lw=.8);ax.set_axisbelow(True);ax.tick_params(length=0,pad=6);ax.set_ylim(0,max(ys)*1.18)
  if pt.startswith('Accuracy'):ax.set_ylim(min(ys)-12,max(ys)+4)
 fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'));plt.close(fig)
pct=lambda v:f'{v:.1f}%';ms=lambda v:f'{v:.0f} ms';usd=lambda v:f'${v:.4f}'
import json
N=json.loads((pathlib.Path(__file__).resolve().parent/'post/numbers.json').read_text())   # built by refresh_numbers.py from one run
H=N['hero']
# 1. holdout: accuracy, speed, cost
triptych('hero_holdout','975-question public holdout: accuracy, speed, cost','Same questions for every system, both called from the same cloud client. Speed = end-to-end p50, one request in flight. Cost per 1,000 answers: Jev as billed, ours at Modal B200 list price at full load.',
 [('Accuracy','% correct',{k:H[k]['accuracy'] for k in H},pct,False),('Speed, end to end','milliseconds',{k:H[k]['p50_ms'] for k in H},ms,True),('Cost','USD per 1,000 answers',{k:H[k]['cost_per_1k'] for k in H},usd,True)])
# 3. latency vs input length, from the concurrency-1 pass of the holdout load test (post/holdout_latency_by_length.json)
import json
BL=json.loads((pathlib.Path(__file__).resolve().parent/'post/holdout_latency_by_length.json').read_text())
fig,ax=plt.subplots(figsize=(8.5,5));fig.subplots_adjust(left=.1,right=.8,top=.8,bottom=.14)
fig.text(.03,.92,'End-to-end latency vs input length',fontsize=16,weight='bold');fig.text(.03,.855,'975-question holdout bucketed by input tokens (Qwen tokenizer), one request in flight, end-to-end p50.',fontsize=10,color=INK2)
xs=range(len(BL['labels']))
for a in ORDER:
 ys=[b['p50'] for b in BL['arms'][a]];ax.plot(list(xs),ys,'-o',lw=2.2,ms=6,color=color(a));ax.annotate(NAMES[a],(len(ys)-1,ys[-1]),xytext=(8,0),textcoords='offset points',va='center',fontsize=10.5)
ax.set_xticks(list(xs),BL['labels']);ax.set_xlabel('Input tokens (state + question + options)');ax.set_ylabel('milliseconds');ax.set_ylim(0,max(b['p50'] for a in ORDER for b in BL['arms'][a])*1.15);ax.set_xlim(-.3,len(BL['labels'])-.5);ax.grid(axis='y',color=GRID,lw=.8);ax.set_axisbelow(True);ax.tick_params(length=0,pad=6)
fig.savefig(out/'latency_by_length.png',dpi=170);fig.savefig(out/'latency_by_length.svg');plt.close(fig)
# 4. latency vs questions per request
sizes=[1,5,20];MQ={k:[N['multiq'][k][str(s_)]['p50_ms'] for s_ in sizes] for k in ORDER if k in N['multiq']}
fig,ax=plt.subplots(figsize=(8.5,5));fig.subplots_adjust(left=.1,right=.8,top=.8,bottom=.14)
fig.text(.03,.92,'End-to-end latency vs questions per request',fontsize=16,weight='bold');fig.text(.03,.855,'Same document, 1 / 5 / 20 questions in one request, one request in flight, end-to-end p50, cache-busted.',fontsize=10.5,color=INK2)
for a in MQ:
 ax.plot(sizes,MQ[a],'-o',lw=2.2,ms=6,color=color(a));ax.annotate(NAMES[a],(sizes[-1],MQ[a][-1]),xytext=(8,0),textcoords='offset points',va='center',fontsize=10.5)
ax.set_xticks(sizes,[str(s) for s in sizes]);ax.set_xlabel('Questions per request');ax.set_ylabel('milliseconds');ax.set_ylim(0,max(v for m in MQ.values() for v in m)*1.15);ax.set_xlim(0,22);ax.grid(axis='y',color=GRID,lw=.8);ax.set_axisbelow(True);ax.tick_params(length=0,pad=6)
fig.savefig(out/'multiq.png',dpi=170);fig.savefig(out/'multiq.svg');plt.close(fig)
# 5. cost vs concurrency
CC={k:(N['cost_curve'][k]['levels'],N['cost_curve'][k]['cost_per_1k']) for k in ['4b','9b','27b'] if k in N['cost_curve']}
fig,ax=plt.subplots(figsize=(8.5,5));fig.subplots_adjust(left=.11,right=.8,top=.8,bottom=.14)
fig.text(.03,.92,'Cost per 1,000 answers vs requests in flight',fontsize=16,weight='bold');fig.text(.03,.855,'975-question holdout, one question per request. Self-hosted = B200 + CPU + RAM list price ÷ measured answers/s.',fontsize=10,color=INK2)
for a in ['4b','9b','27b']:
 xs,ys=CC[a];ax.plot(xs,ys,'-o',lw=2.2,ms=6,color=color(a));ax.annotate(NAMES[a],(xs[-1],ys[-1]),xytext=(8,0),textcoords='offset points',va='center',fontsize=10.5)
ax.axhline(H['jev']['cost_per_1k'],color=JEV,lw=2,ls='--');ax.annotate('Hosted Jev, as billed',(1,H['jev']['cost_per_1k']),xytext=(0,7),textcoords='offset points',fontsize=10.5,weight='bold')
ax.set_xscale('log',base=2);ax.set_xticks([1,8,32,128,256],['1','8','32','128','256']);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());ax.set_xlim(.8,700)
ax.set_yscale('log');ax.set_yticks([.02,.03,.05,.1,.2],['$0.02','$0.03','$0.05','$0.10','$0.20']);ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel('Concurrent requests in flight');ax.set_ylabel('USD per 1,000 answers');ax.grid(axis='y',color=GRID,lw=.8);ax.set_axisbelow(True);ax.tick_params(length=0,pad=6)
fig.savefig(out/'cost_concurrency.png',dpi=170);fig.savefig(out/'cost_concurrency.svg');plt.close(fig)
# 6. journey: accuracy by approach, and time to answer 20 questions by approach
fig,axes=plt.subplots(1,2,figsize=(13,5.2));fig.subplots_adjust(left=.25,right=.97,top=.78,bottom=.14,wspace=1.05)
fig.text(.03,.92,'The journey in two numbers',fontsize=16,weight='bold');fig.text(.03,.855,'Left: holdout accuracy by approach (Qwen3.5-4B unless noted). Right: end-to-end time to answer 20 questions on one document, one request in flight.',fontsize=10,color=INK2)
acc=[('Letter logits, untrained (27B)',85.0,'#9a9a95'),('Head only, best layer',84.3,'#9a9a95'),('LoRA + head',87.6,C['4b']),('LoRA + head (Qwen3.8-27B)',91.1,C['27b'])]
ax=axes[0];ys=range(len(acc));ax.barh(list(ys),[a[1] for a in acc],color=[a[2] for a in acc],height=.55)
for y,a in zip(ys,acc):ax.text(a[1]+.2,y,f'{a[1]:.1f}%',va='center',fontsize=10.5,weight='bold')
ax.axvline(H['jev']['accuracy'],color=JEV,lw=1.5,ls='--');ax.text(H['jev']['accuracy']+.2,-.62,'Hosted Jev %.1f%%'%H['jev']['accuracy'],fontsize=9.5,ha='left',weight='bold')
ax.set_yticks(list(ys),[a[0] for a in acc]);ax.invert_yaxis();ax.set_xlim(78,94);ax.set_xlabel('% correct on the 975-question holdout');ax.grid(axis='x',color=GRID);ax.set_axisbelow(True);ax.tick_params(length=0);ax.spines['left'].set_visible(False)
lat=[('JSON answers from a hosted chat model',5100,'#9a9a95'),('Compact outputs (Y/N letters, bitstrings)',3250,'#9a9a95'),('Hosted Jev',160,JEV),('LoRA + head, batched serving',86,C['4b'])]
ax=axes[1];ys=range(len(lat));ax.barh(list(ys),[l[1] for l in lat],color=[l[2] for l in lat],height=.55)
for y,l in zip(ys,lat):ax.text(l[1]*1.08,y,f'{l[1]:,} ms',va='center',fontsize=10.5,weight='bold')
ax.set_yticks(list(ys),[l[0] for l in lat]);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(50,20000);ax.set_xticks([100,1000,10000],['100 ms','1 s','10 s']);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel('time to answer 20 questions (log scale)');ax.grid(axis='x',color=GRID);ax.set_axisbelow(True);ax.tick_params(length=0);ax.spines['left'].set_visible(False)
fig.savefig(out/'journey.png',dpi=170);fig.savefig(out/'journey.svg');plt.close(fig)
print('wrote',sorted(p.name for p in out.iterdir()))
