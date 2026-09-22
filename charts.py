"""Charts for the final comparison.

usage: charts.py <out_dir> <load_dir> [<multiq_dir>]
Reads <load_dir>/summary.json (from report.py) and optional <multiq_dir>/summary.json (from report_multiq.py).
Writes throughput_cost.png, latency.png, accuracy.png, multiq.png (+ .svg) into <out_dir>.
"""
import sys,json,math,pathlib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=pathlib.Path(sys.argv[1]);out.mkdir(exist_ok=True);load=json.loads((pathlib.Path(sys.argv[2])/'summary.json').read_text())
multiq=json.loads((pathlib.Path(sys.argv[3])/'summary.json').read_text()) if len(sys.argv)>3 else None
# Reference palette (dataviz skill): categorical slots in fixed order; Jev is the neutral reference ink.
SLOTS=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'];JEV='#3b3b3b';BG='#fcfcfb';INK='#0b0b0b';INK2='#52514e';GRID='#e6e6e2'
LABELS={'jev':'Hosted Jev (OpenRouter)','small2h':'Qwen 2B · H100','small2b':'Qwen 2B · B200','small4h':'Qwen 4B · H100','small4b':'Qwen 4B · B200','small9h':'Qwen 9B · H100','small9b':'Qwen 9B · B200','dense27h':'Qwen 27B · H100','dense27b':'Qwen 27B · B200','small4bal':'Qwen 4B balanced · B200','small9bal':'Qwen 9B balanced · B200'}
ORDER=['small2h','small2b','small4h','small4b','small4bal','small9h','small9b','small9bal','dense27h','dense27b']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cfcfca','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'figure.facecolor':BG,'axes.facecolor':BG})
rows=load['rows'];arms=[a for a in ORDER if any(r['arm']==a for r in rows)];color={a:SLOTS[i%len(SLOTS)] for i,a in enumerate(arms)};color['jev']=JEV
jev=[r for r in rows if r['arm']=='jev'];jev1=next((r for r in jev if r['concurrency']==1),jev[0] if jev else None)
def tidy(ax):ax.grid(axis='y',color=GRID,lw=.8);ax.set_axisbelow(True);ax.tick_params(length=0,pad=6)
def label_end(ax,x,y,text,c):ax.annotate(text,(x,y),xytext=(6,0),textcoords='offset points',va='center',fontsize=10,color=INK)
# 1. throughput and cost vs concurrency
fig,axes=plt.subplots(1,2,figsize=(13,5.6));fig.subplots_adjust(left=.07,right=.97,top=.8,bottom=.16,wspace=.35)
fig.text(.04,.93,'Throughput and cost under load: 975 public-holdout questions, one question per request',fontsize=16,weight='bold')
fig.text(.04,.875,'Self-hosted cost = GPU + provisioned CPU/RAM list price ÷ measured answers per second at that concurrency. Jev = billed API usage.',fontsize=10.5,color=INK2)
for a in arms:
 rs=sorted((r for r in rows if r['arm']==a),key=lambda r:r['concurrency']);xs=[r['concurrency'] for r in rs]
 axes[0].plot(xs,[r['answers_per_second'] for r in rs],'-o',lw=2,ms=6,color=color[a]);label_end(axes[0],xs[-1],rs[-1]['answers_per_second'],LABELS[a],color[a])
 axes[1].plot(xs,[r['cost_per_1k'] for r in rs],'-o',lw=2,ms=6,color=color[a]);label_end(axes[1],xs[-1],rs[-1]['cost_per_1k'],LABELS[a],color[a])
if jev:
 xs=[r['concurrency'] for r in sorted(jev,key=lambda r:r['concurrency'])];axes[0].plot(xs,[r['answers_per_second'] for r in sorted(jev,key=lambda r:r['concurrency'])],'-o',lw=2,ms=6,color=JEV);label_end(axes[0],xs[-1],max(r['answers_per_second'] for r in jev),LABELS['jev'],JEV)
 axes[1].axhline(jev1['cost_per_1k'],color=JEV,lw=2,ls='--');axes[1].annotate(f"Hosted Jev  ${jev1['cost_per_1k']:.4f} (billed)",(1,jev1['cost_per_1k']),xytext=(0,7),textcoords='offset points',fontsize=10,color=INK,weight='bold')
for ax,t,yl in zip(axes,['Answers per second','Cost per 1,000 answers (USD)'],['answers / s','$ per 1k']):
 ax.set_xscale('log',base=2);ax.set_xlabel('Concurrent requests in flight');ax.set_ylabel(yl);ax.set_title(t,loc='left',fontsize=13,weight='bold');tidy(ax);ax.set_xlim(right=ax.get_xlim()[1]*3.2)
 levels=sorted({r['concurrency'] for r in rows});ax.set_xticks(levels,[str(x) for x in levels]);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[1].set_yscale('log');axes[1].set_yticks([.01,.02,.03,.05,.1,.2],['$0.01','$0.02','$0.03','$0.05','$0.10','$0.20']);axes[1].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
fig.savefig(out/'throughput_cost.png',dpi=170);fig.savefig(out/'throughput_cost.svg');plt.close(fig)
# 2. latency at concurrency 1 (p50 / p95)
sel=[r for a in ['jev']+arms for r in rows if r['arm']==a and r['concurrency']==1]
fig,ax=plt.subplots(figsize=(10,1.1+.55*len(sel)));fig.subplots_adjust(left=.28,right=.9,top=.78,bottom=.14)
fig.text(.03,.92,'End-to-end latency, one request in flight (p50, with p95 whisker)',fontsize=15,weight='bold');fig.text(.03,.85,'Cloud client in us-west; includes network, tokenization, model forward and response.',fontsize=10,color=INK2)
ys=range(len(sel));ax.barh(ys,[r['p50_ms'] for r in sel],color=[color[r['arm']] for r in sel],height=.5)
for y,r in zip(ys,sel):ax.plot([r['p50_ms'],r['p95_ms']],[y,y],color=INK2,lw=1.2);ax.text(r['p95_ms']+4,y,f"{r['p50_ms']:.0f} ms  (p95 {r['p95_ms']:.0f})",va='center',fontsize=10)
ax.set_yticks(list(ys),[LABELS[r['arm']] for r in sel]);ax.invert_yaxis();ax.set_xlabel('milliseconds');ax.grid(axis='x',color=GRID);ax.set_axisbelow(True);ax.tick_params(length=0);ax.spines['left'].set_visible(False);ax.set_xlim(0,max(r['p95_ms'] for r in sel)*1.45)
fig.savefig(out/'latency.png',dpi=170);fig.savefig(out/'latency.svg');plt.close(fig)
# 3. accuracy (concurrency-1 pass) with Wilson 95% interval
def wilson(k,n,z=1.96):
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
fig,ax=plt.subplots(figsize=(10,1.1+.55*len(sel)));fig.subplots_adjust(left=.28,right=.9,top=.78,bottom=.14)
fig.text(.03,.92,'Accuracy on the 975-question public holdout',fontsize=15,weight='bold');fig.text(.03,.85,'Same questions, options and labels for every system; bars show Wilson 95% intervals.',fontsize=10,color=INK2)
n=975
for y,r in zip(ys,sel):
 k=round(r['accuracy']*n);lo,hi=wilson(k,n);ax.barh(y,r['accuracy']*100,color=color[r['arm']],height=.5);ax.plot([lo*100,hi*100],[y,y],color=INK2,lw=1.2);ax.text(hi*100+.3,y,f"{r['accuracy']*100:.1f}%",va='center',fontsize=10)
ax.set_yticks(list(ys),[LABELS[r['arm']] for r in sel]);ax.invert_yaxis();ax.set_xlim(70,100);ax.set_xlabel('% correct');ax.grid(axis='x',color=GRID);ax.set_axisbelow(True);ax.tick_params(length=0);ax.spines['left'].set_visible(False)
fig.savefig(out/'accuracy.png',dpi=170);fig.savefig(out/'accuracy.svg');plt.close(fig)
# 4. multi-question latency
if multiq:
 mr=multiq['rows'];marms=[a for a in ['jev']+ORDER if any(r['arm']==a for r in mr)]
 fig,ax=plt.subplots(figsize=(9,5.2));fig.subplots_adjust(left=.1,right=.78,top=.8,bottom=.14)
 fig.text(.03,.92,'Latency vs questions per request (same shared context)',fontsize=15,weight='bold');fig.text(.03,.855,'4 invoice contexts, 1 / 5 / 20 questions per request, one request in flight, e2e p50.',fontsize=10,color=INK2)
 for a in marms:
  rs=sorted((r for r in mr if r['arm']==a),key=lambda r:r['size']);ax.plot([r['size'] for r in rs],[r['p50_ms'] for r in rs],'-o',lw=2,ms=6,color=color.get(a,JEV));label_end(ax,rs[-1]['size'],rs[-1]['p50_ms'],LABELS.get(a,a),color.get(a,JEV))
 ax.set_xticks([1,5,20]);ax.set_xlabel('questions per request');ax.set_ylabel('e2e p50 (ms)');ax.set_ylim(bottom=0);tidy(ax)
 fig.savefig(out/'multiq.png',dpi=170);fig.savefig(out/'multiq.svg');plt.close(fig)
print('charts written to',out)
