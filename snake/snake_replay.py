"""Build a self-contained HTML replay page for a Snake arena run.

usage: snake/snake_replay.py <snake_dir>   (fetches <arm>_games.json / <arm>_summary.json from the results volume)
writes <snake_dir>/summary.json, REPORT.md and replay.html (side-by-side boards, play/pause/step, per-tick choice + probabilities).
"""
import sys,json,subprocess,tempfile,pathlib,statistics
PY='/tmp/jev-modal-latency-env/bin/python';ROOT=pathlib.Path(__file__).resolve().parents[1]
def rundir(a):   # a run directory: as given, or by name under <repo>/runs/
 p=pathlib.Path(a);return p if p.exists() or not (ROOT/'runs'/a).exists() else ROOT/'runs'/a
d=rundir(sys.argv[1]);tmp=pathlib.Path(tempfile.mkdtemp())
job=json.loads((d/'job.json').read_text());arms=[a['name'] for a in job['arms']];grid=job['grid']
LABEL={'jev':'Hosted Jev','small4b':'Qwen 4B + head','small9b':'Qwen 9B + head','dense27b':'Qwen 27B + head','small2h':'Qwen 2B + head','vis4b':'Qwen 4B + head (image board)','vis9b':'Qwen 9B + head (image board)','vis27b':'Qwen 27B + head (image board)'}
def fetch(name):
 p=tmp/name;r=subprocess.run([PY,'-m','modal','volume','get','jev-benchmark-results',d.name+'/'+name,str(p),'--force'],capture_output=True,text=True);return json.loads(p.read_text()) if r.returncode==0 else None
games={};summ={}
for a in arms:
 g=fetch(a+'_games.json');s=fetch(a+'_summary.json')
 if g and s:games[a]=[x for x in g if 'ticks' in x];summ[a]=s
rows=[]
for a in arms:
 if a not in summ:continue
 s=summ[a];eaten=[x['eaten'] for x in s];steps=[x['steps'] for x in s];lat=[x['median_ms'] for x in s if x['median_ms']]
 deaths={}
 for x in s:deaths[x['death']]=deaths.get(x['death'],0)+1
 rows.append(dict(arm=a,label=LABEL.get(a,a),games=len(s),food_total=sum(eaten),food_mean=statistics.mean(eaten),food_best=max(eaten),steps_mean=statistics.mean(steps),median_ms=statistics.median(lat) if lat else None,deaths=deaths,cost=sum(x['cost'] for x in s)))
rows.sort(key=lambda r:-r['food_total'])
lines=['# Snake arena '+d.name,'',f"{job['games']} games per player, {grid}x{grid} grid, up to {job['max_steps']} steps, identical seeds. One `choice` question per tick (up/down/left/right) from the same board text. Ranked by total food eaten; steps survived as tie-break.",'','| Player | Food total | Food / game | Best game | Steps / game | Deaths | Median decision | Cost |','|---|---:|---:|---:|---:|---|---:|---:|']
for r in rows:lines.append(f"| {r['label']} | **{r['food_total']}** | {r['food_mean']:.1f} | {r['food_best']} | {r['steps_mean']:.0f} | {', '.join(f'{k} {v}' for k,v in r['deaths'].items())} | {r['median_ms']:.0f} ms | {'$%.4f'%r['cost'] if r['cost'] else '—'} |")
(d/'REPORT.md').write_text('\n'.join(lines)+'\n');(d/'summary.json').write_text(json.dumps(rows,indent=1))
# replay page
data={a:[dict(ticks=[dict(s=t['snake'],f=t['food'],c=t['choice'],p=t['probs']) for t in g['ticks']],final=g['final'],eaten=summ[a][i]['eaten'],death=summ[a][i]['death']) for i,g in enumerate(games[a])] for a in games}
html='''<!doctype html><html><head><meta charset="utf-8"><title>Snake arena</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e2e2dd;--card:#ffffff;--snake:#2a78d6;--head:#0c447c;--food:#e34948;--grid:#f1f1ee}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#fff;--muted:#c3c2b7;--line:#33332f;--card:#232322;--snake:#3987e5;--head:#b5d4f4;--food:#e66767;--grid:#2a2a28}}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
h1{font-size:20px;margin:0 0 4px}p{color:var(--muted);margin:0 0 12px}
table{border-collapse:collapse;margin:0 0 16px;width:100%;max-width:900px}td,th{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:0 0 12px}button,select{font:inherit;padding:6px 12px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px}
.boards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}.board{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.board h2{font-size:15px;margin:0 0 6px}.board canvas{width:100%;height:auto;display:block;background:var(--grid);border-radius:6px}.meta{font-size:13px;color:var(--muted);margin-top:6px;min-height:3.2em}
</style></head><body>
<h1>Snake arena: decision models vs Hosted Jev</h1>
<p>Same board text every tick, one multiple-choice question: which direction? Identical seeds per game. Playback shows each player's chosen move and its probabilities.</p>
<div id="table"></div>
<div class="controls"><label>Game <select id="game"></select></label><button id="play">Play</button><button id="stepb">Step</button><button id="reset">Reset</button><label>Speed <input id="speed" type="range" min="1" max="20" value="8"></label><span id="tick" class="meta"></span></div>
<div class="boards" id="boards"></div>
<script>
const DATA=__DATA__,ARMS=__ARMS__,LABEL=__LABEL__,N=__GRID__,ROWS=__ROWS__;
const tb=document.getElementById('table');tb.innerHTML='<table><tr><th>Player</th><th>Food total</th><th>Food / game</th><th>Best</th><th>Steps / game</th><th>Median decision</th></tr>'+ROWS.map(r=>`<tr><td>${r.label}</td><td><b>${r.food_total}</b></td><td>${r.food_mean.toFixed(1)}</td><td>${r.food_best}</td><td>${r.steps_mean.toFixed(0)}</td><td>${r.median_ms?r.median_ms.toFixed(0)+' ms':''}</td></tr>`).join('')+'</table>';
const boards=document.getElementById('boards'),sel=document.getElementById('game');let t=0,timer=null;
ARMS.forEach(a=>{const b=document.createElement('div');b.className='board';b.innerHTML=`<h2>${LABEL[a]||a}</h2><canvas id="c-${a}" width="300" height="300"></canvas><div class="meta" id="m-${a}"></div>`;boards.appendChild(b)});
const G=DATA[ARMS[0]].length;for(let i=0;i<G;i++){const o=document.createElement('option');o.value=i;o.textContent='#'+(i+1);sel.appendChild(o)}
function draw(a,gi,ti){const c=document.getElementById('c-'+a),ctx=c.getContext('2d'),g=DATA[a][gi],ticks=g.ticks,cell=300/N;ctx.clearRect(0,0,300,300);
 const css=getComputedStyle(document.documentElement);ctx.strokeStyle=css.getPropertyValue('--line');for(let i=0;i<=N;i++){ctx.beginPath();ctx.moveTo(i*cell,0);ctx.lineTo(i*cell,300);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*cell);ctx.lineTo(300,i*cell);ctx.stroke()}
 const k=Math.min(ti,ticks.length-1);const st=ti<ticks.length?ticks[k]:{s:g.final.snake,f:g.final.food,c:null,p:null};
 ctx.fillStyle=css.getPropertyValue('--food');ctx.beginPath();ctx.arc((st.f[0]+.5)*cell,(st.f[1]+.5)*cell,cell*.3,0,7);ctx.fill();
 st.s.forEach((p,i)=>{ctx.fillStyle=css.getPropertyValue(i?'--snake':'--head');ctx.fillRect(p[0]*cell+1,p[1]*cell+1,cell-2,cell-2)});
 const m=document.getElementById('m-'+a);const done=ti>=ticks.length;
 m.textContent=done?`Game over: ${g.death}. Food ${g.eaten}, ${ticks.length} moves.`:`Move ${k+1}/${ticks.length}: ${st.c||'?'}  `+(st.p?Object.entries(st.p).sort((x,y)=>y[1]-x[1]).map(([d,p])=>`${d} ${(p*100).toFixed(0)}%`).join(' · '):'');}
function render(){const gi=+sel.value;ARMS.forEach(a=>draw(a,gi,t));document.getElementById('tick').textContent='tick '+t;const maxT=Math.max(...ARMS.map(a=>DATA[a][gi].ticks.length));if(t>maxT&&timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play'}}
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play';return}document.getElementById('play').textContent='Pause';timer=setInterval(()=>{t++;render()},1000/+document.getElementById('speed').value)};
document.getElementById('stepb').onclick=()=>{t++;render()};document.getElementById('reset').onclick=()=>{t=0;render()};sel.onchange=()=>{t=0;render()};render();
</script></body></html>'''
html=html.replace('__DATA__',json.dumps(data)).replace('__ARMS__',json.dumps([a for a in arms if a in games])).replace('__LABEL__',json.dumps(LABEL)).replace('__GRID__',str(grid)).replace('__ROWS__',json.dumps(rows))
(d/'replay.html').write_text(html);print('\n'.join(lines));print('replay:',d/'replay.html')
