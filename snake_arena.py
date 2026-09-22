"""Snake arena: decision models play Snake from identical board states, one 'choice' question per tick.

Each tick the model sees the board (coordinates, body list, food, current direction, ASCII grid) and picks one of
up/down/left/right. Moving into a wall or its own body ends the game; eating grows the snake. Same seeds for every
arm, so the food sequence is identical until the players diverge. Scores: food eaten, steps survived, cause of death,
median decision latency, and Jev's billed cost. Logs every tick for replay.

usage: snake_arena.py '[{"name":"jev","url":"jev"},{"name":"small4b","url":"https://..."}]' [games=10] [grid=10] [max_steps=200]
"""
import modal,json,sys,datetime,random
from pathlib import Path
app=modal.App('jev-beat-snake');vol=modal.Volume.from_name('jev-benchmark-results')
image=modal.Image.debian_slim(python_version='3.12').pip_install('requests')
DIRS={'up':(0,-1),'down':(0,1),'left':(-1,0),'right':(1,0)}
def new_game(seed,n):
 rnd=random.Random(seed);cx,cy=n//2,n//2;snake=[(cx,cy),(cx-1,cy),(cx-2,cy)]
 return dict(rnd=rnd,n=n,snake=snake,dir='right',food=place_food(rnd,n,snake),eaten=0,steps=0,since_food=0,alive=True,death=None)
def place_food(rnd,n,snake):
 free=[(x,y) for x in range(n) for y in range(n) if (x,y) not in snake];return rnd.choice(free)
def render(g):
 n=g['n'];hx,hy=g['snake'][0];grid=[['.']*n for _ in range(n)]
 for (x,y) in g['snake'][1:]:grid[y][x]='o'
 grid[hy][hx]='H';fx,fy=g['food'];grid[fy][fx]='F'
 rows='\n'.join(f"row {y}: "+''.join(grid[y]) for y in range(n))
 body=', '.join(f"({x},{y})" for x,y in g['snake'][1:])
 return (f"Snake game on a {n}x{n} grid. Coordinates are (x,y): x from 0 (left) to {n-1} (right), y from 0 (top) to {n-1} (bottom).\n"
  f"Head at ({hx},{hy}). Body from neck to tail: {body}. Food at ({fx},{fy}). Current direction: {g['dir']}. Length {len(g['snake'])}. Steps so far: {g['steps']}.\n"
  f"Grid ('H' head, 'o' body, 'F' food, '.' empty):\n{rows}")
def options(g):
 hx,hy=g['snake'][0];n=g['n'];out={}
 for d,(dx,dy) in DIRS.items():
  nx,ny=hx+dx,hy+dy;out[d]=f"{d}: move the head to ({nx},{ny})"
 return out
def step(g,d):
 dx,dy=DIRS[d];hx,hy=g['snake'][0];nx,ny=hx+dx,hy+dy;n=g['n'];g['steps']+=1;g['dir']=d
 if not (0<=nx<n and 0<=ny<n):g['alive']=False;g['death']='wall';return
 tail=g['snake'][-1];body=g['snake'][:-1] if (nx,ny)!=g['food'] else g['snake']
 if (nx,ny) in body:g['alive']=False;g['death']='self';return
 g['snake'].insert(0,(nx,ny))
 if (nx,ny)==g['food']:
  g['eaten']+=1;g['since_food']=0
  if len(g['snake'])==n*n:g['alive']=False;g['death']='won';return
  g['food']=place_food(g['rnd'],n,g['snake'])
 else:g['snake'].pop();g['since_food']+=1
@app.function(image=image,region='us-west',timeout=3*3600,volumes={'/results':vol},secrets=[modal.Secret.from_name('openrouter')])
def play(arms,games,grid,max_steps,runid):
 import os,time,requests,statistics,concurrent.futures
 out=Path('/results')/runid;out.mkdir(exist_ok=True)
 Q=('You are playing Snake. Each turn the snake moves its head one cell in the chosen direction and the body follows. '
    'If the head moves outside the grid, or onto any cell occupied by the body (including the neck, so reversing direction kills), the game ends. '
    'Moving onto the food eats it: the snake grows by one cell and the score goes up. '
    'Goal: stay alive for as many steps as possible and eat as much food as possible. '
    'Which direction should the snake move next? Pick a direction that is safe this turn, brings the head closer to the food, and does not paint the snake into a corner later: avoid moves that lead into dead ends or pockets enclosed by walls and your own body, and keep an open path back to free space.')
 def decide(arm,s,state,crit):
  t=time.perf_counter()
  if arm['url']=='jev':
   r=s.post('https://openrouter.ai/api/alpha/decisions',json=dict(model='typesafe/jev-1.13',provider={'only':['TypeSafe'],'allow_fallbacks':False},state=state,questions={'move':{'type':'choice','instructions':Q,'criteria':crit}}),headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']},timeout=60);r.raise_for_status();d=r.json();a=d['answers']['move'];cost=(d.get('usage') or {}).get('cost',0) or 0
  else:
   r=s.post(arm['url']+'/score',json=dict(state=state,questions=[{'id':'move','instructions':Q,'criteria':crit}]),timeout=120);r.raise_for_status();d=r.json();a=d['answers']['move'];cost=0
  return a['choice'],a['probabilities'],time.perf_counter()-t,cost
 def run_arm(arm):
  s=requests.Session()
  if arm['url']!='jev':
   for _ in range(200):
    try:
     if s.post(arm['url']+'/ping',json={},timeout=1800).status_code==200:break
    except Exception:pass
    time.sleep(10)
  summary=[];log=[]
  for gi in range(games):
   g=new_game(1000+gi,grid);cost=0;lat=[];ticks=[]
   while g['alive'] and g['steps']<max_steps and g['since_food']<grid*grid:
    state=render(g);crit=options(g)
    try:choice,probs,sec,c=decide(arm,s,state,crit)
    except Exception as e:choice,probs,sec,c=None,None,0,0;log.append(dict(game=gi,error=repr(e)[:200]));break
    cost+=c;lat.append(sec);ticks.append(dict(step=g['steps'],snake=list(g['snake']),food=g['food'],choice=choice,probs=probs,seconds=sec))
    step(g,choice)
   if g['alive']:g['death']='timeout' if g['steps']>=max_steps else 'starved'
   summary.append(dict(game=gi,eaten=g['eaten'],steps=g['steps'],death=g['death'],median_ms=statistics.median(lat)*1000 if lat else None,cost=cost));log.append(dict(game=gi,ticks=ticks,final=dict(snake=list(g['snake']),food=g['food'],death=g['death'])))
   print('GAME',arm['name'],gi,g['eaten'],g['steps'],g['death'],flush=True)
  (out/(arm['name']+'_games.json')).write_text(json.dumps(log));(out/(arm['name']+'_summary.json')).write_text(json.dumps(summary));vol.commit()
  return summary
 with concurrent.futures.ThreadPoolExecutor(len(arms)) as pool:res={a['name']:f for a,f in zip(arms,pool.map(run_arm,arms))}
 (out/'status.json').write_text(json.dumps({'status':'complete'}));vol.commit();return res
if __name__=='__main__':
 arms=json.loads(sys.argv[1]);games=int(sys.argv[2]) if len(sys.argv)>2 else 10;grid=int(sys.argv[3]) if len(sys.argv)>3 else 10;max_steps=int(sys.argv[4]) if len(sys.argv)>4 else 200
 here=Path(__file__).resolve().parent;out=here/('snake_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir();(out/'snake_arena.py').write_text(Path(__file__).read_text())
 with app.run(detach=True):
  call=play.spawn(arms,games,grid,max_steps,out.name);(out/'job.json').write_text(json.dumps(dict(app_id=app.app_id,call_id=call.object_id,arms=arms,games=games,grid=grid,max_steps=max_steps),indent=2));print(out.name,app.app_id)
