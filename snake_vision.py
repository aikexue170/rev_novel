"""Snake arena, vision edition: the board is sent as a PNG; the text carries only rules, grid size, direction and length.

Players are the vision servers (vision_server.py). Same engine and seeds as snake_arena.py so scores are comparable.
usage: snake_vision.py '[{"name":"vis27b","url":"https://..."}]' [games=10] [grid=10] [max_steps=200]
"""
import modal,json,sys,datetime,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
app=modal.App('jev-beat-snake-vision');vol=modal.Volume.from_name('jev-benchmark-results')
image=modal.Image.debian_slim(python_version='3.12').pip_install('requests','pillow')
DIRS={'up':(0,-1),'down':(0,1),'left':(-1,0),'right':(1,0)}
def new_game(seed,n):
 rnd=random.Random(seed);cx,cy=n//2,n//2;snake=[(cx,cy),(cx-1,cy),(cx-2,cy)]
 return dict(rnd=rnd,n=n,snake=snake,dir='right',food=place_food(rnd,n,snake),eaten=0,steps=0,since_food=0,alive=True,death=None)
def place_food(rnd,n,snake):
 free=[(x,y) for x in range(n) for y in range(n) if (x,y) not in snake];return rnd.choice(free)
def step(g,d):
 dx,dy=DIRS[d];hx,hy=g['snake'][0];nx,ny=hx+dx,hy+dy;n=g['n'];g['steps']+=1;g['dir']=d
 if not (0<=nx<n and 0<=ny<n):g['alive']=False;g['death']='wall';return
 body=g['snake'][:-1] if (nx,ny)!=g['food'] else g['snake']
 if (nx,ny) in body:g['alive']=False;g['death']='self';return
 g['snake'].insert(0,(nx,ny))
 if (nx,ny)==g['food']:
  g['eaten']+=1;g['since_food']=0
  if len(g['snake'])==n*n:g['alive']=False;g['death']='won';return
  g['food']=place_food(g['rnd'],n,g['snake'])
 else:g['snake'].pop();g['since_food']+=1
def render_png(g,cell=32):
 from PIL import Image,ImageDraw
 import io,base64
 n=g['n'];im=Image.new('RGB',(n*cell,n*cell),(245,245,240));d=ImageDraw.Draw(im)
 for i in range(n+1):d.line([(i*cell,0),(i*cell,n*cell)],fill=(215,215,208));d.line([(0,i*cell),(n*cell,i*cell)],fill=(215,215,208))
 for (x,y) in g['snake'][1:]:d.rectangle([x*cell+2,y*cell+2,(x+1)*cell-3,(y+1)*cell-3],fill=(42,120,214))
 hx,hy=g['snake'][0];d.rectangle([hx*cell+2,hy*cell+2,(hx+1)*cell-3,(hy+1)*cell-3],fill=(12,68,124))
 # eye marks the facing direction
 ex,ey=DIRS[g['dir']];d.ellipse([hx*cell+cell//2+ex*8-4,hy*cell+cell//2+ey*8-4,hx*cell+cell//2+ex*8+4,hy*cell+cell//2+ey*8+4],fill=(255,255,255))
 fx,fy=g['food'];d.ellipse([fx*cell+7,fy*cell+7,(fx+1)*cell-8,(fy+1)*cell-8],fill=(227,73,72))
 b=io.BytesIO();im.save(b,format='PNG');return base64.b64encode(b.getvalue()).decode()
def state_text(g):
 n=g['n'];return f"The image shows a Snake board: a {n}x{n} grid. Blue squares are the snake's body, the dark blue square with a white dot is its head (the dot marks the side it is facing), and the red circle is the food. The snake is currently moving {g['dir']}, is {len(g['snake'])} cells long, and has taken {g['steps']} steps."
OPP={'up':'down','down':'up','left':'right','right':'left'}
def deadly(g,d):
 import copy;h=copy.deepcopy(g);h['rnd']=random.Random(0);step(h,d);return not h['alive']
def plans(g):
 """8 two-move sequences: 4 straight pairs + 4 L-turns (turn toward the food on the other axis; clockwise if aligned)."""
 hx,hy=g['snake'][0];fx,fy=g['food'];out=[]
 for d in ['up','down','left','right']:
  out.append((d,d))
  if d in ('up','down'):t='right' if fx>hx else 'left' if fx<hx else ('right' if d=='up' else 'left')
  else:t='down' if fy>hy else 'up' if fy<hy else ('down' if d=='right' else 'up')
  out.append((d,t))
 return out
def plan_score(g,seq):
 import copy;h=copy.deepcopy(g);h['rnd']=random.Random(0);gain=0
 for d in seq:
  before=h['eaten'];step(h,d)
  if not h['alive']:return None
  gain+=h['eaten']-before
 hx,hy=h['snake'][0];fx,fy=h['food'];return (gain,-(abs(hx-fx)+abs(hy-fy)))
def extra_questions(g):
 kill=[d for d in DIRS if deadly(g,d)];kill_label='none' if not kill else None
 kcrit={d:f"{d}: moving {d} kills the snake (wall or body)" for d in DIRS};kcrit['none']='none: no single move kills the snake this turn'
 ps=plans(g);scores={('%s then %s'%s_):plan_score(g,s_) for s_ in ps};best=max((v for v in scores.values() if v is not None),default=None)
 pcrit={k:f"{k}: first move {k.split(' then ')[0]}, second move {k.split(' then ')[1]}" for k in scores}
 plan_label=[k for k,v in scores.items() if v is not None and v==best]
 return dict(kill=dict(criteria=kcrit,correct=(kill or ['none'])),plan=dict(criteria=pcrit,correct=plan_label))
@app.function(image=image,region='us-west',timeout=3*3600,volumes={'/results':vol})
def play(arms,games,grid,max_steps,runid):
 import time,requests,statistics,concurrent.futures
 out=Path('/results')/runid;out.mkdir(exist_ok=True)
 Q=('You are playing Snake. Each turn the snake moves its head one cell in the chosen direction and the body follows. '
    'If the head moves outside the grid, or onto any cell occupied by the body (including the neck, so reversing direction kills), the game ends. '
    'Moving onto the food eats it: the snake grows by one cell and the score goes up. '
    'Goal: stay alive for as many steps as possible and eat as much food as possible. '
    'Look at the board image. Which direction should the snake move next? Pick a direction that is safe this turn, leaves room to keep moving afterwards, and brings the head closer to the food.')
 crit={'up':'up: move the head one cell up (toward the top of the image)','down':'down: move the head one cell down (toward the bottom of the image)','left':'left: move the head one cell to the left','right':'right: move the head one cell to the right'}
 def run_arm(arm):
  s=requests.Session()
  for _ in range(200):
   try:
    if s.post(arm['url']+'/ping',json={},timeout=1800).status_code==200:break
   except Exception:pass
   time.sleep(10)
  summary=[];log=[]
  for gi in range(games):
   g=new_game(1000+gi,grid);lat=[];ticks=[]
   while g['alive'] and g['steps']<max_steps and g['since_food']<grid*grid:
    t=time.perf_counter();ex=extra_questions(g)
    qs=[{'id':'move','instructions':Q,'criteria':crit},{'id':'kill','instructions':'Look at the board image. Which move is most likely to kill the snake this turn (into a wall or into its own body)?','criteria':ex['kill']['criteria']},{'id':'plan','instructions':'Look at the board image. Which two-move sequence is the best next two moves: both moves must be safe, and the snake should end closer to the food (or eat it)?','criteria':ex['plan']['criteria']}]
    try:
     r=s.post(arm['url']+'/score_image',json=dict(state=state_text(g),image_png_b64=render_png(g),questions=qs),timeout=180);r.raise_for_status();ans=r.json()['answers'];a=ans['move'];choice,probs=a['choice'],a['probabilities']
    except Exception as e:log.append(dict(game=gi,error=repr(e)[:200]));break
    sec=time.perf_counter()-t;lat.append(sec)
    ticks.append(dict(step=g['steps'],snake=list(g['snake']),food=g['food'],choice=choice,probs=probs,seconds=sec,kill=dict(choice=ans['kill']['choice'],correct=ex['kill']['correct'],ok=ans['kill']['choice'] in ex['kill']['correct']),plan=dict(choice=ans['plan']['choice'],correct=ex['plan']['correct'],ok=ans['plan']['choice'] in ex['plan']['correct'])));step(g,choice)
   if g['alive']:g['death']='timeout' if g['steps']>=max_steps else 'starved'
   summary.append(dict(game=gi,eaten=g['eaten'],steps=g['steps'],death=g['death'],median_ms=statistics.median(lat)*1000 if lat else None,cost=0,kill_acc=sum(t['kill']['ok'] for t in ticks)/max(1,len(ticks)),plan_acc=sum(t['plan']['ok'] for t in ticks)/max(1,len(ticks)),ticks_n=len(ticks)));log.append(dict(game=gi,ticks=ticks,final=dict(snake=list(g['snake']),food=g['food'],death=g['death'])))
   print('GAME',arm['name'],gi,g['eaten'],g['steps'],g['death'],flush=True)
  (out/(arm['name']+'_games.json')).write_text(json.dumps(log));(out/(arm['name']+'_summary.json')).write_text(json.dumps(summary));vol.commit();return summary
 with concurrent.futures.ThreadPoolExecutor(len(arms)) as pool:res={a['name']:f for a,f in zip(arms,pool.map(run_arm,arms))}
 (out/'status.json').write_text(json.dumps({'status':'complete'}));vol.commit();return res
if __name__=='__main__':
 arms=json.loads(sys.argv[1]);games=int(sys.argv[2]) if len(sys.argv)>2 else 10;grid=int(sys.argv[3]) if len(sys.argv)>3 else 10;max_steps=int(sys.argv[4]) if len(sys.argv)>4 else 200
 here=Path(__file__).resolve().parent;out=here/('snakevis3q_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir();(out/'snake_vision.py').write_text(Path(__file__).read_text())
 with app.run(detach=True):
  call=play.spawn(arms,games,grid,max_steps,out.name);(out/'job.json').write_text(json.dumps(dict(app_id=app.app_id,call_id=call.object_id,arms=arms,games=games,grid=grid,max_steps=max_steps),indent=2));print(out.name,app.app_id)
