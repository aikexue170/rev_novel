"""Merge load-test summaries from several runs into one directory for charts.py / the final report.

usage: merge.py <out_dir> <load_dir>[:arm1,arm2] [<load_dir>[:arms] ...]
Each source directory must already have summary.json (run report.py on it first). Optional ':arms' keeps only those arms.
"""
import sys,json,pathlib
out=pathlib.Path(sys.argv[1]);out.mkdir(exist_ok=True);rows=[];meta={};sources=[]
for spec in sys.argv[2:]:
 d,_,arms=spec.partition(':');arms=set(arms.split(',')) if arms else None;s=json.loads((pathlib.Path(d)/'summary.json').read_text())
 for r in s['rows']:
  if arms is None or r['arm'] in arms:rows.append(dict(r,source=pathlib.Path(d).name))
 for a,m in s['metadata'].items():
  if arms is None or a in arms:meta[a]=m
 sources.append(dict(dir=str(d),arms=sorted(arms) if arms else 'all'))
(out/'summary.json').write_text(json.dumps(dict(rows=rows,metadata=meta,sources=sources),indent=2));print('merged',len(rows),'rows,',len(meta),'arms ->',out/'summary.json')
