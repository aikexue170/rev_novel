"""JevBench (benchmarkheaven.com/jev-models) public items -> our /score request format, run against a server, score.

usage:
  run.py build                 # write jevbench_public_231.jsonl + README.md from the cloned harness repo
  run.py run  <arm> <url> [n]  # score items on the server, write <arm>.jsonl (n = optional item limit for smoke tests)
  run.py score <arm>           # write summary.json: accuracy / Intelligence / Calibration in the site's terms, plus every
                               # leaderboard system on the same 231 public items from the harness's per-task outcomes

Nothing here is written to data/ or eval_sets/, and nothing is trained on.
"""
import sys, json, time, hashlib, math, collections, concurrent.futures, requests
from pathlib import Path

here = Path(__file__).resolve().parent
REPO = Path('/private/tmp/claude-501/-Users-rob-Documents-ChatGPT-Jev/00778d75-7b56-4974-b1cd-e1de4841c85d/scratchpad/jevbench')
SOURCE_URL = 'https://benchmarkheaven.com/jev-models'
REPO_URL = 'https://github.com/fstandhartinger/jevbench'
ITEMS = here / 'jevbench_public_231.jsonl'
TIER_OF_FILE = {'easy': 'easy', 'original': 'standard', 'hard': 'hard'}
# Harness composite_v13 constants (copied so scoring does not depend on the clone being present).
TIER_WEIGHTS = {'easy': 0.14, 'standard': 0.28, 'judge': 0.28, 'hard': 0.30}
TIER_OPTION_COUNTS = {'easy': {2: 18, 4: 13, 5: 41}, 'standard': {2: 32, 4: 40, 5: 12, 6: 12},
                      'judge': {2: 68, 9: 78}, 'hard': {2: 77, 3: 26, 4: 73, 5: 38, 6: 6}}
SUM_TOL, RENORM_TOL = 1e-3, 2e-2


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def convert(r, src_file):
    """One harness record -> our request. Option keys/descriptions exactly as the benchmark gives them;
    key_map sends the server's option key back to the benchmark's label."""
    q = r['question']; t = q['type']; crit = q.get('criteria')
    if t == 'choice':
        criteria = dict(crit); key_map = {k: k for k in criteria}
        assert set(criteria) == set(r['labels']), r['id']
    elif t == 'noul':
        # benchmark: criteria keyed true/false, labels ["no","yes"]; harness maps Jev's p_yes -> {"yes","no"}
        criteria = dict(crit); key_map = {'true': 'yes', 'false': 'no'}
        assert set(criteria) == {'true', 'false'} and r['labels'] == ['no', 'yes'], r['id']
    elif t == 'score':
        # benchmark: criteria is the ordered list of levels, labels are the level indices as strings
        criteria = {str(i): desc for i, desc in enumerate(crit)}; key_map = {k: k for k in criteria}
        assert list(criteria) == r['labels'], r['id']
    else:
        raise ValueError(t)
    return dict(id=r['id'], tier=TIER_OF_FILE[src_file], family=r['family'], type=t, group=r.get('group'),
                state=r['state'], instructions=q['instructions'], criteria=criteria, key_map=key_map,
                labels=r['labels'], expected=str(r['expected']),
                gold_probs=r.get('provenance', {}).get('gold_probs'), source_file=f'datasets/public/{src_file}.jsonl')


def build():
    manifest = json.loads((REPO / 'datasets/manifest.json').read_text())
    msha = {s['name']: s['sha256'] for s in manifest['splits']}
    rows, srcs = [], []
    for f in ['easy', 'original', 'hard']:
        p = REPO / f'datasets/public/{f}.jsonl'
        h = sha256(p); assert h == msha[f], (f, h, msha[f])
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows += [convert(r, f) for r in recs]; srcs.append((f, len(recs), h))
    ITEMS.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
    n = len(rows); h = sha256(ITEMS)
    counts = collections.Counter((r['tier'], r['type']) for r in rows)
    fam = collections.Counter((r['tier'], r['family']) for r in rows)
    lines = [f'# JevBench public items in our request format', '',
             f'- Source: {SOURCE_URL} (JevBench v1.3.0 scoring, protocol jevbench::v1.2; harness and public tasks: {REPO_URL}, MIT)',
             f'- Items: {n} (all public items; the benchmark has 534, the other 303 are held out or imported and not redistributed)',
             f'- File: `jevbench_public_231.jsonl`, sha256 `{h}`', '',
             '| Upstream file | Items | Upstream sha256 (matches datasets/manifest.json) | Tier |', '|---|---:|---|---|']
    lines += [f'| datasets/public/{f}.jsonl | {k} | `{s}` | {TIER_OF_FILE[f]} |' for f, k, s in srcs]
    lines += ['', 'Public items by tier and question type: ' + ', '.join(f'{t}/{q}: {c}' for (t, q), c in sorted(counts.items())),
              '', 'Per-tier public families: ' + ', '.join(f'{t}/{f}: {c}' for (t, f), c in sorted(fam.items())), '',
              '## Mapping', '',
              '- `choice`: criteria dict sent as-is; keys are the benchmark labels.',
              '- `noul`: criteria keyed `true`/`false` sent as-is (as the benchmark gives them); the answer key is mapped `true->yes`, `false->no`, the labels the harness scores over (its TypeSafe adapter does the same with Jev\'s p_yes).',
              '- `score`: the benchmark\'s ordered level list becomes `{"0": level0, "1": level1, ...}`; our server accepts only dict criteria, and the level indices are exactly the benchmark\'s labels.',
              '- One question per request, as in the harness (each item has its own state).',
              '- `gold_probs` (10 public probability items) is kept for the Calibration axis\'s fidelity term.', '',
              'Rules: not used for training; not copied into data/ or eval_sets/.']
    (here / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {ITEMS} ({n} items, sha256 {h})')


def run(arm, url, limit=None):
    rows = [json.loads(l) for l in ITEMS.read_text().splitlines()]
    if limit: rows = rows[:limit]
    s = requests.Session()
    for _ in range(200):
        try:
            r = s.post(url + '/ping', json={}, timeout=1800)
            if r.status_code == 200: break
        except Exception: pass
        time.sleep(10)
    meta = r.json(); print('READY', arm, meta.get('model'), meta.get('checkpoint_run'), meta.get('training_examples'), flush=True)

    def one(x):
        err = None
        for attempt in range(4):
            try:
                t0 = time.perf_counter()
                resp = s.post(url + '/score', json={'state': x['state'], 'questions': [{'id': x['id'], 'instructions': x['instructions'], 'criteria': x['criteria']}]}, timeout=600)
                lat = time.perf_counter() - t0
                if resp.status_code != 200: raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:200]}')
                d = resp.json(); a = d['answers'][x['id']]
                probs = {x['key_map'][k]: float(v) for k, v in a['probabilities'].items()}
                return dict(id=x['id'], tier=x['tier'], family=x['family'], type=x['type'], expected=x['expected'], labels=x['labels'],
                            choice=x['key_map'][a['choice']], probabilities=probs, raw_choice=a['choice'],
                            server_seconds=d.get('server_seconds'), latency_s=lat, attempts=attempt + 1, gold_probs=x.get('gold_probs'))
            except Exception as e:
                err = repr(e)[:300]; time.sleep(3)
        return dict(id=x['id'], tier=x['tier'], family=x['family'], type=x['type'], expected=x['expected'], labels=x['labels'], choice=None, probabilities=None, error=err, gold_probs=x.get('gold_probs'))

    with concurrent.futures.ThreadPoolExecutor(8) as pool: res = list(pool.map(one, rows))
    out = here / f'{arm}.jsonl'
    out.write_text('\n'.join(json.dumps(r) for r in res) + '\n')
    (here / f'{arm}_server_metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'wrote {out}: {len(res)} rows, {sum(r["choice"] is None for r in res)} errors')


# ---- scoring, following jevbench/scoring.py, metrics.py and composite_v13.py ----
def validate(probs, labels, tol):
    if not isinstance(probs, dict) or set(probs) != set(labels): return None
    vals = {}
    for k, v in probs.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1: return None
        vals[str(k)] = float(v)
    return vals if abs(sum(vals.values()) - 1) <= tol else None


def score_row(r):
    if r.get('probabilities') is None:
        return dict(valid=False, strict_valid=False, correct=False, predicted=None, probs=None)
    p = validate(r['probabilities'], r['labels'], SUM_TOL); strict = p is not None
    if p is None:
        p = validate(r['probabilities'], r['labels'], RENORM_TOL)
        if p is None or sum(p.values()) <= 0: return dict(valid=False, strict_valid=False, correct=False, predicted=None, probs=None)
        tot = sum(p.values()); p = {k: v / tot for k, v in p.items()}
    pred = max(sorted(p), key=lambda k: p[k])  # ties -> lexicographically smallest label, as in the harness
    return dict(valid=True, strict_valid=strict, correct=pred == r['expected'], predicted=pred, probs=p)


def ece(pairs, n_bins=10):
    bins = [[0, 0.0, 0] for _ in range(n_bins)]
    for conf, ok in pairs:
        b = bins[min(int(min(max(conf, 0), 1) * n_bins), n_bins - 1)]; b[0] += 1; b[1] += conf; b[2] += ok
    n = sum(b[0] for b in bins)
    return sum(b[0] / n * abs(b[2] / b[0] - b[1] / b[0]) for b in bins if b[0]) if n else None


def chance_corrected(acc, chance):
    return max(0.0, min(100.0, 100 * (acc - chance) / (1 - chance)))


def tier_chance_published(tier):
    c = TIER_OPTION_COUNTS[tier]; n = sum(c.values()); return sum(k / o for o, k in c.items()) / n


def intelligence(tier_acc, tier_chance):
    s = w = 0.0
    for t, tw in TIER_WEIGHTS.items():
        if tier_acc.get(t) is not None:
            s += tw * chance_corrected(tier_acc[t], tier_chance[t]); w += tw
    return s / w if w else None


def summarize_ours(res):
    items = {x['id']: x for x in (json.loads(l) for l in ITEMS.read_text().splitlines())}
    by_tier = collections.defaultdict(lambda: dict(c=0, w=0, f=0, n=0, chance_sum=0.0))
    by_fam = collections.defaultdict(lambda: dict(c=0, n=0))
    by_type = collections.defaultdict(lambda: dict(c=0, n=0))
    hard_pairs, briers, tvds, per_item = [], [], [], []
    for r in res:
        sc = score_row(r); t = r['tier']; b = by_tier[t]
        b['n'] += 1; b['chance_sum'] += 1 / len(r['labels'])
        if not sc['valid']: b['f'] += 1
        elif sc['correct']: b['c'] += 1
        else: b['w'] += 1
        by_fam[(t, r['family'])]['n'] += 1; by_fam[(t, r['family'])]['c'] += sc['correct']
        by_type[r['type']]['n'] += 1; by_type[r['type']]['c'] += sc['correct']
        if t == 'hard' and sc['valid']:
            hard_pairs.append((max(sc['probs'].values()), sc['correct']))
            briers.append(sum((sc['probs'][k] - (k == r['expected'])) ** 2 for k in r['labels']))
            if r.get('gold_probs'): tvds.append(0.5 * sum(abs(sc['probs'].get(k, 0) - v) for k, v in r['gold_probs'].items()))
        per_item.append(dict(id=r['id'], tier=t, family=r['family'], type=r['type'], expected=r['expected'], predicted=sc['predicted'],
                             correct=sc['correct'], valid=sc['valid'], strict_valid=sc['strict_valid'], probabilities=sc['probs'],
                             top_confidence=max(sc['probs'].values()) if sc['probs'] else None, error=r.get('error'),
                             server_seconds=r.get('server_seconds'), latency_s=r.get('latency_s')))
    tiers = {t: dict(c=b['c'], w=b['w'], f=b['f'], n=b['n'], accuracy=b['c'] / b['n'], chance_public=b['chance_sum'] / b['n'],
                     chance_published_full_tier=tier_chance_published(t),
                     intelligence_public_chance=chance_corrected(b['c'] / b['n'], b['chance_sum'] / b['n']),
                     intelligence_published_chance=chance_corrected(b['c'] / b['n'], tier_chance_published(t))) for t, b in by_tier.items()}
    tier_acc = {t: v['accuracy'] for t, v in tiers.items()}
    e = ece(hard_pairs) if hard_pairs else None; mtvd = sum(tvds) / len(tvds) if tvds else None
    calib = None if e is None else ((100 * max(0.0, 1 - e / 0.5)) + (100 * (1 - mtvd) if mtvd is not None else 0)) / (2 if mtvd is not None else 1)
    n_all = len(res); c_all = sum(v['c'] for v in tiers.values())
    return dict(
        pooled=dict(correct=c_all, n=n_all, accuracy=c_all / n_all, errors=sum(r.get('choice') is None for r in res),
                    invalid_distributions=sum(v['f'] for v in tiers.values()), strict_invalid=sum(not p['strict_valid'] for p in per_item)),
        tiers=tiers,
        intelligence=dict(
            public_items_item_chance=intelligence(tier_acc, {t: v['chance_public'] for t, v in tiers.items()}),
            public_items_published_tier_chance=intelligence(tier_acc, {t: tier_chance_published(t) for t in tiers}),
            note='Tier weights hard .30 / standard .28 / easy .14 renormalised over the tiers we can run; the judge tier (146 items, weight .28) has no public items.'),
        calibration=dict(score=calib, ece_hard_public=e, n_hard_valid=len(hard_pairs), brier_hard_public=sum(briers) / len(briers) if briers else None,
                         mean_tvd_probability_items=mtvd, n_probability_items=len(tvds)),
        by_family={f'{t}/{f}': dict(correct=v['c'], n=v['n'], accuracy=v['c'] / v['n']) for (t, f), v in sorted(by_fam.items())},
        by_type={t: dict(correct=v['c'], n=v['n'], accuracy=v['c'] / v['n']) for t, v in sorted(by_type.items())},
        latency_note='Latency is not comparable: another agent shares this server, and the site measures serially from a Hetzner box; only answers are used.',
    ), per_item


def leaderboard_on_public(items):
    """Every leaderboard system's accuracy on exactly our 231 public items, from the harness per-task outcomes (c/w/f/n)."""
    pt = json.loads((REPO / 'results/v1.2/jevbench-v1.2-per-task.json').read_text())
    full = json.loads((REPO / 'results/v1.2/jevbench-v1.2-results.json').read_text())
    meta = {s['key']: s for s in full['systems']}
    out = {}
    for key, s in pt['systems'].items():
        tasks = s['public_tasks']; bt = collections.defaultdict(lambda: [0, 0, 0.0]); bf = collections.defaultdict(lambda: [0, 0])
        for tid, (code, lat) in tasks.items():
            it = items[tid]; b = bt[it['tier']]; b[1] += 1; b[0] += code == 'c'; b[2] += 1 / len(it['labels'])
            bf[(it['tier'], it['family'])][1] += 1; bf[(it['tier'], it['family'])][0] += code == 'c'
        tiers = {t: dict(correct=v[0], n=v[1], accuracy=v[0] / v[1]) for t, v in bt.items()}
        m = meta.get(key, {})
        out[key] = dict(display=s['display'], rank=m.get('rank'), listing=m.get('listing'),
                        site_jevbench_score=m.get('jevbench_score'), site_axes=m.get('axes'), site_tiers_all_534=m.get('tiers'),
                        public_231=dict(correct=sum(v[0] for v in bt.values()), n=sum(v[1] for v in bt.values()),
                                        accuracy=sum(v[0] for v in bt.values()) / sum(v[1] for v in bt.values()), tiers=tiers,
                                        intelligence_public_item_chance=intelligence({t: v['accuracy'] for t, v in tiers.items()}, {t: bt[t][2] / bt[t][1] for t in bt}),
                                        by_family={f'{t}/{f}': dict(correct=v[0], n=v[1], accuracy=v[0] / v[1]) for (t, f), v in sorted(bf.items())}))
    return out


def score(arm):
    res = [json.loads(l) for l in (here / f'{arm}.jsonl').read_text().splitlines()]
    items = {x['id']: x for x in (json.loads(l) for l in ITEMS.read_text().splitlines())}
    assert len(res) == len(items) == 231, (len(res), len(items))
    ours, per_item = summarize_ours(res)
    (here / f'{arm}.jsonl').write_text('\n'.join(json.dumps(p) for p in per_item) + '\n')  # add scored fields in place
    lb = leaderboard_on_public(items)
    summary = dict(benchmark='JevBench v1.3.0 (benchmarkheaven.com/jev-models), protocol jevbench::v1.2', source=SOURCE_URL, harness=REPO_URL,
                   items_file=str(ITEMS.name), items_sha256=sha256(ITEMS), n_items=231,
                   scope='All 231 public items (easy 48, standard 72, hard 111). The site scores 534; 303 are held out or imported and not redistributed, so the judge tier (146) and the held-out halves are absent here.',
                   arm=arm, server_metadata=json.loads((here / f'{arm}_server_metadata.json').read_text()),
                   ours=ours, leaderboard_on_same_231_public_items=lb)
    (here / 'summary.json').write_text(json.dumps(summary, indent=1))
    o = ours; print(f"{arm}: pooled {o['pooled']['correct']}/{o['pooled']['n']} = {o['pooled']['accuracy']*100:.1f}%  errors {o['pooled']['errors']} invalid {o['pooled']['invalid_distributions']}")
    for t, v in o['tiers'].items(): print(f"  {t:9s} {v['c']}/{v['n']} = {v['accuracy']*100:.1f}%  chance-corrected {v['intelligence_public_chance']:.1f} (published tier chance: {v['intelligence_published_chance']:.1f})")
    print(f"  Intelligence (public items, item chance) {o['intelligence']['public_items_item_chance']:.1f} | (published tier chance) {o['intelligence']['public_items_published_tier_chance']:.1f}")
    c = o['calibration']; print(f"  Calibration {c['score']:.1f}  ECE_hard {c['ece_hard_public']:.3f}  Brier_hard {c['brier_hard_public']:.3f}  mean TVD (10 prob items) {c['mean_tvd_probability_items']:.3f}")
    print('  by family:'); [print(f"    {k:28s} {v['correct']}/{v['n']} = {v['accuracy']*100:.0f}%") for k, v in o['by_family'].items()]
    print('  by type:', {k: f"{v['correct']}/{v['n']}" for k, v in o['by_type'].items()})
    print('\nLeaderboard systems on the same 231 public items (acc, easy/standard/hard, Intelligence public-chance):')
    rows = sorted(lb.values(), key=lambda s: -s['public_231']['accuracy'])
    for s in rows[:20]:
        p = s['public_231']; t = p['tiers']
        print(f"  {str(s['rank']):>4} {s['display'][:48]:48s} {p['accuracy']*100:5.1f}%  " + ' '.join(f"{t[k]['accuracy']*100:5.1f}" for k in ['easy', 'standard', 'hard'] if k in t) + f"  I={p['intelligence_public_item_chance']:.1f}  site score {s['site_jevbench_score']:.1f}" if s['site_jevbench_score'] is not None else '')
    print(f'wrote {here / "summary.json"}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build': build()
    elif cmd == 'run': run(sys.argv[2], sys.argv[3].rstrip('/'), int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif cmd == 'score': score(sys.argv[2])
    else: raise SystemExit(__doc__)
