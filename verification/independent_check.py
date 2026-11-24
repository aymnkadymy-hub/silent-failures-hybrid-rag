"""Independent re-derivation (own code) of the I01 numbers from the per-question records."""
import json, sys, collections
import numpy as np
from scipy.stats import binomtest
X1 = sys.argv[1]; X2 = sys.argv[2]
rows = json.load(open(f'{X1}/per_question.json'))
by = collections.defaultdict(dict)   # (cond,pool) -> qid -> row
for r in rows:
    by[(r['cond'], r['pool'])][r['qid']] = r
def bank_ids(name):
    ex = next(iter(by[('on','bake')].values()))
    ids = []
    for q, r in by[('on','bake')].items():
        if name == 'n5dev' and r['set']=='n5' and r['split']=='dev': ids.append(q)
        if name == 'n5held' and r['set']=='n5' and r['split']=='heldout': ids.append(q)
        if name == 'books' and r['set']=='books_v1': ids.append(q)
        if name == 'rgold' and r['set'] in ('rgold',) : ids.append(q)
    return sorted(ids)
sets = collections.Counter((r['set'], r['split']) for r in by[('on','bake')].values())
print('banks:', dict(sets))
hit = lambda rank, k: isinstance(rank, int) and rank <= k
def recall(cond, pool, ids, field, k):
    d = by[(cond, pool)]
    return np.mean([hit(d[q][field], k) for q in ids])
OUT = {}
for bank in ['n5dev', 'n5held', 'books']:
    ids = bank_ids(bank); print(f'\n== {bank} n={len(ids)}')
    for c in ['on', 'minilm', 'unk', 'rand1', 'off']:
        vals = {}
        if c != 'off': vals['dense_R10'] = recall(c, 'bake', ids, 'dense_gold_rank', 10)
        vals['pool_R20'] = recall(c, 'bake', ids, 'pool_gold_rank', 20)
        vals['pool_R48'] = recall(c, 'bake', ids, 'pool_gold_rank', 48)
        rr = [by[(c,'bake')][q]['rerank_gold_rank'] for q in ids]
        if any(isinstance(x, int) for x in rr):
            vals['rerank_R1'] = recall(c, 'bake', ids, 'rerank_gold_rank', 1)
            vals['rerank_R8'] = recall(c, 'bake', ids, 'rerank_gold_rank', 8)
        OUT[(bank, c)] = vals
        print(f'  {c:7s}', ' '.join(f'{k}={100*v:.1f}' for k, v in vals.items()))
    # halving: fused hit@k under dead arm vs BM25 hit@k/2, bake pool
    for k in (20, 48):
        b_half = [hit(by[('on','bake')][q]['bm25_gold_rank'], k//2) for q in ids]
        b_full = np.mean([hit(by[('on','bake')][q]['bm25_gold_rank'], k) for q in ids])
        agr = []
        for c in ['minilm', 'unk', 'rand1']:
            f = [hit(by[(c,'bake')][q]['pool_gold_rank'], k) for q in ids]
            agr.append((c, 100*np.mean(f), 100*np.mean(np.array(f)==np.array(b_half))))
        print(f'  halving k={k}: BM25 R@{k}={100*b_full:.1f} R@{k//2}={100*np.mean(b_half):.1f} ', ' '.join(f'{c}:{a:.1f}/{g:.1f}%' for c,a,g in agr))
    # dead vs off, pool R20 (bake) exact McNemar
    for c in ['minilm', 'unk', 'rand1']:
        d = [hit(by[(c,'bake')][q]['pool_gold_rank'], 20) for q in ids]
        o = [hit(by[('off','bake')][q]['pool_gold_rank'], 20) for q in ids]
        b = sum(1 for x, y in zip(d, o) if x and not y); cc = sum(1 for x, y in zip(d, o) if y and not x)
        p = binomtest(min(b, cc), b + cc, 0.5).pvalue if b + cc else 1.0
        print(f'  dead-vs-off {c}: dead {sum(d)} off {sum(o)} diff {100*(sum(d)-sum(o))/len(ids):+.1f} pts, discordant {b}/{cc}, p={p:.2g}')
# similarity floor (prod pool): share of questions whose prod head equals BM25-only head is in x1_probes; here: rand1 prod == off prod?
ids = bank_ids('n5dev')
same = np.mean([by[('rand1','prod')][q]['pool_gold_rank'] == by[('off','prod')][q]['pool_gold_rank'] for q in ids])
print('\nrand1 prod pool gold rank identical to BM25-only for', f'{100*same:.1f}% of N5 dev questions')
for c in ['minilm', 'unk', 'rand1', 'off']:
    print(f'  prod pool R@20 {c}: {100*recall(c, "prod", ids, "pool_gold_rank", 20):.1f}')
# ---- X2
q2 = json.load(open(f'{X2}/x2_per_question.json')); n = len(q2); print(f'\n== X2 Quran QA n={n}')
for c in ['on', 'minilm', 'unk', 'rand1', 'rand2', 'rand3', 'off']:
    s = []
    if c != 'off': s.append(f"dense R10 {100*np.mean([hit(r[c+'|dense_rank'],10) for r in q2]):.1f}")
    for N in (48, 200):
        for k in (10, 20):
            s.append(f"N{N} R{k} {100*np.mean([hit(r[f'{c}|N{N}|fused_rank'],k) for r in q2]):.1f}")
    print(f'  {c:7s}', ' | '.join(s))
for c in ['unk', 'rand1', 'rand2', 'rand3', 'minilm']:
    s = []
    for N in (48, 200):
        for k in (10, 20):
            agr = np.mean([hit(r[f'{c}|N{N}|fused_rank'], k) == hit(r['bm25_rank'], k//2) for r in q2])
            s.append(f'N{N} k{k} {100*agr:.1f}%')
    print(f'  halving {c:7s}', ' | '.join(s))
b5, b10, b20 = (np.mean([hit(r['bm25_rank'], k) for r in q2]) for k in (5, 10, 20))
on20 = np.mean([hit(r['on|N48|fused_rank'], 20) for r in q2])
for c in ['unk', 'rand1', 'minilm']:
    d20 = np.mean([hit(r[f'{c}|N48|fused_rank'], 20) for r in q2])
    print(f'  gap R20 N48 {c}: observed {100*(on20-d20):.1f}; predicted net gain {100*(on20-b20):.1f} + halving cost {100*(b20-b10):.1f} = {100*(on20-b20+b20-b10):.1f}')
print(f'  BM25 R5 {100*b5:.1f} R10 {100*b10:.1f} R20 {100*b20:.1f}')
for c in ['on', 'minilm', 'unk', 'rand1']:
    print(f"  overlap48 {c}: mean {np.mean([r[c+'|overlap48'] for r in q2]):.2f}")
