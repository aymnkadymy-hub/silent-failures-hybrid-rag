"""Independent recomputation of the ArguAna damage table (Table 10 KIJOMS / Table 9 short) from the per-stream rows files,
without values_frg. Damage = fault queries (per 1,000) that the healthy pipeline answered with a hit and the served
list missed (src/t1_run.py:180), averaged over streams per scenario, then a simple mean over the 14 single-fault scenarios."""
import glob, json, os, random, statistics as st
R = os.path.join(os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "controller/results")), "v2/tb_arguana")
P = ['P0_rrf', 'CB', 'CIFC_r20', 'CIFC_Q']
rows, agg = {}, {}
for fn in glob.glob(f'{R}/test_pool_*_rows.json'):
    for sc, pv in json.load(open(fn)).items():
        rows[sc] = pv
    for sc, pv in json.load(open(fn.replace('_rows', '')))['agg'].items():
        agg[sc] = pv
F14 = sorted(s for s in rows if s.startswith('S3_') and not s.startswith('S3dur'))
print('fault scenarios:', len(F14), F14)
SHOWN = {'S3_D_adv', 'S3_D_unk', 'S3_D_minilm', 'S3_D_noise', 'S3_D_offset', 'S3_L_shard', 'S3_Q_strip', 'S3_S_shift'}
print('in the 14 but not shown as table rows:', [s for s in F14 if s not in SHOWN])
mean = lambda xs: sum(xs) / len(xs)
D = {s: {p: mean([r['damage'] for r in rows[s][p]]) for p in P} for s in F14}
bad = [(s, p) for s in F14 for p in P if abs(D[s][p] - agg[s][p]['damage']) > 1e-6]
print('rows vs stored agg mismatches:', bad)
print(f"{'scenario':22}" + ''.join(f'{p:>10}' for p in P) + '   streams')
for s in F14:
    print(f'{s:22}' + ''.join(f'{D[s][p]:10.1f}' for p in P) + f'   {len(rows[s]["CB"])}')
M = {p: mean([D[s][p] for s in F14]) for p in P}
print('mean of 14:', {p: round(M[p], 1) for p in P}, '(paper: RRF 233, CB 189, CIFC 160, CIFC-Q 143)')
ex = [s for s in F14 if s != 'S3_Q_strip']
print('without normalizer:', {p: round(mean([D[s][p] for s in ex]), 1) for p in P})
gap = M['CB'] - M['CIFC_Q']; norm = (D['S3_Q_strip']['CB'] - D['S3_Q_strip']['CIFC_Q']) / 14
print(f'CB - CIFC-Q gap {gap:.1f}; normalizer contributes {norm:.1f} ({100 * norm / gap:.0f}% of the gap)')
print('per fault, CIFC-Q better / tie / worse than CB:',
      sum(D[s]['CIFC_Q'] < D[s]['CB'] - .5 for s in F14), sum(abs(D[s]['CIFC_Q'] - D[s]['CB']) <= .5 for s in F14),
      sum(D[s]['CIFC_Q'] > D[s]['CB'] + .5 for s in F14))
print('leave-one-fault-out gap (CB - CIFC-Q):')
for s in F14:
    o = [t for t in F14 if t != s]
    print(f'  drop {s:22} {mean([D[t]["CB"] for t in o]) - mean([D[t]["CIFC_Q"] for t in o]):7.1f}')
# stream bootstrap of the gap (streams resampled within scenario; queries are reused across streams, so this measures
# replay variability only, not sampling of new questions)
random.seed(1)
bs = []
for _ in range(5000):
    tot = 0
    for s in F14:
        n = len(rows[s]['CB']); idx = [random.randrange(n) for _ in range(n)]
        tot += mean([rows[s]['CB'][i]['damage'] - rows[s]['CIFC_Q'][i]['damage'] for i in idx])
    bs.append(tot / 14)
bs.sort(); print(f'stream-bootstrap 95% interval of the gap: {bs[125]:.1f} to {bs[4874]:.1f}')
