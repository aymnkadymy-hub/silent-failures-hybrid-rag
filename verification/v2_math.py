"""Independent checks of Proposition 1 (halving rule) and of the protected-merge guarantee, on random synthetic lists.
RRF is written from scratch here; the merge is the project's frg_core.constrained (the code under test)."""
import os
import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "controller/src"))
from frg_core import constrained

def rrf(lists, kappa=60, depth=None):
    s = {}
    for L in lists:
        for r, d in enumerate(L, 1):
            s[d] = s.get(d, 0) + 1 / (kappa + r)
    # ties: lexical list first (stable order of first appearance, lexical given first) as stated in the paper
    order = []
    for L in zip(*lists) if all(len(L) == len(lists[0]) for L in lists) else []:
        for d in L:
            if d not in order: order.append(d)
    rest = [d for L in lists for d in L if d not in order]
    order += [d for i, d in enumerate(rest) if d not in rest[:i]]
    out = sorted(order, key=lambda d: -s[d])  # Python sort is stable, so ties keep the lexical-first order
    return out[:depth] if depth else out

random.seed(7); viol = 0; n = 0
for _ in range(20000):
    N = random.choice([48, 48, 20]); k = random.choice([8, 10, 20, 48]); k = min(k, N)
    lex = random.sample(range(10**6), N)
    dense = random.sample(range(10**6, 2 * 10**6), N)          # dead arm: shares nothing with the lexical list
    fused = rrf([lex, dense], depth=k)
    n += 1; viol += set(fused) & set(lex) != set(lex[:math.ceil(k / 2)])
print(f'Prop 1, disjoint dead arm, equal lengths: {viol} violations in {n} trials (lexical top ceil(k/2) kept)')
# overlap breaks the rule: count how often the lexical share exceeds k/2 when the dense list shares items
over = 0
for _ in range(5000):
    lex = random.sample(range(200), 48); dense = random.sample(range(200), 48)
    fused = rrf([lex, dense], depth=20)
    over += len(set(fused) & set(lex)) > 10
print(f'with overlapping lists the lexical share exceeds k/2 in {over}/5000 trials (expected: assumption needed)')
# protected merge: every prefix i holds trusted[:ceil(beta*i)], whatever the base list
bad = 0; t = 0
for _ in range(20000):
    beta = random.choice([0.5, 0.625, 0.75, 0.875, 1.0])
    trusted = random.sample(range(300), 48); base = random.sample(range(300), 48)
    s = constrained(trusted, base, beta, depth=48); t += 1
    bad += any(not set(trusted[:math.ceil(beta * i - 1e-9)]) <= set(s[:i]) for i in range(1, 49))
print(f'protected merge guarantee: {bad} violations in {t} trials (beta in 0.5..1, depth 48, adversarial random base)')
