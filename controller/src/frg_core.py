"""Core of the Failure-Aware Retrieval Governor study: fusion, prefix-constrained fusion, fault worlds.

Everything here is pure Python/numpy and works on ranked id lists, so the same code serves the frozen classical-Arabic
package (T1, replay of stored lists) and the recomputed public testbeds (T2, T3).

Terms
  L  lexical (BM25) list            D  dense list (+ cosine scores)
  F  fused head (RRF, kappa = 60)   R  reranker order of the served head
  beta  reservation share: every prefix of length i of the served list holds the trusted list's top ceil(beta * i)
"""
import math

KAPPA = 60


def rrf(lists, weights=None, depth=48):
    """Reciprocal rank fusion. Ties are broken by first appearance, list order L then D (stable, as in x1_steps.pools,
    where the dense list was passed first; see reproduce check in t1_data.check_reproduction)."""
    acc = {}
    order = {}
    w = weights or [1.0] * len(lists)
    for lst, wt in zip(lists, w):
        if wt == 0:
            continue
        for r, i in enumerate(lst):
            if i not in acc:
                order[i] = len(order)
                acc[i] = 0.0
            acc[i] += wt / (KAPPA + r + 1)
    return [i for i, _ in sorted(acc.items(), key=lambda x: (-x[1], order[x[0]]))][:depth]


def constrained(trusted, base, beta, depth=48):
    """Prefix-constrained merge. Walk positions 1..depth; at position i, if fewer than ceil(beta*i) items of `trusted`
    have been placed, place the best-ranked unplaced item of `trusted`; otherwise place the best unplaced item of `base`.
    Guarantee (Proposition 2): for every k <= depth, served[:k] contains trusted[:ceil(beta*k)] (when trusted has that
    many items), whatever `base` holds. beta = 0 returns base; beta = 1 returns trusted."""
    if beta <= 0:
        return list(base[:depth])
    out, placed = [], set()
    ti = bi = 0          # ti: first index of `trusted` not yet placed (all of trusted[:ti] are in the output)
    for i in range(1, depth + 1):
        while ti < len(trusted) and trusted[ti] in placed:
            ti += 1
        pick = None
        if ti < len(trusted) and ti < math.ceil(beta * i - 1e-9):
            pick = trusted[ti]
        if pick is None:
            while bi < len(base) and base[bi] in placed:
                bi += 1
            if bi < len(base):
                pick = base[bi]
            elif ti < len(trusted):
                pick = trusted[ti]
            else:
                break
        out.append(pick)
        placed.add(pick)
    return out


def m_reserved(beta, k):
    """Number of trusted-list items in the first k positions of constrained(trusted, base, beta) when base is the RRF of
    trusted with a dead list that shares nothing with it (so base alternates t1, d1, t2, d2, ...). Used for the
    closed-form dead-arm curve h_dead(beta, k) = h_trusted(m_reserved(beta, k)) (Proposition 3)."""
    trusted = [("t", j) for j in range(10 * k + 10)]
    dead = [("d", j) for j in range(10 * k + 10)]
    base = []
    for a, b in zip(trusted, dead):
        base += [a, b]
    srv = constrained(trusted, base, beta, depth=k)
    return sum(1 for x in srv[:k] if x[0] == "t")


def first_hit(lst, rel, depth=None):
    """1-based rank of the first gold item, or None."""
    for r, i in enumerate(lst if depth is None else lst[:depth], 1):
        if i in rel:
            return r
    return None


def overlap(a, b, k=48):
    return len(set(a[:k]) & set(b[:k]))
