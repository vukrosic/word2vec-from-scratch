"""Step 6 - look inside the trained table.

Nearest neighbours, word arithmetic, and a flat map of 100 numbers.
"""
import os

import numpy as np

from step2_pairs import HERE

COMMON = 30_000      # search only the most common words


def load_vectors(name='vectors.npy'):
    """words, row numbers, and every vector scaled to length 1."""
    with open(os.path.join(HERE, 'words.txt')) as f:
        words = f.read().split('\n')
    vectors = np.load(os.path.join(HERE, name)).astype(float)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return words, {w: i for i, w in enumerate(words)}, unit


def closest(unit, target, skip, n):
    """The n common words pointing most nearly the way target does."""
    score = (unit[:COMMON] * target).sum(axis=1)
    score /= np.linalg.norm(target)
    for i in skip:
        if i < COMMON:
            score[i] = -2
    best = np.argsort(-score)[:n]
    return [(int(i), float(score[i])) for i in best]


def neighbours(words, row, unit, word, n=5):
    i = row[word]
    return [(words[j], s) for j, s in closest(unit, unit[i], [i], n)]


def analogy(words, row, unit, a, b, c, n=3):
    """a is to b as c is to ...?   Computed as b - a + c."""
    ia, ib, ic = row[a], row[b], row[c]
    target = unit[ib] - unit[ia] + unit[ic]
    return [(words[j], s)
            for j, s in closest(unit, target, [ia, ib, ic], n)]


def flat_map(points):
    """Squash 100 numbers per word down to the 2 that vary most."""
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    return centred @ axes[:2].T


if __name__ == '__main__':
    words, row, unit = load_vectors()
    print('nearest neighbours')
    for w in ('king', 'paris', 'apple', 'car', 'six', 'she', 'hot'):
        near = neighbours(words, row, unit, w)
        print(f'  {w:6s}', ', '.join(f'{x} {s:.2f}' for x, s in near))
    print()
    print('word arithmetic: b - a + c')
    for a, b, c in (('man', 'king', 'woman'),
                    ('france', 'paris', 'germany'),
                    ('france', 'paris', 'japan'),
                    ('man', 'actor', 'woman'),
                    ('he', 'his', 'she'),
                    ('big', 'bigger', 'small'),
                    ('good', 'better', 'bad'),
                    ('france', 'paris', 'italy')):
        got = analogy(words, row, unit, a, b, c)
        print(f'  {b} - {a} + {c}'.ljust(28),
              ', '.join(f'{x} {s:.2f}' for x, s in got))
    print()
    places = ['france', 'paris', 'germany', 'berlin', 'japan', 'tokyo',
              'russia', 'moscow', 'spain', 'madrid', 'italy', 'rome',
              'greece', 'athens', 'china', 'beijing']
    flat = flat_map(unit[[row[w] for w in places]])
    print('flat map (2 numbers per word instead of 100)')
    for w, (x, y) in zip(places, flat):
        print(f'  {w:8s} {x:+.3f} {y:+.3f}')
