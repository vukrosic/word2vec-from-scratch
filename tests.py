"""Quick checks for every step. Run: python3 tests.py"""
import os

import numpy as np

from step1_vectors import WORDS, cosine, nearest
from step2_pairs import HERE, pairs
from step3_model import prob, sigmoid, step
from step4_negative import make_table, update
from step5_train import keep_chance, thin


def test_step1():
    assert abs(cosine(WORDS['king'], WORDS['king']) - 1) < 1e-12
    answer = WORDS['king'] - WORDS['man'] + WORDS['woman']
    assert nearest(answer, WORDS, ('king', 'man', 'woman'))[0] == 'queen'


def test_step2():
    made = pairs([7, 8, 9, 10], window=1)
    assert made == [(7, 8), (8, 7), (8, 9), (9, 8), (9, 10), (10, 9)]
    assert len(pairs(list(range(8)), window=2)) == 26


def test_step3():
    assert sigmoid(0) == 0.5
    w_in = np.array([[0.4, -0.2, 0.1]])
    w_out = np.array([[0.1, 0.3, -0.2]])
    before = prob(w_in, w_out, 0, 0)
    step(w_in, w_out, 0, 0, label=1, lr=1.0)
    assert prob(w_in, w_out, 0, 0) > before
    step(w_in, w_out, 0, 0, label=0, lr=1.0)
    step(w_in, w_out, 0, 0, label=0, lr=1.0)
    assert prob(w_in, w_out, 0, 0) < before


def test_step4():
    counts = np.array([1000, 100, 10])
    table = make_table(counts, size=100_000)
    share = np.bincount(table) / len(table)
    assert share[0] < 1000 / 1110          # common word drawn less
    assert share[2] > 10 / 1110            # rare word drawn more
    # the batch update gives the same result as step() pair by pair
    rng = np.random.default_rng(0)
    w_in = rng.normal(size=(4, 3))
    w_out = rng.normal(size=(4, 3))
    a_in, a_out = w_in.copy(), w_out.copy()
    update(a_in, a_out, np.array([0]), np.array([1]),
           np.array([[2, 3]]), lr=0.1)
    b_in, b_out = w_in.copy(), w_out.copy()
    v = b_in[0].copy()
    push = np.zeros(3)
    for o, label in ((1, 1), (2, 0), (3, 0)):
        err = sigmoid(v @ b_out[o]) - label
        push += err * b_out[o]
        b_out[o] -= 0.1 * err * v
    b_in[0] -= 0.1 * push
    assert np.allclose(a_in, b_in) and np.allclose(a_out, b_out)


def test_step5():
    counts = np.array([900_000, 90_000, 10])
    keep = keep_chance(counts)
    assert keep[0] < keep[1] < keep[2] == 1.0
    ids = np.zeros(10_000, dtype=np.int32)
    kept = thin(ids, keep, np.random.default_rng(0))
    assert len(kept) < 300


def test_trained_vectors():
    if not os.path.exists(os.path.join(HERE, 'vectors.npy')):
        print('  (vectors.npy missing: skipped)')
        return
    from step6_look import analogy, load_vectors, neighbours
    words, row, unit = load_vectors()
    assert analogy(words, row, unit, 'man', 'king', 'woman')[0][0] == 'queen'
    assert analogy(words, row, unit,
                   'france', 'paris', 'germany')[0][0] == 'berlin'
    assert neighbours(words, row, unit, 'six')[0][1] > 0.9


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok ', name)
    print('all tests passed')
