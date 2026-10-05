"""Step 4 - the useful bug, and negative sampling.

Train on real pairs only and the model learns to say "yes" to every
pair. The fix: for each real pair, also show K made-up pairs and
teach the model to say "no" to those.
"""
import time

import numpy as np

from step2_pairs import load
from step3_model import DIM, make_tables, sigmoid

WINDOW = 5
NEGATIVES = 5        # made-up neighbours per real pair
CHUNK = 1024         # words handled per batch
LR = 0.025


def make_table(counts, power=0.75, size=10_000_000):
    """A long list of word rows. Picking a random entry picks a word
    with chance proportional to count ** power."""
    share = counts.astype(np.float64) ** power
    share /= share.sum()
    copies = np.round(share * size).astype(np.int64)
    return np.repeat(np.arange(len(counts)), copies).astype(np.int32)


def update(w_in, w_out, centre, context, negatives, lr):
    """step() from step 3, for a whole column of pairs at once."""
    targets = np.concatenate([context[:, None], negatives], axis=1)
    v = w_in[centre]                     # n x DIM
    u = w_out[targets]                   # n x (1 + k) x DIM
    belief = sigmoid(np.einsum('nd,nkd->nk', v, u))
    err = belief.copy()
    err[:, 0] -= 1                       # column 0 is the real pair
    push_v = np.einsum('nk,nkd->nd', err, u)
    push_u = err[:, :, None] * v[:, None, :]
    np.add.at(w_in, centre, -lr * push_v)
    np.add.at(w_out, targets, -lr * push_u)
    loss = -np.log(belief[:, 0]).sum() - np.log(1 - belief[:, 1:]).sum()
    return float(loss)


def train_chunk(w_in, w_out, tok, table, k, lr, rng, window=WINDOW):
    """Every pair inside one chunk of text, one distance at a time."""
    loss, count = 0.0, 0
    for off in range(1, window + 1):
        left, right = tok[:-off], tok[off:]
        for centre, context in ((left, right), (right, left)):
            pick = rng.integers(0, len(table), (len(centre), k))
            loss += update(w_in, w_out, centre, context, table[pick], lr)
            count += len(centre)
    return loss, count


def beliefs(w_in, w_out, a, b):
    """The model's belief for each pair (a[i], b[i])."""
    return sigmoid(np.einsum('nd,nd->n', w_in[a], w_out[b]))


def experiment(ids, table, k, n_words, seed=0):
    rng = np.random.default_rng(seed)
    w_in, w_out = make_tables(table.max() + 1, DIM, rng)
    for start in range(0, n_words, CHUNK):
        tok = ids[start:start + CHUNK]
        train_chunk(w_in, w_out, tok, table, k, LR, rng)
    return w_in, w_out


if __name__ == '__main__':
    vocab, counts, ids = load()
    row = {w: i for i, w in enumerate(vocab)}
    table = make_table(counts)

    print('how often a word is drawn as a fake neighbour')
    raw = counts / counts.sum()
    drawn = np.bincount(table, minlength=len(vocab)) / len(table)
    print('  word        share of text   share of draws')
    for w in ('the', 'king', 'queen', 'navarre'):
        i = row[w]
        print(f'  {w:10s}  {100 * raw[i]:10.4f} %  {100 * drawn[i]:12.4f} %')
    print()

    # a test the model has not trained on: 2,000 real pairs from far
    # into the text, and 2,000 pairs of words picked at random
    rng = np.random.default_rng(1)
    spot = rng.integers(10_000_000, 16_000_000, 2000)
    real_a, real_b = ids[spot], ids[spot + 1]
    fake_a = ids[rng.integers(0, len(ids), 2000)]
    fake_b = ids[rng.integers(0, len(ids), 2000)]

    n_words = 1_000_000
    for k, name in ((0, 'real pairs only'), (NEGATIVES, '5 fakes per pair')):
        t0 = time.time()
        w_in, w_out = experiment(ids, table, k, n_words)
        real = beliefs(w_in, w_out, real_a, real_b).mean()
        fake = beliefs(w_in, w_out, fake_a, fake_b).mean()
        print(f'{name}: first {n_words:,} words, '
              f'{time.time() - t0:.0f} s')
        print(f'  belief in 2,000 real pairs    {real:.3f}')
        print(f'  belief in 2,000 random pairs  {fake:.3f}')
        print(f'  belief in (king, queen)       '
            f'{beliefs(w_in, w_out, [row["king"]], [row["queen"]])[0]:.3f}')
        print(f'  belief in (king, banana)      '
            f'{beliefs(w_in, w_out, [row["king"]], [row["banana"]])[0]:.3f}')
