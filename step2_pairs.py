"""Step 2 - read the text, pick a vocabulary, cut it into pairs.

The only training signal in this whole course: which words appear
near which other words.
"""
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
MIN_COUNT = 5        # words seen fewer times than this are dropped


def read_words():
    with open(os.path.join(DATA, 'text8')) as f:
        return f.read().split()


def build_vocab(words):
    """Every word seen MIN_COUNT times or more, most common first."""
    counts = Counter(words)
    vocab = [w for w, c in counts.most_common() if c >= MIN_COUNT]
    return vocab, np.array([counts[w] for w in vocab])


def to_ids(words, vocab):
    """Swap each word for its row number. Dropped words vanish."""
    row = {w: i for i, w in enumerate(vocab)}
    return np.array([row[w] for w in words if w in row], dtype=np.int32)


def load():
    """vocab, counts, ids. Built once (about 20 s), then cached."""
    cache = os.path.join(DATA, 'ids.npy')
    if not os.path.exists(cache):
        words = read_words()
        vocab, counts = build_vocab(words)
        np.save(cache, to_ids(words, vocab))
        np.save(os.path.join(DATA, 'counts.npy'), counts)
        with open(os.path.join(DATA, 'vocab.txt'), 'w') as f:
            f.write('\n'.join(vocab))
    with open(os.path.join(DATA, 'vocab.txt')) as f:
        vocab = f.read().split('\n')
    counts = np.load(os.path.join(DATA, 'counts.npy'))
    return vocab, counts, np.load(cache)


def pairs(ids, window):
    """(centre, neighbour) for every neighbour within `window` words.
    Lines the text up against a copy of itself shifted by 1, 2, ..."""
    out = []
    for off in range(1, window + 1):
        left, right = ids[:-off], ids[off:]
        for a, b in zip(left, right):
            out.append((int(a), int(b)))     # neighbour on the right
            out.append((int(b), int(a)))     # neighbour on the left
    return out


def company(ids, word_id, window, vocab_size):
    """How often each word appears within `window` of word_id."""
    where = np.flatnonzero(ids == word_id)
    seen = np.zeros(vocab_size, dtype=np.int64)
    for off in range(1, window + 1):
        for pos in (where - off, where + off):
            pos = pos[(pos >= 0) & (pos < len(ids))]
            seen += np.bincount(ids[pos], minlength=vocab_size)
    return seen


def find(ids, phrase_ids):
    """First position where the run of ids `phrase_ids` appears."""
    hit = np.flatnonzero(ids == phrase_ids[0])
    for k, wid in enumerate(phrase_ids[1:], start=1):
        hit = hit[hit + k < len(ids)]
        hit = hit[ids[hit + k] == wid]
    return int(hit[0])


if __name__ == '__main__':
    words = read_words()
    print(f'words in the text      {len(words):>11,}')
    print(f'different words        {len(set(words)):>11,}')
    vocab, counts, ids = load()
    row = {w: i for i, w in enumerate(vocab)}
    print(f'seen {MIN_COUNT} times or more    {len(vocab):>11,}')
    print(f'words left in the text {len(ids):>11,}')
    print()
    print('most common:', ', '.join(
        f'{vocab[i]} {counts[i]:,}' for i in range(6)))
    for w in ('king', 'queen', 'man', 'woman'):
        print(f'  {w:6s} row {row[w]:>5,}   seen {counts[row[w]]:>6,} times')
    print()

    phrase = 'of navarre became king henry iv of france'.split()
    start = find(ids, [row[w] for w in phrase])
    piece = ids[start:start + len(phrase)]
    print('a piece of the text, word', f'{start:,}:')
    print('  ', ' '.join(vocab[i] for i in piece))
    made = pairs(piece, window=2)
    print(f'window 2 gives {len(made)} pairs. Those with centre "king":')
    for c, o in made:
        if vocab[c] == 'king':
            print(f'   (king, {vocab[o]})')
    print()

    total = 2 * sum(len(ids) - off for off in range(1, 6))
    print(f'window 5 over the whole text: {total:,} pairs')
    print()
    print('the company each word keeps (window 5, without the 150')
    print('most common words):')
    for w in ('king', 'queen', 'apple'):
        seen = company(ids, row[w], 5, len(vocab))
        seen[:150] = 0
        seen[row[w]] = 0
        top = np.argsort(-seen)[:7]
        print(f'  {w:6s}', ', '.join(f'{vocab[i]} {seen[i]}' for i in top))
