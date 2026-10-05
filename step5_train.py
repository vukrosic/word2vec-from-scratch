"""Step 5 - train on the whole text.

Three additions to step 4: frequent words are thinned out before each
pass, the step size shrinks to zero as training goes on, and several
copies of the loop run at once on the same two tables.
A full run is 10 passes. The tables live in files, so a stopped run
picks up where it left off.
"""
import os
import sys
import time
from multiprocessing import Process, Queue

import numpy as np

from step2_pairs import HERE, load
from step3_model import DIM, make_tables, step
from step4_negative import (CHUNK, LR, NEGATIVES, WINDOW, beliefs,
                            make_table, train_chunk)

EPOCHS = 10          # passes over the text
THRESHOLD = 1e-4     # words with a larger share of the text are thinned
WORKERS = 8          # copies of the loop running at once


def keep_chance(counts, t=THRESHOLD):
    """Chance of keeping each copy of a word: sqrt(t / share)."""
    share = counts / counts.sum()
    return np.minimum(1.0, np.sqrt(t / share))


def thin(ids, keep, rng):
    """Drop copies of frequent words at random."""
    return ids[rng.random(len(ids)) < keep[ids]]


def open_tables(folder, vocab_size):
    """Both tables as files on disk that every worker can write to."""
    shape = (vocab_size, DIM)
    return [np.memmap(os.path.join(folder, name), np.float32, 'r+',
                      shape=shape) for name in ('w_in.dat', 'w_out.dat')]


def work(folder, part, epoch, epochs, window, results):
    """One worker: one slice of the text, one pass."""
    vocab, counts, ids = load()
    w_in, w_out = open_tables(folder, len(vocab))
    table = make_table(counts)
    rng = np.random.default_rng([epoch, part])
    lo = len(ids) * part // WORKERS
    hi = len(ids) * (part + 1) // WORKERS
    text = thin(ids[lo:hi], keep_chance(counts), rng)
    loss, count = 0.0, 0
    for start in range(0, len(text), CHUNK):
        done = (epoch + start / len(text)) / epochs
        lr = LR * max(1e-4, 1 - done)
        got, pairs = train_chunk(w_in, w_out, text[start:start + CHUNK],
                                 table, NEGATIVES, lr, rng, window)
        loss, count = loss + got, count + pairs
    results.put((loss, count))


def neighbours(vectors, i, n=3):
    """Rows whose arrows point most nearly the same way as row i."""
    vectors = np.asarray(vectors, dtype=float)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    score = (unit * unit[i]).sum(axis=1)
    score[i] = -2
    return np.argsort(-score)[:n]


def report(w_in, w_out, vocab, check):
    real = beliefs(w_in, w_out, check[0], check[1]).mean()
    fake = beliefs(w_in, w_out, check[2], check[3]).mean()
    near = neighbours(w_in, vocab.index('king'))
    return (f'real {real:.3f}  random {fake:.3f}  king -> '
            + ', '.join(vocab[i] for i in near))


def train(folder, epochs=EPOCHS, window=WINDOW, stop_after=None):
    vocab, counts, ids = load()
    os.makedirs(folder, exist_ok=True)
    done_path = os.path.join(folder, 'done.txt')
    if not os.path.exists(done_path):
        first = make_tables(len(vocab), DIM, np.random.default_rng(0))
        first[0].tofile(os.path.join(folder, 'w_in.dat'))
        first[1].tofile(os.path.join(folder, 'w_out.dat'))
        with open(done_path, 'w') as f:
            f.write('0')
    w_in, w_out = open_tables(folder, len(vocab))
    start = int(open(done_path).read())

    rng = np.random.default_rng(1)
    spot = rng.integers(10_000_000, 16_000_000, 2000)
    check = (ids[spot], ids[spot + 1],
             ids[rng.integers(0, len(ids), 2000)],
             ids[rng.integers(0, len(ids), 2000)])

    for epoch in range(start, epochs):
        if stop_after is not None and epoch - start >= stop_after:
            break
        t0 = time.time()
        results = Queue()
        crew = [Process(target=work, args=(folder, part, epoch, epochs,
                                           window, results))
                for part in range(WORKERS)]
        for p in crew:
            p.start()
        got = [results.get() for p in crew]
        for p in crew:
            p.join()
        loss = sum(g[0] for g in got) / sum(g[1] for g in got)
        line = (f'pass {epoch + 1:2d}  loss {loss:.3f}  '
                f'{time.time() - t0:3.0f} s  '
                + report(w_in, w_out, vocab, check))
        print(line, flush=True)
        with open(os.path.join(folder, 'log.txt'), 'a') as f:
            f.write(line + '\n')
        np.save(os.path.join(folder, f'pass{epoch + 1}.npy'),
                np.asarray(w_in).astype(np.float16))
        with open(done_path, 'w') as f:
            f.write(str(epoch + 1))


def slow_speed(ids, n=20_000):
    """Nudges per second when step() handles one pair at a time."""
    rng = np.random.default_rng(0)
    w_in, w_out = make_tables(int(ids.max()) + 1, DIM, rng)
    t0 = time.time()
    for i in range(n):
        step(w_in, w_out, ids[i], ids[i + 1], 1, LR)
    return n / (time.time() - t0)


if __name__ == '__main__':
    vocab, counts, ids = load()
    row = {w: i for i, w in enumerate(vocab)}
    keep = keep_chance(counts)
    print('chance of keeping each copy of a word')
    for w in ('the', 'of', 'king', 'queen'):
        print(f'  {w:6s} {100 * keep[row[w]]:6.1f} %')
    kept = int((counts * keep).sum())
    print(f'words per pass: {len(ids):,} -> about {kept:,}')
    nudges = kept * 2 * WINDOW * (1 + NEGATIVES)
    print(f'nudges per pass: about {nudges:,}')
    speed = slow_speed(ids)
    print(f'one pair at a time: {speed:,.0f} nudges a second, '
          f'{nudges * EPOCHS / speed / 3600:.1f} hours for '
          f'{EPOCHS} passes')
    print()

    folder = os.path.join(HERE, 'ck')
    if '--show' in sys.argv:       # print the saved log of our own run
        print(open(os.path.join(HERE, 'train-log.txt')).read())
        sys.exit()
    stop = int(sys.argv[1]) if len(sys.argv) > 1 else None
    train(folder, stop_after=stop)
    last = os.path.join(folder, f'pass{EPOCHS}.npy')
    if os.path.exists(last):
        np.save(os.path.join(HERE, 'vectors.npy'), np.load(last))
        with open(os.path.join(HERE, 'words.txt'), 'w') as f:
            f.write('\n'.join(vocab))
        print('saved vectors.npy and words.txt')
