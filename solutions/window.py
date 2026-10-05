"""Your turn - what does the window size change?

Trains the same model with a window of 2 words instead of 5, then
puts the two side by side. Training takes about 6 minutes; if
vectors_window2.npy is already here it is skipped.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from step2_pairs import HERE                         # noqa: E402
from step5_train import EPOCHS, train                # noqa: E402
from step6_look import load_vectors, neighbours      # noqa: E402
from step7_test import run_test                      # noqa: E402

SMALL = os.path.join('solutions', 'vectors_window2.npy')

if __name__ == '__main__':
    if not os.path.exists(os.path.join(HERE, SMALL)):
        folder = os.path.join(HERE, 'ck_window2')
        train(folder, window=2)
        last = np.load(os.path.join(folder, f'pass{EPOCHS}.npy'))
        np.save(os.path.join(HERE, SMALL), last)

    words, row, wide = load_vectors()
    _, _, narrow = load_vectors(SMALL)
    print('nearest neighbours, window 5 against window 2')
    for w in ('paris', 'apple', 'war', 'running', 'einstein'):
        for name, unit in (('5', wide), ('2', narrow)):
            near = neighbours(words, row, unit, w)
            print(f'  {w:9s} window {name}: '
                  + ', '.join(x for x, _ in near))
    print()
    a, b = run_test(row, wide), run_test(row, narrow)
    print('section                      window 5  window 2')
    tot = [0, 0, 0]
    for name in a:
        print(f'{name:28s} {100 * a[name][0] / a[name][1]:6.1f} % '
              f'{100 * b[name][0] / b[name][1]:7.1f} %')
        tot = [tot[0] + a[name][0], tot[1] + b[name][0],
               tot[2] + a[name][1]]
    print(f'{"total":28s} {100 * tot[0] / tot[2]:6.1f} % '
          f'{100 * tot[1] / tot[2]:7.1f} %')
