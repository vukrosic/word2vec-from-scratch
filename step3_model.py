"""Step 3 - the model: two tables, a dot product and a squash.

Question the model answers: "is word o a real neighbour of word c?"
Answer: sigmoid(w_in[c] . w_out[o]), a number between 0 and 1.
"""
import numpy as np

DIM = 100            # numbers per word


def make_tables(vocab_size, dim, rng):
    """w_in: tiny random numbers. w_out: all zeros."""
    w_in = (rng.random((vocab_size, dim)) - 0.5) / dim
    w_out = np.zeros((vocab_size, dim))
    return w_in.astype(np.float32), w_out.astype(np.float32)


def sigmoid(x):
    """Squash any number into the range 0..1."""
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def prob(w_in, w_out, c, o):
    """The model's belief that o is a real neighbour of c."""
    return float(sigmoid(w_in[c] @ w_out[o]))


def step(w_in, w_out, c, o, label, lr):
    """One pair, one nudge. label 1 = real pair, 0 = fake pair."""
    v = w_in[c].copy()
    u = w_out[o].copy()
    err = sigmoid(v @ u) - label     # how wrong, and which way
    w_in[c] -= lr * err * u          # move v along u
    w_out[o] -= lr * err * v         # move u along v
    return float(err)


if __name__ == '__main__':
    # three numbers per word so every value fits on screen
    w_in = np.array([[0.4, -0.2, 0.1]])      # row 0: the centre word
    w_out = np.array([[0.1, 0.3, -0.2],      # row 0: a real neighbour
                      [0.2, -0.1, 0.3]])     # row 1: a fake neighbour
    print('real pair, label 1: the score should go up')
    print(f'  start   belief {prob(w_in, w_out, 0, 0):.3f}')
    for n in range(1, 7):
        err = step(w_in, w_out, 0, 0, label=1, lr=1.0)
        print(f'  step {n}  error {err:+.3f}  '
              f'belief {prob(w_in, w_out, 0, 0):.3f}')
    print('fake pair, label 0: the score should go down')
    print(f'  start   belief {prob(w_in, w_out, 0, 1):.3f}')
    for n in range(1, 7):
        err = step(w_in, w_out, 0, 1, label=0, lr=1.0)
        print(f'  step {n}  error {err:+.3f}  '
              f'belief {prob(w_in, w_out, 0, 1):.3f}')
    print()
    rng = np.random.default_rng(0)
    big_in, big_out = make_tables(71290, DIM, rng)
    print('full size tables:', big_in.shape, 'and', big_out.shape)
    print(f'numbers to learn: {big_in.size + big_out.size:,}')
    print(f'belief for any pair before training: '
          f'{prob(big_in, big_out, 187, 1108):.3f}')
