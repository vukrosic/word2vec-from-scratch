"""Step 1 - a word as a list of numbers.

Four words, two numbers each, written by hand. The first number says
how royal the word is, the second how female. With numbers like these,
subtracting and adding words is ordinary arithmetic.
"""
import numpy as np

WORDS = {
    'man':   np.array([0.1, 0.1]),
    'woman': np.array([0.1, 0.9]),
    'king':  np.array([0.9, 0.1]),
    'queen': np.array([0.9, 0.9]),
    'apple': np.array([0.0, 0.3]),
}


def cosine(a, b):
    """1 = same direction, 0 = unrelated, -1 = opposite."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def nearest(vec, words, skip=()):
    """The word whose arrow points most nearly the same way as vec."""
    best, best_score = None, -2.0
    for word, other in words.items():
        if word in skip:
            continue
        score = cosine(vec, other)
        if score > best_score:
            best, best_score = word, score
    return best, best_score


if __name__ == '__main__':
    w = WORDS
    print('word      royal  female')
    for word, vec in w.items():
        print(f'{word:8s}  {vec[0]:5.1f}  {vec[1]:6.1f}')
    print()
    print(f"cosine(king, queen) = {cosine(w['king'], w['queen']):.3f}")
    print(f"cosine(king, woman) = {cosine(w['king'], w['woman']):.3f}")
    print(f"cosine(king, apple) = {cosine(w['king'], w['apple']):.3f}")
    print()
    answer = w['king'] - w['man'] + w['woman']
    print('king - man + woman =', np.round(answer, 2))
    word, score = nearest(answer, w, skip=('king', 'man', 'woman'))
    print(f'nearest word: {word}  (cosine {score:.3f})')
    print()
    print('These ten numbers were typed in by hand.')
    print('The real table needs 71,290 words x 100 numbers = '
          f'{71290 * 100:,} numbers.')
