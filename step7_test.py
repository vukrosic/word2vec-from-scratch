"""Step 7 - the test: 19,544 analogy questions.

Each line of questions-words.txt is "a b c d": a is to b as c is to d.
We give the model a, b and c and count how often it answers d.
Like the original test, we only use the 30,000 most common words.
"""
import os

import numpy as np

from step2_pairs import DATA
from step6_look import COMMON, load_vectors

np.seterr(all='ignore')   # Apple's maths library raises false alarms


def read_questions():
    """{section name: [(a, b, c, d), ...]}"""
    sections, name = {}, None
    with open(os.path.join(DATA, 'questions-words.txt')) as f:
        for line in f:
            parts = line.lower().split()
            if parts[0] == ':':
                name = parts[1]
                sections[name] = []
            else:
                sections[name].append(tuple(parts))
    return sections


def answer(unit, rows):
    """rows: n x 3 row numbers (a, b, c). Returns n answers."""
    a, b, c = rows[:, 0], rows[:, 1], rows[:, 2]
    target = unit[b] - unit[a] + unit[c]
    score = target @ unit[:COMMON].T
    lines = np.arange(len(rows))
    for col in (a, b, c):
        score[lines, col] = -np.inf      # never answer with a question word
    return score.argmax(axis=1)


def run_test(row, unit):
    """{section: (right, asked, skipped, wrong answers)}"""
    out = {}
    for name, questions in read_questions().items():
        known = [q for q in questions
                 if all(w in row and row[w] < COMMON for w in q)]
        rows = np.array([[row[w] for w in q] for q in known])
        got = answer(unit, rows)
        wrong = [(q, int(g)) for q, g in zip(known, got) if g != row[q[3]]]
        out[name] = (len(known) - len(wrong), len(known),
                     len(questions) - len(known), wrong)
    return out


if __name__ == '__main__':
    words, row, unit = load_vectors()
    result = run_test(row, unit)
    right = asked = skipped = 0
    print('section                      right  asked   score  skipped')
    for name, (r, n, s, _) in result.items():
        print(f'{name:28s} {r:5d}  {n:5d}  {100 * r / n:5.1f} %  {s:6d}')
        right, asked, skipped = right + r, asked + n, skipped + s
    print(f'{"total":28s} {right:5d}  {asked:5d}  '
          f'{100 * right / asked:5.1f} %  {skipped:6d}')
    print()
    print('some wrong answers')
    for name in ('capital-common-countries', 'family', 'gram2-opposite'):
        for (a, b, c, d), g in result[name][3][:4]:
            print(f'  {b} - {a} + {c}: said {words[g]}, wanted {d}')
