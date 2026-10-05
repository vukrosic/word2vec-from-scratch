# word2vec from scratch

Skip-gram with negative sampling in plain Python and numpy: 821 lines in 10 files.
It trains on 17 million words of Wikipedia in about 15 minutes on a laptop and
answers `king - man + woman` with `queen`.

This is the code from the video **How AI Learned That King − Man + Woman = Queen**.
Every number in the video came from running these files.

## Run it

```bash
pip install numpy
./run_all.sh
```

`run_all.sh` downloads the data (31 MB), runs every step in video order and takes
about one minute after the download. It uses the trained vectors that ship in this repo
(`vectors.npy`, `words.txt`), so nothing is trained. The output of our run is in
`demo-output.txt`.

To train the vectors yourself (about 15 minutes, 8 processes):

```bash
python3 step5_train.py
```

## The steps

| File | What it builds |
|---|---|
| `get_data.py` | downloads text8 and the analogy questions into `data/` |
| `step1_vectors.py` | words as vectors by hand, and the cosine that compares them |
| `step2_pairs.py` | the vocabulary and the (word, neighbour) pairs from a sliding window |
| `step3_model.py` | two tables, a dot product, a sigmoid, and one update rule |
| `step4_negative.py` | fake pairs drawn with the 3/4 power, and a chunked training step |
| `step5_train.py` | thinning common words, a shrinking step size, 8 workers on shared tables |
| `step6_look.py` | nearest neighbours and vector arithmetic |
| `step7_test.py` | the 19,544-question analogy test from the original release |
| `tests.py` | checks for every step |
| `solutions/window.py` | the exercise: the same model with a window of 2 |

## What our run measured

- Text: 17,005,207 words, 253,854 distinct, 71,290 kept (seen 5 times or more).
- Settings: 100 numbers per word, window 5, 5 fakes per real pair, step 0.025
  shrinking to zero, 10 passes, 877 seconds of training.
- Analogy test, scored the way the original `compute-accuracy` tool is run in its demo
  script (questions with a word outside the 30,000 most common are skipped):
  **5,371 right of 12,268 asked = 43.8 %**; 7,276 skipped.

| Section | Right | Asked | Score |
|---|---:|---:|---:|
| capital-common-countries | 390 | 506 | 77.1 % |
| capital-world | 795 | 1,452 | 54.8 % |
| currency | 49 | 268 | 18.3 % |
| city-in-state | 522 | 1,571 | 33.2 % |
| family | 171 | 306 | 55.9 % |
| gram1-adjective-to-adverb | 139 | 756 | 18.4 % |
| gram2-opposite | 28 | 306 | 9.2 % |
| gram3-comparative | 637 | 1,260 | 50.6 % |
| gram4-superlative | 128 | 506 | 25.3 % |
| gram5-present-participle | 200 | 992 | 20.2 % |
| gram6-nationality-adjective | 1,161 | 1,371 | 84.7 % |
| gram7-past-tense | 394 | 1,332 | 29.6 % |
| gram8-plural | 574 | 992 | 57.9 % |
| gram9-plural-verbs | 183 | 650 | 28.2 % |

The first word2vec paper reports 65.6 % for its best skip-gram model. That is not the
same exam: it used 6 billion words, 1,000 numbers per word, and a different last step
(hierarchical softmax).

## Your turn

Change the window from 5 words to 2 and train again (under 7 minutes). Then run the
test. Some sections go up and some go down. Before you run it, guess: does a narrow
window help with meaning, or with grammar?

One answer: `python3 solutions/window.py`. The window-2 vectors are in
`solutions/vectors_window2.npy`, so it does not need to train.

## Sources

- Mikolov, Chen, Corrado, Dean, *Efficient Estimation of Word Representations in Vector
  Space*, 2013. https://arxiv.org/abs/1301.3781
- Mikolov, Sutskever, Chen, Corrado, Dean, *Distributed Representations of Words and
  Phrases and their Compositionality*, 2013. https://arxiv.org/abs/1310.4546
- The original C code and `questions-words.txt`: https://github.com/tmikolov/word2vec
- text8, the first 100 million characters of a cleaned English Wikipedia dump:
  http://mattmahoney.net/dc/textdata.html

## Licence

MIT. The data files are downloaded from their own sources and are not part of this repo.
