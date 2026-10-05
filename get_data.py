"""Download the two public files this course uses.

text8            100,000,000 characters of cleaned English Wikipedia
questions-words  19,544 analogy questions from the word2vec release
"""
import os
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
TEXT8 = 'http://mattmahoney.net/dc/text8.zip'
QUESTIONS = ('https://raw.githubusercontent.com/tmikolov/word2vec/'
             'master/questions-words.txt')


def fetch(url, path):
    if os.path.exists(path):
        print('have', os.path.basename(path))
        return
    print('downloading', url)
    urllib.request.urlretrieve(url, path)


if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    fetch(QUESTIONS, os.path.join(DATA, 'questions-words.txt'))
    if not os.path.exists(os.path.join(DATA, 'text8')):
        zip_path = os.path.join(DATA, 'text8.zip')
        fetch(TEXT8, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extract('text8', DATA)
        os.remove(zip_path)
    for name in ('text8', 'questions-words.txt'):
        size = os.path.getsize(os.path.join(DATA, name))
        print(f'{name:22s} {size:>11,} bytes')
