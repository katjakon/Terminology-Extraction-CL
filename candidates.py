# -*- coding: utf-8 -*-
"""
Kandidatenauswahl
"""
import os
import string
import math

from nltk import bigrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import WordPunctTokenizer


class Candidates:

    PUNC = set(string.punctuation)

    def __init__(self, corpus, stops=None, from_dir=True):
        if stops is None:
            stops = []
        self.corpus = corpus
        self.stops = stops
        self.bigrams = FreqDist()
        self.unigrams = FreqDist()
        self.from_dir = from_dir
        self._count()

    @property
    def sum_unigrams(self):
        count = 0
        for token in self.unigrams:
            count += self.unigrams[token]
        return count

    @property
    def sum_bigrams(self):
        count = 0
        for bigram in self.bigrams:
            count += self.bigrams[bigram]
        return count

    def _count(self):
        if self.from_dir:
            txt = [file for file in os.listdir(self.corpus_dir) if file.endswith(".txt")]
            for file in txt:
                with open(os.path.join(self.corpus, file), encoding="utf-8") as file:
                    for line in file:
                        line = WordPunctTokenizer().tokenize(line.rstrip().lower())
                        self.count_bigrams(bigrams(line))
                        self.count_unigrams(line)
        else:
            self.count_bigrams(self.corpus)
            self.count_unigrams(self.corpus)

    def count_bigrams(self, bigram_list):
        for bigram in bigram_list:
            self.bigrams[bigram] += 1

    def count_unigrams(self, uni_list):
        for unigram in uni_list:
            self.unigrams[unigram] += 1

    def get_candidates(self, prop=-15):
        candidates = set()
        sum_bigrams = self.sum_bigrams
        for word_i, word_j in self.bigrams:
            if word_i not in self.stops and word_j not in self.stops:
                if word_i not in self.PUNC and word_j not in self.PUNC:
                    log_bi = math.log(self.bigrams[(word_i, word_j)]/sum_bigrams)
                    if log_bi >= prop:
                        candidates.add((word_i, word_j))
        return candidates

if __name__ == "__main__":
    c = Candidates("acl_texts", stops=stopwords.words("english"))
    d = c.get_candidates()
    print(d)
    print(len(d))