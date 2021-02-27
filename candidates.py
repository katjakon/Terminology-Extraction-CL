# -*- coding: utf-8 -*-
"""
Kandidatenauswahl
"""
import os

from nltk import bigrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import WordPunctTokenizer


class Candidates:

    def __init__(self, corpus_dir, stops=None):
        self.corpus_dir = corpus_dir
        self.stops = stops
        self.bigrams = FreqDist()
        self.unigrams = FreqDist()

        self._count()

    @property
    def corpus_len(self):
        count = 0
        for token in self.unigrams:
            count += self.unigrams[token]
        return count

    def _count(self):
        txt = [file for file in os.listdir(self.corpus_dir) if file.endswith(".txt")]
        for file in txt:
            with open(os.path.join(self.corpus_dir, file), encoding="utf-8") as file:
                for line in file:
                    line = WordPunctTokenizer().tokenize(line.rstrip().lower())
                    self.count_bigrams(bigrams(line))
                    self.count_unigrams(line)

    def count_bigrams(self, bigram_list):
        for bigram in bigram_list:
            self.bigrams[bigram] += 1

    def count_unigrams(self, uni_list):
        for unigram in uni_list:
            self.unigrams[unigram] += 1

    def get_candidates(self):
        # Only read in txt files
        print(self.bigrams.most_common(100))

if __name__ == "__main__":
    cand = Candidates("acl_texts")
    print(cand.unigrams.most_common(100))