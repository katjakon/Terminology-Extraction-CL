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
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


class Preprocess:

    PUNC = set(string.punctuation)

    def __init__(self, corpus, from_dir=True):
        if from_dir:
            corpus = PlaintextCorpusReader(corpus, ".*")
        self.corpus = corpus
        self.bigrams = FreqDist()
        self._count()

    @property
    def sum_bigrams(self):
        return sum(self.bigrams[bigram] for bigram in self.bigrams)

    def _count(self):
        words = [word.lower() for word in self.corpus.words()]
        bigrams_words = bigrams(words)
        for bigram in bigrams_words:
            self.bigrams[bigram] += 1

    def get_candidates(self, prop=-15, stops=None):
        if stops is None:
            stops = []
        candidates = dict()
        sum_bigrams = self.sum_bigrams
        for word_i, word_j in self.bigrams:
            if word_i not in stops and word_j not in stops:
                if word_i not in self.PUNC and word_j not in self.PUNC:
                    log_bi = math.log(self.bigrams[(word_i, word_j)]/sum_bigrams)
                    if log_bi >= prop:
                        candidates[(word_i, word_j)] = self.bigrams[(word_i, word_j)]
        return candidates

    def bigram_freq(self, bigram_list):
        return {bigram: self.bigrams.get(bigram, 0) for bigram in bigram_list}

    def bigrams_to_file(self, bigrams, file_dir="preprocess", name="candidates", ext=".txt"):
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
        in_dir = [file for file in os.listdir(file_dir)]
        count = 1
        file_name = "{}{}{}".format(name, count, ext)
        while file_name in in_dir:
            count += 1
            file_name = "{}{}{}".format(name, count, ext)
        bi_freq = self.bigram_freq(bigrams)
        with open(os.path.join(file_dir, file_name), "w", encoding="utf-8") as file:
            for word_i, word_j in bigrams:
                file.write("{} {}\t{}\n".format(word_i,
                                                word_j,
                                                bi_freq[(word_i, word_j)]))

    def bigrams_from_file(self, fileid):
        assert fileid in self.corpus.fileids(), "File needs to be in instance corpus."
        file_words = [word.lower() for word in self.corpus.words(fileid)]
        bigrams_file = bigrams(file_words)
        return FreqDist(bigrams_file)



if __name__ == "__main__":
    d = Preprocess("acl_texts/")
    print(dict(d.bigrams_from_file("text_paper_1.txt")))