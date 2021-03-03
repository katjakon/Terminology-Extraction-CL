# -*- coding: utf-8 -*-
"""
Preprocess Terminology extraction
"""
import os
import math

from nltk import bigrams
from nltk.corpus import reuters
from nltk.tokenize import WordPunctTokenizer

from candidates import Candidates

class Terminology:

    def __init__(self, domain, reference, candidates):
        self.domain = domain
        self.reference = reference
        self.candidates = candidates
        self.domain_relevance = dict()
        self.domain_consens = dict()
        self.files = {term: dict() for term in self.candidates}
        self.r = Candidates(self.reference, from_dir=False)

    def _domain_relevance(self):
        s = self._count_reference()
        for candidate in self.candidates:
            if s == 0:
                p = 0
            else:
                p = self.r.bigrams.get(candidate, 0) / s
            term_relevance = self.candidates[candidate] / (self.candidates[candidate]
                                                          + p)
            self.domain_relevance[candidate] = term_relevance

    def _count_reference(self):
        s = 0
        for c in self.candidates:
            s += self.r.bigrams.get(c, 0)
        return s

    def _domain_consens(self):
        txt = [file for file in os.listdir(self.domain) if file.endswith(".txt")]
        for index, file in enumerate(txt):
            with open(os.path.join(self.domain, file)) as f:
                for line in f:
                    line = WordPunctTokenizer().tokenize(line.rstrip().lower())
                    line = bigrams(line)
                    match = [bigram for bigram in line if bigram in self.candidates]
                    for bi in match:
                        self.files[bi].setdefault(index, 0)
                        self.files[bi][index] += 1
        for term in self.files:
            s = 0
            for doc in self.files[term]:
                s += self.files[term][doc]
            for doc in self.files[term]:
                self.files[term][doc] = self.files[term][doc]/s
            n = 0
            for doc in self.files[term]:
                n += self.files[term][doc] * math.log(1/self.files[term][doc])
            self.domain_consens[term] = n

    def extract_terminology(self, alpha=0.5, theta=0.7):
        self._domain_consens()
        print("Domain consens")
        self._domain_relevance()
        print("Domain relevance")
        term = set()
        for candidate in self.candidates:
            value = alpha * self.domain_relevance[candidate] + (1-alpha)*self.domain_consens[candidate]
            if value >= 5:
                term.add(candidate)
        return term

if __name__ == "__main__":
    can = dict()
    with open("cand.txt") as file:
        for line in file:
            line = line.rstrip().split("\t")
            words = tuple(line[0].split())
            can[words] = int(line[1])
    t = Terminology("acl_texts", reuters.words(), can)
    terms = t.extract_terminology()
    print(terms)
    print(len(terms))
