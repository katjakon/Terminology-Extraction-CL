# -*- coding: utf-8 -*-
"""
Preprocess Terminology extraction
"""
from nltk.corpus import reuters

from candidates import Candidates

class Terminology:

    def __init__(self, domain, reference, candidates):
        self.domain = domain
        self.reference = reference
        self.candidates = candidates
        self.domain_relevance = dict()
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
        pass

if __name__ == "__main__":
    can = dict()
    with open("cand.txt") as file:
        for line in file:
            line = line.rstrip().split("\t")
            words = tuple(line[0].split())
            can[words] = int(line[1])
    t = Terminology("acl_texts", reuters.words(), can)
    print("Done")
    t._domain_relevance()