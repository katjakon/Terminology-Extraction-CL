# -*- coding: utf-8 -*-
"""
Preprocess Terminology extraction
"""
import os
import math

from nltk import bigrams
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

from preprocess import Preprocess


class Terminology:

    def __init__(self, domain, reference, candidates=None):
        self.domain = Preprocess(domain)
        self.reference = Preprocess(reference, from_dir=False)

        if candidates is None:
            candidates = self.domain.get_candidates(stops=stopwords.words("english"))
        self.candidates = candidates
        self.domain_relevance = self._domain_relevance()
        self.domain_consensus = self._domain_consensus()

    @staticmethod
    def _prob_in_corpus(freq, freq_sum):
        if freq_sum == 0:
            return 0
        return freq / freq_sum

    def _domain_relevance(self):
        print("Relevance")
        domain_relevance = dict()
        freq_dom = self.domain.bigram_freq(self.candidates)
        freq_ref = self.reference.bigram_freq(self.candidates)
        sum_dom = sum(freq_dom[bigram] for bigram in freq_dom)
        sum_ref = sum(freq_ref[bigram] for bigram in freq_ref)
        for candidate in self.candidates:
            prob_ref = self._prob_in_corpus(freq_ref[candidate], sum_ref)
            prob_dom = self._prob_in_corpus(freq_dom[candidate], sum_dom)
            if prob_dom == 0 and prob_ref == 0:
                term_relevance = 0
            else:
                term_relevance = prob_dom / (prob_dom + prob_ref)
            domain_relevance[candidate] = term_relevance
        return domain_relevance

    def _domain_consensus(self):
        print("Consensus")
        domain_consensus = dict()
        files = {term: dict() for term in self.candidates}
        c = 0
        for file in self.domain.corpus.fileids():
            if c % 1000 == 0:
                print(c)
            cand_freq = self.domain.bigram_freq(self.candidates, file)
            for term in cand_freq:
                files[term][file] = cand_freq[term]
            c += 1
        for term in files:
            sum_files = sum(files[term][file] for file in files[term])
            for file in files[term]:
                files[term][file] = files[term][file] / sum_files
            cons = sum(files[term][doc] * math.log(1/files[term][doc])
                       for doc in files[term])
            domain_consensus[term] = cons
        return domain_consensus

    def extract_terminology(self, alpha=0.1, theta=1):
        term = set()
        for candidate in self.candidates:
            value = (alpha * self.domain_relevance[candidate]
                     + (1-alpha)*self.domain_consensus[candidate])
            if value >= theta:
                term.add(candidate)
        return term

if __name__ == "__main__":
    can = dict()
    with open("preprocess/candidates1.txt") as file:
        for line in file:
            line = line.rstrip().split("\t")
            words = tuple(line[0].split())
            can[words] = int(line[1])
    t = Terminology("acl_texts", reuters, can)
    terms = t.extract_terminology(theta=2.5)
    print(terms)
    print(len(terms))