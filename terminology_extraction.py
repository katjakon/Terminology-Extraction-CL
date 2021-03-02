# -*- coding: utf-8 -*-
"""
Preprocess Terminology extraction
"""
from candidates import Candidates

class Terminology:

    def __init__(self, domain, reference, candidates):
        self.domain = domain
        self.reference = reference
        self.candidates = candidates
        self.domain_relevance = dict()

    def _domain_relevance(self):
        d = Candidates(self.domain)
        r = Candidates(self.reference, from_file=False)
        d_sum = d.sum_unigrams
        r_sum = r.sum_unigrams
        for canidate in self.candidates:
            pass