# -*- coding: utf-8 -*-
"""
Evaluation
"""
import os

from terminology_extraction import Terminology

class Evaluation:

    def __init__(self, term_file, gold_file):
        self.terms = self._read_from_file(term_file)
        self.golds = self._read_from_file(gold_file)

    def _read_from_file(self, file):
        words = set()
        with open(os.path.join(file), encoding="utf-8") as file:
            for line in file:
                line = line.rstrip().split()
                if len(line) == 2:
                    words.add(tuple(line))
        return words

    def precision(self):
        correct = len(self.terms.intersection(self.golds))
        return correct / len(self.terms)

    def recall(self):
        correct = len(self.terms.intersection(self.golds))
        return correct / len(self.golds)

    def f1(self):
        prec = self.precision()
        rec = self.recall()
        return (2 * prec * rec) / (prec + rec)

if __name__ == "__main__":
    eval_terms = Evaluation("extracted_terms.txt", "gold_terminology.txt")
    print(eval_terms.precision())
    print(eval_terms.recall())
    print(eval_terms.f1())