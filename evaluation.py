# -*- coding: utf-8 -*-
# Katja Konermann
# 802658
"""
Evaluation of a set of extracted terms.
"""


class Evaluation:

    DEMO = {"terms": {('machine', 'translation'),
                      ('computational', 'linguistics'),
                      ('use', 'machine')},
            "golds": {('machine', 'translation'),
                      ('speech', 'recognition')}}

    """
    A class that evaluates a set of extracted terms.

    Attributes:
        terms (set): A set of extraced bigrams (two-tuples of strings)
        golds (set): A set of gold standard bigrams (two-tuples of strings)
        correct_terms (set): Intersection of terms and golds.

    Methods:
        precision():
            Number of correct terms divided by number of extracted terms.
        recall():
            Number of correct terms divided by number of gold terms.
        f1():
            Harmonic medium of precision and recall.
    """

    def __init__(self, terms, golds):
        """
        Construct an instance of Evaluation class.

        Args:
            terms:
                Iterable of bigrams (two-tuples of strings) that should
                be evaluated.
            golds:
                Iterable of bigrams (two-tuples of strings) that are
                considered the standard.

        Returns:
            None.
        """
        self.terms = set(terms)
        self.golds = set(golds)
        self.correct_terms = self.terms.intersection(self.golds)

    def precision(self):
        """
        Compute precision by dividing number of correct terms by
        number of extracted terms.

        Returns:
            Precision value (float)

        """
        if len(self.terms) == 0:
            return 0
        return len(self.correct_terms) / len(self.terms)

    def recall(self):
        """
        Compute recall by dividing number of correct terms by
        number of gold standard terms.

        Returns:
            Recall value (float)

        """
        if len(self.golds) == 0:
            return 0
        return len(self.correct_terms) / len(self.golds)

    def f1(self):
        """Returns harmonic medium of recall and precision value.
        """
        prec = self.precision()
        rec = self.recall()
        if not prec and not rec:
            return 0
        return (2 * prec * rec) / (prec + rec)

    @classmethod
    def demo(cls):
        print("\tDemo for class Evaluation\n"
              "For each method, you can see its arguments and output. "
              "For more information use the help function.\n\n"
              "Arguments used for instanciating the class:\n"
              "\tExtracted terms - {}\n"
              "\tGold terms - {}".format(cls.DEMO["terms"], cls.DEMO["golds"]))
        eva = cls(**cls.DEMO)
        print("{:=^90}".format("recall()"))
        print(eva.recall())
        print("{:=^90}".format("precision()"))
        print(eva.precision())
        print("{:=^90}".format("f1()"))
        print(eva.f1())


if __name__ == "__main__":
    Evaluation.demo()
